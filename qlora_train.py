#!/usr/bin/env python3
"""qlora_train.py -- the standard QLoRA recipe on train.py's data, for comparison with the exact-grid arm.

  qlora_train.py <samples.jsonl> <out_dir> [--epochs E] [--lr LR] [--rank R] [--alpha A] [--batch B]
                 [--max-len N] [--max-steps N]

transformers Trainer + PEFT LoRA on a bitsandbytes NF4 base, loaded from the QAT checkpoint the served
GGUF was built from. Same selection, same sequences and same settings as train.py: LoRA rank and alpha on
q/k/v/o/gate/up/down, AdamW, warmup over a twentieth of the steps then cosine, gradient clipping at 1.0,
the batch as gradient accumulation of single sequences, loss summed over the batch's answer tokens and
divided by their count. Trainer shuffles with its own sampler, so the order differs from train.py's.

Two changes to Trainer's defaults, both to fit 8 GB: the per-layer embedding table (2.35B values) moves to
the CPU after loading, and the loss is computed in chunks as in qat_lora.masked_loss, so the 262,144-word
output layer never holds a whole sequence.

out_dir/lora.pt uses qat_lora's names, so export_gguf.py writes it into the served GGUF the same way as the
exact-grid adapter. The two arms then differ only in the 4-bit weights the adapter trained against.
"""

import argparse
import json
import math
import os
import re
import time

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

import lazy_eval
import qat_lora
import train

KEY = re.compile(r"^base_model\.model\.model\.(layers\.\d+\.\w+\.\w+_proj)\.(lora_[AB])\.default\.weight$")


def qat_name(peft_name):
    """PEFT parameter name -> qat_lora.lora_state name, or None for anything else."""
    m = KEY.match(peft_name)
    return f"{m[1]}.{m[2]}" if m else None


def ple_on_cpu(text_model):
    """Move the per-layer embedding table to the CPU; lookups run there and the rows come back to the GPU."""
    ple = text_model.embed_tokens_per_layer.to("cpu")
    ple.register_forward_pre_hook(lambda mod, args: (args[0].to("cpu"),))
    ple.register_forward_hook(lambda mod, args, out: out.to("cuda"))
    torch.cuda.empty_cache()


def chunked_loss_sum(text_model, lm_head, softcap, input_ids, labels, chunk=256):
    """Summed next-token cross-entropy over labels != -100, logits built chunk by chunk."""
    h = text_model(input_ids=input_ids, use_cache=False).last_hidden_state
    h, tgt = h[:, :-1], labels[:, 1:]
    total = h.new_zeros((), dtype=torch.float32)

    def piece(hc, tc):
        logits = lm_head(hc).float()
        if softcap:
            logits = torch.tanh(logits / softcap) * softcap
        return F.cross_entropy(logits.flatten(0, 1), tc.flatten(), ignore_index=-100, reduction="sum")

    for s in range(0, h.shape[1], chunk):
        tc = tgt[:, s : s + chunk]
        if (tc != -100).any():
            total = total + checkpoint(piece, h[:, s : s + chunk], tc, use_reentrant=False)
    return total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("samples")
    p.add_argument("out")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=float, default=32.0)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--max-len", type=int, default=2048)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=-1, help="stop early (smoke test); no lora.pt is written")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoTokenizer,
        BitsAndBytesConfig,
        Gemma4ForCausalLM,
        Trainer,
        TrainerCallback,
        TrainingArguments,
    )

    tok = AutoTokenizer.from_pretrained(qat_lora.HF_DIR)
    assert tok.convert_tokens_to_ids("<turn|>") == train.TURN_END

    chosen, log = train.select(lazy_eval.load(a.samples))
    with open(os.path.join(a.out, "selection.jsonl"), "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in log)
    data = []
    for r in chosen:
        gen = r["gen_ids"] if r["gen_ids"][-1:] == [train.TURN_END] else [*r["gen_ids"], train.TURN_END]
        if len(r["prompt_ids"]) + len(gen) <= a.max_len:
            data.append({"input_ids": r["prompt_ids"] + gen, "labels": [-100] * len(r["prompt_ids"]) + gen})
    n_gen = [sum(t != -100 for t in d["labels"]) for d in data]
    print(json.dumps({"problems": len(log), "with_right": len(chosen), "train_seqs": len(data),
                      "mean_gen_len": round(sum(n_gen) / max(len(data), 1), 1)}), flush=True)

    torch.manual_seed(a.seed)
    t_load = time.time()
    model = Gemma4ForCausalLM.from_pretrained(
        qat_lora.HF_DIR, dtype=torch.bfloat16, device_map={"": 0},
        key_mapping={r"^model\.language_model\.": "model."},
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                               bnb_4bit_use_double_quant=True,
                                               bnb_4bit_compute_dtype=torch.bfloat16))
    ple_on_cpu(model.model)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=0.0, bias="none",
                                             target_modules=list(qat_lora.TARGETS), task_type="CAUSAL_LM"))
    names = {qat_name(n) for n, p in model.named_parameters() if p.requires_grad}
    assert None not in names, "a trainable parameter outside the LoRA targets"
    print(json.dumps({"load_s": round(time.time() - t_load, 1),
                      "load_peak_mib": round(torch.cuda.max_memory_allocated() / 2**20),
                      "lora_params": sum(p.numel() for p in model.parameters() if p.requires_grad)}), flush=True)

    text_model = model.base_model.model.model
    lm_head = model.base_model.model.lm_head
    softcap = model.config.final_logit_softcapping

    class ChunkedLossTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            total = chunked_loss_sum(text_model, lm_head, softcap, inputs["input_ids"], inputs["labels"])
            return total / num_items_in_batch

    logf = open(os.path.join(a.out, "train.log.jsonl"), "w")  # noqa: SIM115 -- open for the whole run
    t0 = time.time()

    class Log(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kw):
            if not logs or "loss" not in logs:
                return
            rec = {"step": state.global_step, "of": state.max_steps, "epoch": int(state.epoch or 0),
                   "loss": round(logs["loss"], 4), "gnorm": round(float(logs.get("grad_norm", float("nan"))), 3),
                   "lr": logs.get("learning_rate"), "elapsed_s": round(time.time() - t0),
                   "peak_mib": round(torch.cuda.max_memory_allocated() / 2**20)}
            logf.write(json.dumps(rec) + "\n")
            logf.flush()
            if state.global_step % 10 == 0 or state.global_step == state.max_steps:
                print(json.dumps(rec), flush=True)

    steps = a.epochs * math.ceil(len(data) / a.batch)
    args = TrainingArguments(
        output_dir=os.path.join(a.out, "trainer"), num_train_epochs=a.epochs, max_steps=a.max_steps,
        per_device_train_batch_size=1, gradient_accumulation_steps=a.batch, learning_rate=a.lr,
        lr_scheduler_type="cosine", warmup_steps=max(1, steps // 20), weight_decay=0.0, max_grad_norm=1.0,
        optim="adamw_torch", bf16=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}, logging_steps=1, save_strategy="no",
        report_to=[], seed=a.seed, remove_unused_columns=False, dataloader_pin_memory=False)
    trainer = ChunkedLossTrainer(
        model=model, args=args, train_dataset=data, callbacks=[Log()],
        data_collator=lambda b: {k: torch.tensor([b[0][k]]) for k in ("input_ids", "labels")})
    trainer.model_accepts_loss_kwargs = True  # Trainer counts the batch's answer tokens and leaves the division to us
    trainer.train()
    logf.close()
    if a.max_steps > 0:
        return
    lora = {qat_name(n): p.detach().float().cpu() for n, p in model.named_parameters() if p.requires_grad}
    torch.save({"lora": lora, "rank": a.rank, "alpha": a.alpha, "args": vars(a), "base": "nf4"},
               os.path.join(a.out, "lora.pt"))


if __name__ == "__main__":
    main()
