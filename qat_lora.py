"""qat_lora.py -- Gemma 4 E2B text model on its exact Q4_0 grid, with LoRA trained through that grid.

Every Q4_0 tensor of the served GGUF is held as int8 levels q in [-8, 7] and one fp16 step d per 32
values, read from the GGUF's own blocks, so the frozen model is the shipped file value for value. A
LoRA delta is trained through the same grid:

    W = d * clamp(q + round(delta / d), -8, 7)

with a straight-through gradient across the rounding. Since the base is on the grid, this equals
rounding (W_base + delta) with the base's steps, which is what export writes back into the GGUF, so
the adapter is learned for the weights that ship. The per-layer embedding table (2.35B values) stays
on the CPU; everything else goes to the GPU. Tensors not stored as Q4_0 (norms, scalars) come from
the QAT safetensors.
"""

import json
import os
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, os.path.expanduser("~/llama.cpp/gguf-py"))
import gguf

HF_DIR = os.path.expanduser(
    "~/.cache/huggingface/hub/models--google--gemma-4-E2B-it-qat-q4_0-unquantized/snapshots/"
    "6befbaca7398925921802abd1f277b495b78b738"
)
GGUF_PATH = os.path.expanduser("~/models/gemma-4-E2B-it-qat-q4_0-exact-v2/gemma-4-E2B-it-q4_0-exact.gguf")
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

# GGUF tensor -> module path under Gemma4TextModel (names from gguf_exact.py, 2026-09-30-exact-gguf-e4b-1650ti)
BLK = {"attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj", "attn_v": "self_attn.v_proj",
       "attn_output": "self_attn.o_proj", "ffn_gate": "mlp.gate_proj", "ffn_up": "mlp.up_proj",
       "ffn_down": "mlp.down_proj", "inp_gate": "per_layer_input_gate", "proj": "per_layer_projection"}
TOP = {"token_embd.weight": "embed_tokens", "per_layer_token_embd.weight": "embed_tokens_per_layer",
       "per_layer_model_proj.weight": "per_layer_model_projection"}


def module_path(gguf_name):
    if gguf_name in TOP:
        return TOP[gguf_name]
    p = gguf_name.split(".")
    if p[0] == "blk" and p[2] in BLK and p[3] == "weight":
        return f"layers.{p[1]}.{BLK[p[2]]}"
    return None


def q4_0_levels(t):
    """GGUF Q4_0 tensor -> (q int8 [rows, cols], d fp16 [rows, cols/32]), ggml block_q4_0 layout."""
    cols, rows = (int(x) for x in t.shape)
    raw = np.asarray(t.data).reshape(rows, cols // 32, 18)
    d = raw[..., :2].copy().view(np.float16)[..., 0]
    qs = raw[..., 2:]
    q = np.concatenate([qs & 0x0F, qs >> 4], axis=-1).astype(np.int8) - 8
    return torch.from_numpy(q.reshape(rows, cols)), torch.from_numpy(d)


def dequant(q, d, dtype):
    rows, cols = q.shape
    return (q.view(rows, cols // 32, 32).to(dtype) * d.to(dtype).unsqueeze(-1)).view(rows, cols)


class Q4Linear(nn.Module):
    def __init__(self, q, d, rank=0, alpha=0.0):
        super().__init__()
        self.register_buffer("q", q)
        self.register_buffer("d", d)
        self.out_features, self.in_features = q.shape
        self.rank = rank
        if rank:
            self.scale = alpha / rank
            self.lora_A = nn.Parameter(torch.empty(rank, self.in_features, dtype=torch.float32))
            nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
            self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank, dtype=torch.float32))

    def weight(self, dtype):
        if not self.rank:
            return dequant(self.q, self.d, dtype)
        rows, cols = self.q.shape
        dd = self.d.float().unsqueeze(-1)
        x = ((self.lora_B @ self.lora_A) * self.scale).view(rows, cols // 32, 32) / dd
        step = x + (torch.round(x) - x).detach()  # straight-through rounding
        lv = torch.clamp(self.q.view(rows, cols // 32, 32).float() + step, -8, 7)
        return (lv * dd).view(rows, cols).to(dtype)

    def levels(self):
        """Integer levels the trained weight takes, for export."""
        if not self.rank:
            return self.q
        with torch.no_grad():
            rows, cols = self.q.shape
            x = ((self.lora_B @ self.lora_A) * self.scale).view(rows, cols // 32, 32) / self.d.float().unsqueeze(-1)
            lv = torch.clamp(self.q.view(rows, cols // 32, 32).float() + torch.round(x), -8, 7)
            return lv.view(rows, cols).to(torch.int8)

    def forward(self, x):
        return F.linear(x, self.weight(x.dtype))


class Q4Embedding(nn.Module):
    """Frozen scaled embedding on the Q4_0 grid; lookups run where the table lives."""

    def __init__(self, q, d, embed_scale, dtype):
        super().__init__()
        self.register_buffer("q", q)
        self.register_buffer("d", d)
        self.embed_scale = embed_scale
        self.dtype = dtype

    def forward(self, ids):
        out_dev = ids.device
        ids = ids.to(self.q.device)
        rows = dequant(self.q[ids.reshape(-1)], self.d[ids.reshape(-1)], torch.float32)
        rows = (rows * self.embed_scale).to(self.dtype)
        return rows.view(*ids.shape, -1).to(out_dev)


def set_module(root, path, mod):
    parent, _, name = path.rpartition(".")
    setattr(root.get_submodule(parent) if parent else root, name, mod)


def load(rank=16, alpha=32.0, device="cuda", dtype=torch.bfloat16, ple_device="cpu"):
    """-> (Gemma4TextModel with Q4 modules, text config, lm_head weight [vocab, hidden] on device)."""
    from safetensors import safe_open
    from transformers import Gemma4Config
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextModel

    with open(os.path.join(HF_DIR, "config.json")) as f:
        cfg = Gemma4Config(**json.load(f)).text_config
    cfg._attn_implementation = "sdpa"
    with torch.device("meta"):
        model = Gemma4TextModel(cfg)

    r = gguf.GGUFReader(GGUF_PATH)
    done = set()
    for t in r.tensors:
        path = module_path(t.name)
        if path is None:
            continue
        assert t.tensor_type == gguf.GGMLQuantizationType.Q4_0, (t.name, t.tensor_type)
        q, d = q4_0_levels(t)
        if path == "embed_tokens":
            mod = Q4Embedding(q, d, cfg.hidden_size**0.5, dtype).to(device)
        elif path == "embed_tokens_per_layer":
            mod = Q4Embedding(q, d, cfg.hidden_size_per_layer_input**0.5, dtype).to(ple_device)
        else:
            lora = rank if path.rsplit(".", 1)[-1] in TARGETS else 0
            mod = Q4Linear(q, d, lora, alpha).to(device)
        set_module(model, path, mod)
        done.add(path)

    # Everything left on meta: norms, scalars, rotary buffers. Materialize, then fill from safetensors.
    rest = {n for n, _ in model.named_parameters() if not n.startswith(tuple(done))}
    rest |= {n for n, b in model.named_buffers() if b.is_meta}
    for mod in model.modules():  # empty storage for what is still on meta only; to_empty() would wipe the Q4 buffers
        for n, t in list(mod._parameters.items()):
            if t is not None and t.is_meta:
                mod._parameters[n] = nn.Parameter(torch.empty_like(t, device=device), requires_grad=t.requires_grad)
        for n, t in list(mod._buffers.items()):
            if t is not None and t.is_meta:
                mod._buffers[n] = torch.empty_like(t, device=device)
    state = {}
    prefix = "model.language_model."
    with safe_open(os.path.join(HF_DIR, "model.safetensors"), "pt") as f:
        for k in f.keys():  # noqa: SIM118 -- safe_open has no __iter__
            if k.startswith(prefix) and k[len(prefix):] in rest:
                state[k[len(prefix):]] = f.get_tensor(k)
    missing = model.load_state_dict(state, strict=False, assign=False)
    for mod in model.modules():  # non-persistent buffers computed in __init__ were lost on meta
        if hasattr(mod, "rope_init_fn") or type(mod).__name__.endswith("RotaryEmbedding"):
            fresh = type(mod)(cfg).to(device)
            for n, b in fresh.named_buffers():
                setattr(mod, n, b)
    still = [k for k in missing.missing_keys if k in rest and not k.endswith(("embed_scale", "inv_freq"))]
    for n, p in model.named_parameters():
        p.requires_grad_("lora_" in n)
        if not p.is_floating_point() or "lora_" in n:
            continue
        p.data = p.data.to(dtype)
    lm_head = model.embed_tokens
    return model, cfg, lm_head, still


def lm_logits(cfg, lm_head, h):
    w = dequant(lm_head.q, lm_head.d, h.dtype)
    logits = F.linear(h, w).float()
    cap = cfg.final_logit_softcapping
    return torch.tanh(logits / cap) * cap if cap else logits


def masked_loss(model, cfg, lm_head, input_ids, labels, chunk=256):
    """Mean next-token cross-entropy over positions with labels != -100, logits built chunk by chunk."""
    h = model(input_ids=input_ids, use_cache=False).last_hidden_state
    h, tgt = h[:, :-1], labels[:, 1:]
    n = (tgt != -100).sum()
    total = h.new_zeros((), dtype=torch.float32)

    def piece(hc, tc):
        return F.cross_entropy(lm_logits(cfg, lm_head, hc).flatten(0, 1), tc.flatten(), ignore_index=-100, reduction="sum")

    for s in range(0, h.shape[1], chunk):
        tc = tgt[:, s : s + chunk]
        if (tc != -100).any():
            total = total + checkpoint(piece, h[:, s : s + chunk], tc, use_reentrant=False)
    return total / n.clamp(min=1)


def lora_state(model):
    return {n: p.detach().cpu() for n, p in model.named_parameters() if "lora_" in n}

