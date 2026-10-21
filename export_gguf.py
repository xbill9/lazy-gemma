#!/usr/bin/env python3
"""export_gguf.py -- write a trained QAT-LoRA into a copy of the served Q4_0 GGUF.

  export_gguf.py <lora.pt> <out.gguf>

Each trained tensor gets its new integer levels under the base's unchanged fp16 steps, packed as
ggml block_q4_0, written in place at that tensor's offset. Metadata, tensor order and every other
byte of the file are the served file's.
"""

import shutil
import sys

import numpy as np
import torch

import qat_lora


def pack(levels, d):
    """levels int8 [rows, cols] in [-8, 7], d fp16 [rows, cols/32] -> uint8 bytes in block_q4_0 layout."""
    rows, cols = levels.shape
    u = (levels.numpy().astype(np.int16) + 8).astype(np.uint8).reshape(rows, cols // 32, 32)
    out = np.empty((rows, cols // 32, 18), np.uint8)
    out[..., :2] = d.numpy().astype(np.float16).view(np.uint8).reshape(rows, cols // 32, 2)
    out[..., 2:] = u[..., :16] | (u[..., 16:] << 4)
    return out.tobytes()


def main():
    ckpt, out = sys.argv[1], sys.argv[2]
    saved = torch.load(ckpt, map_location="cpu")
    model, _, _, _ = qat_lora.load(rank=saved["rank"], alpha=saved["alpha"], device="cpu")
    res = model.load_state_dict(saved["lora"], strict=False)
    assert not res.unexpected_keys, res.unexpected_keys[:5]
    assert not [k for k in res.missing_keys if "lora_" in k], "adapter is missing LoRA tensors"
    shutil.copyfile(qat_lora.GGUF_PATH, out)
    r = qat_lora.gguf.GGUFReader(qat_lora.GGUF_PATH)
    changed = total = 0
    with open(out, "r+b") as f:
        for t in r.tensors:
            path = qat_lora.module_path(t.name)
            if path is None:
                continue
            mod = model.get_submodule(path)
            if not isinstance(mod, qat_lora.Q4Linear) or not mod.rank:
                continue
            lv = mod.levels().cpu()
            raw = pack(lv, mod.d.cpu())
            assert len(raw) == t.n_bytes, (t.name, len(raw), t.n_bytes)
            f.seek(t.data_offset)
            f.write(raw)
            changed += int((lv != mod.q.cpu()).sum())
            total += lv.numel()
    print(f"levels changed {changed} of {total} in trained tensors -> {out}")


if __name__ == "__main__":
    main()
