# SPDX-License-Identifier: MIT
"""W8A8 赛道端到端:{fp16 | naive | smooth} + wikitext-2 PPL。

naive = 权重 per-输出行对称 INT8 + 激活 per-token 对称 INT8(无迁移);
smooth = 先按 s_j = actmax_j^α / wmax_j^{1-α} 迁移再同样量化。
PPL 协议与 run_w4a16.py 完全一致(窗 2048/步 1536,同计分 token 集)。
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from smoothquant import apply_w8a8  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_w4a16 import MODEL, LINEAR_NAMES, get_calib, eval_ppl  # noqa: E402


@torch.no_grad()
def collect_actmax(model, calib, dev):
    stats, handles = {}, []
    for li, layer in enumerate(model.model.layers):
        for n in LINEAR_NAMES:
            m = layer.get_submodule(n)
            key = (li, n)
            stats[key] = torch.zeros(m.in_features, device=dev)

            def mk(k):
                def hook(mod, args):
                    x = args[0].reshape(-1, args[0].shape[-1]).float()
                    stats[k] = torch.maximum(stats[k], x.abs().amax(dim=0))
                return hook
            handles.append(m.register_forward_pre_hook(mk(key)))
    for batch in calib:
        model(batch.to(dev))
    for h in handles:
        h.remove()
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["fp16", "naive", "smooth"], required=True)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--out", required=True)
    ap.add_argument("--provenance", default="")
    args = ap.parse_args()
    dev = "cuda:0"
    torch.manual_seed(3407)

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float16).to(dev).eval()

    t0 = time.time()
    if args.mode != "fp16":
        calib = get_calib(tokenizer)
        stats = collect_actmax(model, calib, dev)
        alpha = args.alpha if args.mode == "smooth" else -1.0
        for li, layer in enumerate(model.model.layers):
            for n in LINEAR_NAMES:
                apply_w8a8(layer.get_submodule(n), stats[(li, n)], alpha)
        print(f"w8a8 applied ({args.mode}, alpha={alpha})", flush=True)
    t_quant = time.time() - t0

    ppl, ntok = eval_ppl(model, tokenizer, dev)
    result = {"model": MODEL, "track": "w8a8", "mode": args.mode,
              "alpha": args.alpha if args.mode == "smooth" else None,
              "w_quant": "int8 sym per-out-row", "a_quant": "int8 sym per-token",
              "ppl": round(ppl, 4), "tokens_scored": ntok,
              "window": 2048, "stride": 1536,
              "quant_seconds": round(t_quant, 1)}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"provenance": args.provenance, **result}, f, indent=1)
    print("RESULT", json.dumps(result))


if __name__ == "__main__":
    main()
