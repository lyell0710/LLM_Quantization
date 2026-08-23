# SPDX-License-Identifier: MIT
"""W4A16 赛道端到端:Qwen2.5-0.5B 五臂 {fp16|rtn|gptq|awq|awq_gptq} + wikitext-2 PPL。

用法:
  python run_w4a16.py --mode {fp16|rtn|gptq|awq|awq_gptq} --out data/raw/EXP-NNN/<name>.json
流程(gptq):校准集 128×2048 → 捕获第 0 层输入 → 逐 decoder 层:
  hook 累积各 Linear 的 H → 逐 Linear 量化 → 用量化后权重重算本层输出作为
  下一层输入(sequential,误差随深度传播的真实设定)。
PPL 协议与 vllm/experiments#EXP-016 同族:窗 2048 / 步 1536,test 全串。
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from gptq import GPTQ  # noqa: E402
from awq import AWQ  # noqa: E402
from quant_linear import pack_check  # noqa: E402

MODEL = "Qwen/Qwen2.5-0.5B"
WINDOW, STRIDE = 2048, 1536
N_CALIB, CALIB_LEN, SEED = 128, 2048, 3407
LINEAR_NAMES = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj",
                "mlp.down_proj"]


def get_calib(tokenizer):
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(r["text"] for r in ds if r["text"].strip())
    ids = tokenizer.encode(text)
    g = torch.Generator().manual_seed(SEED)
    starts = torch.randint(0, len(ids) - CALIB_LEN - 1, (N_CALIB,), generator=g)
    return [torch.tensor(ids[s:s + CALIB_LEN]).unsqueeze(0) for s in starts]


@torch.no_grad()
def capture_layer0_inputs(model, calib, dev):
    """跑到第 0 个 decoder 层即截获 (hidden_states, forward kwargs)。"""
    inps, caches = [], []

    class Catcher(torch.nn.Module):
        def __init__(self, mod):
            super().__init__()
            self.mod = mod

        def forward(self, hidden_states, **kw):
            inps.append(hidden_states.detach())
            caches.append({k: v for k, v in kw.items()})
            raise RuntimeError("__captured__")

    layer0 = model.model.layers[0]
    model.model.layers[0] = Catcher(layer0)
    for batch in calib:
        try:
            model(batch.to(dev))
        except RuntimeError as e:
            if "__captured__" not in str(e):
                raise
    model.model.layers[0] = layer0
    return inps, caches


def _run_block(layer, inps, caches, hooks):
    handles = list(hooks)
    for x, kw in zip(inps, caches):
        layer(x, **kw)
    for h in handles:
        h.remove()


@torch.no_grad()
def quantize_model(model, calib, dev, mode, group_size, log):
    inps, caches = capture_layer0_inputs(model, calib, dev)
    per_layer = []
    for li, layer in enumerate(model.model.layers):
        linears = {n: layer.get_submodule(n) for n in LINEAR_NAMES}

        awq_s = {}
        if mode in ("awq", "awq_gptq"):
            awq = {n: AWQ(m, group_size=group_size) for n, m in linears.items()}
            _run_block(layer, inps, caches, [
                m.register_forward_pre_hook(
                    (lambda a: lambda mod, args: a.add_batch(args[0]))(awq[n]))
                for n, m in linears.items()])
            if mode == "awq":
                for n in LINEAR_NAMES:
                    t0 = time.time()
                    res = awq[n].search_and_apply()
                    err = pack_check(res["scaled_fq"], res["qidx"],
                                     res["scales"], res["zeros"], group_size)
                    assert err < 1e-3, f"pack mismatch {err}"
                    per_layer.append({"layer": li, "name": n,
                                      "alpha": res["alpha"], "mse": res["mse"],
                                      "pack_maxerr": err,
                                      "sec": round(time.time() - t0, 2)})
                    awq[n].free()
            else:
                # 只取 s;量化交给 GPTQ(对 W·s,H 取自 X/s——数学等价拆分)
                for n in LINEAR_NAMES:
                    res = awq[n].search_scale_only()
                    awq_s[n] = res["s"]
                    per_layer.append({"layer": li, "name": n,
                                      "alpha": res["alpha"], "stage": "awq_s"})
                    awq[n].free()

        if mode in ("gptq", "rtn", "awq_gptq"):
            gptq = {n: GPTQ(m, group_size=group_size)
                    for n, m in linears.items()}
            if mode != "rtn":
                def mk(g, n):
                    if n in awq_s:
                        inv = (1.0 / awq_s[n]).half()
                        return lambda mod, args: g.add_batch(args[0] * inv)
                    return lambda mod, args: g.add_batch(args[0])
                _run_block(layer, inps, caches, [
                    m.register_forward_pre_hook(mk(gptq[n], n))
                    for n, m in linears.items()])
            for n in LINEAR_NAMES:
                t0 = time.time()
                if n in awq_s:
                    linears[n].weight.data = (
                        linears[n].weight.data.float()
                        * awq_s[n].unsqueeze(0)).half()
                res = gptq[n].quantize(use_hessian=(mode != "rtn"))
                err = pack_check(linears[n].weight.data, res["qidx"],
                                 res["scales"], res["zeros"], group_size)
                wmax = float(linears[n].weight.data.abs().max())
                tol = max(1e-3, wmax * 2 ** -10)  # fp16 相对 ulp,幅值感知
                assert err < tol, f"pack mismatch {err} (tol {tol})"
                if n in awq_s:
                    linears[n].weight.data = (
                        linears[n].weight.data.float()
                        / awq_s[n].unsqueeze(0)).half()
                per_layer.append({"layer": li, "name": n, "loss": res["loss"],
                                  "pack_maxerr": err,
                                  "sec": round(time.time() - t0, 2)})
                gptq[n].free()

        new_inps = []
        for x, kw in zip(inps, caches):
            o = layer(x, **kw)
            if isinstance(o, tuple):
                o = o[0]
            new_inps.append(o.detach())
        inps = new_inps
        log(f"layer {li} done")
    return per_layer


@torch.no_grad()
def eval_ppl(model, tokenizer, dev):
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    ids = tokenizer.encode("\n\n".join(r["text"] for r in ds if r["text"].strip()))
    nll, count, pos = 0.0, 0, 0
    while pos + WINDOW <= len(ids):
        window = torch.tensor(ids[pos:pos + WINDOW]).unsqueeze(0).to(dev)
        logits = model(window).logits.float()
        score_from = 0 if pos == 0 else WINDOW - STRIDE
        lp = torch.log_softmax(logits[0, :-1], dim=-1)
        tgt = window[0, 1:]
        tok_nll = -lp.gather(1, tgt.unsqueeze(1)).squeeze(1)
        nll += tok_nll[score_from:].sum().item()
        count += tok_nll[score_from:].numel()
        pos += STRIDE
    return math_exp(nll / count), count


def math_exp(x):
    import math
    return math.exp(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["fp16", "rtn", "gptq", "awq",
                                       "awq_gptq"], required=True)
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--out", required=True)
    ap.add_argument("--provenance", default="")
    args = ap.parse_args()
    dev = "cuda:0"
    torch.manual_seed(SEED)

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float16).to(dev).eval()

    t0 = time.time()
    per_layer = []
    if args.mode != "fp16":
        calib = get_calib(tokenizer) if args.mode != "rtn" else []
        per_layer = quantize_model(model, calib, dev, mode=args.mode,
                                   group_size=args.group_size,
                                   log=lambda s: print(s, flush=True))
    t_quant = time.time() - t0

    ppl, ntok = eval_ppl(model, tokenizer, dev)
    result = {"model": MODEL, "mode": args.mode, "bits": 4,
              "group_size": args.group_size, "sym": False,
              "ppl": round(ppl, 4), "tokens_scored": ntok,
              "window": WINDOW, "stride": STRIDE,
              "calib": {"set": "wikitext2-train", "n": N_CALIB,
                        "len": CALIB_LEN, "seed": SEED}
              if args.mode not in ("fp16", "rtn") else None,
              "quant_seconds": round(t_quant, 1),
              "sum_gptq_loss": round(sum(p.get("loss", 0) for p in per_layer), 2),
              "max_pack_err": max((p["pack_maxerr"] for p in per_layer
                                   if "pack_maxerr" in p), default=0)}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        if args.provenance:
            pass
        json.dump({"provenance": args.provenance, **result,
                   "per_layer": per_layer}, f, indent=1)
    print("RESULT", json.dumps(result))


if __name__ == "__main__":
    main()
