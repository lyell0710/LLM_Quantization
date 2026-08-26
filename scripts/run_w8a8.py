# SPDX-License-Identifier: MIT
"""W8A8 赛道端到端:{fp16 | naive | smooth} + wikitext-2 PPL。

naive = 权重 per-输出行对称 INT8 + 激活 per-token 对称 INT8(无迁移);
smooth = 先按 s_j = actmax_j^α / wmax_j^{1-α} 迁移再同样量化。两臂共用
同一 INT8 格式与管线,唯一差异 = 是否迁移(EXP-003（SmoothQuant 从零实现）的控制变量设计)。
PPL 协议与 run_w4a16.py 完全一致(窗 2048/步 1536,同计分 token 集)——
直接 import 其 get_calib/eval_ppl,协议一致性由共用代码保证而非口头约定。

实测锚(EXP-003;0.5B 激活 outlier 温和的语境):naive 12.1227(vs fp16
11.9152,Δ+0.2075);smooth α=0.75 → 12.0221 收回缺口 48%;α=0.5 →
12.0394;α=0.25 → 12.2332 反例更差(迁移不足先伤权重端)。
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
    """一遍 fp16 全模型前向,收齐全部 Linear 的逐输入通道 |x| max。

    面试点:与 W4A16 的 sequential 逐层标定不同——SmoothQuant 的 s 只依赖
    激活统计(静态标定,不含量化误差项),可在原始 fp16 模型上一次收齐再
    统一改装,不存在"量化后激活分布变化需逐层重标"的问题。amax 而非
    分位数:s 要保证迁移后激活侧 absmax 可控,取校准集内最坏值(原论文
    同口径)。
    """
    stats, handles = {}, []
    for li, layer in enumerate(model.model.layers):
        for n in LINEAR_NAMES:
            m = layer.get_submodule(n)
            key = (li, n)
            stats[key] = torch.zeros(m.in_features, device=dev)

            def mk(k):
                # 工厂函数按值绑定 key,防闭包晚绑定(同 run_w4a16)
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
    torch.manual_seed(3407)  # 与 W4A16 同 seed:校准段采样一致,赛道间可对

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float16).to(dev).eval()

    t0 = time.time()
    if args.mode != "fp16":
        # 顺序不可倒:actmax 必须在任何权重被改写前于原始模型上收集
        calib = get_calib(tokenizer)
        stats = collect_actmax(model, calib, dev)
        # -1 作哨兵 → apply_w8a8 内 s≡1(naive 臂与 smooth 臂共用同一改装)
        alpha = args.alpha if args.mode == "smooth" else -1.0
        for li, layer in enumerate(model.model.layers):
            for n in LINEAR_NAMES:
                # 对全部 7 类 Linear 施加迁移(原论文只对 post-LN linears;
                # 实现选择,EXP-003 §2 注明);lm_head 不量化
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
