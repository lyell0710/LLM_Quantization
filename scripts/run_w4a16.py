# SPDX-License-Identifier: MIT
"""W4A16 赛道端到端:Qwen2.5-0.5B 五臂 {fp16|rtn|gptq|awq|awq_gptq} + wikitext-2 PPL。

解决什么问题:GPTQ/AWQ 论文各用各的模型与评测口径,数字互不可比;本脚本
把五臂放进同一模型、同一量化网格(共用 GroupQuantizer)、同一 PPL 协议,
使臂间差异可完全归因于机制本身(控制变量:RTN↔GPTQ 唯一差异=补偿开关,
AWQ↔RTN 唯一差异=s 预缩放)。

用法:
  python run_w4a16.py --mode {fp16|rtn|gptq|awq|awq_gptq} --out data/raw/EXP-NNN/<name>.json
流程(gptq):校准集 128×2048 → 捕获第 0 层输入 → 逐 decoder 层:
  hook 累积各 Linear 的 H → 逐 Linear 量化 → 用量化后权重重算本层输出作为
  下一层输入(sequential,误差随深度传播的真实设定)。
PPL 协议与 vllm/experiments#EXP-016（D4 FP8 vs W4A16 同卡对比）同族:窗 2048 / 步 1536,test 全串;
各臂同 298302 计分 token,确定性 greedy scoring,单轮可逐位复算。

实测锚(EXP-001（GPTQ 从零实现）(EXP-002（AWQ 从零实现 + AWQ×GPTQ 叠加）臂经幅值感知容差,最大 pack 误差 1.22e-3)):fp16 11.9152 / RTN 14.1154 / AWQ 13.4127 /
GPTQ 12.7600 / AWQ+GPTQ 12.7376——GPTQ 收回 RTN 损失 61.6%、AWQ 31.9%、
叠加 62.6%(vs 单独 GPTQ 仅 +1.0pp,0.5B 上两机制高度重叠);量化耗时
30/43/135/258 s,pack↔fake 最大误差 ≤7.3e-4。
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
# 窗 2048 = 模型训练上下文量级;步 1536 → 相邻窗重叠 512 token 只作上文
# 不计分,每个 token 恰计分一次(协议同 vllm/experiments#EXP-016 家族)
WINDOW, STRIDE = 2048, 1536
# 128 段×2048 token ≈ 26 万校准 token(GPTQ 论文量级);seed 固定使
# 校准采样可复现——PPL 才能逐位复算
N_CALIB, CALIB_LEN, SEED = 128, 2048, 3407
# 量化范围 = 每 decoder 层 7 个 Linear;lm_head 不量化(惯例,EXP-001 §2)
LINEAR_NAMES = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj",
                "mlp.down_proj"]


def get_calib(tokenizer):
    # 校准取 train 全串随机段(与 test 评测集无重叠),定长 2048 保证
    # 每段都跑满注意力窗口
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(r["text"] for r in ds if r["text"].strip())
    ids = tokenizer.encode(text)
    g = torch.Generator().manual_seed(SEED)
    starts = torch.randint(0, len(ids) - CALIB_LEN - 1, (N_CALIB,), generator=g)
    return [torch.tensor(ids[s:s + CALIB_LEN]).unsqueeze(0) for s in starts]


@torch.no_grad()
def capture_layer0_inputs(model, calib, dev):
    """跑到第 0 个 decoder 层即截获 (hidden_states, forward kwargs)。

    面试点:用"抛哨兵异常"截断 forward,而非跑完整模型——逐层量化只需要
    第 0 层的输入(embedding 输出 + rotary/mask 等 kwargs),后续 23 层的
    计算纯属浪费;kwargs 原样缓存,之后每层重放时按批回填。
    """
    inps, caches = [], []

    class Catcher(torch.nn.Module):
        def __init__(self, mod):
            super().__init__()
            self.mod = mod

        def forward(self, hidden_states, **kw):
            inps.append(hidden_states.detach())
            caches.append({k: v for k, v in kw.items()})
            raise RuntimeError("__captured__")  # 截获即中止,省掉后续层前向

    layer0 = model.model.layers[0]
    model.model.layers[0] = Catcher(layer0)
    for batch in calib:
        try:
            model(batch.to(dev))
        except RuntimeError as e:
            if "__captured__" not in str(e):
                raise  # 只吞哨兵,真实错误照常抛出
    model.model.layers[0] = layer0
    return inps, caches


def _run_block(layer, inps, caches, hooks):
    # 把整套校准批重放过单个 decoder 层,统计由挂上的 pre-hook 收集;
    # 结束立即摘 hook——防泄漏到同层的下一臂/下一阶段重放
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

        # ── 阶段 A:AWQ 统计与 best-scale(awq / awq_gptq 臂)──
        awq_s = {}
        if mode in ("awq", "awq_gptq"):
            awq = {n: AWQ(m, group_size=group_size) for n, m in linears.items()}
            # (lambda a: ...)(awq[n]):立即调用外层 lambda 按值绑定当前
            # awq[n]——Python 闭包晚绑定会让 7 个 hook 全指向最后一个 n
            _run_block(layer, inps, caches, [
                m.register_forward_pre_hook(
                    (lambda a: lambda mod, args: a.add_batch(args[0]))(awq[n]))
                for n, m in linears.items()])
            if mode == "awq":
                for n in LINEAR_NAMES:
                    t0 = time.time()
                    res = awq[n].search_and_apply()
                    # pack 断言对 Q(W·s) 本体(scaled_fq)——除回 s 的有效
                    # 权重不在整数网格上,无法逐元素对齐
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
                # 面试点:叠加臂的恒等拆分 (W·s)·(X/s) ≡ W·X——权重半边
                # W·s 进 GPTQ 量化,激活半边 X/s 进 H 统计,最终权重 =
                # Q_gptq(W·s)/s;两机制正交可叠加的原因即在此拆分不改数学
                for n in LINEAR_NAMES:
                    res = awq[n].search_scale_only()
                    awq_s[n] = res["s"]
                    per_layer.append({"layer": li, "name": n,
                                      "alpha": res["alpha"], "stage": "awq_s"})
                    awq[n].free()

        # ── 阶段 B:GPTQ / RTN 量化(gptq / rtn / awq_gptq 臂)──
        if mode in ("gptq", "rtn", "awq_gptq"):
            gptq = {n: GPTQ(m, group_size=group_size)
                    for n, m in linears.items()}
            if mode != "rtn":
                # RTN 臂不重放、不挂 hook(不需要 H)——但仍走同一段量化
                # 代码(use_hessian=False),对照臂共享一切、唯一差异=开关
                def mk(g, n):
                    # 工厂函数按值绑定 (g, n),同上防闭包晚绑定;
                    # awq_gptq 时 hook 现场把激活换到 X/s 域再喂 H
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
                    # 临时切到 W·s 域做 GPTQ:qidx/scale/zero 描述的是该域
                    linears[n].weight.data = (
                        linears[n].weight.data.float()
                        * awq_s[n].unsqueeze(0)).half()
                res = gptq[n].quantize(use_hessian=(mode != "rtn"))
                err = pack_check(linears[n].weight.data, res["qidx"],
                                 res["scales"], res["zeros"], group_size)
                # 容差幅值感知:fp16 尾数 10 位,权重经 fp16 存储的舍入
                # ~|w|·2^-10;W·s 域幅值可远超 1,固定 1e-3 会误报
                wmax = float(linears[n].weight.data.abs().max())
                tol = max(1e-3, wmax * 2 ** -10)  # fp16 相对 ulp,幅值感知
                assert err < tol, f"pack mismatch {err} (tol {tol})"
                if n in awq_s:
                    # 量化完除回 s:评测态权重 = Q(W·s)/s,激活侧零改动
                    linears[n].weight.data = (
                        linears[n].weight.data.float()
                        / awq_s[n].unsqueeze(0)).half()
                per_layer.append({"layer": li, "name": n, "loss": res["loss"],
                                  "pack_maxerr": err,
                                  "sec": round(time.time() - t0, 2)})
                gptq[n].free()

        # ── 阶段 C:sequential 前推 ──
        # 面试点:下一层的输入用"本层量化后"的输出重算,而非 fp16 激活
        # ——下层的 H/统计见到的是前序误差已累积的真实分布(部署时即如此);
        # 用 fp16 激活校准会系统性高估深层的量化质量
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
    # 尾部不足一窗即停:丢掉尾巴保证各臂计分 token 集完全一致(298302)
    while pos + WINDOW <= len(ids):
        window = torch.tensor(ids[pos:pos + WINDOW]).unsqueeze(0).to(dev)
        logits = model(window).logits.float()  # fp32 做 log_softmax,防 fp16 下溢
        # 滑窗去重:首窗全计分;其后每窗前 WINDOW−STRIDE=512 token 已在
        # 上一窗计过,只作上下文——每 token 恰计一次且至少带 512 token 上文
        score_from = 0 if pos == 0 else WINDOW - STRIDE
        # 移位对齐:位置 t 的 logits 预测 token t+1,故 logits 去尾一位、
        # target 去头一位(首 token 无人预测,不计分)
        lp = torch.log_softmax(logits[0, :-1], dim=-1)
        tgt = window[0, 1:]
        tok_nll = -lp.gather(1, tgt.unsqueeze(1)).squeeze(1)
        nll += tok_nll[score_from:].sum().item()
        count += tok_nll[score_from:].numel()
        pos += STRIDE
    return math_exp(nll / count), count


def math_exp(x):
    # 薄包装:math 仅此一处使用,函数内 import 保持模块顶栏干净
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
        # RTN 不用校准集(无统计可收集),给空表仍走同一 quantize_model
        # 代码路径——对照臂的代码共享由此保证
        calib = get_calib(tokenizer) if args.mode != "rtn" else []
        per_layer = quantize_model(model, calib, dev, mode=args.mode,
                                   group_size=args.group_size,
                                   log=lambda s: print(s, flush=True))
    t_quant = time.time() - t0

    ppl, ntok = eval_ppl(model, tokenizer, dev)
    # sum_gptq_loss 是量纲未标定的诊断值(只作回归监控,不进对外表格,
    # LEDGER 红线);max_pack_err 是 real-quant 闭环的全局最坏值
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
            pass  # 历史残留的空分支,保持原样;provenance 恒作 JSON 首字段写入
        json.dump({"provenance": args.provenance, **result,
                   "per_layer": per_layer}, f, indent=1)
    print("RESULT", json.dumps(result))


if __name__ == "__main__":
    main()
