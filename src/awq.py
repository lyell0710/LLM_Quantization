# SPDX-License-Identifier: MIT
"""AWQ 从零实现(核心 = per-input-channel best scale 网格搜索)。

解决什么问题:少数输入通道的激活幅值远大于其余通道,这些通道上的权重
量化误差会被激活等比放大——RTN 对所有权重一视同仁,恰恰亏待了少数
"重要权重"。

思想(算法一句话):激活大的输入维上,权重先乘 s 再量化(数值上等价除回),
让"重要权重"占据更细的量化格。搜索目标(与既有实践一致):
    min_s ‖ Q(W·diag(s))·diag(s)^{-1}·X − W·X ‖²,  s = absmean(X)^α, α∈[0,1) 网格。
只用一阶信息(激活幅值统计)+ 输出 MSE 打分:无反传、无 Hessian——这也是
它与 GPTQ 的本质分界(一阶重要性加权 vs 二阶误差补偿)。

数据布局:
    absmean   (in,)  逐输入通道 |x| 的运行均值(重要性代理)
    x_sample  list   ≤mse_cap 条校准 token(half 存),仅用于 α 打分
    scales/zeros/qidx 与 GPTQ 同布局(共用 GroupQuantizer)

接口契约:add_batch() 累积统计;search_and_apply() 就地把 weight 改写为
Q(W·s)/s,返回 {alpha, mse, qidx, scales, zeros, scaled_fq};
search_scale_only() 只返回 s,供 awq_gptq 叠加臂(量化交给 GPTQ)。

性能特征(实测锚,EXP-002):AWQ PPL 13.4127,收回 RTN 损失 31.9%
(GPTQ 61.6%——一阶与二阶的差距即"二阶信息的定价");量化耗时 43 s;
per-layer best-α 中位 0.30、主体 0.15–0.45,两个强 outlier 层顶到
网格上限 0.95。本仓为 per-linear 简化 + 无 clip(EXP-002 §2 诚实标注),
与 AutoAWQ 完整实现(块级 MSE、共享输入组共享 s、weight clip)不同口径。

实现说明(面试点):本仓采用 fake-quant 评测形式——把 Q(W·s)/s 直接写回
weight,激活不动,数学上与"runtime 给激活除 s"等价;因此 q/k/v 可各自持有
独立 s。若要把 s 折叠进前置 RMSNorm 做零开销 runtime(AutoAWQ 的做法),
共享同一输入的 linears 必须共享同一个 s——那是部署折叠的约束,不是算法的。
量化网格与 GPTQ 臂完全一致(GroupQuantizer, INT4 g128 非对称),保证臂间可比。
"""

import torch

from gptq import GroupQuantizer


@torch.no_grad()
def group_fakequant(W: torch.Tensor, group_size: int, quantizer: GroupQuantizer):
    """对 (out,in) 权重做 per-group fake quant,返回 (fq, qidx, scales, zeros)。

    量化本身是纯 RTN(无补偿):AWQ 的全部收益必须只来自 s 预缩放,
    归因才干净;网格与 GPTQ 臂共用同一 GroupQuantizer。
    """
    out, cols = W.shape
    fq = torch.zeros_like(W)
    n_groups = (cols + group_size - 1) // group_size
    scales = torch.zeros(out, n_groups, device=W.device)
    zeros = torch.zeros(out, n_groups, device=W.device)
    qidx = torch.zeros(out, cols, device=W.device, dtype=torch.uint8)
    for g in range(n_groups):
        c0, c1 = g * group_size, min((g + 1) * group_size, cols)
        s, z = quantizer.find_params(W[:, c0:c1])
        scales[:, g], zeros[:, g] = s, z
        q = torch.clamp(torch.round(W[:, c0:c1] / s.unsqueeze(1)) + z.unsqueeze(1),
                        0, quantizer.maxq)
        qidx[:, c0:c1] = q.to(torch.uint8)
        fq[:, c0:c1] = (q - z.unsqueeze(1)) * s.unsqueeze(1)
    return fq, qidx, scales, zeros


class AWQ:
    """单个 Linear:激活 absmean 累积 + best-scale 搜索 + 应用。"""

    def __init__(self, layer: torch.nn.Linear, bits: int = 4,
                 group_size: int = 128, n_grid: int = 20,
                 mse_sample_tokens: int = 4096):
        self.layer = layer
        self.dev = layer.weight.device
        self.columns = layer.weight.shape[1]
        self.absmean = torch.zeros(self.columns, device=self.dev)
        self.ntok = 0
        self.x_sample = []          # 少量校准输入,用于 MSE 打分
        self.mse_cap = mse_sample_tokens
        self.quantizer = GroupQuantizer(bits)
        self.group_size = group_size
        self.n_grid = n_grid        # 20 档 → α 步长 0.05,搜索成本线性于档数

    @torch.no_grad()
    def add_batch(self, inp: torch.Tensor):
        x = inp.reshape(-1, self.columns).float()
        n = x.shape[0]
        # absmean 而非 absmax:量通道的"系统性"幅值,对个别极端 token 稳健
        # (AWQ 论文的重要性代理);运行均值防长校准流的数值漂移
        self.absmean = (self.absmean * self.ntok + x.abs().sum(0)) / (self.ntok + n)
        self.ntok += n
        have = sum(t.shape[0] for t in self.x_sample)
        if have < self.mse_cap:
            # 打分样本封顶 mse_cap=4096 token:MSE 只用来给 α 排序,对样本量
            # 不敏感;封顶控制每档 α 一次 (tok,in)×(in,out) 的打分开销,
            # half 存省显存(打分时再升 float)
            self.x_sample.append(x[: self.mse_cap - have].half())

    @torch.no_grad()
    def _search(self):
        W = self.layer.weight.data.clone().float()
        X = torch.cat(self.x_sample).float()          # (tok, in)
        ref = X @ W.t()                               # fp 权重的参考输出,MSE 的锚
        base = self.absmean.clamp(min=1e-4)           # 防死通道 0^α → s=0 除零

        best = (None, float("inf"), 0.0)
        for i in range(self.n_grid):
            # α 网格 {0, 0.05, …, 0.95}:α=0 即 s≡1(退化为 RTN),网格必含
            # "不保护"选项,搜索结果不会劣于 RTN 起点;α=1 为纯激活幅值
            # 主导(常过度),不在网格内
            alpha = i / self.n_grid
            s = base.pow(alpha)
            # 面试点:几何归一 s/√(smax·smin) 让 log s 关于 0 对称。不归一
            # 时 s 整体 >1(或 <1)会系统性撑大(缩小)每组的 min-max 范围,
            # 各 α 的 MSE 差异混入"网格整体变粗/变细"的混杂因素,α 之间
            # 不再可比——归一后比较的才是"幅值再分配"本身。
            s = s / (s.max() * s.min()).sqrt()        # 几何归一,防网格漂移
            fq, *_ = group_fakequant(W * s.unsqueeze(0), self.group_size,
                                     self.quantizer)
            # 打分对象 = Q(W·s)/s 这一"有效权重":激活不动时它与部署形态
            # (激活除 s)端到端等价,MSE 即真实输出误差
            w_eff = fq / s.unsqueeze(0)
            loss = ((X @ w_eff.t()) - ref).pow(2).mean().item()
            if loss < best[1]:
                best = (s, loss, alpha)
        return W, best

    @torch.no_grad()
    def search_scale_only(self):
        # awq_gptq 叠加臂入口:只要 s,不量化——量化交给 GPTQ 对 W·s 做
        _, (s, loss, alpha) = self._search()
        return {"s": s, "alpha": alpha, "mse": loss}

    @torch.no_grad()
    def search_and_apply(self):
        W, (s, loss, alpha) = self._search()
        fq, qidx, scales, zeros = group_fakequant(
            W * s.unsqueeze(0), self.group_size, self.quantizer)
        # 写回 Q(W·s)/s:fake-quant 评测形式,激活侧零改动(等价性见文件头)
        self.layer.weight.data = (fq / s.unsqueeze(0)).to(self.layer.weight.dtype)
        # scaled_fq = Q(W·s) 本体:qidx/scales/zeros 描述的是它;pack_check
        # 必须对它断言——除回 s 后的有效权重已不在整数网格上,无法逐元素对齐
        return {"alpha": alpha, "mse": loss, "qidx": qidx.cpu(),
                "scales": scales.cpu(), "zeros": zeros.cpu(),
                "scaled_fq": fq}                       # pack 校验对 Q(W·s) 本体

    def free(self):
        # x_sample(≤4096×in half)与统计逐 Linear 释放,防 168 个实例累积
        self.absmean = None
        self.x_sample = []
        torch.cuda.empty_cache()
