# SPDX-License-Identifier: MIT
"""AWQ 从零实现(核心 = per-input-channel best scale 网格搜索)。

思想:激活大的输入维上,权重量化误差被放大 → 先给这些维的权重乘 s 再量化
(数值上等价除回),让"重要权重"占据更细的量化格。搜索目标(与既有实践一致):
    min_s ‖ Q(W·diag(s))·diag(s)^{-1}·X − W·X ‖²,  s = absmean(X)^α, α∈[0,1] 网格。

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
    """对 (out,in) 权重做 per-group fake quant,返回 (fq, qidx, scales, zeros)。"""
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
        self.n_grid = n_grid

    @torch.no_grad()
    def add_batch(self, inp: torch.Tensor):
        x = inp.reshape(-1, self.columns).float()
        n = x.shape[0]
        self.absmean = (self.absmean * self.ntok + x.abs().sum(0)) / (self.ntok + n)
        self.ntok += n
        have = sum(t.shape[0] for t in self.x_sample)
        if have < self.mse_cap:
            self.x_sample.append(x[: self.mse_cap - have].half())

    @torch.no_grad()
    def _search(self):
        W = self.layer.weight.data.clone().float()
        X = torch.cat(self.x_sample).float()          # (tok, in)
        ref = X @ W.t()
        base = self.absmean.clamp(min=1e-4)

        best = (None, float("inf"), 0.0)
        for i in range(self.n_grid):
            alpha = i / self.n_grid
            s = base.pow(alpha)
            s = s / (s.max() * s.min()).sqrt()        # 几何归一,防网格漂移
            fq, *_ = group_fakequant(W * s.unsqueeze(0), self.group_size,
                                     self.quantizer)
            w_eff = fq / s.unsqueeze(0)
            loss = ((X @ w_eff.t()) - ref).pow(2).mean().item()
            if loss < best[1]:
                best = (s, loss, alpha)
        return W, best

    @torch.no_grad()
    def search_scale_only(self):
        _, (s, loss, alpha) = self._search()
        return {"s": s, "alpha": alpha, "mse": loss}

    @torch.no_grad()
    def search_and_apply(self):
        W, (s, loss, alpha) = self._search()
        fq, qidx, scales, zeros = group_fakequant(
            W * s.unsqueeze(0), self.group_size, self.quantizer)
        self.layer.weight.data = (fq / s.unsqueeze(0)).to(self.layer.weight.dtype)
        return {"alpha": alpha, "mse": loss, "qidx": qidx.cpu(),
                "scales": scales.cpu(), "zeros": zeros.cpu(),
                "scaled_fq": fq}                       # pack 校验对 Q(W·s) 本体

    def free(self):
        self.absmean = None
        self.x_sample = []
        torch.cuda.empty_cache()
