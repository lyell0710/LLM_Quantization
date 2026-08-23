# SPDX-License-Identifier: MIT
"""GPTQ 从零实现(对齐论文 arXiv:2210.17323 与 GPTQ-for-LLaMa 参考行为)。

核心思想:逐列量化权重,每列的量化误差按二阶信息(Hessian H = 2*X*X^T 的
Cholesky 逆)补偿到尚未量化的列上,使层输出误差 ||WX - QX||^2 最小,
而非逐元素最近取整(RTN)。

量化设定与既有 AWQ 实践对齐:per-group(默认 g=128)非对称 INT4,
scale/zero 按输出行×组存储。
"""

import math

import torch


class GroupQuantizer:
    """per-(输出行, 组) 非对称 min-max INT4。"""

    def __init__(self, bits: int = 4):
        self.maxq = 2 ** bits - 1

    def find_params(self, w: torch.Tensor):
        # w: (out, group) —— 用当前(可能已被误差更新的)权重求组参数
        wmin = w.min(dim=1).values.clamp(max=0)
        wmax = w.max(dim=1).values.clamp(min=0)
        wmax = torch.where((wmin == 0) & (wmax == 0), torch.ones_like(wmax), wmax)
        scale = ((wmax - wmin) / self.maxq).clamp(min=1e-8)
        zero = torch.round(-wmin / scale)
        return scale, zero  # each: (out,)

    def quantize(self, w: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor):
        # w: (out,) 单列;返回反量化值(fake quant)与整数码
        q = torch.clamp(torch.round(w / scale) + zero, 0, self.maxq)
        return (q - zero) * scale, q


class GPTQ:
    """单个 Linear 的 GPTQ 状态:H 累积 + 逐列量化。"""

    def __init__(self, layer: torch.nn.Linear, bits: int = 4, group_size: int = 128):
        self.layer = layer
        self.dev = layer.weight.device
        self.rows, self.columns = layer.weight.shape
        self.H = torch.zeros((self.columns, self.columns),
                             device=self.dev, dtype=torch.float32)
        self.nsamples = 0
        self.quantizer = GroupQuantizer(bits)
        self.group_size = group_size

    @torch.no_grad()
    def add_batch(self, inp: torch.Tensor):
        # inp: (..., in) → (tokens, in);H 用移动平均保持数值尺度稳定
        inp = inp.reshape(-1, self.columns).float()
        n = inp.shape[0]
        self.H *= self.nsamples / (self.nsamples + n)
        self.nsamples += n
        inp = inp * math.sqrt(2.0 / self.nsamples)
        self.H += inp.t() @ inp

    @torch.no_grad()
    def quantize(self, percdamp: float = 0.01, blocksize: int = 128,
                 use_hessian: bool = True):
        """use_hessian=False 即 RTN 对照(同一量化网格,无误差补偿)。"""
        W = self.layer.weight.data.clone().float()
        gs = self.group_size

        if use_hessian:
            H = self.H.clone()
            dead = torch.diag(H) == 0
            H[dead, dead] = 1.0
            W[:, dead] = 0.0
            damp = percdamp * torch.mean(torch.diag(H))
            H += torch.eye(self.columns, device=self.dev) * damp
            # Hinv 的上三角 Cholesky 因子:对角元即 [H^-1]_jj^(1/2) 的行主元
            Hinv = torch.linalg.cholesky(H)
            Hinv = torch.cholesky_inverse(Hinv)
            Hinv = torch.linalg.cholesky(Hinv, upper=True)

        # 组边界与误差补偿块边界对齐,组参数才能取自"补偿后"的最新权重
        assert blocksize == gs, "实现约束: blocksize 必须等于 group_size"

        Q = torch.zeros_like(W)
        n_groups = math.ceil(self.columns / gs)
        scales = torch.zeros((self.rows, n_groups), device=self.dev)
        zeros = torch.zeros((self.rows, n_groups), device=self.dev)
        qidx = torch.zeros((self.rows, self.columns), device=self.dev,
                           dtype=torch.uint8)
        total_loss = 0.0

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            if use_hessian:
                Hinv1 = Hinv[i1:i2, i1:i2]

            g = i1 // gs
            # 组参数取自当前(前序块误差已补偿到位的)权重——GPTQ+groups 标准行为
            scales[:, g], zeros[:, g] = self.quantizer.find_params(W1)

            for i in range(i2 - i1):
                w = W1[:, i]
                dq, q = self.quantizer.quantize(w, scales[:, g], zeros[:, g])
                Q1[:, i] = dq
                qidx[:, i1 + i] = q.to(torch.uint8)
                if use_hessian:
                    err = (w - dq) / Hinv1[i, i]
                    W1[:, i:] -= err.unsqueeze(1) * Hinv1[i, i:].unsqueeze(0)
                    Err1[:, i] = err
                    total_loss += ((w - dq) ** 2 / Hinv1[i, i] ** 2).sum().item() / 2

            Q[:, i1:i2] = Q1
            if use_hessian and i2 < self.columns:
                W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]

        self.layer.weight.data = Q.to(self.layer.weight.dtype)
        return {"loss": total_loss, "scales": scales.cpu(),
                "zeros": zeros.cpu(), "qidx": qidx.cpu()}

    def free(self):
        self.H = None
        torch.cuda.empty_cache()
