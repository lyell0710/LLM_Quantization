# SPDX-License-Identifier: MIT
"""SmoothQuant 从零实现(W8A8 赛道)。

目标与 W4A16 组不同:激活也要量化(INT8)。LLM 激活有少数通道的系统性
outlier,per-token 对称 INT8 会被 absmax 撑爆分辨率;SmoothQuant 用等价变换
    y = (X·diag(s)^{-1}) · (diag(s)·W^T),  s_j = actmax_j^α / wmax_j^{1-α}
把激活侧难度按 α 迁移到权重侧(权重分布平坦,吃得下)。

评测实现:hook 形式的 fake quant——weight 预先做 per-输出行对称 INT8
(对 W·diag(s));forward pre-hook 把输入替换为 per-token 对称 INT8(对 X/s)。
数学上与"s 折叠进前置 RMSNorm"的零开销部署完全等价(折叠仅是 runtime 优化,
且要求共享输入的 linears 共享 s——见 docs/theory/03)。naive 臂即 s=1。
"""

import torch


@torch.no_grad()
def fakeq_int8_per_token(x: torch.Tensor) -> torch.Tensor:
    s = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 127.0
    return torch.round(x / s).clamp(-127, 127) * s


@torch.no_grad()
def fakeq_int8_per_row(w: torch.Tensor) -> torch.Tensor:
    s = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 127.0
    return torch.round(w / s).clamp(-127, 127) * s


class ActMaxCollector:
    def __init__(self, columns: int, dev):
        self.actmax = torch.zeros(columns, device=dev)

    def hook(self, mod, args):
        x = args[0].reshape(-1, x_cols(args[0]))
        self.actmax = torch.maximum(self.actmax, x.abs().amax(dim=0).float())


def x_cols(x):
    return x.shape[-1]


@torch.no_grad()
def apply_w8a8(linear: torch.nn.Linear, actmax: torch.Tensor, alpha: float):
    """给单个 Linear 装上 W8A8 fake quant;alpha<0 表示 naive(s=1)。"""
    W = linear.weight.data.float()
    if alpha >= 0:
        wmax = W.abs().amax(dim=0).clamp(min=1e-4)     # 每输入通道
        s = actmax.clamp(min=1e-4).pow(alpha) / wmax.pow(1.0 - alpha)
        s = s.clamp(min=1e-4)
    else:
        s = torch.ones_like(actmax)
    linear.weight.data = fakeq_int8_per_row(W * s.unsqueeze(0)).to(
        linear.weight.dtype)

    inv_s = (1.0 / s).to(linear.weight.dtype)

    def pre_hook(mod, args):
        x = args[0]
        return (fakeq_int8_per_token(x * inv_s),) + args[1:]

    return linear.register_forward_pre_hook(pre_hook)
