# SPDX-License-Identifier: MIT
"""real quant 存储与前向:INT4 码按 uint8 两枚一字节打包,forward 反量化。

正确性契约:dequant(pack(qidx)) 与 GPTQ 输出的 fake-quant 权重逐元素相等
(同 scale/zero 下反量化是确定映射),由 pack_check 验证。
"""

import torch


class QuantLinear4(torch.nn.Module):
    def __init__(self, qidx: torch.Tensor, scales: torch.Tensor,
                 zeros: torch.Tensor, bias, group_size: int = 128):
        super().__init__()
        out, cols = qidx.shape
        assert cols % 2 == 0
        packed = (qidx[:, 0::2] | (qidx[:, 1::2] << 4)).contiguous()
        self.register_buffer("qweight", packed)          # (out, cols/2) uint8
        self.register_buffer("scales", scales.half())    # (out, n_groups)
        self.register_buffer("zeros", zeros.half())
        self.bias = bias
        self.group_size = group_size
        self.out_features, self.in_features = out, cols

    def dequant(self) -> torch.Tensor:
        lo = self.qweight & 0xF
        hi = self.qweight >> 4
        q = torch.empty(self.out_features, self.in_features,
                        device=self.qweight.device, dtype=torch.uint8)
        q[:, 0::2], q[:, 1::2] = lo, hi
        g = torch.arange(self.in_features, device=q.device) // self.group_size
        return ((q.float() - self.zeros.float()[:, g])
                * self.scales.float()[:, g])

    def forward(self, x):
        w = self.dequant().to(x.dtype)
        return torch.nn.functional.linear(x, w, self.bias)


@torch.no_grad()
def pack_check(fake_weight: torch.Tensor, qidx, scales, zeros,
               group_size: int = 128) -> float:
    """打包→反量化 与 fake-quant 权重的最大绝对误差(应为 float32 舍入级)。"""
    ql = QuantLinear4(qidx.to(fake_weight.device), scales.to(fake_weight.device),
                      zeros.to(fake_weight.device), None, group_size)
    return (ql.dequant() - fake_weight.float()).abs().max().item()
