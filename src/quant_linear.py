# SPDX-License-Identifier: MIT
"""real quant 存储与前向:INT4 码按 uint8 两枚一字节打包,forward 反量化。

解决什么问题:fake quant 只证明"数值网格受得了",不证明"码真的能紧凑
存下并原样取回";本文件把 GPTQ/AWQ 产出的 INT4 码落到每字节两码的真实
存储,再从存储态反量化做 forward,闭合 real-quant 链路。

数据布局(ASCII):
    qidx 列:   c0   c1   c2   c3  ...        每列一个 4-bit 码(0..15)
    packed:   [c1<<4|c0] [c3<<4|c2] ...       偶列→低 nibble,奇列→高 nibble
    qweight (out, in/2) uint8;scales/zeros (out, n_groups) half;
    列 c 的组号 g = c // group_size(g=128)。

接口契约(正确性):dequant(pack(qidx)) 与 GPTQ 输出的 fake-quant 权重
逐元素相等(同 scale/zero 下反量化是确定映射),由 pack_check 验证。
实测锚(EXP-001):全部 168 个 Linear 断言通过,最大误差 ≤7.3e-4(EXP-001 口径;EXP-002 臂在幅值感知容差下最大 1.22e-3),
非零部分全部来自 scale/zero 以 half 存储的 fp16 舍入,而非打包本身。

性能特征:存储 4.25 bit/权重(码 4 bit + fp16 scale/zero 摊到 g=128);
forward 为"先反量化、再 fp GEMM"的教学实现,只为正确性闭环,不追求
速度——生产内核(如 Marlin)做 fused dequant-GEMM,见
vllm/experiments#EXP-016 的部署侧。
"""

import torch


class QuantLinear4(torch.nn.Module):
    def __init__(self, qidx: torch.Tensor, scales: torch.Tensor,
                 zeros: torch.Tensor, bias, group_size: int = 128):
        super().__init__()
        out, cols = qidx.shape
        assert cols % 2 == 0  # 两码一字节的配对前提(本仓各层 in 皆 128 倍数)
        # 偶列进低 4 位、奇列进高 4 位:与 dequant 的 0::2 / 1::2 切片互为逆
        packed = (qidx[:, 0::2] | (qidx[:, 1::2] << 4)).contiguous()
        self.register_buffer("qweight", packed)          # (out, cols/2) uint8
        # scale/zero 存 half:模拟真实部署的存储 dtype;pack_check 的非零
        # 误差(~1e-4 量级)即源于这次降精度,见 run_w4a16 的幅值感知容差
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
        q[:, 0::2], q[:, 1::2] = lo, hi   # 交错写回原列序,与打包严格互逆
        # g[c] = c // group_size:每列查自己的组参数;scales[:, g] 是花式索引
        # gather 出 (out, in) 全尺寸矩阵——组粒度反量化的向量化写法
        g = torch.arange(self.in_features, device=q.device) // self.group_size
        # 面试点:(q−zero)·scale 在 float32 域算而非 half——若在 fp16 域,
        # 乘法舍入会叠进误差,pack_check 就无法把残差干净地归因于
        # "scale/zero 的存储精度"这一单一来源
        return ((q.float() - self.zeros.float()[:, g])
                * self.scales.float()[:, g])

    def forward(self, x):
        w = self.dequant().to(x.dtype)
        return torch.nn.functional.linear(x, w, self.bias)


@torch.no_grad()
def pack_check(fake_weight: torch.Tensor, qidx, scales, zeros,
               group_size: int = 128) -> float:
    """打包→反量化 与 fake-quant 权重的最大绝对误差(应为 float32 舍入级)。

    逐元素断言为何成立:给定同一 (q, zero, scale),反量化是确定映射,
    打包/解包只搬比特不碰数值;残差仅可能来自 scale/zero 的 half 存储与
    fake_weight 的 fp16 写回,皆为 ulp 级。误差一旦超出该量级,必是打包
    位序或组索引的 bug——断言即抓(gather 无原子、无竞态,纯确定性)。
    """
    ql = QuantLinear4(qidx.to(fake_weight.device), scales.to(fake_weight.device),
                      zeros.to(fake_weight.device), None, group_size)
    return (ql.dequant() - fake_weight.float()).abs().max().item()
