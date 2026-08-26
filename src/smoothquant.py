# SPDX-License-Identifier: MIT
"""SmoothQuant 从零实现(W8A8 赛道)。

解决什么问题:目标与 W4A16 组不同——激活也要量化(INT8)。LLM 激活有
少数通道的系统性 outlier,per-token 对称 INT8 的 scale 被 absmax 通道独占,
其余通道分辨率被撑爆;权重分布平坦,难度余量充足。

算法一句话:对每个 Linear 做逐输入通道的等价变换
    y = (X·diag(s)^{-1}) · (diag(s)·W^T),  s_j = actmax_j^α / wmax_j^{1-α}
把激活侧难度按 α 迁移到权重侧。迁移后激活通道 j 的 absmax 变为
actmax_j^{1-α}·wmax_j^{1-α},权重侧列 absmax 变为 actmax_j^α·wmax_j^α:
α=0.5 时两侧难度恰好几何均衡,α 越大激活越轻、权重越重——α 是真实的
权衡旋钮,不是"加了就好"(EXP-003（SmoothQuant 从零实现）的 α=0.25 反例臂即为此设)。

接口契约:apply_w8a8(linear, actmax, alpha) 就地把权重改写为
fakeq(W·diag(s)) 并注册 pre-hook 做 fakeq(x/s);actmax 须由调用方在改动
权重前于原始模型上收集(见 scripts/run_w8a8.collect_actmax);alpha<0
为 naive 臂(s≡1)——与 smooth 臂共用同一 INT8 格式与管线,唯一差异 =
是否迁移(EXP-003 控制变量设计)。

性能特征(实测锚,EXP-003;0.5B 激活 outlier 温和的语境):naive W8A8
PPL 12.1227(Δ vs fp16 +0.2075);smooth α=0.75 → 12.0221,收回 naive
缺口 48%;α=0.5 → 12.0394;α=0.25 → 12.2332 反而更差——迁移不足时
权重端先被撑大的 scale 所伤,激活端却没换来足够收益。

评测实现:hook 形式的 fake quant——weight 预先做 per-输出行对称 INT8
(对 W·diag(s));forward pre-hook 把输入替换为 per-token 对称 INT8(对 X/s)。
数学上与"s 折叠进前置 RMSNorm"的零开销部署完全等价(折叠仅是 runtime 优化,
且要求共享输入的 linears 共享 s——见 docs/theory/03)。naive 臂即 s=1。

面试点:粒度为什么是"激活 per-token × 权重 per-输出行"?INT8 GEMM 要求
scale 可从整数乘加中整体提出:y_ij = s_x[i]·s_w[j]·Σ_k qx_ik·qw_jk,即
scale 必须呈行×列外积形状。激活沿 in 维(k 维)做 per-channel 就提不
出来——outlier 只能靠"迁移"消解而不能靠"更细粒度"硬扛,这正是
SmoothQuant 存在的理由。
"""

import torch


@torch.no_grad()
def fakeq_int8_per_token(x: torch.Tensor) -> torch.Tensor:
    # 动态量化:每 token(沿最后一维)一个 scale,推理时在线即得,无需校准。
    # /127 而非 /128:码域取对称的 [-127,127],舍弃 -128 避免正负不对称
    s = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 127.0
    return torch.round(x / s).clamp(-127, 127) * s


@torch.no_grad()
def fakeq_int8_per_row(w: torch.Tensor) -> torch.Tensor:
    # 静态量化:每输出行一个 scale——GEMM 可折叠(外积形状)的最细权重粒度
    s = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 127.0
    return torch.round(w / s).clamp(-127, 127) * s


class ActMaxCollector:
    """actmax 收集器的库形态最小实现(逐输入通道 |x| 的 running max)。

    注:当前脚本 run_w8a8.py 用其内联的 collect_actmax(等价逻辑,便于
    按 (layer, name) 建 key);本类保留为可复用的单层版本。
    """

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
        # wmax 沿 dim=0(跨输出行)取:s 是输入通道方向的量,必须与
        # actmax 同轴;clamp(1e-4) 三处都在防死通道/零权重列把 s 推向
        # 0 或 ∞(等价变换在数学上允许任意正 s,数值上不允许)
        wmax = W.abs().amax(dim=0).clamp(min=1e-4)     # 每输入通道
        s = actmax.clamp(min=1e-4).pow(alpha) / wmax.pow(1.0 - alpha)
        s = s.clamp(min=1e-4)
    else:
        s = torch.ones_like(actmax)
    # 权重半边:先乘 s(接下难度)再量化——此后 weight 里存的是
    # fakeq(W·s);若先量化再乘 s,迁移就发生在网格确定之后,完全无效
    linear.weight.data = fakeq_int8_per_row(W * s.unsqueeze(0)).to(
        linear.weight.dtype)

    # 预先取倒数并降到权重 dtype:每次 forward 只做一次逐元素乘,
    # 不在热路径重复除法/类型转换
    inv_s = (1.0 / s).to(linear.weight.dtype)

    def pre_hook(mod, args):
        # 激活半边:先除 s(x·inv_s)再 per-token 量化。顺序不可换——
        # 量化若在除 s 之前,scale 仍被原 outlier 撑爆,迁移白做
        x = args[0]
        return (fakeq_int8_per_token(x * inv_s),) + args[1:]

    return linear.register_forward_pre_hook(pre_hook)
