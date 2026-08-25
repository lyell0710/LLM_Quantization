# SPDX-License-Identifier: MIT
"""GPTQ 从零实现(对齐论文 arXiv:2210.17323 与 GPTQ-for-LLaMa 参考行为)。

解决什么问题:逐元素最近取整(RTN)只最小化权重误差 ‖W−Q‖²,而推理在乎
的是层输出误差 ‖WX−QX‖²——激活分布高度各向异性时两个目标差距巨大。

算法一句话:固定列序逐列量化,每列的量化误差按二阶信息(Hessian
H = 2·X·Xᵀ 的 Cholesky 逆)补偿到"尚未量化"的列上,即 OBQ 的
固定序 + Cholesky 一次分解版本。

数据布局:
    W / Q            (out, in)      float32 工作副本;结果写回 layer.weight
    H                (in, in)       float32,校准期运行均值累积
    scales / zeros   (out, n_groups) per-(输出行, 组) 非对称参数,g=128
    qidx             (out, in) uint8 INT4 码(0..15),供 real-quant 打包链路

接口契约:add_batch() 可多次调用(H 为运行均值,顺序无关);quantize()
要求 blocksize == group_size(原因见断言处),就地把 layer.weight 改写为
fake-quant 权重,返回 {loss, scales, zeros, qidx};use_hessian=False 即
RTN 对照臂——与 GPTQ 臂共用同一量化网格,唯一差异 = 补偿开关,恢复量
可完全归因于二阶补偿(EXP-001 的实验设计本身)。

性能特征(实测锚,EXP-001):Qwen2.5-0.5B,24 层×7 Linear,INT4-g128:
GPTQ PPL 12.7600 / RTN 14.1154 / fp16 11.9152——二阶补偿收回 RTN 质量
损失 61.6%;量化耗时 135 s(RTN 30 s);pack↔fake 最大误差 5.5e-4。

量化设定与既有 AWQ 实践对齐:per-group(默认 g=128)非对称 INT4,
scale/zero 按输出行×组存储。
"""

import math

import torch


class GroupQuantizer:
    """per-(输出行, 组) 非对称 min-max INT4。

    非对称(带 zero point):权重组的 min/max 通常不关于 0 对称,对称量化
    会浪费半边码域。min-max 而非 MSE 搜网格:网格保持最简,RTN/GPTQ/AWQ
    三臂共用同一网格,臂间差异才可归因于各自机制而非网格差异。
    """

    def __init__(self, bits: int = 4):
        self.maxq = 2 ** bits - 1  # INT4 → 15:码域 [0, maxq] 共 2^bits 档

    def find_params(self, w: torch.Tensor):
        # w: (out, group) —— 用当前(可能已被误差更新的)权重求组参数
        # clamp 强制 0 ∈ [wmin, wmax]:保证实数 0 可被精确表示,且
        # zero = round(-wmin/scale) 落在码域内(全正/全负组若不 clamp 会越界)
        wmin = w.min(dim=1).values.clamp(max=0)
        wmax = w.max(dim=1).values.clamp(min=0)
        # 全零组:给一个无害网格(wmax=1),防 scale=0 → 除零/NaN
        wmax = torch.where((wmin == 0) & (wmax == 0), torch.ones_like(wmax), wmax)
        scale = ((wmax - wmin) / self.maxq).clamp(min=1e-8)
        zero = torch.round(-wmin / scale)
        return scale, zero  # each: (out,)

    def quantize(self, w: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor):
        # w: (out,) 单列;返回反量化值(fake quant)与整数码
        # 两者都要:dq 留在 float 域继续参与误差补偿,q 供 real-quant 打包
        q = torch.clamp(torch.round(w / scale) + zero, 0, self.maxq)
        return (q - zero) * scale, q


class GPTQ:
    """单个 Linear 的 GPTQ 状态:H 累积 + 逐列量化。"""

    def __init__(self, layer: torch.nn.Linear, bits: int = 4, group_size: int = 128):
        self.layer = layer
        self.dev = layer.weight.device
        self.rows, self.columns = layer.weight.shape
        # H 用 float32:半精度下 26 万 token 的外积累加会明显丢位,
        # 且 Cholesky 对条件数敏感
        self.H = torch.zeros((self.columns, self.columns),
                             device=self.dev, dtype=torch.float32)
        self.nsamples = 0
        self.quantizer = GroupQuantizer(bits)
        self.group_size = group_size

    @torch.no_grad()
    def add_batch(self, inp: torch.Tensor):
        # inp: (..., in) → (tokens, in);H 用移动平均保持数值尺度稳定:
        # 维护 H = (2/N)·Σ xxᵀ 而非裸 Σ——裸累加的量级随 token 数线性增长,
        # fp32 会"大数吃小数"。先把旧 H 缩到新总数占比,再加上按
        # sqrt(2/N) 预缩放的新批,恒等于全量均值(与到达顺序无关)。
        # 注:补偿量对 H 的整体缩放不变(H→cH ⇒ err×√c、Hinv 行×1/√c,
        # 乘积不变),故此归一只影响数值稳定与 percdamp 的相对含义,不改结果。
        inp = inp.reshape(-1, self.columns).float()
        n = inp.shape[0]
        self.H *= self.nsamples / (self.nsamples + n)
        self.nsamples += n
        inp = inp * math.sqrt(2.0 / self.nsamples)  # 2 来自 ∂²‖WX−QX‖²/∂W²
        self.H += inp.t() @ inp

    @torch.no_grad()
    def quantize(self, percdamp: float = 0.01, blocksize: int = 128,
                 use_hessian: bool = True):
        """use_hessian=False 即 RTN 对照(同一量化网格,无误差补偿)。"""
        W = self.layer.weight.data.clone().float()
        gs = self.group_size

        if use_hessian:
            # ── 阶段 1:H 预处理(死列 → 阻尼 → 三步分解)──
            H = self.H.clone()
            # 死列:校准集中恒为 0 的输入通道,H 行列全 0,无二阶信息可用;
            # 对角置 1 保证可分解,权重清零(该通道输入恒 0,清零不影响
            # 校准分布上的输出,却让量化码省出码域)
            dead = torch.diag(H) == 0
            H[dead, dead] = 1.0
            W[:, dead] = 0.0
            # 阻尼:校准样本有限/通道强相关时 H 近奇异,H⁻¹ 在弱激发方向
            # 爆炸 → 补偿量爆炸反而毁掉权重。加 1% 平均对角的 ridge 保证
            # 正定可 Cholesky,同时抑制病态方向的过度补偿;取"相对平均
            # 对角"而非绝对常数,对 H 的整体量级自适应(percdamp=0.01 为
            # GPTQ 论文默认,EXP-001 同设)。
            damp = percdamp * torch.mean(torch.diag(H))
            H += torch.eye(self.columns, device=self.dev) * damp
            # 面试点:为什么"H⁻¹ 的上三角 Cholesky 因子"一次分解就够?
            # OBQ 每量化一列要用"剩余未量化列"子矩阵的逆 [H_F⁻¹],朴素
            # 做法是每列重求一次逆(O(n·n³))。固定左→右列序后,"删去已
            # 量化列"恰是对 H⁻¹ 依次取 Schur 补,而 Cholesky 分解本身就是
            # 逐行 Schur 补的过程:对 H⁻¹ 做一次上三角分解(O(n³)),其第
            # i 行就同时给出 U[i,i] = √([H_F⁻¹]_ii) 与 U[i,i:] ∝ [H_F⁻¹]_{i,i:}
            # ——第 i 列补偿所需的全部量(GPTQ 论文的 Cholesky 技巧)。
            # Hinv 的上三角 Cholesky 因子:对角元即 [H^-1]_jj^(1/2) 的行主元
            Hinv = torch.linalg.cholesky(H)
            Hinv = torch.cholesky_inverse(Hinv)  # 经 Cholesky 求逆,比直接 inv 稳
            Hinv = torch.linalg.cholesky(Hinv, upper=True)

        # 组边界与误差补偿块边界对齐,组参数才能取自"补偿后"的最新权重
        # 面试点:若 blocksize≠gs,组会横跨块边界——组 scale/zero 将取自
        # "部分已补偿、部分未补偿"的混合权重,与实际被量化的数值系统性
        # 错位;对齐后每组参数恰在其所有列被量化前、误差已传播到位时确定
        # (GPTQ-for-LLaMa 的 groups 标准行为)。
        assert blocksize == gs, "实现约束: blocksize 必须等于 group_size"

        Q = torch.zeros_like(W)
        n_groups = math.ceil(self.columns / gs)
        scales = torch.zeros((self.rows, n_groups), device=self.dev)
        zeros = torch.zeros((self.rows, n_groups), device=self.dev)
        qidx = torch.zeros((self.rows, self.columns), device=self.dev,
                           dtype=torch.uint8)
        total_loss = 0.0

        for i1 in range(0, self.columns, blocksize):
            # ── 阶段 2:块内逐列量化 + 立即补偿 ──
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
                    # OBQ 最优补偿 δ = −(w−dq)/[H_F⁻¹]_ii · [H_F⁻¹]_{i,i:},
                    # 用 U 行改写即下面两行(1/U[i,i] 进 err,方向在 U[i,i:])。
                    # 只向右传播(i:):列序固定后,右侧列尚未定码、还有自由
                    # 度吸收误差;已量化列的整数码不可再动——这就是误差
                    # 传播方向必然"向未量化侧"的原因。
                    err = (w - dq) / Hinv1[i, i]
                    W1[:, i:] -= err.unsqueeze(1) * Hinv1[i, i:].unsqueeze(0)
                    Err1[:, i] = err
                    # 论文式逐列损失 (w−dq)²/(2[H_F⁻¹]_ii):仅作诊断/回归
                    # 监控,量纲未标定,不进任何对外表格(LEDGER 红线)
                    total_loss += ((w - dq) ** 2 / Hinv1[i, i] ** 2).sum().item() / 2

            Q[:, i1:i2] = Q1
            # ── 阶段 3:块间 lazy 补偿 ──
            # 块内对 W1 已就地逐列更新;块外列则攒满一块后用一次 GEMM 批量
            # 补偿。不这样:每列对全宽 W 做 rank-1 更新,访存量 O(rows×in)
            # × in 次,纯带宽浪费——"块内立即 + 块间延迟"正是 GPTQ 对 OBQ
            # 的关键工程改造(数学结果与逐列全宽更新严格相同)。
            if use_hessian and i2 < self.columns:
                W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]

        # ── 阶段 4:写回 ──
        # fake-quant 权重写回原层(PPL 评测直接用);整数码与组参数返回给
        # real-quant 链路(quant_linear.QuantLinear4 打包 + pack_check 断言)
        self.layer.weight.data = Q.to(self.layer.weight.dtype)
        return {"loss": total_loss, "scales": scales.cpu(),
                "zeros": zeros.cpu(), "qidx": qidx.cpu()}

    def free(self):
        # H 是 (in,in) float32,大 in(如 MLP down_proj)时达百 MB 级;
        # 逐 Linear 用完即释,防 24 层×7 个 H 同时驻留挤爆显存
        self.H = None
        torch.cuda.empty_cache()
