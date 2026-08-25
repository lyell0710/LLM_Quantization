# 01 · 三种训练后量化方法合讲:均匀量化基础 → GPTQ → AWQ → SmoothQuant

> 深度讲义。配套代码 `src/`,配套数据 `data/raw/EXP-00{1,2,3}/`,全部数字带
> EXP 锚,可从 raw 复算。阅读顺序即讲解顺序:先把"量化"这件事本身讲透,
> 再逐个拆三种方法的机制、代码与实验证据。

## 1. 这一篇回答什么问题

GPTQ、AWQ、SmoothQuant 这三种主流训练后量化(PTQ)方法,各自到底在优化什么、
差在哪一阶、贡献几何?本篇从均匀量化的第一性原理讲起,把三者的核心公式逐步
推完,再对照本仓从零实现的真实代码与同协议对照实验(EXP-001/002/003)逐段走读。
读完你应当能:①白板手推 GPTQ 的逐列补偿公式(含 Cholesky 技巧为什么成立);
②解释 61.6%/31.9%/48% 三个恢复率各自的控制变量设计与含义;③答上
"为什么叠加只 +1.0pp""容差为什么要幅值感知"这类三层追问。

## 2. 直觉与第一性原理

**没有量化,世界会怎样。** LLM 推理的 decode 阶段每生成一个 token,都要把
全部权重从显存读一遍——权重带宽是第一瓶颈(实测证据见
vllm/experiences 侧 vllm/experiments#EXP-014,转引于 `docs/talk/quant_walkthrough.md` §0)。
fp16 权重每参数 16 bit;若能压到 4 bit,同一张卡上 decode 的权重读取量
即降为 1/4,显存占用同理。这就是 W4A16 赛道的物理动机。W8A8 赛道另有所图:
激活也换成 INT8,连算力路径(INT8 TensorCore)一起换,prefill/大 batch 受益。
两个赛道压的东西不同、快的 regime 不同——这是全篇的骨架。

**类比:有限刻度的尺子。** 量化就是把连续实数逼到一把只有 16 个刻度
(INT4)的尺子上,每个数只能记"离它最近的刻度是第几格"。scale 是格距,
zero-point 是"第 0 格对准哪里"。类比在两处失效:①尺子量身高,误差就是误差;
量化误差却要再过一次矩阵乘,被激活放大——同样大小的误差,落在不同输入
通道上伤害差几个数量级,这是三种方法全部故事的起点;②量身高不能换坐标系,
量化却可以先给数乘个 s 再量、事后除回(数学恒等)——AWQ 与 SmoothQuant
的全部操作空间都在这个"类比里不存在"的自由度上。

**再给 GPTQ 一个类比:逐件装箱。** 往一个不规则的箱子里逐件放行李,每放
一件都会留下缝隙(量化误差);聪明的做法是每放完一件,立刻调整还没放的
行李的摆法,把缝隙吃掉。失效点:行李之间没有"相关性矩阵",而权重列之间
有——校准集激活的二阶矩 XXᵀ 精确刻画了"调整哪些列、按什么比例调整"最划算,
这正是 GPTQ 用 H⁻¹ 做的事。

## 3. 完整推导与机制

### 3.1 均匀量化:scale、zero-point、粒度谱系

b bit 非对称均匀量化把实数 w 映为整数码 q 再映回:

$q = \mathrm{clamp}(\mathrm{round}(w/s) + z,\ 0,\ 2^b{-}1)$,  $\hat w = (q - z)\cdot s$

- 为什么 $s = (w_{max}-w_{min})/(2^b-1)$:要让 $[w_{min}, w_{max}]$ 恰好铺满
  $2^b$ 个刻度,格距就是区间长度除以格数减一(两端各占一格)。
- 为什么要 zero-point $z = \mathrm{round}(-w_{min}/s)$:权重组的 min/max 通常
  不关于 0 对称,对称量化(z 固定为码域中点)会浪费半边码域;z 把网格平移到
  贴合实际区间。
- 为什么要强制 $0 \in [w_{min}, w_{max}]$(实现见 `src/gptq.py:51-52` 的
  clamp):保证实数 0 有精确表示(padding/稀疏权重都依赖它),且全正/全负组
  的 z 不会越出码域。全零组给无害网格防除零(`src/gptq.py:54`)。
- 舍入本身:round 到最近刻度,单点误差落在 $[-s/2, s/2]$,近似均匀分布时
  均方误差 $s^2/12$——这笔账后面反复用。

**粒度谱系**(scale/zero 共享到什么范围):per-tensor(整层 1 套)→
per-channel/per-输出行(out 套)→ per-group(本仓 g=128,每输出行每 128 个
输入维一套)→ 极限是 per-element(等于不量化,信息全搬进 scale)。粒度越细,
网格越贴合局部分布、误差越小,代价是元数据存储与 kernel 复杂度。本仓 W4A16
的存储账:码 4 bit + fp16 scale/zero 各 16 bit 摊到 128 个权重,
$4 + 32/128 = 4.25$ bit/权重(`src/quant_linear.py:19`)。激活侧另有一条谱系:
per-tensor 静态 → per-token 动态(本仓 W8A8 用)→ per-channel——最后一档与
INT8 GEMM 的数学不相容,见 §3.5,这是 SmoothQuant 存在的理由。

### 3.2 RTN 为什么不够:一笔误差账

RTN(round-to-nearest)= 上面的公式直接逐元素套用,优化目标是权重误差
$\|W-\hat W\|^2$。但推理在乎的是**输出**。设某输出行权重误差向量为
$\delta = \hat w - w$,输入为 x,则输出误差为 $\delta^\top x$,在校准分布上:

$\mathbb{E}[(\delta^\top x)^2] = \delta^\top\, \mathbb{E}[xx^\top]\, \delta$

- 为什么可以这么写:平方展开后取期望,$\delta$ 是常量可提出,剩下的正是
  激活二阶矩矩阵。
- 含义:权重误差被激活二阶矩**加权**。若某输入通道的 $\mathbb{E}[x_j^2]$ 比
  别的通道大 100 倍,该通道上同样大小的 $\delta_j$ 伤害大 100 倍。RTN 对此
  完全无感——它把每个通道的误差压到同样的 $s^2/12$,等于在错误的度量下
  做了最优。

实测代价:同一 INT4-g128 网格下,RTN 把 Qwen2.5-0.5B 的 wikitext-2 PPL 从
11.9152 打到 14.1154(+2.2002;EXP-001 §5,`data/raw/EXP-001/rtn_g128.json`,
单轮确定性评测,下同)。三种方法都是对这笔"加权误差账"的不同回应:
GPTQ 在固定网格内**重新分配**误差(二阶),AWQ **重塑**误差落点的难度分布
(一阶),SmoothQuant 把难度在激活/权重两侧**搬家**(W8A8 专属)。

### 3.3 GPTQ:从 OBS 到逐列补偿公式

**起源。** OBS(Optimal Brain Surgeon)是九十年代的剪枝理论:删一个权重时,
用损失函数的二阶信息求"其余权重怎么动能把伤害降到最小"的闭式解。
OBQ 把"删权重"换成"量化权重"逐个应用;GPTQ 再把它工程化到十亿参数规模。
推导只需要一页纸,逐步走:

**第 0 步:层内目标是精确二次型。** 取一个输出行 $w \in \mathbb{R}^d$
(行与行独立,后面解释),量化后 $\hat w = w + \delta$,层输出误差

$L(\delta) = \|\hat w^\top X - w^\top X\|^2 = \|\delta^\top X\|^2 = \delta^\top (XX^\top)\, \delta = \tfrac{1}{2}\,\delta^\top H\,\delta,\quad H = 2XX^\top$

为什么可以这么做:这不是泰勒展开——层内目标对 $\delta$ 本来就是二次的,
一步都没截断。$H$ 就是 $L$ 对该行权重的 Hessian,且**与行无关**(只含 X),
所以全部输出行共享同一个 H、同一次分解——这是 GPTQ 能把整层向量化的根基。

**第 1 步:单列量化的约束优化。** 量化第 j 列:把 $w_j$ 固定到网格点 $q_j$,
记 $\varepsilon = w_j - q_j$,即约束 $\delta_j = -\varepsilon$;其余坐标自由,
用来吸收伤害。问题:$\min_\delta \tfrac{1}{2}\delta^\top H \delta$
s.t. $e_j^\top \delta = -\varepsilon$。

- 为什么可以这么建模:量化是硬约束(码必须落网格),但尚未量化的坐标还在
  连续域,可以连续优化补救——"牺牲是强制的,补偿是自由的"。

**第 2 步:拉格朗日求解,每步一行理由。**

1. $\mathcal{L} = \tfrac{1}{2}\delta^\top H\delta + \lambda(e_j^\top\delta + \varepsilon)$ —— 等式约束的标准拉格朗日化。
2. $\partial\mathcal{L}/\partial\delta = H\delta + \lambda e_j = 0 \Rightarrow \delta = -\lambda H^{-1} e_j$ —— H 正定(阻尼保证,见下)故可逆,凸问题一阶条件即全局最优。
3. 代回约束:$e_j^\top\delta = -\lambda\,[H^{-1}]_{jj} = -\varepsilon \Rightarrow \lambda = \varepsilon / [H^{-1}]_{jj}$ —— $e_j^\top H^{-1} e_j$ 就是 $H^{-1}$ 的第 (j,j) 元。
4. 最优补偿:$\delta^* = -\dfrac{\varepsilon}{[H^{-1}]_{jj}}\, H^{-1} e_j$ —— 方向是 $H^{-1}$ 第 j 列,强度按其对角元归一。
5. 代回目标:$\Delta L = \tfrac{1}{2}\delta^{*\top} H \delta^* = \dfrac{\varepsilon^2}{2\,[H^{-1}]_{jj}}$ —— 这就是代码里那个"量纲未标定、只作诊断"的逐列 loss(`src/gptq.py:173-175`,EXP-001 §7)。

**第 3 步:固定列序后,"剩余子问题"的 H 是 Schur 补。** OBQ 每次挑
$\Delta L$ 最小的列,每列都要对"尚未量化的列集 F"重求 $[H_F^{-1}]$
(注意:是**子矩阵的逆**,不是逆的子矩阵——经典易错点),朴素做法
$O(d\cdot d^3)$。GPTQ 的观察:按固定列序(直接从左到右)量化,精度几乎不掉,
而固定序让全部行共享同一列序,且 $[H_F^{-1}]$ 有一次性解法。用块矩阵逆恒等式:

$[H_F]^{-1} = [H^{-1}]_{F} - [H^{-1}]_{F,F^c}\,([H^{-1}]_{F^c})^{-1}\,[H^{-1}]_{F^c,F}$

即:对 $H^{-1}$ 关于已量化块 $F^c$ 取 Schur 补。

**第 4 步:Cholesky 逆上三角为什么"恰好可用"。** 设
$H^{-1} = U^\top U$(U 上三角,即代码第三步 `cholesky(Hinv, upper=True)`)。
按前缀/后缀分块 $U = \begin{pmatrix} U_{11} & U_{12} \\ 0 & U_{22} \end{pmatrix}$,直接乘开:

$H^{-1} = \begin{pmatrix} U_{11}^\top U_{11} & U_{11}^\top U_{12} \\ U_{12}^\top U_{11} & U_{12}^\top U_{12} + U_{22}^\top U_{22} \end{pmatrix}$

对后缀块取 Schur 补(为什么可以:$U_{11}$ 是正定矩阵的 Cholesky 因子,
对角非零必可逆):

$U_{12}^\top U_{12} + U_{22}^\top U_{22} - U_{12}^\top U_{11}(U_{11}^\top U_{11})^{-1}U_{11}^\top U_{12} = U_{22}^\top U_{22}$

所以 $[H_F^{-1}] = U_{22}^\top U_{22}$:**一次上三角分解,右下角每个后缀块
自动就是对应剩余列集的逆**。取该块的 (1,1) 元与第一行
($U_{22}$ 上三角,其第一列只有对角元非零):

$[H_F^{-1}]_{jj} = U_{jj}^2,\qquad [H_F^{-1}]_{j,\,j:} = U_{jj}\cdot U_{j,\,j:}$

代入第 2 步的 $\delta^*$,补偿公式化简为:

$\delta^*_{j:} = -\dfrac{\varepsilon}{U_{jj}^2}\cdot U_{jj}\,U_{j,j:} = -\dfrac{\varepsilon}{U_{jj}}\cdot U_{j,j:}$

这正是 `src/gptq.py:170-171` 那两行。一次 $O(d^3)$ 分解替代 d 次重求逆;
"上"三角而非"下",是因为固定序从左到右、被冻结的是前缀、自由的是后缀,
所有后缀块沿上三角因子的行依次排开(若量化序反向则用下三角)。补偿只向
右传播:左侧列的整数码已定、不可再动,右侧列还有自由度——误差传播方向
必然"向未量化侧"。

**阻尼的作用。** 校准样本有限或输入通道强共线时 H 近奇异,$H^{-1}$ 在弱激发
方向的元素爆炸,补偿量随之爆炸——本想救误差,反把权重毁了。加
$\lambda = \text{percdamp}\cdot\mathrm{mean}(\mathrm{diag}\,H)$ 的 ridge
(`src/gptq.py:117-118`,percdamp=0.01,EXP-001 §2 同设)有三重效果:
①保证正定,Cholesky 不失败;②把病态方向的补偿强度压回有界;③取"相对
平均对角"而非绝对常数,对 H 的整体量级自适应(H 本身经运行均值归一,
见走读第 2 段)。太小:分解失败或补偿爆炸;太大:H 趋向 $\lambda I$,
补偿方向失去激活信息,退化向 RTN。

### 3.4 AWQ:显著权重直觉与 best-scale 目标

**直觉。** 极少数输入通道的激活系统性偏大,这些通道上的权重是"显著权重"
——由 §3.2 的加权误差账,它们的量化误差被放大最多,却在 RTN 里和普通权重
享受同样的格距。AWQ 不改取整规则,改坐标系:对通道 j 的权重先乘 $s_j>1$
再量化,运行时等价除回:

$y = XW^\top = (X\,\mathrm{diag}(s)^{-1})\,(\mathrm{diag}(s)\,W^\top)$ —— 数学恒等,只改变误差落在哪里。

**为什么乘 s 能保护。** per-group min-max 网格的格距 $\Delta_g$ 由组内极值
决定。通道 j 乘 $s_j$ 后,它在 $W\cdot s$ 域的舍入误差仍是 $\Delta'_g/2$ 级,
除回 $s_j$ 后有效误差 $\approx \Delta'_g/(2s_j)$:若组极值不由 j 主导
(放大后 $\Delta'_g \approx \Delta_g$),j 的有效误差整整缩小 $s_j$ 倍;
若放大过猛、j 撑爆组极值,$\Delta'_g \propto s_j$,全组通道跟着遭殃。
**保护是有代价的转移,不是免费午餐**——这就是 α 需要搜索的原因。

**best-scale 目标函数**(EXP-002 §1 口径):

$\min_s\ \|\,Q(W\cdot\mathrm{diag}(s))\,\mathrm{diag}(s)^{-1}X - WX\,\|^2,\quad s_j = \mathrm{absmean}(X_j)^\alpha,\ \alpha \in \{0, 0.05, \dots, 0.95\}$

三个设计决定,各有理由:①打分在**输出空间**(MSE 对 ref = XW^⊤),不在
权重空间——优化目标必须与推理在乎的东西一致,这是全篇的主旋律;
②重要性代理用 absmean 不用 absmax——量通道的系统性幅值,不被单个极端
token 绑架;③s 做几何归一 $s/\sqrt{s_{max}s_{min}}$——不归一时 s 整体大于
(小于)1 会系统性撑大(缩小)组的 min-max 范围,α 间比较混入"网格整体
变粗/变细"的混杂因素(`src/awq.py:111-115` 注释)。

**per-linear 简化的代价(限定语教学)。** 本仓为臂间可比取 per-linear 独立
s + per-linear MSE 打分,且未实现 clip;AutoAWQ 完整实现为块级 MSE +
共享输入组共享 s + weight clip(EXP-002 §2 诚实标注)。因此本仓 AWQ 的
31.9% 是"**per-linear 简化、无 clip** 口径下"的数字,不能拿去与 AutoAWQ
完整实现的相对表现直接比——引用数字时这个定语不许丢。这同时是一个
可讲的活例:简化的代价是可测的(EXP-002 §6)。

### 3.5 SmoothQuant:等价迁移的数学与 α 的选取

**W8A8 的病灶在激活。** LLM 激活的 outlier 集中在固定的少数通道(模型属性,
与 token 无关),幅值可比其余通道大百倍。per-token 对称 INT8 的 scale 取
该 token 全通道 absmax,被 outlier 通道独占后其余通道分辨率崩塌。为什么
不能对激活做 per-channel?INT8 GEMM 要求 scale 能从整数乘加中整体提出:
$y_{ij} = s^x_i \cdot s^w_j \cdot \sum_k qx_{ik}\,qw_{jk}$——scale 必须呈
行×列外积形状;沿求和维(输入通道)的 per-channel scale 提不出来
(`src/smoothquant.py:180-184`)。outlier 只能靠**迁移**消解,不能靠更细
粒度硬扛。

**迁移公式与两侧难度的推导。** 恒等变换 $y = (X\,\mathrm{diag}(s)^{-1})(\mathrm{diag}(s)W^\top)$,取

$s_j = \mathrm{actmax}_j^{\alpha} \,/\, \mathrm{wmax}_j^{1-\alpha}$

迁移后两侧的通道幅值(逐步代入):

- 激活侧:$\mathrm{actmax}_j / s_j = \mathrm{actmax}_j^{1-\alpha}\cdot \mathrm{wmax}_j^{1-\alpha}$
- 权重侧:$\mathrm{wmax}_j \cdot s_j = \mathrm{actmax}_j^{\alpha}\cdot \mathrm{wmax}_j^{\alpha}$

α=0.5 时两侧恰为几何均值 $\sqrt{\mathrm{actmax}_j\,\mathrm{wmax}_j}$——难度
几何均衡;α 越大激活越轻、权重越重。**α 是天平不是开关**:激活侧确实变好,
权重侧的 per-输出行 scale 却被撑大的列拉粗,其余列分辨率下降。最优 α 取决
于两侧的难度对比:激活 outlier 越猛,α 越大越划算。本仓 α∈{0.25,0.5,0.75}
三点扫描给出 0.5B 上的形状:单调向好、最优点偏大(权重侧余量足),而
α=0.25 比不迁移还差——迁移不足以救激活、却已开始伤权重(EXP-003 §6)。

## 4. 代码逐段走读

按 gptq 臂的执行顺序走 `scripts/run_w4a16.py` 与 `src/`,再补 AWQ/SmoothQuant
与 real-quant 链路。引文逐字拷贝自仓内现行代码,标注 文件:起-止行。

**第 1 段 · 校准输入截获**(`scripts/run_w4a16.py:72-91`)——逐层量化只
需要第 0 层的输入,跑完整模型纯属浪费。用"哨兵异常"在第 0 层截断前向:

```python
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
```

角色:为逐层 sequential 量化备好第 0 层的 (hidden_states, kwargs)。关键行:
`raise RuntimeError("__captured__")` 抓到即弃剩余 23 层计算;except 里只吞
哨兵——若改成裸 `except: pass`,真实 bug(如 OOM)会被静默吞掉,校准输入
悄悄少一批,H 偏了都无从发现。后续每层量化完,用量化后权重重算输出作为
下层输入(`run_w4a16.py:189-199` 阶段 C):下层的 H 见到的是前序误差已
累积的真实分布,改用 fp16 激活校准会系统性高估深层质量。

**第 2 段 · H 的累积**(`src/gptq.py:81-94`)——H = 2XX^⊤ 的运行均值形态:

```python
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
```

角色:把约 26 万校准 token(128×2048,EXP-001 §2)的二阶统计压进 (in,in)
矩阵。为什么均值而不是裸和:26 万 token 的外积裸累加量级线性膨胀,fp32
后来的批会"大数吃小数";而补偿公式对 H 的整体缩放不变(注释里那行代数),
归一只买数值稳定,不改结果。改错会怎样:若漏掉 `self.H *=` 的旧值收缩,
H 变成"越晚到的批权重越低"的错误加权;若用 fp16 存 H,Cholesky 对条件数
敏感,病态层直接分解失败。

**第 3 段 · H 预处理与三步分解**(`src/gptq.py:109-118`、`126-129`)——
死列、阻尼、再到 $H^{-1}$ 的上三角因子:

```python
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
```

```python
            # Hinv 的上三角 Cholesky 因子:对角元即 [H^-1]_jj^(1/2) 的行主元
            Hinv = torch.linalg.cholesky(H)
            Hinv = torch.cholesky_inverse(Hinv)  # 经 Cholesky 求逆,比直接 inv 稳
            Hinv = torch.linalg.cholesky(Hinv, upper=True)
```

角色:一次性算出 §3.3 第 4 步的 U。死列(校准集中恒为零的输入通道)无二阶
信息可言,对角置 1 保可分解、权重清零省码域。三步分解对应:分解 H → 由
因子求 $H^{-1}$(比直接 inv 稳)→ 对 $H^{-1}$ 再做上三角分解得 U。改错会
怎样:少了阻尼,强共线层的 cholesky 直接抛错或补偿量爆炸;把 upper=True
丢掉拿到下三角因子,行方向与"后缀自由集"错位,补偿方向整个转置,PPL 不会
报错但会显著劣化——这类"数学错了代码还能跑"的 bug 只有对照臂能抓出来。

**第 4 段 · 逐列量化与两级补偿**(`src/gptq.py:155-172`、`177-184`)——
算法心脏,blocksize=group_size 的对齐原因在此:

```python
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
```

```python
            Q[:, i1:i2] = Q1
            # ── 阶段 3:块间 lazy 补偿 ──
            # 块内对 W1 已就地逐列更新;块外列则攒满一块后用一次 GEMM 批量
            # 补偿。不这样:每列对全宽 W 做 rank-1 更新,访存量 O(rows×in)
            # × in 次,纯带宽浪费——"块内立即 + 块间延迟"正是 GPTQ 对 OBQ
            # 的关键工程改造(数学结果与逐列全宽更新严格相同)。
            if use_hessian and i2 < self.columns:
                W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
```

角色:§3.3 推导的最终落地。`err = (w-dq)/Hinv1[i,i]` 与下一行合起来就是
$-(\varepsilon/U_{jj})\cdot U_{j,j:}$。**blocksize 必须等于 group_size**
(`src/gptq.py:136` 断言):组参数 find_params 在块头一次性确定,此刻该组
所有列的"前序块误差"已通过块间 lazy 更新传播到位——组网格描述的正是即将
被量化的那批数值。若 blocksize≠gs,组会横跨块边界,scale/zero 取自"部分
已补偿、部分未补偿"的混合权重,与实际被量化的数值系统性错位。
`use_hessian=False` 时跳过全部补偿路径、其余一切共享——这个开关就是
EXP-001 的实验设计本身:61.6% 的恢复量可完全归因于二阶补偿。改错会怎样:
把 `W1[:, i:]` 写成 `W1[:, i+1:]` 看似更合理(别改自己),实际等价(第 i 列
的更新在赋值 Q1 之后不再被读),但把块间更新的 `Hinv[i1:i2, i2:]` 切片切错
一列,后续所有组的网格都建立在错位的权重上,误差逐块滚雪球。

**第 5 段 · AWQ 的 α 网格搜索**(`src/awq.py:104-123`)——一阶方法的全部
机关:

```python
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
```

角色:对每档 α 构造 s、fake-quant、按输出 MSE 打分取最优。三个关键行:
α=0 必在网格内(结果不劣于 RTN 起点,搜索有安全底);几何归一(§3.4 的
第③点);打分对象是 `fq / s`——除回 s 的**有效权重**,与部署形态端到端
等价。改错会怎样:若打分时忘了除回 s(直接用 fq 对 ref 算 MSE),比较的
是两个不同坐标系里的输出,α 越大 MSE 天然越大,搜索永远选 α=0,AWQ 静默
退化为 RTN——PPL 只会告诉你"AWQ 没用",不会告诉你哪错了。量化内核用
`group_fakequant`(`src/awq.py:42-62`),纯 RTN、与 GPTQ 臂共用同一
GroupQuantizer:AWQ 的全部收益必须只来自 s 预缩放,归因才干净。

**第 6 段 · SmoothQuant 的迁移与双端 fake quant**(`src/smoothquant.py:229-250`):

```python
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
```

角色:§3.5 公式的落地,两侧各一半。两处"顺序不可换"是同一件事的镜像:
权重先乘 s 再量化(网格在迁移后的分布上确定),激活先除 s 再量化(scale
在 outlier 已消解的分布上取)。哪边顺序反了,迁移就在那边完全失效——
且不报任何错,只有 PPL 对照能抓到。`alpha < 0` 走 else 分支 s≡1,naive 臂
与 smooth 臂共用同一改装管线,唯一差异 = 是否迁移(EXP-003 控制变量设计)。
actmax 由调用方在**改动权重之前**于原始 fp16 模型上一遍收齐
(`scripts/run_w8a8.py:31-58`)——与 W4A16 的逐层 sequential 标定不同,
s 只依赖激活统计,不含量化误差项,可一次收齐统一改装。

**第 7 段 · INT4 打包与反量化**(`src/quant_linear.py:34-39`、`45-58`)——
real quant 闭环的存储半边:

```python
        # 偶列进低 4 位、奇列进高 4 位:与 dequant 的 0::2 / 1::2 切片互为逆
        packed = (qidx[:, 0::2] | (qidx[:, 1::2] << 4)).contiguous()
        self.register_buffer("qweight", packed)          # (out, cols/2) uint8
        # scale/zero 存 half:模拟真实部署的存储 dtype;pack_check 的非零
        # 误差(~1e-4 量级)即源于这次降精度,见 run_w4a16 的幅值感知容差
        self.register_buffer("scales", scales.half())    # (out, n_groups)
```

```python
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
```

角色:fake quant 只证明"数值网格受得了",不证明"码真的能紧凑存下并原样
取回"。两枚 INT4 码合一字节(偶列低 nibble、奇列高 nibble),dequant 的
切片互逆还原。改错会怎样:高低 nibble 写反,dequant 出的权重列序两两对调
——forward 不报错,PPL 灾难性劣化;这类 bug 恰是下一段的逐元素断言存在的
理由。反量化在 float32 域做乘法:让断言残差可以**干净地归因于单一来源**
(scale/zero 的 half 存储),这是"每个误差都要有账"的工程习惯。

**第 8 段 · pack↔fake 断言与幅值感知容差**(`scripts/run_w4a16.py:172-178`):

```python
                err = pack_check(linears[n].weight.data, res["qidx"],
                                 res["scales"], res["zeros"], group_size)
                # 容差幅值感知:fp16 尾数 10 位,权重经 fp16 存储的舍入
                # ~|w|·2^-10;W·s 域幅值可远超 1,固定 1e-3 会误报
                wmax = float(linears[n].weight.data.abs().max())
                tol = max(1e-3, wmax * 2 ** -10)  # fp16 相对 ulp,幅值感知
                assert err < tol, f"pack mismatch {err} (tol {tol})"
```

角色:对每个 Linear 断言 dequant(pack(qidx)) 与 fake-quant 权重逐元素一致。
为什么断言能成立:给定同一 (q, zero, scale),反量化是确定映射,打包只搬
比特不碰数值,残差仅可能来自 half 存储的 ulp 级舍入(`src/quant_linear.py:
65-77`)。容差工程的完整故事见 §6 误区四:固定 1e-3 曾两次误杀正确实现。

## 5. 实验数据怎么读

**主对照表**(全部单轮确定性 greedy scoring;协议:Qwen2.5-0.5B,
wikitext-2 PPL,窗 2048/步 1536,各臂同 298302 计分 token;PPL 为自定义
协议,只作协议内臂间相对比较,不与文献绝对值对比):

| 臂 | PPL | Δ vs fp16 | 恢复率 | 出处 |
|---|---|---|---|---|
| fp16 | 11.9152 | — | — | EXP-001,`data/raw/EXP-001/fp16_g128.json` |
| RTN INT4-g128 | 14.1154 | +2.2002 | 0%(定义基线) | EXP-001,`rtn_g128.json` |
| AWQ | 13.4127 | +1.4975 | 31.9% | EXP-002,`awq_g128.json` |
| GPTQ | 12.7600 | +0.8448 | 61.6% | EXP-001,`gptq_g128.json` |
| AWQ+GPTQ | 12.7376 | +0.8224 | 62.6% | EXP-002,`awq_gptq_g128.json` |
| naive W8A8 | 12.1227 | +0.2075 | 0%(W8A8 基线) | EXP-003,`naive.json` |
| smooth α=0.25 | 12.2332 | +0.3180 | 为负(反例臂) | EXP-003,`smooth_a0.25.json` |
| smooth α=0.50 | 12.0394 | +0.1242 | 40% | EXP-003,`smooth_a0.5.json` |
| smooth α=0.75 | 12.0221 | +0.1069 | 48% | EXP-003,`smooth_a0.75.json` |

**恢复率怎么算(列算式)。** 定义:收回基线缺口的比例,W4A16 赛道以 RTN
为 0%、fp16 为 100%(`scripts/plot_recovery.py:5-7` 同定义):

- GPTQ:$(14.1154-12.7600)/(14.1154-11.9152) = 1.3554/2.2002 = 61.6\%$
- AWQ:$(14.1154-13.4127)/2.2002 = 0.7027/2.2002 = 31.9\%$
- 叠加:$(14.1154-12.7376)/2.2002 = 1.3778/2.2002 = 62.6\%$
- smooth α=0.75(W8A8 赛道,以 naive 为基线):
  $(12.1227-12.0221)/(12.1227-11.9152) = 0.1006/0.2075 = 48.5\%$,
  EXP-003 §6 取整表述为 48%(图 fig2 脚注同此说明)。

**这个实验设计防了哪些坑。** ①**控制变量到开关级**:RTN 与 GPTQ 共用同一
GroupQuantizer、同一 find_params、同一网格,唯一差异是 use_hessian 布尔值
(`src/gptq.py:99`);AWQ 的量化内核是纯 RTN,收益只可能来自 s。归因链条
"数字差异 → 机制开关"中间没有任何别的变量。②**反例臂主动保留**:α=0.25
比 naive 更差不是失败,是 α 作为真实权衡旋钮的证据——只报单调向好的臂,
读者无法排除"α 随便设都行"。③**同一计分 token 集**:eval 尾部不足一窗即停
(`scripts/run_w4a16.py:209-210`),各臂 298302 个 token 逐位相同;滑窗
去重保证每 token 恰计分一次、且至少带 512 token 上文。④**单轮的资格**:
关键数字通常要求多轮 mean±std,本仓 PPL 为确定性 greedy scoring(无采样、
seed 固定、fp32 log_softmax),同机重跑逐位一致,EXP-001 §6 据此显式豁免
——"单轮"限定语因此必须随数字出现,这是豁免的对价。

**图怎么读。** fig2(`figures/fig2_recovery_rates.png`):横轴恢复率 0-100%,
各赛道内以自家基线为 0%、fp16 为 100% 归一——两赛道的条**不可跨读**
(W8A8 的 48% 收的是 +0.2075 的小缺口,W4A16 的 61.6% 收的是 +2.2002 的
大缺口,绝对收益差一个量级)。fig1(`figures/fig1_awq_alpha_dist.png`):
横轴是 per-layer best-α(搜索网格 0-0.95、步 0.05,无量纲),纵轴是 Linear
层数,n=168(24 层×7 Linear);读点有三:中位 0.30、主体 0.15-0.45 说明
多数层只要温和保护;两层顶到网格上限 0.95 是强 outlier 层被自动识别
(注意"顶到上限"也提示可能欠搜索,EXP-002 §7);α=0 恰 1 层——该层激活
均匀,AWQ 无事可做,网格里的"不保护"选项被真实用到。

**数字背后的机理账。** 为什么叠加只 +1.0pp(62.6% vs 61.6%)?AWQ 修的是
"显著通道误差被放大",GPTQ 的输出空间补偿修的是"误差总量在列间的最优
分配"——机制上正交(一个改坐标系、一个改取整),但两者救的都是同一批
对输出伤害最大的权重。0.5B 上 GPTQ 已把 AWQ 能救的大部分吸收掉,叠加的
边际收益只剩 +1.0pp(EXP-002 §6)。这解释了工程实践中两者通常二选一。
再算一笔时间账:RTN 30 s、AWQ 43 s、GPTQ 135 s(EXP-001/002 §5)——GPTQ
的溢价花在校准重放(26 万 token 过每层收 H)与 168 次 Cholesky 上;AWQ 的
溢价只是 20 档 α × 4096 token 的打分 GEMM。二阶信息的定价,时间侧同样成立。
存储账:4.25 bit/权重(§3.1),对 fp16 是 3.76× 压缩。

## 6. 误区与边界

**误区一:"权重误差小,模型就好。"** 聪明人会想当然地用 $\|W-\hat W\|$
选方案。本仓的直接反证:RTN 与 GPTQ 在同一网格上,逐元素误差同为
$s^2/12$ 量级,PPL 却差 1.36(14.1154 vs 12.7600,EXP-001 §5)——差的
全部是"误差往哪些列放"。度量必须取在输出空间,这是三方法共同的第一课。

**误区二:"两个好方法叠加,收益近似相加。"** 31.9% + 61.6% 远大于实测的
62.6%(EXP-002 §5)。正交性是机制层面的(改坐标系 ⊥ 改取整),不保证
收益可加——两者保护的是同一批难量化权重,0.5B 上高度重叠。措辞上"正交
可叠加"只能说**方向成立**(EXP-002 §6),不得说"显著提升":+1.0pp 的
增益配不上这个词。

**误区三:"SmoothQuant 加了就好,α 越大越保险。"** 本仓的反例臂:α=0.25
的 PPL 12.2332 比 naive 的 12.1227 还差(EXP-003 §5)——迁移不足以救激活、
却已开始伤权重。且 0.5B 上 naive 缺口本来只有 +0.2075:激活 outlier 温和
的语境下,可迁移的难度本来就少。48% 这个数字必须带"0.5B outlier 温和"
定语,且不得外推大模型幅度(EXP-003 §6:该方法的价值随模型规模/outlier
严重度增长,本仓小缺口恰是反向证据)。

**误区四:"断言容差设个 1e-3 就稳了。"** 本仓被实测证伪过两次(EXP-002
§7):AWQ+GPTQ 臂在 W·s 域做 pack 校验,权重幅值被 s 撑大后,fp16 存储的
相对舍入(尾数 10 位,~|w|·2⁻¹⁰)在 wmax≈1.7 时就是 1.7e-3——正确实现
被固定容差误杀两次。修复是幅值感知容差 max(1e-3, wmax·2⁻¹⁰)
(`scripts/run_w4a16.py:176-177`)。所以本仓的口径是两段式的:EXP-001
口径最大误差 ≤7.3e-4,EXP-002 臂在幅值感知容差下最大 1.22e-3
(`src/quant_linear.py:16`)——**容差必须与被检对象的数值幅度挂钩**,
浮点世界里"绝对误差阈值"几乎总是错的。

**误区五:"per-layer loss 可以当质量指标排层。"** GPTQ 的逐列损失
$\varepsilon^2/(2[H_F^{-1}]_{jj})$ 数值量级 1e-20~1e-15(EXP-001 §7):
量纲随 H 的运行均值归一化缩放,未标定,仅可作层间相对诊断,不进任何表格。
量化正确性由 PPL 与 pack 断言独立支撑,不依赖它。

**适用边界(引用这些数字时限定语不许丢)。** ①全部结论出自 Qwen2.5-0.5B
单模型、wikitext-2 单任务、自定义 PPL 协议——臂间相对比较可信,绝对值
不跨协议比较;②AWQ 数字是 per-linear 简化 + 无 clip 口径;③blocksize=
group_size 是本仓实现约束(EXP-001 §7),通用化未做;④α 网格 0.95 封顶,
两层顶到上界可能欠搜索;⑤SmoothQuant 对全部 7 类 Linear 施加迁移,与原
论文"仅 post-LN linears"口径不同(EXP-003 §2,实现选择)。

## 7. 连环追问

**Q1 非对称量化的 zero-point 到底买到了什么?** 权重组 min/max 不对称时,
对称网格有半边码域低利用。z 把网格平移贴合真实区间,等效于免费多出近
1 bit 的动态范围利用率;代价是 dequant 多一次减法(`src/gptq.py:63`)。

**Q2 H = 2XX^⊤ 是什么的 Hessian?为什么全部输出行共享?** 是层输出 MSE
对单个输出行权重的 Hessian。目标 $\|\delta^\top X\|^2$ 展开后只含 X,
不含行号——所以 168 个 Linear 每个只做一次分解,全部行共享(§3.3 第 0 步)。

**Q3 为什么 Cholesky 一次分解就够,OBQ 却要逐列重算?** OBQ 按"当前伤害
最小"动态挑列,自由集 F 的变化无规律,$[H_F^{-1}]$ 只能重算。GPTQ 固定
列序后 F 恒为后缀,块矩阵逆恒等式 + Cholesky 的 Schur 补结构让 U 的每一行
恰好携带对应后缀的 $[H_F^{-1}]_{jj}$ 与方向(§3.3 第 3-4 步)。

**Q4 阻尼 percdamp 调大调小各会怎样?** 太小:病态层 Cholesky 失败或弱激发
方向补偿爆炸;太大:H → λI,补偿失去激活各向异性信息,退化向 RTN。取
1% 平均对角是论文默认,且因 H 已做运行均值归一,该相对量对不同层自适应。

**Q5 为什么 blocksize 必须等于 group_size?** 组参数必须描述"即将被量化的
那批数值"。块间 lazy 更新以块为单位传播误差;组横跨块边界时,组内一部分列
的误差已传播、另一部分没有,find_params 见到的是混合态,scale/zero 与实际
量化值系统性错位(`src/gptq.py:131-136` 断言与注释,EXP-001 §7)。

**Q6 AWQ 的 s 为什么用 absmean 不用 absmax?为什么要几何归一?** absmax 被
单个极端 token 绑架,absmean 量的是通道系统性幅值;几何归一让各 α 档比较
的是"幅值再分配"本身,而非网格整体变粗变细的混杂效应(§3.4)。

**Q7 awq_gptq 叠加臂里,GPTQ 的 H 为什么必须取自 X/s?** 恒等拆分
$(W s)(X/s) \equiv WX$:权重半边进 GPTQ 量化,激活半边就必须同步换到 X/s
域——H 描述的是量化对象 W·s 所面对的激活分布。若 H 仍用原 X,补偿方向
对错了坐标系(`scripts/run_w4a16.py:136-139` 注释)。

**Q8 为什么激活不能 per-channel 量化,权重却能 per-输出行?** INT8 GEMM
要求 scale 呈行×列外积:激活行方向是 token(per-token 可提出),列方向是
输入通道(在求和维内,提不出);权重的输出行在求和维外,天然可提
(`src/smoothquant.py:180-184`)。

**Q9 部署时 AWQ/SmoothQuant 的 s 怎么做到零开销?约束是什么?** s 折叠进
前置 RMSNorm/Linear 的权重,runtime 无额外算子。约束:共享同一输入的
linears(q/k/v;gate/up)必须共享同一个 s——这是折叠的约束不是算法的;
本仓 fake-quant 评测形式下 per-linear s 合法(`src/awq.py:29-33`)。

**Q10 4.25 bit/权重怎么算出来的?还能再压吗?** 4 bit 码 + (16+16) bit
scale/zero 摊到 g=128:32/128=0.25。再压的路子:zero 用 4 bit 整数存、
scale 二次量化(如部分生产格式),或加大 g——都在精度与元数据之间换。

**Q11(压力)61.6% 能外推到 7B、外推到下游任务吗?** 不能直接外推,这是
单模型单任务单协议的数字。方向性论据可以给:大模型参数冗余更多、RTN 缺口
相对更小,文献中 GPTQ 相对优势依模型而变;但本仓没有 7B 实测,任务面也
只有 PPL。诚实答法:"在我的控制变量协议内,二阶补偿收回 61.6%;跨模型
跨任务的量级要重测,我的协议与脚本可直接换模型复跑。"

**Q12(压力)你的 AWQ 只有 31.9%,比社区口碑差,是不是实现错了?** 首先
承认口径差异:本仓是 per-linear 简化 + 无 clip,AutoAWQ 是块级 MSE + 共享
s + clip,数字不同口径不可直接比(EXP-002 §2/§6)。其次给证据链:α 分布
形状合理(中位 0.30、outlier 层自动顶格)、α=0 档保底存在、pack 断言全过
——实现正确性有独立支撑。最后给可证伪路径:补 clip 与共享 s 的增量是
现成的后续实验设计。"数字低"与"实现错"之间,隔着口径与消融两层证据。

## 8. 工业对照与延伸

**与生产实现的差距各在哪一层。**

- **GPTQ 侧**:GPTQ-for-LLaMa/AutoGPTQ 的算法主循环与本仓同构(H 滑动缩放、
  percdamp、cholesky(inverse(H), upper) 三处数值锚点逐一对齐,见
  `docs/theory/01_gptq.md` §5);差距在 act_order(按 diag(H) 降序先量化
  重要列,本仓留位未开)、更低位宽(W3/W2)与超大模型的分层调度。
- **推理侧**:本仓 `QuantLinear4.forward` 是"先反量化、再 fp GEMM"的教学
  实现,只为正确性闭环(`src/quant_linear.py:20-22`);生产内核(vLLM 的
  Marlin 家族)做 fused dequant-GEMM,INT4 权重直接进寄存器再融合反量化
  ——速度差距在 kernel 层,不在算法层。serving 侧实测见
  vllm/experiments#EXP-016:W4A16(Marlin)decode 快 23-48%,但大 M
  (prefill)下反量化开销显形,FP8 在 c128 TTFT 反超——按 regime 选型。
- **AWQ 侧**:AutoAWQ `quantizer.py` 为块级输出 MSE + 共享输入组共享 s +
  weight clip 三件本仓皆简化/未做——差距在搜索目标的粒度与 clip,
  不在恒等变换本身(EXP-002 §2)。
- **SmoothQuant 侧**:原实现把 s 折叠进 RMSNorm(ln.weight/=s)做零开销
  部署、仅对 post-LN linears 迁移;本仓 hook 形式数学等价但有 runtime
  开销,且对全部 7 类 Linear 迁移(EXP-003 §2)。生产 W8A8 还会配静态
  激活 scale(免每 token 求 absmax),本仓未做,是 per-tensor 静态对照的
  自然后续。

**延伸阅读(带锚)。**

1. Frantar et al., *GPTQ*, arXiv:2210.17323——§3 的三步工程化(固定序/
   lazy batch/Cholesky)对应本篇 §3.3 与走读第 3-4 段。
2. Lin et al., *AWQ*, arXiv:2306.00978——§3 显著通道观察与 s 搜索;对照
   本仓 `src/awq.py:97-124` 的 per-linear 简化版。
3. Xiao et al., *SmoothQuant*, arXiv:2211.10438——§4 迁移公式与折叠;对照
   `src/smoothquant.py:224-252` 的 hook 等价实现。
4. GPTQ-for-LLaMa `gptq.py`——参考实现;本仓三个数值锚点(H 缩放/阻尼/
   组参数取补偿后权重)的对齐对象(`docs/theory/01_gptq.md` §5)。
5. vllm/experiments#EXP-016——GPTQ-Int4+Marlin 在线部署实测;与本仓
   EXP-001 构成"离线算法 → 在线部署"完整链(EXP-001 §8)。
