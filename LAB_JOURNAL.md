# LAB_JOURNAL — LLM_Quantization

## §1 建仓 + GPTQ 从零实现三臂对照(2026-08-23,EXP-001)

- **做了什么**:按 /root/standards/CORE.md 骨架建仓;从零实现 GPTQ 关键路径
  (H 累积/Cholesky 逆上三角/逐列补偿/块化延迟更新/INT4-g128 非对称)+
  INT4 两枚一字节打包(pack↔fake 断言);Qwen2.5-0.5B 三臂
  {fp16, RTN, GPTQ} 各评 wikitext-2 PPL(窗2048/步1536)。
- **为什么**:简历量化段此前只有"方法跑通"没有效果数字;GPTQ 是明确的洞
  (面试必问 GPTQ vs AWQ);对照设计沿用 CUDA 项目的控制变量反例方法论
  ——同一量化网格,唯一差异=补偿开关。
- **关键数字**:PPL 11.9152 / 14.1154 / **12.7600**;
  **补偿收回 RTN 损失 61.6%**;全模型量化 135s;pack 校验 168/168 层
  ≤7.3e-4。假设(>50%)成立。
- **产物**:src/{gptq,quant_linear}.py、records/EXP-001、
  docs/theory/01_gptq.md(五节全回填)、
  docs/HOW_TO_LEARN_A_QUANT_METHOD.md(六步 playbook)、
  data/raw/EXP-001/(provenance 全)。
- **下一步**:AWQ/SmoothQuant 收进同协议补数据(用户指令:整理在一起)。

## §2 AWQ + SmoothQuant 收编同协议,数据补全(2026-08-23,EXP-002/003)

- **做了什么**:①从零实现 AWQ(α 网格 n_grid=20 best-scale,输出 MSE
  裁决)入 W4A16 第四臂 + AWQ×GPTQ 叠加第五臂(H 取自 X/s 的恒等拆分);
  ②从零实现 SmoothQuant(s=actmax^α/wmax^{1−α},hook 式 W8A8 fake quant)
  开 W8A8 赛道:naive 对照 + α∈{0.25,0.5,0.75} 扫描;③脚本按赛道改名
  (run_w4a16.py / run_w8a8.py),README 重写为"一张主对照表"。
- **为什么**:此前 AWQ/SmoothQuant 是异机 AutoAWQ 跑通、无任何效果数字
  ("跑的数据不全");收进同一仓、同模型、同 PPL 口径、同 INT4 网格,
  臂间才可比,一张表讲完三方法。
- **关键数字**:AWQ 13.4127(收回 31.9%);AWQ+GPTQ **12.7376(62.6%**,
  vs GPTQ 单独 61.6%——正交方向成立、小模型上高度重叠);per-layer α
  中位 0.30、两层 0.95(按层自适应)。W8A8:naive 仅 +0.21(0.5B outlier
  温和),smooth α=0.75 收回缺口 48%,α=0.25 反而更差(天平不足先伤权重)
  ——反面印证 SmoothQuant 价值随模型规模增长。
- **教训(两则)**:①pack 校验容差必须与被检对象幅值挂钩(W·s 幅值大,
  fp16 ulp 1.7e-3 两次误杀断言;修为 max(1e-3, wmax·2⁻¹⁰));
  ②讲解型结论要带分寸词入红线表("正交可叠加"限定为"方向成立 +1.0pp",
  SmoothQuant 收益必须带小模型语境)。
- **产物**:src/{awq,smoothquant}.py、records/EXP-002/003、theory/02、03
  (五节全回填)、docs/talk/quant_walkthrough.md(明日 30 分钟讲解提纲,
  问题驱动五段 + 自家数字全程带跑)、README 主对照表 + 红线表(7 条)。
- **下一步**:明日按 talk 提纲讲解;可选 EXP-004(clip/块级共享 s/act_order)、
  EXP-005(α>0.75 尾部/静态激活量化);简历量化段按 README 红线表升格。


## §3 收编 LLMQT_Example,量化工程归一(2026-08-23 晚)

- **做了什么**:发现本机存在第二个量化工程 LLMQT_Example(用户既有 AutoAWQ
  侧实践本体,GitHub 远程 lyell0710/LLMQT_Example,5 commit);以
  `merge -s ours + read-tree --prefix` 连**完整 git 历史**并入本仓
  `llmqt_example/`,随后移除本地重复目录(远程 + 并入历史双重保底)。
- **为什么**:用户指令"两个 quantization 工程整理成一个";CORE 单一事实源。
- **关键发现**:旧仓 HEAD 里有 289QS 课程交付(论文/slides/**真实 PPL 与
  latency JSON**:AWQ-INT4 Qwen2-1.5B 8.933 vs fp16 8.474 等)——
  但旧工作区里 289QS 被**未提交地删除**;本次以 HEAD 为准并入,数据找回。
  旧工作区另有一个未跟踪 CLAUDE.md(标准指针,本仓已有,弃)。
- **口径勘注**:EXP-002/003 记录中"异机 AutoAWQ 实践"指实践发生地;
  其仓副本今晨已在本机(08:38),现已并入本仓。289QS 数据为**异协议**
  (HF eval,不同模型),README 单列一节,不与主对照表混排。
- **产物**:merge commit 1feb83e;README 增"既有实践数据"节与结构更新。
- **下一步**:明日讲解(talk 提纲);llmqt_example 内代码与本仓从零实现的
  对照阅读(AutoAWQ 的块级 MSE / clip vs 本仓简化)可作讲解加餐。

## §4 2026-08-24 · 审计收尾批次

- **做了什么**:逐条闭环外部审计确认的 9 项问题——①run_all*.sh 改 UTC 前缀
  新文件 + 同名拒覆盖 + worktree dirty 拒跑(旧固定命名 raw 在各 manifest 标
  grandfather);②README 顶部加"无远端待用户建仓"警示(不代建远端);
  ③EXP-001 §7 勘注 sha=worktree(代码即 274acb2 所提交内容)+ README
  "provenance 全"改如实;④EXP-001 §3/§7 勘注 run_gptq.py→run_w4a16.py 改名;
  ⑤新建根 ENV.md(链 llmqt_example/ENV.md 为子范围);⑥records/TEMPLATE.md
  自 vllm/experiments 拷入;⑦scripts/plot_alpha_dist.py 纯 CPU 出
  figures/fig1(α 分布),talk §3 挂图;⑧三篇 theory frontmatter 补
  date/exp;⑨新建 llmqt_example/README.md 注明 submit/ 为交付快照。
- **为什么**:审计收尾;硬约束(raw 不可变 / GPU 被占禁跑 / 禁建远端)下
  全部以"勘注留痕 + 脚本加固 + 新增文档"完成,零重跑、零改 raw 本体。
- **关键数字**:无新测量。fig1 由既有 raw 重算:per-layer best-α 中位 0.30,
  n=168 层,0.95 两层(data/raw/EXP-002/awq_g128.json,与 EXP-002 §5 一致)。
- **产物路径**:scripts/run_all{,2}.sh、scripts/plot_alpha_dist.py、
  figures/fig1_awq_alpha_dist.png、ENV.md、records/TEMPLATE.md、
  llmqt_example/README.md、data/raw/EXP-00{1,2,3}/manifest.txt(勘注)、
  records/EXP-001 §3/§7(勘注)、README(警示+如实表述)、docs/theory/01–03
  (frontmatter)、docs/talk/quant_walkthrough.md(挂图)。
- **下一步**:用户创建 github.com/lyell0710/LLM_Quantization 远端后首推全量;
  GPU 空闲后可选 EXP-004(clip/块级共享 s/act_order)、EXP-005(α 尾部)。

## §4 README 升级为门面级 + 恢复率图(2026-08-25)

- **做了什么**:README 重排为门面结构(🎯结果一览带指针 → 📊图表 →
  🔬代码导览(src/gptq.py Cholesky+补偿主循环节选)→ 复现 → 结构 → 台账 →
  红线表+诚实度方法论 → 相关仓 GitHub 链接);新增 scripts/plot_recovery.py
  从 data/raw/EXP-00{1,2,3} 各臂 JSON 复算恢复率,产出
  figures/fig2_recovery_rates.png(水平条形,双赛道分组,条端标值);
  fig1 复用。台账与红线表原样保留(仅位置下移)。
- **为什么**:秋招投递前门面化——面试官 30 秒扫读可见数据/图/代码/方法论;
  数字全部走 raw 复算或现行文档转录,不新造。
- **关键数字**:fig2 复算 62.6 / 61.6 / 31.9 / 48.5%,与 records 一致
  (48.5% 即 EXP-003 §6 取整表述的 48%,图脚注注明);无新增测量。
- **产物**:README.md、scripts/plot_recovery.py、figures/fig2_recovery_rates.png、
  本节。
- **下一步**:用户建远端 github.com/lyell0710/LLM_Quantization 后 push;
  可选 EXP-004(clip/块级共享 s)/EXP-005(α 尾部/静态激活)。

## §5 README 对外/对内分家,新建 LEDGER.md(2026-08-25)

- **做了什么**:①新建 LEDGER.md 为状态与措辞唯一权威,从 README 搬入:
  实验台账(日期/状态列)、措辞红线表(7 条)、无远端警示与待办
  (远端/GPU/EXP-004/005)、勘误与审计留痕、289QS 异协议说明、内部硬约定;
  ②README 重写为纯对外门面(读者=陌生面试官):一句话+动机 → 🎯核心结果
  (测量条件作自然定语)→ 📊图表 → 🧠关键发现(四段机制解释:二阶 vs 一阶
  定价/叠加重叠/SmoothQuant 规模论证+α=0.25 反例/per-layer α 自适应)→
  🔬代码导览 → 🚀复现 → 结构 → 📚实验记录索引(无日期无状态,一行一句
  结论)→ 🧪测量方法(对外化:可溯源/误差条/对照反例臂/负结果照报)→
  🔗相关项目;README 禁词逐一清零(日期/勘误/审计/红线/台账/待办类);
  ③出图脚本脚注去日期(保留源数据+硬件+轮数),fig1/fig2 重出;
  ④CLAUDE.md 加附则(README=对外门面;唯一权威=LEDGER.md);
  HOW_TO_LEARN §6 与 llmqt_example/README 的红线/异协议引用改指 LEDGER.md。
- **为什么**:用户反馈 README 读起来像作者实验记录,而读者应是第一次打开
  仓库、只有 60 秒的面试官——对外讲机制与数字,对内流程全部收进账本。
- **关键数字**:无新测量;fig1/fig2 由既有 raw 重算,数值不变
  (62.6/61.6/31.9/48.5%;α 中位 0.30,n=168)。
- **产物**:LEDGER.md、README.md(重写)、scripts/plot_{recovery,alpha_dist}.py
  (脚注)、figures/fig{1,2}(重出)、CLAUDE.md 附则、
  docs/HOW_TO_LEARN_A_QUANT_METHOD.md、llmqt_example/README.md、本节。
- **下一步**:待用户建远端后首推;可选 EXP-004/005(见 LEDGER 待办)。
