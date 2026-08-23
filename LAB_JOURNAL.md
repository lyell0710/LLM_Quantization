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
