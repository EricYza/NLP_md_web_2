# About Me

> **Ph.D. in Computer Science & Technology · Large Language Model Researcher · Tech Blogger**
> 面向“研究—工程—产品—传播”闭环的 LLM 研究者，用可复现实验、可维护系统与清晰写作把 SOTA 变成 SOP。

---

## 一页速览（Executive Summary）

| 维度   | 亮点                                                          | 证明方式       |
| ---- | ----------------------------------------------------------- | ---------- |
| 研究方向 | LLM 对齐（DPO/RLHF/RLAIF）、推理与代理、RAG 可信引用、轻量化训练（LoRA/QLoRA/MoE） | 代表性项目与开源仓库 |
| 工程化  | 分布式训练/推理、评测平台搭建、A/B 与灰度发布、可观测性                              | 系统架构与迭代节奏  |
| 影响力  | 技术博客长期连载、内部工作坊/分享可复用                                        | 阅读/复用与反馈数据 |
| 方法论  | “评测先行、数据为王、工程为本、合规内建”                                       | 项目复盘与指标闭环  |

---

## 核心价值（What I Bring）

* **把 SOTA 变成 SOP**：把最新论文沉淀为可执行流程与模板。
* **把模型变成系统**：兼顾可部署、可观测、可回滚。
* **把指标变成体验**：设计可信评测，让离线提升能迁移到线上。
* **把复杂变简单**：以结构化提示、代码样例与可复现实验降低门槛。

---

## 能力矩阵（Skill Matrix）

| 能力域    | 子项                                          | 熟练度   | 备注           |
| ------ | ------------------------------------------- | ----- | ------------ |
| 训练与优化  | PyTorch、DeepSpeed、FSDP、ZeRO、LoRA/QLoRA、混合精度 | ★★★★★ | 端到端自建/迁移经验   |
| 对齐与偏好  | SFT、DPO/IPO、RLHF/RLAIF、偏好数据生成与清洗            | ★★★★★ | 注重稳定性与样本难度控制 |
| 推理与代理  | CoT/ToT/GoT、函数调用、工具生态与规划执行                  | ★★★★☆ | 读—思—查—写闭环    |
| RAG    | 分片与召回、重排、可信引用、向量库                           | ★★★★☆ | 长文档与法规场景落地   |
| 评测与可靠性 | 幻觉检测、鲁棒性、事实一致性与不确定性                         | ★★★★★ | 自动回归测试与告警    |
| 系统工程   | 分布式训练集群、CI/CD、Tracing/Profiling/Logging     | ★★★★☆ | 支撑高频迭代与灰度    |

---

## 研究兴趣（Research Interests）

* **对齐与反馈学习**：DPO/ORPO vs RLHF 的稳态训练与样本难度分配。
* **推理与工具使用**：从提示工程到多步规划与环境交互。
* **高效化与稀疏化**：QLoRA、量化/蒸馏、MoE 路由与稳定性。
* **评测与安全**：事实一致性、幻觉治理、红队/风控与合规。
* **多模态与长上下文**：跨模态检索与记忆、长文档状态管理。

---

## 代表性项目（Selected Projects）

1. **Reasoning-First LLM（读—思—查—写闭环）**

   * 提升复杂任务成功率与可解释性，通过**分层提示模板 + 检索重排 + 奖励重排序**减少幻觉。
2. **轻量化对齐流水线（小算力可复现）**

   * **偏好数据自举 → SFT → DPO** 一体化，稳定收敛，效果接近重算力 RLHF。
3. **可信 RAG for 长文档/法规**

   * **语义分片 + 检索重排 + 引用可追溯**，将答案与来源强绑定，降低误导成本。
4. **统一评测与可观测平台**

   * 覆盖学术集与业务私有集，支持**自动回归测试、对比实验与指标告警**。

---

## 方法论（Playbooks）

* **数据治理**：去重/去毒化、难例挖掘、样本配比搜索（Mixture-of-Datasets）。
* **稳定训练**：梯度裁剪、loss 曲线卫星指标、对齐阶段学习率热启策略。
* **可信评测**：场景—能力—风险三维度，指标与用户体验对齐。
* **上线策略**：灰度发布、保护阈值与回滚、冷启动与缓存治理。

---

## 论文与出版物（Publications & Writing）

> 说明：以下分为两部分。**A. 真实/可核验条目（留空或待补）**；**B.【示例/占位】条目**仅用于版式演示与叙事占位，**不代表真实已发表**。待有真实论文后可一键替换。

### A. 真实/可核验（示例：目前待补）

* （请在有真实发表后添加：作者、题目、会议/期刊、年份、链接/DOI）

### B. 【示例/占位】（可改为“在投/预印本草稿（示例）”）

| 类型 | 作者     | 标题                                                                                      | 会议/期刊           | 年份   | 备注                |
| -- | ------ | --------------------------------------------------------------------------------------- | --------------- | ---- | ----------------- |
| 占位 | 你的名字 等 | **Direct Preference Optimization at Scale: Stabilizing Alignment with Hard-Neg Mining** | ICLR（【示例】在投）    | 2026 | 大规模 DPO 稳定训练与难例挖掘 |
| 占位 | 你的名字 等 | **Retrieval-First Agents: Closing the Loop for Read-Think-Search-Act**                  | NeurIPS（【示例】在投） | 2026 | 代理式检索与执行闭环        |
| 占位 | 你的名字 等 | **Trustable RAG: Calibrated Citations and Hallucination Audits**                        | TACL（【示例】预印本）   | 2026 | 可信引用与幻觉审计         |
| 占位 | 你的名字 等 | **QLoRA++: Memory-Efficient Alignment under Budget Constraints**                        | JMLR（【示例】草稿）    | 2026 | 资源受限对齐的参数高效化      |


---

## 博客与分享（Blogging & Talks）

| 形式    | 主题               | 受众/场景     | 产出         |
| ----- | ---------------- | --------- | ---------- |
| 系列长文  | LLM 对齐实战（从数据到上线） | 工程/研究混合团队 | 教程、脚本与清单   |
| 技术清单  | RAG 可信引用与评测      | 法规/企业文档问答 | Demo 与指标报表 |
| 内部工作坊 | 分布式训练与可观测        | 平台与基础架构   | 诊断手册与最佳实践  |


---

## 开源与社区（Open Source & Community）

* 贡献训练/评测工具链、提示模板与数据处理组件。
* 维护“高质量最小可复现（MRR）”范例，便于教学与二次开发。
* 为初学者提供路线建议与代码审阅，推动规范与文化共建。

---

## 教育背景（Education）

* **Ph.D., Computer Science & Technology**
  研究方向：LLM 对齐、推理与可靠性；兼顾系统与工程实践。

---

## 联系方式（Get in Touch）

* **Email**：your.name [at] example.com
* **Blog**：yourblog.example.com
* **GitHub**：github.com/yourhandle
* **WeChat/Telegram**：来信索取

---

## 附录

### 1) 论文 BibTeX（占位）

```bibtex
@inproceedings{yourkey2026dpo,
  author = {Your Name and Coauthor, X.},
  title = {Direct Preference Optimization at Scale: Stabilizing Alignment with Hard-Neg Mining},
  booktitle = {ICLR},
  year = {2026},
  note = {【示例/占位】替换为真实发表信息}
}
```

### 2) 指标看板（占位表）

| 指标                 | 定义          | 当前值 | 目标      |
| ------------------ | ----------- | --- | ------- |
| Hallucination Rate | 不可验证回答占比    | —   | < 5%    |
| Faithfulness@K     | 与引用一致的答案占比  | —   | > 95%   |
| Latency (p95)      | 端到端响应 95 分位 | —   | < 800ms |


