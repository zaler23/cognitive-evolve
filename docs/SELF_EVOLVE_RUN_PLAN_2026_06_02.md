# 自举运行优化计划 · 2026-06-02

> 状态：仅计划，未改代码。来源：2026-06-02 第二次自举 run（stop @ round 7/48，67 候选全 Dormant，final gate 未过）的复盘，对照早先 L1–L4 杠杆方案。
> 总目标：让这套自举 run 偏发散/探索；前中期不卡死，只在关键节点验证。渠道问题暂不处理（计划改用稳定渠道）。

## 杠杆对账（L1–L4）

| 杠杆 | 状态 | 说明 |
|---|---|---|
| L1 拆分意图 | 设计完成，待生效 | 改为**一份**探索版 goal（非两份 profile）；已写 `goal-探索版.md`，但需接进 run（见 Part B） |
| L2 通道韧性 | 主动搁置 | 换稳定渠道后再处理；已落地的空回复/截断/429 分类保留 |
| L3 patch preflight | 就绪 | (a) token 不足由 A1/A2 解决；(b) 截断 diff 进 repair lane 已实现（`failure_classifier.py:284,293`） |
| L4 探索类 final gate | **未解决，本计划新增 A3** | artifact contract 仍只认可执行 work_product，会反噬探索 goal |

---

## Part A · 代码改动（仅计划）

### A1 — 列表型生成节点纳入长输出预算
- 文件：`cognitive_evolve_runtime/llm/transport.py:40` `LONG_CONTEXT_OUTPUT_REQUESTS`
- 当前集合：`nexus_synthesize_result / nexus_diagnose_search_state / nexus_generate_offspring`
- 加入：`nexus_seed_population`、`nexus_plan_mutations`、`nexus_critique_candidates`
- 理由：三者都返回列表（`model_adapter.py:80/139/104`），却只拿 4096 light 预算。`seed_population` 是初始种群、发散广度主战场，从第 0 轮就被压窄——与"目标偏窄、Active 掉回 Dormant"症状一致。
- 风险：低。只抬上限，token 实际用量由模型决定。

### A2 — 长输出默认 16384 → 32768
- 文件：`cognitive_evolve_runtime/llm/transport.py:60`
- 改 `env_int(LLM_LONG_MAX_TOKENS_ENV, 16384)` 默认值为 `32768`
- 理由：reasoning-heavy OpenAI-compatible upstream 的 `max_tokens` = 思考 + 可见输出之和；16K 常被思考烧光，导致完整 unified diff 写一半 EOF（本轮 `unexpected end of file`×17）。
- 保留：`COGEV_LLM_LONG_MAX_TOKENS` / `COGEV_LLM_MAX_TOKENS` 覆盖、light 4096 不动。

### A3 — artifact contract 承认探索期产物（design_candidate）【本轮新增，闭合 L4】
- 问题：探索版 goal 在**提示层**说"design_candidate 合法、别判死"，但运行时 gate 仍按"必须有可执行 work_product"拦截（本轮 `required_work_product_absent` 等各 134 次），会把模型按 goal 交的设计候选压回 Dormant——goal 与 contract 矛盾。
- 落点 1：`nexus/artifact_contract.py:213` `evaluate_candidate_against_dynamic_contract`，第 228–238 行的 `final_eligible`/`rank_eligible` 判定。
  - 引入"探索期/设计候选"通道：当候选具备**机制描述 + evaluation_dimensions + 与现有设计的 diff + 失败条件**（结构完整）但尚无可执行 artifact 时，**不计入 `rank_eligible` 的否决项**（当前 `artifact_object_absent`/`meta_commentary_only` 会直接踢掉它），让它能继续被排名、繁殖、留在 Active/Incubating，而不是沉 Dormant。
  - `final_eligible` 仍**保持严格**：design_candidate 永远 `final_eligible=False`，并标 `non_final / 待落地`。
- 落点 2：`contracts/objective_contract.py:606` 契约生成处的 `required_work_product`。
  - 在探索运行下，允许 `allowed_artifact_shapes` 包含一类 `design_candidate` 形状（结构化设计描述即合法 work product），使 `validate_dynamic_artifact_contract`（`artifact_contract.py:194-209`）不因 contract 本身缺可执行产物而判 `required_work_product_absent`。
- 护栏（守诚实红线）：
  - design_candidate **绝不** `final_eligible`，输出必标 `non_final`，不冒充已验证/客观最优。
  - 不能退化成"叙述即通过"的后门——必须满足"结构完整"四要素（机制 + evaluation_dimensions + diff + 失败条件），否则仍判 `meta_commentary_only`。
  - 该通道应**仅在探索运行下放开**（由 goal/契约信号控制），落地运行（窄 run）仍走严格 work_product gate。
- 与 A1/A2 的关系：A1/A2 让模型**能**产出更完整的候选；A3 让"完整但非 patch"的设计候选**不被 gate 误杀**。两者配合才让探索 goal 真正生效。

### A4 — 严格 final gate 只用于最终 final answer【新增边界】
- 问题：最近一轮已经能生成 patch、repair offspring 和设计候选，但大量候选在中间轮次因 `required_work_product_absent`、`final_gate_absent`、`meta_commentary_only`、`proof_object_absent`、`evidence_ref_absent` 等**最终答案门禁类诊断**被过早压成 Dormant，最终触发 `no_parents_available`。
- 原则：中间演化阶段只判断候选是否有继续发展的价值；严格 final gate 只在最终答案选择/合成时执行。
- 落点：
  - `nexus/stage_policy.py`：把动态 artifact contract / final gate 完整性问题归为 pre-final repair diagnostics；只要候选非空、有机制/来源/证据/repair target，就保留为 Incubating/parent material，而不是硬拒绝。
  - `archives/manager.py`：Active floor 不再只限 early/middle；任何最终合成前的搜索轮都不能因为候选尚未 final-eligible 而归零 Active。
- 护栏：
  - `ArchiveManager.is_final_answer_eligible()` 与 `nexus/final_gate.py` 仍保持严格；Incubating、未验证、未补证据、未补 final gate 的候选永远不能作为 final answer。
  - docs-only、seed-note-only、第二套 runtime、隐藏 fallback、明显跑题、终态失败等仍硬拒绝。
  - 这不是降低最终门禁，而是把"可发展材料"和"可发布最终答案"拆开。

### A5 — 搜索空间宽度由模型定义，避免低层实现面垄断【新增】
- 问题：即使 A1–A4 生效，真实 run 仍可能因项目快照里的易验证文件而收敛到 `engine_runner / executor / preflight` 等低层 surface，和"整体自举/自进化核心机制"这类输入期望不一致。
- 原则：文件、测试、工具、patch surface 只是 grounding context，不是搜索目标本身。搜索空间应由模型从用户目标/动态 artifact contract 中声明，而不是由 runtime 注入 `minimal_patch`、`test_first` 等低层项目默认族群。
- 落点：
  - `nexus/search_space.py`：删除 proof/code 等有限领域搜索族群；优先使用模型给出的 `search_space_plan / exploration_planes / candidate_families`，缺失时只生成 objective-derived placeholder，并显式要求模型替换。
  - `nexus/policy.py`：项目 fallback 不再自动注入 `minimal_patch / architecture_refactor / test_first` 等默认 niche；若没有模型 search plan，只标记需要模型声明搜索空间。
  - `nexus/prompt_view.py`：seed / mutation / offspring / diagnose 请求都携带 `search_space_contract`，提示模型先覆盖不同 objective-level planes，再深入某个局部 surface；population stats 暴露 `top_search_planes`、`top_source_surfaces` 和 surface concentration warning。
  - `nexus/exploration.py`：探索种子优先来自模型定义 search planes；项目任务不再默认只走项目 patch seed。
- 护栏：
  - 不把“高层”硬编码成科学、代码、文章、小说等领域类别。
  - 仍要求候选产生实际 artifact / repair obligation；只是不要让最容易 patch 的局部文件挤掉候选生命周期、父代选择、materialization、final gate 等更高层方向。

---

## Part B · 让探索版 goal 进入 run（L1 生效）

- 现状：本轮 `bootstrap-goal.md` 是**手工拼装**的旧前言（前言 13 处 patch/low-risk → 逼模型收敛到安全小补丁）；`run-core-self-evolve-openai.py` 只认 `--goal-file`，对 `input_zip` 不做处理。探索版 goal 曾位于本地输入包目录，公开版不保留操作者下载路径；需通过 `--goal-file` 显式传入。
- B1：下次 run 的 `bootstrap-goal.md` 用 `goal-探索版.md` 前言 + 解压的输入包正文（`00–06`+附录）重拼，经 `--goal-file` 传入。
- B2（可选治本）：把"goal 前言 + 解压输入包 → bootstrap-goal.md"固化成 `scripts/` 下的小包装（接 `--input-zip` + `--goal-preamble`），避免每次手工拼、也避免再拼回旧前言。使探索 goal 成为可复现运行参数。

---

## Part C · 验证

1. `python -m compileall -q cognitive_evolve_runtime` —— 编译通过。
2. `python -m pytest -q` —— 确认仍 309 passed/1 skipped。
   - 重点：A1/A2 看涉及 token 预算/长输出集合的测试（如 `test_nexus_api_budget_model_binding`）是否断言旧值 16384 或旧集合，若是则同步更新。
   - 重点：A3 看 artifact contract 相关测试（如 `test_dynamic_artifact_contract`）是否需新增 design_candidate 通道的用例，确保 `final_eligible` 仍对 design_candidate 为 False（护栏不被破）。

## 待拍板

1. A1 范围：三个全加，还是只加 `seed_population`（最痛点）？
2. A2 数值：32K 起步，或更激进 48K？
3. A3 放开条件：design_candidate 通道"仅探索运行放开"由什么信号控制（goal 标记 / 契约字段 / 环境变量）？
4. B2 是否固化成脚本，还是这次仍手工拼前言？
