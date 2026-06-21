# M5.1 Latent Objective Evolution 接入计划

更新时间：2026-06-04

## 0. 当前代码事实

已核对本地真实文件：

- `cognitive_evolve_runtime/outcomes/improvement.py` 已提供 M5 Outcome Improvement Kernel：
  - `OutcomeContract`、`OutcomeMetric`、`TrialObservation`
  - `ImprovementCertificate`、`ImprovementEdge`
  - `compare_outcomes()`、`verify_certificate()`、`improvement_edge()`
  - 证书语义是 fail-closed：只有同一 contract/basis/evaluator、硬约束通过、独立 verifier、LCB 达标时才是 verified。
- `cognitive_evolve_runtime/outcomes/latent.py` 已提供 M5.1 Latent Problem-Space Kernel：
  - `IntentHypothesis`、`LatentProblemState`、`FrontierCandidate`、`ExplorationAction`
  - `rank_candidates()`、`pareto_frontier()`、`select_exploration_action()`
  - `freeze_outcome_contract()`、`assess_convergence()`
- 当前 M5/M5.1 主要停留在独立模块和单测，尚未系统接入 Nexus contract、candidate ranking、closure certificate、stop/continue 语义。
- `NexusObjectiveContract` 当前是 Nexus 运行时的冻结目标边界；`NexusRuntime.run_text/run_project/resume_from_checkpoint` 会持久化 contract/world/evolution。
- `RoundPipeline.rank()` 是每轮 candidate ranking 后、archive assignment 前的低风险接入点。
- `_closure_certificate()` 是 `objective_solved` 的最终本地门禁；当前只检查 stop reason / interruption / synthesis status。
- `StopDecisionEngine.stop_reason_after_round()` 是每轮 stop/continue 的本地决策入口。

## 1. 总体边界

固定层级：

```text
M4 = 开放世界演化底座：任意 artifact、候选、事件、档案、验证、生成计划。
M5 = Outcome Improvement Kernel：证明 challenger 是否相对 baseline 更优。
M5.1 = Latent Problem-Space Evolution Layer：在目标/空间/评价标准都模糊时，逐步发现什么叫“更优”。
```

核心原则：

```text
M5.1 负责发现/收缩潜在目标空间；
M5 负责把局部候选改进冻结成可验证证书；
solved 不能只靠模型叙事、排名、former routed-output label 或 evaluator 分数。
```

本轮不做范围：

- 不重写整个 `NexusRuntime`。
- 不改 LLM provider / HTTP retry / transport。
- 不做 Git commit / push / PR。
- 不启动长期外部模型自举 run。
- 不做无关 UI/API 大改。
- 不把所有任务强行判成 latent ambiguity。

## 2. 需要修改的文件列表

计划修改/新增：

1. `cognitive_evolve_runtime/outcomes/runtime_bridge.py`：新增 M5/M5.1 与 Nexus 运行路径之间的轻量连接层。
2. `cognitive_evolve_runtime/outcomes/__init__.py`：导出 runtime bridge helper。
3. `cognitive_evolve_runtime/contracts/objective_contract.py`：给 `NexusObjectiveContract` 增加可序列化 `metadata`，并在 builder 中按需附加 latent state 摘要/hash。
4. `cognitive_evolve_runtime/nexus/runtime.py`：确保 checkpoint 恢复路径中的 contract/world 能保留 latent metadata。
5. `cognitive_evolve_runtime/nexus/loop/`：每轮 ranking 后写入 latent ranking signals；closure certificate 接入 M5 certificate 字段；finalize 时执行 latent convergence override。
6. `cognitive_evolve_runtime/nexus/stop_decision.py`：stop/continue 前读取 latent convergence gate，防止高熵无证书任务过早 solved。
7. `cognitive_evolve_runtime/nexus/synthesis.py`：former routed-output label/answer ordering 使用 latent rank signal，但不把它等同于 solved。
8. `cognitive_evolve_runtime/ranking/parent_selection.py`：父代选择纳入 latent reproductive / Pareto signal，帮助不同 latent intent 下的候选保留繁殖机会。
9. 新增测试：`tests/test_m5_1_runtime_integration.py`。
10. 更新本文件作为实施报告。

## 3. 任务、成功标准与测试

### Task 1：M5.1 状态接到 contract/world metadata

成功标准：

- `NexusObjectiveContract` 能序列化/反序列化 `metadata`。
- 模糊/开放目标会生成 `latent_problem_state_summary`、`latent_problem_state_hash`、`latent_problem_state`。
- 明确任务不被错误扩大成 latent ambiguity。
- model/provider 可覆盖本地 fallback；本地 fallback 只作为 degraded initialization。

测试：

- `test_contract_builder_attaches_latent_state_for_ambiguous_goal()`
- `test_clear_goal_does_not_force_latent_state()`
- contract roundtrip metadata preservation。

### Task 2：candidate ranking 与 latent posterior 对接

成功标准：

- 候选可通过 `metadata` 或 `verification_result` 携带：
  - `latent_intent_scores`
  - `latent_uncertainty`
  - `latent_risk`
  - `latent_cost`
- 每轮 ranking 后写入：
  - `metadata.latent_ranking`
  - `multihead_scores.latent_reproductive_signal`
  - `multihead_scores.latent_expected_utility`
  - `metadata.latent_pareto_frontier`
- 高单一总分但高风险/高不确定候选不会压过稳定候选。
- 不同 latent intent 下的 Pareto 候选都能保留正向 selection signal。

测试：

- `test_latent_ranking_penalizes_high_risk_candidate_in_runtime_metadata()`
- `test_latent_pareto_frontier_keeps_distinct_intent_candidates()`

### Task 3：M5 certificate 接入 closure certificate

成功标准：

- `_closure_certificate()` 兼容新增字段：
  - `improvement_certificate_hash`
  - `improvement_verified`
  - `baseline_id`
  - `challenger_id`
  - `aggregate_lcb`
  - `improvement_critical_failures`
- 旧路径无 certificate 不崩溃。
- `requires_verified_solution=true` 时，缺证书不能 solved，critical failures 包含 `missing_verified_improvement_certificate`。
- verified certificate 存在时暴露 certificate hash 与 delta，不由 former routed-output label 冒充。

测试：

- `test_requires_verified_solution_blocks_solved_without_m5_certificate()`
- `test_verified_m5_certificate_is_exposed_in_closure_certificate()`

### Task 4：latent convergence 接入 stop/continue 语义

成功标准：

- 高熵 latent objective + valuable exploration remains + 无 verified improvement certificate 时，不允许提前 solved。
- 低熵 latent objective + verified certificate + 低价值下一步动作时，允许 solved。
- 对无 latent state 或明确短任务保持现有行为。

测试：

- `test_stop_decision_blocks_solved_when_latent_space_unresolved()`
- `test_stop_decision_allows_solved_when_latent_converged_and_verified()`

### Task 5：文档与验证报告

成功标准：

- 本文件更新实际改动、任务状态、测试结果、达成度与下一步风险。
- 指定回归与全量测试完成。

## 4. 验证命令

```bash
cd <repo-root>
./.venv/bin/pytest tests/test_m5_1_runtime_integration.py -q
./.venv/bin/pytest tests/test_latent_problem_space_kernel.py tests/test_outcome_improvement_kernel.py tests/test_generation_plan.py tests/test_security_config_and_stop_decision.py tests/test_nexus_adaptive_semantics.py tests/test_nexus_text_evolution_loop.py -q
./.venv/bin/pytest -q
```

## 5. 当前状态

- 计划已落盘。
- 实现与验证待完成。

## 6. 实施报告（2026-06-04）

### 6.1 实际改动文件

- `cognitive_evolve_runtime/outcomes/runtime_bridge.py`：新增 M5/M5.1 runtime bridge helper。
- `cognitive_evolve_runtime/outcomes/__init__.py`：导出 runtime bridge helper。
- `cognitive_evolve_runtime/contracts/objective_contract.py`：
  - `NexusObjectiveContract` 增加可序列化 `metadata`。
  - builder 在 contract 创建/模型返回后尝试附加 latent objective state。
  - `canonical_payload()` 排除 runtime metadata，避免 posterior/latent 状态漂移破坏冻结目标 hash。
- `cognitive_evolve_runtime/nexus/runtime.py`：run/resume world payload 附带 latent state summary/hash，便于持久化和审计。
- `cognitive_evolve_runtime/nexus/loop/`：
  - ranking 后写入 latent posterior / Pareto signals。
  - generation plan 在写入 latent ranking summary 后重新计算 plan_id，保证 checkpoint/resume 验证不漂移。
  - final closure certificate 接入 M5 improvement certificate 摘要。
  - finalization 增加 latent convergence override：latent 未收敛时，solved 降级为 needs_continuation。
- `cognitive_evolve_runtime/nexus/stop_decision.py`：model/self-observed solved 前增加 latent convergence guard。
- `cognitive_evolve_runtime/nexus/synthesis.py`：best-current/reference 排序纳入 latent signal，但仍不把 best-current 当 solved。
- `cognitive_evolve_runtime/ranking/parent_selection.py`：父代 reproductive value 纳入 latent posterior/Pareto 保留信号。
- `tests/test_m5_1_runtime_integration.py`：新增 8 条 runtime 集成测试。

### 6.2 任务完成状态

- Task 1：完成。模糊目标会生成并携带 `LatentProblemState` 摘要/hash；明确短任务不会被强制扩大成 latent ambiguity。
- Task 2：完成。candidate metadata / multihead scores 会携带 latent ranking、uncertainty/risk/cost penalty、Pareto frontier signal，并被 parent selection/answer ordering 消费。
- Task 3：完成。closure certificate 暴露 M5 certificate 字段；`requires_verified_solution` 缺 verified M5 certificate 时不能 `objective_solved=true`。
- Task 4：完成。latent convergence assessment 已接入 stop/continue 与 finalization；高熵/仍有高信息增益/缺证书会降级为 continuation。
- Task 5：完成。本文件已更新实际改动、测试结果、达成度和风险。

### 6.3 验证结果

已执行：

```bash
./.venv/bin/python -m compileall -q cognitive_evolve_runtime tests/test_m5_1_runtime_integration.py
./.venv/bin/pytest tests/test_m5_1_runtime_integration.py -q
# 8 passed
./.venv/bin/pytest tests/test_latent_problem_space_kernel.py tests/test_outcome_improvement_kernel.py tests/test_generation_plan.py tests/test_security_config_and_stop_decision.py tests/test_nexus_adaptive_semantics.py tests/test_nexus_text_evolution_loop.py tests/test_m5_1_runtime_integration.py -q
# 53 passed
./.venv/bin/pytest -q
# 368 passed, 1 skipped
```

### 6.4 当前达成度估计

- M5 Outcome Improvement Kernel 本体：约 70%。核心证书、比较、verify、closure 暴露已具备；尚未覆盖真实外部 verifier 的完整证据采集自动化。
- M5.1 Latent Problem-Space Evolution Layer 本体：约 55%。已有 latent posterior、candidate ranking、Pareto、convergence，且已接入 Nexus contract/ranking/stop/final closure；但 posterior 更新仍主要依赖候选 metadata/运行证据，尚未形成跨轮主动 probe 的完整闭环。
- M5.1 接入真实运行路径：约 45%。已进入 contract、world payload、ranking、parent selection、synthesis、closure、stop decision；下一阶段应让 critique/verifier/archive 产出的 evidence 自动更新 latent posterior，而不是只作为静态初始化和 ranking signal。

### 6.5 下一步剩余风险

1. `LatentProblemState` 当前通过 contract metadata 携带；hash 不影响 frozen contract hash，但 posterior 跨轮更新还没有单独 durable ledger。
2. M5 certificate 当前可从 candidate metadata / verification_result / obligation_delta 中提取；真实 verifier 还需把 `TrialObservation` 与 certificate 自动写回候选。
3. Pareto 多样性已进入 parent selection 信号，但还不是独立 archive lane；极端 compaction 下仍可能丢失少数 intent frontier。
4. stop/closure 已防止 latent 未收敛时伪 solved，但不会主动生成下一步高信息增益 probe；这应由后续 mutation/critique planner 接入。

### 6.6 子代理复核后的补充修复

只读子代理复核确认：M5.1 主链路基本覆盖 Task 1-5，未发现 `former routed-output label` 冒充 solved 或 `requires_verified_solution` 缺证书仍 solved 的直接绕过；但指出 `generation_plan.plan_id` 在后续写入 `parent_ids` / `mutation_objectives` 后可能漂移。

已补充修复：

- `cognitive_evolve_runtime/nexus/loop/` 增加 `_refresh_generation_plan_id()`。
- 在 reproduction 阶段写入 `parent_ids` 与 `mutation_objectives` 后立即重新计算 `plan_id`。
- 保留此前 latent ranking 写入后的 `plan_id` 刷新，确保 checkpoint/resume 的 `expected_generation_plan_id == plan_id`。

补充验证：

```bash
./.venv/bin/pytest tests/test_nexus_remaining_capabilities.py::test_nexus_project_runtime_verifies_context_and_resume tests/test_m5_1_runtime_integration.py tests/test_generation_plan.py -q
# 18 passed
./.venv/bin/python -m compileall -q cognitive_evolve_runtime tests/test_m5_1_runtime_integration.py
./.venv/bin/pytest tests/test_latent_problem_space_kernel.py tests/test_outcome_improvement_kernel.py tests/test_generation_plan.py tests/test_security_config_and_stop_decision.py tests/test_nexus_adaptive_semantics.py tests/test_nexus_text_evolution_loop.py tests/test_m5_1_runtime_integration.py -q
# 53 passed
./.venv/bin/pytest -q
# 368 passed, 1 skipped
```
