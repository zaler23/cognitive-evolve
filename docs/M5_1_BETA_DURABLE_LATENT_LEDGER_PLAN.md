# M5.1-beta: Durable Latent Ledger + Evidence-to-Posterior Feedback Loop

更新时间：2026-06-05

## 0. 当前结论

M5.1-alpha 已经把 latent objective/search-space layer 接入 Nexus 真实运行路径：contract、world payload、ranking、parent selection、synthesis、closure certificate、stop decision 都已经能感知 latent state 或 M5 improvement certificate。

三路外部模型审计结论一致：下一阶段应命名为：

```text
M5.1-beta = Durable Latent Ledger + Evidence-to-Posterior Feedback Loop
```

核心判断：

```text
M5.1-alpha 让 latent objective 可以影响运行路径；
M5.1-beta 必须让 latent objective 的变化可持久、可审计、可重放、可由真实 evidence 自动更新。
```

否则 latent posterior 只是 runtime metadata 和 ranking signal，不是可信的长期演化状态。

## 1. 多模型审计来源（公开版摘要）

来源形态：

- 通过操作者配置的 OpenAI-compatible `/v1/chat/completions` 审计端点完成；
- 使用短审计包与串行间隔请求；
- 原始供应商响应、操作者账号环境、私有工具路径和运行目录不纳入公开源码；
- 公开版只保留去标识化后的模型共识，不保留私有接入层、私有路径或具体账号/模型路由。

模型共识：

1. 接受 M5.1-alpha 当前方向。
2. M5.1-beta 应先做 durable latent ledger，再做 evidence-to-posterior feedback loop。
3. feedback loop 必须有 provenance、dedup、replay、bounded update、source weighting、retraction/supersession。
4. ExplorationAction → mutation/critique planner、Pareto intent archive lane、完整 TrialObservation → ImprovementCertificate 自动生成应作为 beta 后续或受控集成，不应抢在 ledger 之前扩张。

## 2. 当前代码事实

已完成 M5/M5.1-alpha 事实：

- `cognitive_evolve_runtime/outcomes/improvement.py`
  - `OutcomeContract`
  - `TrialObservation`
  - `ImprovementCertificate`
  - `compare_outcomes()`
  - `verify_certificate()`

- `cognitive_evolve_runtime/outcomes/latent.py`
  - `IntentHypothesis`
  - `PreferenceEvidence`
  - `LatentProblemState`
  - `FrontierCandidate`
  - `ExplorationAction`
  - `rank_candidates()`
  - `pareto_frontier()`
  - `assess_convergence()`

- `cognitive_evolve_runtime/outcomes/runtime_bridge.py`
  - latent state attaches to contract metadata
  - world payload carries latent summary/hash
  - candidate ranking writes latent signals
  - M5 certificate summary feeds closure certificate
  - latent convergence can block premature solved

- `NexusObjectiveContract.metadata`
  - carries latent metadata
  - excluded from canonical contract hash to avoid posterior drift mutating frozen objective identity

- `nexus/loop/`
  - closure certificate exposes M5 fields
  - `requires_verified_solution` without verified M5 certificate cannot become objective_solved
  - generation plan ID refresh prevents checkpoint/resume drift after latent ranking / parent / mutation metadata changes

当前验证：

```text
tests/test_m5_1_runtime_integration.py: 8 passed
M5/M5.1 + Nexus regression: 53 passed
full suite: 368 passed, 1 skipped
```

## 3. 当前最大缺口

M5.1-alpha 的缺口不是字段不够，而是缺少闭环：

```text
真实运行 evidence
  -> PreferenceEvidence
  -> durable latent ledger
  -> deterministic posterior update
  -> posterior snapshot / cursor
  -> ranking / parent selection / synthesis / closure / stop
  -> decision trace
```

当前还没有做到：

1. `LatentProblemState` 主要由 contract metadata 携带，没有独立 durable ledger。
2. critique / verifier / archive / trial / certificate 输出还没有自动转换为 `PreferenceEvidence`。
3. posterior 更新没有 append-only 事件源，也不能通过 replay 确定性重建。
4. ranking/parent/synthesis/stop 还没有读取“固定 ledger cursor 的 posterior snapshot”。
5. duplicate/retraction/supersession/conflict 语义还没定义。
6. posterior update 缺少 damping、source cap、decay 和 collapse guard。

## 4. M5.1-beta 范围边界

### 4.1 本阶段必须做

- durable latent ledger
- typed `PreferenceEvidence` ingestion
- critique/verifier/archive/certificate evidence adapters
- deterministic posterior replay
- posterior snapshot/cache
- idempotent dedup
- retraction/supersession
- bounded update / source weighting / decay
- decision trace
- 至少一个 evidence -> posterior -> downstream decision 的闭环集成测试

### 4.2 本阶段只做接口或 shadow，不做大扩张

- `ExplorationAction -> mutation/critique planner`
- Pareto intent archive lane
- 完整自动 trial/certificate 生产系统
- 大规模 exploration policy 重写

这些应进入 M5.1-rc 或 M5.2。Beta 只保留 schema forward-compat 字段和最小 hook。

## 5. 主要风险与防护

### 风险 1：posterior corruption / non-idempotent replay

如果同一 critique/verifier/archive evidence 被重复摄入，posterior 会漂移，ranking/selection/stop 会被静默污染。

防护：

- content-addressed evidence id
- monotonic ledger offset
- idempotency key
- duplicate ingest no-op
- replay equivalence test

### 风险 2：弱证据或循环证据污染 latent objective

critique、verifier、archive signal 并不等价。archive frequency 不是 desirability；model critique 不是 user preference；ranking result 不能当独立 evidence 回灌。

防护：

- source_type
- provenance_ref
- confidence/calibration
- source-specific weight cap
- derivative evidence discount
- direct / inferred / model-generated / archive-prior 分层
- quarantine unsupported evidence

### 风险 3：runaway confidence / search collapse

posterior 一旦过快集中，会重新造成搜索空间过窄。

防护：

- bounded update step
- prior floor
- entropy collapse guard
- stale evidence decay
- per-source update cap
- shadow-mode disagreement report

### 风险 4：durable schema 过早锁死

Ledger 一旦落盘就是长期兼容边界。

防护：

- schema_version
- update_model_version
- forward-compatible fields for `ExplorationAction`, `ParetoIntent`, `TrialObservation`, `ImprovementCertificate`
- migration event
- snapshots 只是 cache，event log 才是真相

### 风险 5：证书语义和 posterior 语义混淆

M5 certificate 证明局部改进；M5.1 posterior 表示潜在目标信念。不能让 posterior 直接伪造 improvement certificate。

防护：

- certificate-derived evidence 必须引用 verified certificate
- unverified observation 只能生成 weak/ambiguous/negative evidence
- posterior 不能单独让 solved=true
- `requires_verified_solution` 仍必须依赖 verified M5 certificate

## 6. 实施任务计划

### Task 1：定义 durable latent ledger schema

建议新增：

- `cognitive_evolve_runtime/outcomes/latent_ledger.py`
- `tests/test_latent_ledger.py`

核心类型：

```text
LatentLedgerEvent
LatentEvidenceRecord
LatentPosteriorSnapshot
LatentLedger
LatentLedgerStore
LatentLedgerReplayResult
```

最小字段：

```text
event_id
sequence
schema_version
event_type
run_id
round_index
candidate_id
trial_id
source_type
source_ref
provenance_ref
idempotency_key
intent_id
support
contradiction
weight
confidence
prior_state_hash
posterior_state_hash
update_model_version
created_at_utc
```

事件类型：

```text
latent_state_initialized
evidence_added
evidence_deduplicated
evidence_rejected
evidence_retracted
evidence_superseded
posterior_updated
posterior_snapshot_materialized
migration_applied
decision_bound_to_posterior
```

成功标准：

- append-only
- event id content-addressed
- sequence monotonic
- duplicate evidence by idempotency key is no-op
- retraction/supersession 不改写历史
- replay 能重建 posterior snapshot

测试：

- `test_latent_ledger_appends_events_with_monotonic_offsets`
- `test_latent_ledger_deduplicates_by_idempotency_key`
- `test_latent_ledger_replay_reconstructs_same_posterior`
- `test_latent_ledger_retraction_rebuilds_expected_posterior`
- `test_latent_ledger_snapshot_is_cache_not_authority`

### Task 2：实现 evidence -> PreferenceEvidence 适配器

建议新增：

- `cognitive_evolve_runtime/outcomes/evidence_feedback.py`
- `tests/test_latent_evidence_feedback.py`

适配来源：

```text
critique result -> PreferenceEvidence
verification result -> PreferenceEvidence
archive assignment / failure archive -> weak PreferenceEvidence
verified ImprovementCertificate -> strong PreferenceEvidence
TrialObservation without verified certificate -> weak/ambiguous evidence only
```

控制要求：

- source-specific default weight
- malformed evidence quarantine
- duplicate/correlated evidence discount
- conflict preserved as uncertainty，不静默覆盖
- model narrative alone cannot create strong evidence

成功标准：

- 每个 posterior-changing evidence 都可追溯到 raw source artifact / candidate / round。
- 重复 critique/verifier 不会重复加权。
- contradictory evidence 增加 uncertainty 或降低 confidence，而不是直接覆盖。

测试：

- `test_critique_adapter_emits_directional_preference_evidence_only_for_specific_tradeoff`
- `test_verifier_adapter_separates_hard_constraint_failure_from_preference_signal`
- `test_archive_adapter_emits_weak_prior_not_desirability_truth`
- `test_verified_certificate_adapter_emits_strong_but_deduplicated_evidence`
- `test_malformed_evidence_is_quarantined_not_ingested`

### Task 3：实现 deterministic posterior update engine

建议扩展：

- `cognitive_evolve_runtime/outcomes/latent.py`
- 或新增 `cognitive_evolve_runtime/outcomes/posterior_update.py`

能力：

```text
apply_preference_evidence(prior_state, ledger_events, update_policy) -> posterior_state
```

策略：

- bounded log update
- per-source weight cap
- stale evidence decay
- prior floor
- entropy floor / collapse guard
- deterministic ordering by ledger sequence
- update_model_version hash-bound

成功标准：

- 相同 ledger event stream 产生相同 posterior。
- evidence 顺序由 ledger sequence 决定，不受 dict/list 原始顺序影响。
- 单一来源不能把 posterior 直接打满。
- 高冲突 evidence 保留 entropy，不虚假收敛。

测试：

- `test_posterior_update_is_deterministic_from_ledger_order`
- `test_posterior_update_caps_single_source_influence`
- `test_conflicting_evidence_preserves_uncertainty`
- `test_stale_evidence_decay_reduces_old_signal`
- `test_prior_floor_prevents_total_intent_extinction`

### Task 4：把 posterior snapshot 绑定到现有决策点

建议修改：

- `cognitive_evolve_runtime/outcomes/runtime_bridge.py`
- `cognitive_evolve_runtime/nexus/loop/`
- `cognitive_evolve_runtime/ranking/parent_selection.py`
- `cognitive_evolve_runtime/nexus/synthesis.py`
- `cognitive_evolve_runtime/nexus/stop_decision.py`

核心要求：

```text
ranking / parent selection / synthesis / stop
读取同一个 pinned posterior_snapshot_id / ledger_cursor
```

不要每个 consumer 重新即兴计算 posterior。

新增 metadata：

```text
latent_ledger_cursor
latent_posterior_snapshot_hash
latent_update_model_version
latent_decision_trace_ref
```

成功标准：

- 每个 latent-informed decision 都能指出它读了哪个 posterior snapshot。
- replay 后同一个 ledger cursor 下 decision trace 可解释。
- posterior stale / missing 时 fail closed 到当前 M5.1-alpha 行为，不伪造 solved。

测试：

- `test_ranking_records_posterior_snapshot_cursor`
- `test_parent_selection_uses_same_pinned_posterior_snapshot`
- `test_stop_decision_records_latent_decision_trace`
- `test_missing_posterior_snapshot_falls_back_without_objective_solved`

### Task 5：做一个最小闭环集成切片

目标：证明 M5.1-beta 不是只记账，而是 evidence 能自动改变真实决策。

最小流程：

```text
candidate/verifier/critique evidence
  -> evidence adapter
  -> ledger event
  -> posterior replay/update
  -> posterior snapshot
  -> ranking or parent selection changes
  -> decision trace records cause
```

优先选择 ranking 或 parent selection，不先动 stop/closure。

测试：

- `test_verifier_evidence_updates_posterior_and_changes_parent_selection`
- `test_retracted_evidence_restores_prior_decision`
- `test_duplicate_evidence_does_not_change_decision_twice`
- `test_replay_after_restart_yields_same_decision_trace`

成功标准：

- 至少一个 production path 消费自动 derived evidence。
- changed decision 能从 raw evidence 追溯到 posterior delta。
- duplicate/retraction/replay 全部确定性。

### Task 6：beta 阶段只留 ExplorationAction / Pareto archive forward-compatible hook

不在 beta 里完整实现 planner 和 archive lane，但 ledger schema 必须能承接：

```text
exploration_action_id
exploration_action_kind
intent_dimension_refs
pareto_frontier_ref
trial_observation_ref
improvement_certificate_ref
```

成功标准：

- schema 不阻塞 M5.1-rc 的 ExplorationAction planner 接入。
- 不因提前实现 archive lane 污染当前 archive/fate 语义。

测试：

- schema roundtrip with future action/certificate fields
- unknown forward-compatible fields preserved or ignored safely

## 7. Beta 退出标准

M5.1-beta 完成条件：

1. durable latent ledger append-only 可持久化。
2. replay 后 posterior 与 live posterior 等价。
3. critique/verifier/archive/certificate 至少三类 evidence 有 typed adapter。
4. duplicate ingest 是 no-op。
5. retraction/supersession 可重建 posterior。
6. posterior update 有 source cap / bounded step / entropy guard。
7. 至少一个 downstream decision 由自动 evidence feedback 产生确定性改变。
8. 每个 latent-informed decision 能记录 posterior snapshot / ledger cursor。
9. 全量测试保持通过。
10. 不允许 posterior 单独让 `objective_solved=true`，M5 verified certificate gate 继续生效。

## 8. 建议验证命令

```bash
cd <repo-root>
./.venv/bin/python -m compileall -q cognitive_evolve_runtime tests
./.venv/bin/pytest tests/test_latent_ledger.py tests/test_latent_evidence_feedback.py -q
./.venv/bin/pytest tests/test_m5_1_runtime_integration.py tests/test_latent_problem_space_kernel.py tests/test_outcome_improvement_kernel.py tests/test_generation_plan.py tests/test_security_config_and_stop_decision.py tests/test_nexus_adaptive_semantics.py tests/test_nexus_text_evolution_loop.py -q
./.venv/bin/pytest -q
```

## 9. 当前优先级排序

最终排序：

1. Durable latent ledger schema + replay + idempotency。
2. Evidence adapters with provenance + quarantine。
3. Bounded deterministic posterior update。
4. Posterior snapshot pinned reads and decision trace。
5. Minimal evidence-to-ranking/parent-selection closed-loop slice。
6. Forward-compatible hooks for ExplorationAction / Pareto archive / TrialObservation certificate。

## 10. 不做事项

本 beta 不做：

- 不重写 NexusRuntime。
- 不改 LLM provider。
- 不做 UI/API 大改。
- 不把 posterior 当成 solved 证明。
- 不让 archive frequency 自动等同 desirability。
- 不让 model narrative 直接变成 strong preference evidence。
- 不实现完整 Pareto intent archive lane。
- 不完整接入 mutation/critique planner，只保留可审计接口。
