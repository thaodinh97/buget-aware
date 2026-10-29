# Agent implementation guide: Sample-Level Federated Unlearning + Budget-Aware Digital Twin on ToN-IoT

> **Purpose:** Execution specification for a coding agent to build reproducible experiments for the thesis **“Resource orchestration via digital twin for federated unlearning services in IoT network anomaly detection systems.”**
>
> **Primary contribution:** Budget-aware Digital Twin (DT) state synchronization, cost prediction and recovery-client selection. **Supporting component:** Empirically validated *approximate sample-level* federated unlearning (FU). **Do not present an empirical similarity test as a formal deletion/DP guarantee.**
>
> **Language:** Report and explanatory documentation in Vietnamese; Python module/function names, command-line flags, metric keys and file names in English.

## 0. Source-of-truth and attribution

Read the three supplied PDFs **before implementing**. If inaccessible in the agent workspace, ask the operator for their paths; do not invent missing details.

1. **Thesis proposal:** `DeCuongKLTN_23521465_DinhCongThao_23520738_HuynhMinhKhoa (1)(2).pdf`, particularly pp. 3–9: ToN-IoT Network, binary anomaly detection, MLP/FedAvg, sample-level FU with an online target client, federated recovery on retained samples, a DT state/cost/planner loop, budget constraints, runtime/network costs, MIA and retraining-reference comparisons. The proposal limits the principal study to five simulated clients and a containerized proof of concept.
2. **ConPFU:** Yanghe Pan et al., *The Right to Be Forgotten Versus the Need to Be Remembered: Efficient Personalized Federated Unlearning With Optimal Incentives*, IEEE TNSE (2025/2026 volume), DOI `10.1109/TNSE.2025.3597640`. Paper §IV and Algorithm 1: client-local alternating gradient ascent on forget samples / descent on retained samples, regularization and clipping, followed by client-side knowledge distillation. §V uses ToN-IoT, Dirichlet non-IID β=0.5, 20 clients, forget 50% of one client's samples; test accuracy, unlearning accuracy, membership-inference UASR, execution time and transmitted traffic. Contract/incentive optimization is **not** within this thesis scope.
3. **FedKD-Unlearn:** Xiaojing Cao et al., *Heterogeneous Federated Unlearning Algorithm Based on Knowledge Distillation and Differential Privacy in Network Intelligence*, IEEE IoT Journal accepted author's version (2026), DOI `10.1109/JIOT.2026.3734241`. §IV Algorithm 1: projected stochastic gradient ascent; Algorithm 2: server-side knowledge distillation using original global model as teacher, modified model as student and an *auxiliary labeled distillation dataset*. §V: non-IID sensitivity, retraining reference, MIA, backdoor attack success rate (for explicit backdoor scenarios), distance to retraining, accuracy, execution/storage/communication costs. Its DP claims require a correctly implemented and accounted DP mechanism; **gradient clipping or ordinary noise alone does not establish the published privacy guarantee**.

### Attribution rules

- Tag every algorithm in metadata and reports as `conpfu_inspired`, `fedkd_inspired` or `hybrid_projected_alternating`, **not** as a faithful reproduction unless each step and experimental assumption matches the paper.
- **Adaptations proposed for this thesis, not claimed as paper steps:** starting sample-level FU directly from the original global checkpoint (as an optional safer baseline), retain-only KD ablation, mandatory FedAvg recovery with DT-selected clients, five clients rather than twenty, group-aware splits, and budget-aware orchestration.
- ConPFU's initial removal of the target client's **last-round contribution** and FedKD-Unlearn's suggested `remaining model` construction must not be casually interpreted as exact removal of a subset's contribution across all FL rounds. If implementing them, label and test them as distinct initialization ablations, accounting for FedAvg sample-weighted aggregation (avoid using a uniform average formula in a weighted system).
- Neither paper's reported metrics constitute results for *our* dataset split or code. Never copy experimental numbers into generated result tables.

## 1. Agent working contract

1. **Inspect before editing:** recursively inventory the repository; locate existing preprocessing, `fedavg_core.py`, training, forget-set, reference retraining, evaluation, any run manifests/checkpoints and dependency files. Record findings in `docs/implementation_audit.md`.
2. **Reuse stable functionality** rather than rewriting it. If previous artifacts/checkpoints are missing, provide executable regeneration commands; do not fabricate them.
3. No silent fallbacks, especially for invalid forget requests, empty/single-class client splits, absent checkpoints, incompatible feature dimensions, unsupported paper settings, or unavailable MIA dependencies.
4. All experiments must work **without DT** first; do not proceed to claim DT improvements until the FU evaluation suite is operational.
5. Preserve raw data locality in logical client workers. Server/DT receive aggregated metadata/updates, not raw client records. A separate offline audit runner may access evaluation data in a research-only environment; label this boundary in docs.
6. Separate immutable **experimental manifest** from mutable logs. Save seed, dataset hash, row IDs/group IDs, input schema, preprocessing version, partition/forget hashes, model architecture, training/optimizer config, model checkpoint hashes, code revision, hardware and wall-clock method.
7. For every feature: implement tests, one cheap smoke run, then gated larger experiments. Store actual measured values and errors; never emit plausible placeholder metrics.
8. Do not change the experimental method just to improve a metric without reporting the change. Hyperparameters are tuned only on pilot/validation, never on held-out test results.
9. Agent may implement and execute code if resources permit, but must distinguish **implemented**, **smoke-tested**, **experiment completed**, and **empirically supported**.

## 2. Study questions and acceptance claims

**RQ-FU1** — Is the unlearned model behavior closer to retraining on `D \ F` than the original model is, particularly on forget records, while maintaining useful anomaly detection?

**RQ-FU2** — Is the result robust across forget fractions, target clients, Dirichlet heterogeneity and random seeds, versus no-op / retain-only alternatives?

**RQ-DT1** — Does a DT with measured state, online-updated cost prediction and budget-constrained planning choose less costly recovery plans than fair static/random/greedy alternatives without violating a predeclared detection-utility floor?

**RQ-DT2** — How do plan quality, total orchestration overhead and budget violation change under variable CPU/load/bandwidth and delayed DT state?

**Outcome wording:** If tests pass, state only that the approximate method **shows evidence of reduced influence** under studied assumptions. Do **not** assert “all information has been erased”, certified unlearning, or `(ε,δ)`-DP without a formal proof and validated privacy accountant.

## 3. Data preparation (ToN-IoT Network)

### 3.1 Input and provenance

- Use the specified **network CSV** version. Capture path, SHA-256, row count, schema, label frequencies, missing-value counts and exact duplicates. Do not substitute an unrelated ToN-IoT subset.
- Binary label: original `label` normal vs attack. Retain `type` only as **audit metadata** for attack-family analysis; never include `label`/`type` as predictors.
- Drop or explicitly justify direct identifiers and leakage-prone fields (IP, ports, identifiers, timestamps, `dns_query`, SSL/HTTP string fields, etc.) following current pipeline. Dropping all such columns is a design choice; preserve a documented approved-feature list and feature provenance.
- Check exact duplicates before splitting; do not silently deduplicate conflicting labels. For rows with identical model-feature signatures but opposite labels, report conflict counts and choose a **predeclared** policy.
- For missing values and categorical unknowns: fit imputation/one-hot or ordinal mapping and numeric scaling **on training/approved calibration data only**; freeze transformers across Original, FU, No-op and Retraining. Never fit on test or forget-specific evaluation splits.

### 3.2 Group-aware split, with leakage audit

- Use credible group IDs (connection/5-tuple grouping when session IDs are unavailable). A 5-tuple group **is not necessarily a true session**; describe it accurately.
- Prefer group-based holdout for train / validation / calibration / test. Test must not be involved in hyperparameter selection, auxiliary distillation construction, model-selection decisions, or DT planner learning.
- Explicitly audit cross-split `group_id`, connection identifiers and feature-signature overlap. Any residual overlap must be disclosed; high group disjointness alone is not proof of no leakage.
- Training and calibration roles must be unambiguous. If calibration is used to fit preprocessing, document why and ensure any calibration rows are excluded from all forget requests; preferable: fit transforms on *training only* and use calibration exclusively for thresholds, MIA controls or cost-model calibration.
- Train-only Dirichlet partition into **5 disjoint clients**. Target client holds both F and R after constructing the request. Reject invalid partitions instead of quietly creating degenerate or empty clients.
- Candidate heterogeneity conditions: Dirichlet `alpha ∈ {0.1, 0.5}`; add IID as optional control. Save per-client sample count, normal/attack fraction, group composition and seed.

### 3.3 Two forget-request designs

- **Primary:** stratified random fraction of target client's train-only data, `fraction ∈ {0.10,0.30,0.50}`; fractions are of **target-client size**, not global training size. Prefer nested forget sets for the same seed/client when analyzing fraction sensitivity.
- **Secondary:** group-aware deletion of a coherent connection/device-derived group, if metadata supports it; report actual fraction and class mixture. This is more realistic but can be confounded by group-specific distribution shifts.
- Compare at least **two target clients** with different sizes/class mixtures, with **three independent seeds** for final claims. Run a smaller pilot first.
- For each request: save unique row IDs, `group_id`s, target client ID, class counts, forget/retain hashes; assert `F ⊂ D_target_train`, `F ∩ R = ∅`, `F ∩ test = ∅`, `F ∩ calibration = ∅`.
- Do not physically erase source experiment data until the *authorized offline verification setup* has completed its required fixed measurements. In a real deployment, audit-only copies must be access-controlled and retained only according to policy; **no forgotten raw data is used in recovery or subsequent model selection**.

## 4. Shared model and training baselines

- Binary MLP with two hidden layers, using current repository's proven architecture and feature pipeline. Use FedAvg with **sample-count-weighted aggregation**, matching the number of retained training samples after deletion.
- `M_original`: FedAvg trained on all participating training rows `D`.
- `M_retrain`: initialize fresh and run the same FL protocol on `D \ F`, with matching initialization seed/policy, preprocessing, training budget and selection rules. Document whether round/client randomness is coupled to the original; reruns across seeds measure retrain variability.
- `M_noop`: original checkpoint carried through unlearning API but with `unlearn_steps=0, recovery_rounds=0`.
- `M_retain_only`: fine-tuning/recovery using only retained data without a destructive forgetting step.
- `M_unlearned(method, planner)`: selected FU algorithm + **identical recovery procedure** across planner comparisons.
- Maintain separate checkpoints **before FU**, **after local FU/KD**, **after FedAvg recovery**; measure intermediate stages so a harmful recovery stage is detectable.
- Validate a healthy, non-trivial original vs retrain separation (`M_original` already extremely close to `M_retrain` can make deletion influence unidentifiable); do not inflate the split, tune using test data, or cherry-pick deletions to force separation.

## 5. FU algorithms to implement (staged)

### 5.1 `conpfu_inspired` — **primary**

**Not a full ConPFU reproduction** (drop contract theory, use global-model start by default, add DT-directed FL recovery).

**Inputs:** original global `θ_O`, local forget F, retain R, tuned λ, learning rates, clipping bound, alternating schedule, KD settings.

1. Initialize `θ_local := θ_O` (recommended primary adaptation). Optionally implement a separate `last_round_subtraction` ablation as in the source paper, **without** claiming sample-level exactness.
2. Repeat predefined alternating cycles:
   - Forget step(s): optimize `L_forget = - CE(θ;F) + λ ||θ − θ_init||²` using gradient descent, equivalent to ascent on CE under regularization; clip gradients and log norms/losses.
   - Retain step(s): gradient descent on `CE(θ;R)`; log utility loss.
3. KD: **source-inspired** client-side student/teacher distillation with temperature T and hard/soft terms, clearly recording teacher/student initializations, data access and output. Run a **retain-only KD** ablation as the safer thesis adaptation; forbid F in any subsequent recovery once the local forget operation is complete.
4. Transfer the resulting student model/update to the server. Keep a frozen intermediate checkpoint.
5. Federated recovery on `D \ F` by planner-selected client subset (target client may join with R), using fixed recovery-round count in main planner comparisons. Record sample-count weights and all update bytes.

**Key ablations:** without GA, without alternating retain steps, without KD, without recovery, with/without optional last-round subtraction. Keep comparisons feasible (do not create a full Cartesian grid needlessly).

### 5.2 `fedkd_inspired` — **secondary experimental baseline**

1. Implement target-local projected ascent `θ ← θ + η∇L_F(θ)` with projection to `||θ − θ_center||₂ ≤ r` and gradient clipping. Default `θ_center := θ_O` to preserve compatibility with partial deletion. If using the paper's `remaining model` approximation, record the change and test separately.
2. Optional server-side KD: original global teacher, modified model student, separate **approved auxiliary labeled distillation dataset**; never re-use held-out test or forget F in this dataset. Explicitly declare where auxiliary data comes from and what data-locality assumption changes.
3. Follow with the **same federated recovery** as primary to make DT planner comparison fair. This is an adaptation of FedKD-Unlearn; it is not its original server-only optimized-recovery design.
4. **No DP claim by default.** A DP variant is out of core scope unless implemented with per-sample clipping, correct noise calibration, composition accounting, published sampling assumptions and external verification. Do not copy reported `ε`.

### 5.3 `hybrid_projected_alternating` — optional *after* ablation

Use projected GA from FedKD in alternating GA/GD from ConPFU, optionally retain-only KD, and the same federated recovery. Run only if it addresses a documented primary-method failure. Label this a **new hybrid adaptation**, not a source-paper algorithm.

### 5.4 Interface contract

```python
class UnlearningMethod(Protocol):
    def run(self, original_checkpoint, target_client, forget_request,
            config, output_dir) -> UnlearningResult: ...

# UnlearningResult: intermediate_checkpoint, final_checkpoint,
# timing_breakdown, client_bytes_up/down, local_losses,
# config_hash, forget_request_hash, structured_warnings
```

All variants support a CLI-level `--method`, `--seed`, `--forget-request`, `--original-run`, `--recovery-rounds`, `--planner` and fail early on incompatible checkpoints.

## 6. Evaluation suite: influence reduction versus retraining

All metrics below are computed for `M_original`, `M_noop`, `M_retain_only`, `M_retrain`, each FU intermediate and FU final. Use same fixed evaluation samples and record confidence intervals across seeds.

### 6.1 Utility (held-out test; optionally per client)

`macro_f1`, `precision_attack`, `recall_attack`, `pr_auc`, `fpr`, `roc_auc` (secondary), `log_loss`, confusion matrix. Do not rely on raw accuracy alone for imbalanced normal/attack classes.

### 6.2 Retraining similarity (main empirical verification)

Binary output is attack probability `p_M(x)`:

```
prediction_mae_F(A, B) = mean_{x in F} |p_A(x) - p_B(x)|
prediction_mae_test(A, B) = mean_{x in test} |p_A(x) - p_B(x)|
disagreement_X(A, B) = mean_{x in X} 1[ class_A(x) != class_B(x) ]
```

Also compute Jensen–Shannon/KL divergence on probability vectors (with clipped probabilities) and **optionally** normalized parameter distance for identically structured models. Parameter closeness alone is NOT evidence of deletion.

**Primary contrast:** `d_F(M_unlearned, M_retrain)` versus `d_F(M_original, M_retrain)`, paired by request/split/seed, plus the analogous test/retain contrasts. Report the **absolute distances**, signed improvement and uncertainty. If `d_F(original,retrain)` is already near zero, mark the request as low identifiable signal, not a success by default.

**Important:** `MAE` in this section means *mean absolute error between model prediction probabilities*, not the MAE of a traffic-flow regression problem and not the DT cost-forecast MAE. Use separate metric names `prediction_mae` vs `cost_mae_seconds`.

### 6.3 Membership Inference Attack (MIA)

- Build one explicitly specified attack protocol with train/evaluation separation (e.g. shadow models or calibrated loss/confidence attacks). Each attack's available information and thresholds must be recorded.
- Non-members matched by class and, where possible, by connection/group characteristics; keep forget, retained-member, and non-member groups separated.
- Report `mia_auc`, `mia_balanced_accuracy`, `mia_tpr_at_fpr_0_01` (if sample size supports it), and a forget-member identification rate under a fixed operating point.
- Evaluate on original, retrain and all FU versions; include retained-member positive controls and a no-op negative control. Compare `M_unlearned` **to retrain**, not just to chance level 50%. A weak MIA with every model near chance is inconclusive.
- Do not claim that lower forget-set **classification accuracy** automatically indicates better forgetting. Retrained models can still predict F correctly from generalizable patterns.
- Optional privacy-oriented backdoor ASR experiment **only if a controlled training-time backdoor is deliberately and explicitly implemented**; never conflate backdoor ASR and MIA success rate.

### 6.4 Robustness and statistics

- Predeclare pilot hyperparameter grid and hold it fixed for final runs.
- Test `fraction={0.10,0.30,0.50}`, `alpha={0.1,0.5}`, `target_clients=2`, `seeds=3` (36 cases per evaluated FU method, before baselines).
- Use paired plots/paired differences and 95% bootstrap confidence intervals when feasible; describe sample/seed units correctly (do not bootstrap individual records as if they were independent experimental seeds).
- Report failure/degeneracy rate and any skipped case in a `coverage.csv`; do not silently omit cases where FU performed poorly.

## 7. Budget-aware DT: main contribution

### 7.1 Execution-state twin

Capture per client at time `t`: `{client_id, model_version, sample_count_retained, class_summary, cpu_capacity/usage, available_memory, estimated bandwidth, observed upload/download throughput, last_round_compute_time, state_timestamp}`. Avoid raw personal/network traffic records. The DT must have **timestamped state**, actual-vs-predicted cost feedback and model refresh; a one-off constant cost formula by itself is merely a profiler.

### 7.2 Cost model and measurement

Measure per plan `P`:

- `wall_time_total_seconds`: all stages (API/orchestrator, state collection, FU, planner, recovery, validation, update), accounting for parallel client work;
- `compute_seconds_sum`: sum of client compute times (not equivalent to makespan);
- `bytes_sent/received`, `peak_memory_mb`, `clients_selected`;
- `dt_overhead_seconds` and prediction-error metrics `cost_mae_seconds`, `cost_mape` with zero-denominator rules;
- `budget_violation` from **actual measured** cost, not predicted cost alone.

Prediction should start with an interpretable measured-cost model (e.g. local steps, samples, observed throughput, load); calibrate/update on *past* runs and avoid seeing target run's true cost before selecting its plan. Include cost of collecting and refreshing DT state.

### 7.3 Recovery planner

Plans are subsets `P ⊆ eligible_clients` (target client uses only retained R), with a **fixed** recovery-round count and fixed FU intermediate state in the main experiment. Define a budget explicitly, e.g. `B_time_seconds`, optionally `B_upload_bytes`. Do not mix time and bytes in one scalar without explicit conversion/weights.

**Primary optimization:** find lowest predicted wall time *among feasible plans* satisfying budget **and predefined data-representation/utility-proxy constraints**. A cost-only optimizer selecting zero clients is invalid. If no feasible plan exists, return a distinct `NO_FEASIBLE_PLAN` result rather than silently violating constraints.

Do not use actual test/FU/MIA results from the same run to guide selection: those are post hoc audit measurements, not available decision-time information.

### 7.4 Planner comparators and fairness

- `target_only`, `all_clients`, `random_k`, `greedy_low_cost`, `static_cost_planner`, `dt_adaptive_planner`;
- optional `offline_oracle` by enumerating subsets with five clients **only as an evaluation upper bound**, using measurements not available online;
- hold identical FL start, FU step, recovery rounds, data, seeds and resource scenario across planners. State if a comparator is infeasible under a budget; don't unfairly count unavailable all-client as budget compliant.

### 7.5 Stress experiments and ablation

Run at **tight / moderate / loose** budgets derived from observed baseline costs, under **stable / high CPU load / reduced bandwidth / stale DT state** scenarios. Measure prediction MAE, budget-violation rate, end-to-end latency, communication, utility and retrain similarity. Compare static versus continuously updated DT, and include the DT's own overhead. Present cost/utility Pareto plots (with actual—not solely forecast—costs).

## 8. Run matrix and staged execution gates

| Stage | Work | Exit gate |
|---|---|---|
| 0 | Repository audit, dataset/hash audit | Current pipeline and available artifacts documented |
| 1 | Group-aware data split, train-only clients, immutable request manifests | No cross-split group overlap; deterministic regeneration; all schema checks pass |
| 2 | `M_original`, `M_retrain`, `M_noop`, `M_retain_only` | Reproducible checkpoints; healthy test metrics; clear no-op equivalence |
| 3 | `conpfu_inspired` primary implementation with KD/no-KD ablations | One pilot successful end-to-end; F excluded from recovery; intermediate/final checkpoints |
| 4 | FU evaluation including retrain similarity + MIA controls | Complete small pilot report; no data/test leakage; unexpected results diagnosed |
| 5 | `fedkd_inspired` optional baseline and hybrid only if needed | Fair matched comparison; clear algorithm attribution |
| 6 | DT state/cost measurement + feasibility planner | Budget checking tested; forecast-vs-observed logging and adaptive update |
| 7 | Budget sweep + planner ablation + stress scenarios | Measured quality-cost trade-offs, overheads, violations and scenario coverage |
| 8 | Final repeated-seed runs and report | Every claim backed by figures, configurations and source-run identifiers |

**Minimal pilot**: `alpha=0.5`, one target client, one seeded request each for fraction 10% and 50%, all baseline models and primary FU, one no-DT recovery setting. **Do not run the entire grid before pilot metrics/bugs are reviewed.**

## 9. Repo layout (adapt to existing repo; do not duplicate modules)

```text
configs/
  data.yaml                 # split, approved columns, grouped leakage rules
  fl.yaml                   # architecture, training, seed
  fu_conpfu.yaml            # GA/GD/KD and fixed recovery
  fu_fedkd.yaml             # optional projected GA and auxiliary KD
  dt.yaml                   # budget, cost prediction, planner policy
src/
  data/prepare.py
  data/partition.py
  data/forget_requests.py
  fl/train_fedavg.py
  fu/conpfu_inspired.py
  fu/fedkd_inspired.py
  fu/recovery.py
  eval/retrain_reference.py
  eval/prediction_similarity.py
  eval/membership_inference.py
  eval/aggregate_metrics.py
  dt/state_twin.py
  dt/cost_model.py
  dt/planner.py
  dt/orchestrator.py
scripts/
  run_pilot.py
  run_fu_matrix.py
  run_dt_matrix.py
  plot_results.py
runs/<experiment_id>/
  manifest.json
  checkpoints/
  metrics.json
  metrics.csv
  timings.json
  network_bytes.json
  figures/
  audit.json
  error.log                  # if failed
results/
  coverage.csv
  fu_comparison.csv
  dt_comparison.csv
  figures/
docs/
  implementation_audit.md
  experiment_protocol.md
  algorithm_adaptations.md
  validation_report.md
```

Retain existing file names when present (`fedavg_core`, `make_forget_set`, retrain scripts, etc.). Names above are a logical layout, **not instructions to overwrite working code**.

### Example CLI design (agent to implement; do not claim commands already exist)

```bash
python scripts/run_pilot.py --config configs/fu_conpfu.yaml --seed 42
python scripts/run_fu_matrix.py --config configs/fu_conpfu.yaml --methods conpfu_inspired,retain_only,no_op,retrain
python scripts/run_dt_matrix.py --config configs/dt.yaml --planners target_only,random_k,greedy_low_cost,static_cost_planner,dt_adaptive_planner
python scripts/plot_results.py --runs runs/ --output results/
```

## 10. Automated tests and correctness checks

Write and run at least these tests:

1. **Data:** deterministic split; zero group-ID overlap; no label/type target leakage; train-only transform fitting; mutually exclusive train/test/calibration/forget; stable feature dimension.
2. **Forget request:** sample IDs are from one target client's training data, stratification bounds as specified, requests are reproducible and no forgotten sample enters recovery.
3. **FedAvg:** aggregation matches weights `n_i / Σ_j n_j`, including when target-client retained sample count changes.
4. **FU loss:** one forget gradient step has expected sign (increases CE absent regularizer/clipping interference); projection never exceeds radius; no-op is numerically equivalent to original within serialization tolerance.
5. **Retraining:** checkpoint is fresh initialized; F absent; same model schema and fair training policy.
6. **KD:** teacher frozen, temperature scaling correct for binary logits, gradients only into student; auxiliary data independent when FedKD-inspired.
7. **MIA:** balanced or clearly weighted evaluation, disjoint attack train/test, exact evaluation-point reproducibility, non-member matching and retained-member control.
8. **DT:** no empty-client 'solution', constraints enforced, predicted ≠ actual costs kept distinct, parallel makespan measurement sensible, stale-state timestamps enforced, no future information in planning.
9. **Artifact audits:** required manifest/log fields, full case coverage accounting, deterministic plot regeneration; all failures visible, no fabricated success reports.

## 11. Required plots, tables and report sections

**FU:** (a) per-method macro-F1/PR-AUC/FPR; (b) prediction MAE to retrain on F and test; (c) MIA AUC / TPR@FPR with retained-member control; (d) forget fraction sensitivity; (e) Dirichlet/target-client sensitivity; (f) FU stage ablation; (g) model utility vs actual unlearning time.

**DT:** (h) predicted vs observed cost with MAE; (i) budget violations by method/constraint; (j) actual cost vs quality/Pareto scatter; (k) selected-client frequencies and total bytes; (l) static vs adaptive DT under fluctuating load/stale state; (m) DT own overhead.

Create a **Vietnamese experimental report** with dataset/version, leakage controls, exact FU adaptation, hardware and random seeds, all baselines, matched statistical comparisons, failed cases, validity threats and conclusions limited to evidence. Include source-paper comparison only as contextual discussion, **not** as if benchmark splits match.

## 12. Final task instruction for the coding agent

> Inspect the repository and three reference PDFs; produce a short implementation audit and a detailed phased plan **before changes**. Then implement the smallest reproducible end-to-end pilot: ToN-IoT grouped data pipeline → 5-client FedAvg → original/no-op/retain-only/retrain checkpoints → ConPFU-inspired alternating local FU with optional retain-only KD → common FedAvg recovery → utility/prediction-similarity/MIA verification. Execute available tests and a smoke run, explicitly report failures and missing data. Only after a credible FU pilot, implement the timestamped cost-predicting budget-aware DT recovery planner and fair planner comparisons. Do not expand into full ConPFU incentives, unverified FedKD differential privacy claims, whole-client/class unlearning, or continuous deletion outside the proposal's scope. Never claim forgetting success from forget accuracy alone or from MIA near 50% without controls. Deliver source code changes, configs, tests, runnable scripts, metrics/figures generated from actual runs, and a Vietnamese evaluation report. If execution is impossible, deliver validated code plus exact reproduction commands and mark numerical results as **not run**.

---

**Reference scope:** This guide combines the provided thesis proposal, ConPFU §IV–V, and FedKD-Unlearn §IV–V. Whenever the guide makes a recommendation not specified in those texts (e.g. retain-only KD, five-client planner-controlled FL recovery, stringent grouped leakage checks, paired confidence intervals), treat it as a **thesis-specific adaptation or proposed research control**, not a quote or experimental finding of the source papers.
