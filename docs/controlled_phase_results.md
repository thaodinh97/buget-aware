# Controlled FU pilot — seed 42

Artifact directory: `runs/controlled_phase_v2_seed42`. Protocol: `docs/controlled_phase_plan.md`; configuration: `configs/controlled_phase.json`.

Completed 8/8 requests; 168 evaluated checkpoints; 72 recovery runs, 8 fresh retrains, 3 shadow fits. Total runner time 207.3s including diagnostics. Frozen preprocessing and client partitions reused. No Digital Twin planner implemented. This is a single-seed diagnostic pilot, not evidence of successful unlearning.

## Request sensitivity

| request | actual fraction | F normal/attack | R normal/attack | feature overlap | selected |
| --- | --- | --- | --- | --- | --- |
| client0_low_overlap_groups_0.1 | 0.09997 | [993, 453] | [7664, 5355] | 0.00000 | alt_1_1 |
| client0_low_overlap_groups_0.5 | 0.40207 | [2458, 3358] | [6199, 2450] | 0.46131 | alt_3_1 |
| client0_random_stratified_0.1 | 0.09990 | [865, 580] | [7792, 5228] | 0.72526 | alt_1_3 |
| client0_random_stratified_0.5 | 0.49997 | [4328, 2904] | [4329, 2904] | 0.71391 | alt_1_1 |
| client4_low_overlap_groups_0.1 | 0.09999 | [95, 7255] | [654, 65500] | 0.00000 | alt_3_1 |
| client4_low_overlap_groups_0.5 | 0.37990 | [748, 27176] | [1, 45579] | 0.42723 | alt_1_3 |
| client4_random_stratified_0.1 | 0.09998 | [74, 7275] | [675, 65480] | 0.77262 | alt_1_3 |
| client4_random_stratified_0.5 | 0.49999 | [374, 36377] | [375, 36378] | 0.77212 | alt_1_1 |

Group requests preserve whole local portions of connected components. Actual deletion fractions differ when a group exceeds the budget. Overlap is the fraction of forgotten rows with feature-equivalent records remaining globally. Labels and overlap change together across request designs; comparisons do not isolate overlap causally. In particular, the larger client-4 group request leaves only ONE normal example locally, a major distribution-shift confounder. All planned requests are included.

## Paired FU versus retained controls

| request | FU F1 | retain F1 | FU test MAE | retain test MAE | GD24 test MAE |
| --- | --- | --- | --- | --- | --- |
| client0_low_overlap_groups_0.1 | 0.88952 | 0.88948 | 0.00747 | 0.00768 | 0.00774 |
| client0_low_overlap_groups_0.5 | 0.89054 | 0.89054 | 0.00813 | 0.00855 | 0.00870 |
| client0_random_stratified_0.1 | 0.88917 | 0.88917 | 0.00886 | 0.00916 | 0.00911 |
| client0_random_stratified_0.5 | 0.89054 | 0.89054 | 0.00839 | 0.00835 | 0.00873 |
| client4_low_overlap_groups_0.1 | 0.88922 | 0.88918 | 0.01648 | 0.01643 | 0.01648 |
| client4_low_overlap_groups_0.5 | 0.89340 | 0.89340 | 0.09301 | 0.09299 | 0.09299 |
| client4_random_stratified_0.1 | 0.88921 | 0.88930 | 0.00867 | 0.00873 | 0.00866 |
| client4_random_stratified_0.5 | 0.89210 | 0.89210 | 0.02615 | 0.02615 | 0.02631 |

FU is the alternating ratio selected using validation MAE to fresh retraining, subject to validation F1/FPR constraints. Test metrics above are reporting only. Smaller MAE indicates prediction similarity, not a deletion guarantee. Inspect paired requests rather than interpreting their mean as independent repeated trials.

## Descriptive method means across eight requests

| method | macro_f1 | pr_auc_ap | fpr | prediction_mae_test | wall_seconds | optimizer_steps | payload_bytes_up |
| --- | --- | --- | --- | --- | --- | --- | --- |
| conpfu_inspired/post | 0.89003 | 0.98934 | 0.27204 | 0.02231 | 1.15558 | 443.50000 | 204248.00000 |
| ga_gd_with_kd/post | 0.89003 | 0.98934 | 0.27204 | 0.02229 | 1.27067 | 443.50000 | 204248.00000 |
| ga_only/post | 0.89048 | 0.98934 | 0.27075 | 0.02228 | 0.81066 | 427.50000 | 204248.00000 |
| gd_only/post | 0.89003 | 0.98934 | 0.27206 | 0.02234 | 1.06534 | 427.50000 | 204248.00000 |
| no_update/post | 0.89046 | 0.98934 | 0.27075 | 0.02225 | 0.81623 | 403.50000 | 185680.00000 |
| original | 0.88970 | 0.98946 | 0.27228 | 0.02453 | 0.00000 | 0.00000 | 0.00000 |
| retain_kd_matched/post | 0.89003 | 0.98934 | 0.27204 | 0.02231 | 1.13985 | 443.50000 | 204248.00000 |
| retrain | 0.88907 | 0.98875 | 0.27490 | 0.00000 | 5.41922 | 3026.25000 | 1392600.00000 |

Retain-only is no_update/post (two retained FedAvg rounds). GD24 adds 24 retained local updates; retain_kd_matched adds 16 retained KD steps. Cached original/no-op have zero incremental training cost, not zero historical cost. Per-method timings include their reused local-stage cost; summing rows double-counts shared computation. Payload is simulated tensor upload; download is also in metrics.csv. Sampled RSS and CPU profiles are in each request/profiles.json. Training and recovery rounds are recorded separately.

## Local stage changes

| arm | forget CE change | retain CE change | parameter L2 | F prediction change |
| --- | --- | --- | --- | --- |
| alt_1_1 | 0.00150 | 0.00020 | 0.00129 | 0.00022 |
| alt_1_3 | -0.00084 | -0.00086 | 0.00180 | 0.00037 |
| alt_3_1 | 0.00396 | 0.00134 | 0.00334 | 0.00080 |
| ga_only | 0.00637 | 0.00247 | 0.00550 | 0.00138 |
| gd_only | -0.00313 | -0.00187 | 0.00344 | 0.00087 |

Full per-stage CE, gradient norms, displacement, prediction changes and checkpoint reload checks are in *_diagnostics.json; actual batches and optimizer steps are in *_steps.json. Root checkpoints record the actual input-state hash; KD checkpoints also record the teacher hash. Direct GA/GD stage ancestry is recorded by diagnostic checkpoints.

## Learned membership inference

Held-out shadow AUC: **0.498422**. Calibration realized FPR: 0.009688; nominal target 0.01. The attack is **inconclusive** because the independent shadow control is below the predeclared 0.55 usefulness threshold. No score direction was flipped using target results.

Two group-disjoint shadows train a standardized logistic attack; a third independent shadow validates it. A separate original-model nonmember bucket fixes the threshold for every comparator. Target F, all-retained and group-disjoint-retained cohorts are matched by label/type and shared across models. Coverage and unsupported strata are saved in mia_coverage.json; actual nonmember FPR and sample counts are in metrics.json. A calibrated nominal 1% threshold does not guarantee target FPR=1% under distribution shift. Shadows have much smaller training sets and fewer optimizer steps than the target despite matching epochs/rounds; this limits transfer. The preprocessing transformer was fit on original training data and reused by design; results concern model-weight membership conditional on that fixed pipeline.

## Integrity and limitations

20 unit tests passed, including the corrected profiler round-role checks. Run scripts/verify_controlled_phase.py for artifact, checkpoint, update-budget, auxiliary-disjointness and no-op verification. Source metrics.csv differs from its historic hash and is explicitly recorded as an unused changed report in the manifest; all other source hashes remain mandatory. Original datasets and source runs were not edited.

The global-start model, fixed-step schedules, full-local KD baseline, small MLP and auxiliary clients are paper-inspired adaptations, not a faithful ConPFU reproduction. Full-local KD accesses F during the unlearning operation. Recovery uses retained data only. No confidence intervals or multi-seed confirmation are claimed. Next work should validate attack power on independent auxiliary data with an appropriate training budget before interpreting privacy changes; any subsequent FU settings must again be selected on validation, not these test results.