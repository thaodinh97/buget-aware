# Membership protocol audit and validation

Artifacts: `runs/membership_validation_v3_seed42`. Four controlled shadows plus one constructed memorization control. Same target seed/checkpoints and frozen data; no test-driven attack tuning. 108 target comparison rows across six models, three cohorts and three fixed attacks.

## Audit of the old protocol

The previous shadows had75 optimizer steps each over15rounds; the target had3375. Their roughly2,400-member datasets and balanced auxiliary clients differ from the114,284-row non-IID target federation. The original held-out learned attack AUC≈0.4984 did not establish attack power. The table below checks utility and loss gaps rather than inferring power from training steps alone.

| model | member N | nonmember N | member F1 | nonmember F1 | member loss | nonmember loss | loss gap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| improved_shadow0 | 2374 | 2379 | 0.86788 | 0.85401 | 0.19163 | 0.26255 | 0.07091 |
| improved_shadow1 | 2388 | 2372 | 0.86782 | 0.84038 | 0.19345 | 0.23990 | 0.04646 |
| improved_shadow2 | 2383 | 2398 | 0.86947 | 0.84905 | 0.21627 | 0.22141 | 0.00514 |
| improved_shadow3 | 1188 | 1190 | 0.85436 | 0.86000 | 0.26632 | 0.25575 | -0.01057 |
| old_shadow0 | 2374 | 2379 | 0.75951 | 0.71986 | 0.63760 | 0.63867 | 0.00107 |
| old_shadow1 | 2389 | 2372 | 0.43482 | 0.43869 | 0.61639 | 0.61483 | -0.00157 |
| old_shadow2 | 2383 | 2399 | 0.87022 | 0.84959 | 0.64415 | 0.64633 | 0.00218 |
| target_original | 114284 | 28571 | 0.96765 | 0.90155 | 0.07614 | 0.27563 | 0.19948 |

model_audit.csv additionally contains label counts and loss quantiles0/10/50/90/100. Shadow nonmembers are their paired independent auxiliary groups; target nonmembers in this audit are validation rows. These differing evaluation distributions mean utility numbers are descriptive, not directly matched estimates.

The budget correction improves shadow fitting but does not reproduce the target generalization gap: target mean logit-CE gap is about0.1995, while the final shadow gap is about−0.0106. Thus inadequate old training was a real mismatch, but fixing update count alone did not establish a useful target-membership proxy. The two excluded auxiliary rows do not by themselves explain the old near-chance AUC.

## Controlled changes and leakage checks

| shadow | member_count | nonmember_count | batch_size | cached | optimizer_steps | target_optimizer_steps | rounds | local_epochs | wall_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2374 | 2379 | 10 | True | 3600 | 3375 | 15 | 1 | 8.65719 |
| 1 | 2388 | 2372 | 10 | True | 3600 | 3375 | 15 | 1 | 4.23860 |
| 2 | 2383 | 2398 | 10 | True | 3600 | 3375 | 15 | 1 | 3.71089 |
| 3 | 1188 | 1190 | 8 | True | 2250 | 3375 | 15 | 1 | 2.43099 |

Final v3 evaluation reuses all four improved shadow checkpoints and the positive control from v2. Shadow training times in the table/logs are historical cached times, not newly measured fit costs. The first run stopped at the encoded-role leakage preflight; v2 is retained to document rounded-probability baseline saturation. No target or shadow was refit to optimize these results.

Only the shadow batch budget was adjusted by a fixed formula using target optimizer counts; architecture, lr,15rounds and balanced-client construction were retained. Smaller batch sizes approximately match total update count, not sample exposures, gradient noise, client skew or privacy difficulty. This is a justified undertraining check, not proof of shadow-target equivalence.

Preflight found one encoded signature shared across old buckets2 and5. Whole components spanning old roles were excluded, with IDs/counts saved in auxiliary_role_audit.json and excluded_auxiliary_ids.npz; no rows were reassigned and no primary split changed. Buckets0/1 and2/3 train the classifier;4/5 are development validation;6 calibrates original-model nonmember thresholds; previously unused bucket7 supplies a new group-disjoint final shadow pair. Explicit encoded-row checks passed between auxiliary buckets and target train/validation/test. Original train-fitted preprocessing is held fixed, so these results assess model weights conditional on that transformer, not end-to-end deletion of preprocessing influence.

Logistic attack: standardized log true-label probability, entropy, margin and label, fixed C=1. Baselines: negative CE and maximum confidence. All score directions were fixed in advance. Label+family matching caps each stratum at500 per side, with unsupported strata retained in coverage reports. F and all-retained cohorts reuse previous IDs; the independent retained control is rebuilt to exclude all encoded-equivalence components touching F. All cohorts are fixed across models. This avoids comparator sampling drift but limits conclusions to covered strata.

## Independent shadow evaluation

| stage | attack | auc | auc_ci_low | auc_ci_high | tpr_0.01 | actual_fpr_0.01 | member_count | nonmember_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| development | learned | 0.51227 | 0.49292 | 0.53108 | 0.05885 | 0.04644 | 2175 | 2175 |
| development | negative_loss | 0.51150 | 0.49380 | 0.53150 | 0.00092 | 0.00046 | 2175 | 2175 |
| development | confidence | 0.51396 | 0.49573 | 0.53382 | 0.00138 | 0.00046 | 2175 | 2175 |
| final | learned | 0.49695 | 0.47326 | 0.52287 | 0.00353 | 0.00000 | 1134 | 1134 |
| final | negative_loss | 0.50020 | 0.47251 | 0.52335 | 0.00088 | 0.00000 | 1134 | 1134 |
| final | confidence | 0.50220 | 0.47330 | 0.52528 | 0.00176 | 0.00000 | 1134 | 1134 |

For the final baseline scores, negative CE is computed stably from logits; confidence uses absolute binary logit margin, which is strictly monotonic in ideal max-softmax confidence. This avoids float32 probability rounding at1 and spurious0/1 calibration ties. The learned classifier and its clipped-probability features remain unchanged. Thresholds are in each score’s own units.

Intervals use200 group-bootstrap draws and require20 groups per side. They condition on a fixed attack and threshold; they do not quantify model-training/attack-fit seed uncertainty. Final-shadow AUC>=0.55 with lower95% bound>0.5 and the same condition on Original group-disjoint-retained target control are both required to call an attack detectable. Otherwise the exact status is MIA_INCONCLUSIVE.

## Positive and negative controls

| control | attack | auc | tpr_0.01 | actual_fpr_0.01 |
| --- | --- | --- | --- | --- |
| random_label_memorization | learned | 0.58972 | 0.29688 | 0.26562 |
| identical_score_distribution | learned | 0.50000 | 0.26562 | 0.26562 |
| random_label_memorization | negative_loss | 0.60791 | 0.01562 | 0.00000 |
| identical_score_distribution | negative_loss | 0.50000 | 0.00000 | 0.00000 |
| random_label_memorization | confidence | 0.43018 | 0.01562 | 0.00000 |
| identical_score_distribution | confidence | 0.50000 | 0.00000 | 0.00000 |

The attempted positive control trains on64 random-label development examples for a fixed2,000-step budget; independent nonmembers receive random labels too. It achieved only59.4% member classification accuracy and therefore did NOT establish a strong memorization positive control. Its modest attack AUC must not be described as a power validation. The failure is retained without tuning or retries (see positive_control_diagnostics.json). The negative control duplicates the score distribution across membership labels and checks AUC≈0.5. Neither control is used to pick attack hyperparameters or override weak final-shadow/retained-member controls.

## Target forget-cohort comparison

| case | model | attack | auc | auc_ci_low | auc_ci_high | auc_minus_retrain | tpr_0.01 | actual_fpr_0.01 | interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| client0_random_stratified_0.1 | original | learned | 0.67244 | 0.44175 | 0.82244 | 0.00011 | 0.33616 | 0.03712 | MIA_INCONCLUSIVE |
| client0_random_stratified_0.1 | original | negative_loss | 0.65439 | 0.51074 | 0.74876 | 0.00010 | 0.08378 | 0.02545 | MIA_INCONCLUSIVE |
| client0_random_stratified_0.1 | original | confidence | 0.62624 | 0.47023 | 0.74149 | 0.00005 | 0.08378 | 0.02545 | MIA_INCONCLUSIVE |
| client0_random_stratified_0.1 | retain_only | learned | 0.67265 | 0.44110 | 0.82318 | 0.00032 | 0.38282 | 0.11983 | MIA_INCONCLUSIVE |
| client0_random_stratified_0.1 | retain_only | negative_loss | 0.65704 | 0.50852 | 0.75084 | 0.00275 | 0.00000 | 0.00000 | MIA_INCONCLUSIVE |
| client0_random_stratified_0.1 | retain_only | confidence | 0.63008 | 0.47797 | 0.74315 | 0.00390 | 0.00000 | 0.00000 | MIA_INCONCLUSIVE |
| client0_random_stratified_0.1 | gd_recovery | learned | 0.67190 | 0.44115 | 0.82247 | -0.00043 | 0.38282 | 0.11983 | MIA_INCONCLUSIVE |
| client0_random_stratified_0.1 | gd_recovery | negative_loss | 0.65642 | 0.50850 | 0.75101 | 0.00214 | 0.00000 | 0.00000 | MIA_INCONCLUSIVE |
| client0_random_stratified_0.1 | gd_recovery | confidence | 0.62944 | 0.47658 | 0.74303 | 0.00326 | 0.00000 | 0.00000 | MIA_INCONCLUSIVE |
| client0_random_stratified_0.1 | alternating_recovery | learned | 0.67220 | 0.44197 | 0.82226 | -0.00012 | 0.38282 | 0.11983 | MIA_INCONCLUSIVE |
| client0_random_stratified_0.1 | alternating_recovery | negative_loss | 0.65628 | 0.50827 | 0.75097 | 0.00199 | 0.00000 | 0.00000 | MIA_INCONCLUSIVE |
| client0_random_stratified_0.1 | alternating_recovery | confidence | 0.62949 | 0.47727 | 0.74348 | 0.00331 | 0.00000 | 0.00000 | MIA_INCONCLUSIVE |
| client0_random_stratified_0.1 | fu_kd | learned | 0.67158 | 0.44059 | 0.82222 | -0.00075 | 0.38282 | 0.11983 | MIA_INCONCLUSIVE |
| client0_random_stratified_0.1 | fu_kd | negative_loss | 0.65625 | 0.50848 | 0.75099 | 0.00196 | 0.00000 | 0.00000 | MIA_INCONCLUSIVE |
| client0_random_stratified_0.1 | fu_kd | confidence | 0.62923 | 0.47648 | 0.74302 | 0.00304 | 0.00000 | 0.00000 | MIA_INCONCLUSIVE |
| client0_random_stratified_0.1 | retrain | learned | 0.67233 | 0.44313 | 0.82228 | 0.00000 | 0.04348 | 0.01697 | MIA_INCONCLUSIVE |
| client0_random_stratified_0.1 | retrain | negative_loss | 0.65428 | 0.51081 | 0.74880 | 0.00000 | 0.13998 | 0.10180 | MIA_INCONCLUSIVE |
| client0_random_stratified_0.1 | retrain | confidence | 0.62618 | 0.46966 | 0.74201 | 0.00000 | 0.13998 | 0.10180 | MIA_INCONCLUSIVE |
| client4_random_stratified_0.5 | original | learned | 0.55342 | 0.44586 | 0.64471 | -0.02369 | 0.06716 | 0.00677 | MIA_INCONCLUSIVE |
| client4_random_stratified_0.5 | original | negative_loss | 0.54483 | 0.42518 | 0.67343 | -0.01697 | 0.11569 | 0.03160 | MIA_INCONCLUSIVE |
| client4_random_stratified_0.5 | original | confidence | 0.54191 | 0.42203 | 0.66994 | -0.01821 | 0.11569 | 0.03160 | MIA_INCONCLUSIVE |
| client4_random_stratified_0.5 | retain_only | learned | 0.54465 | 0.43898 | 0.63776 | -0.03246 | 0.07844 | 0.02455 | MIA_INCONCLUSIVE |
| client4_random_stratified_0.5 | retain_only | negative_loss | 0.54711 | 0.43373 | 0.67233 | -0.01469 | 0.00056 | 0.00056 | MIA_INCONCLUSIVE |
| client4_random_stratified_0.5 | retain_only | confidence | 0.54291 | 0.42298 | 0.66829 | -0.01721 | 0.00056 | 0.00056 | MIA_INCONCLUSIVE |
| client4_random_stratified_0.5 | gd_recovery | learned | 0.54464 | 0.43902 | 0.63768 | -0.03246 | 0.07844 | 0.02455 | MIA_INCONCLUSIVE |
| client4_random_stratified_0.5 | gd_recovery | negative_loss | 0.54711 | 0.43354 | 0.67233 | -0.01469 | 0.00056 | 0.00056 | MIA_INCONCLUSIVE |
| client4_random_stratified_0.5 | gd_recovery | confidence | 0.54292 | 0.42302 | 0.66829 | -0.01720 | 0.00056 | 0.00056 | MIA_INCONCLUSIVE |
| client4_random_stratified_0.5 | alternating_recovery | learned | 0.54464 | 0.43897 | 0.63775 | -0.03246 | 0.07844 | 0.02455 | MIA_INCONCLUSIVE |
| client4_random_stratified_0.5 | alternating_recovery | negative_loss | 0.54712 | 0.43365 | 0.67228 | -0.01468 | 0.00056 | 0.00056 | MIA_INCONCLUSIVE |
| client4_random_stratified_0.5 | alternating_recovery | confidence | 0.54293 | 0.42300 | 0.66816 | -0.01719 | 0.00056 | 0.00056 | MIA_INCONCLUSIVE |
| client4_random_stratified_0.5 | fu_kd | learned | 0.54461 | 0.43896 | 0.63766 | -0.03250 | 0.07844 | 0.02455 | MIA_INCONCLUSIVE |
| client4_random_stratified_0.5 | fu_kd | negative_loss | 0.54714 | 0.43374 | 0.67238 | -0.01466 | 0.00056 | 0.00056 | MIA_INCONCLUSIVE |
| client4_random_stratified_0.5 | fu_kd | confidence | 0.54295 | 0.42305 | 0.66830 | -0.01717 | 0.00056 | 0.00056 | MIA_INCONCLUSIVE |
| client4_random_stratified_0.5 | retrain | learned | 0.57711 | 0.47540 | 0.65725 | 0.00000 | 0.07844 | 0.02286 | MIA_INCONCLUSIVE |
| client4_random_stratified_0.5 | retrain | negative_loss | 0.56180 | 0.44210 | 0.67274 | 0.00000 | 0.00056 | 0.00028 | MIA_INCONCLUSIVE |
| client4_random_stratified_0.5 | retrain | confidence | 0.56012 | 0.43960 | 0.67204 | 0.00000 | 0.00056 | 0.00028 | MIA_INCONCLUSIVE |

The same learned classifier and Original-calibrated nominal1%/5% thresholds are used for all six models. comparison.csv reports both thresholds, actual shifted FPR, balanced accuracy, sample/group counts, retained controls, paired AUC-difference intervals and point differences in TPR/FPR relative to retraining. Nominal calibration does not guarantee1% target FPR; ROC curves are descriptive and never used to recalibrate on test.

Interpretation counts: {('confidence', 'MIA_INCONCLUSIVE'): 36, ('learned', 'MIA_INCONCLUSIVE'): 36, ('negative_loss', 'MIA_INCONCLUSIVE'): 36}

For the F cohort, positive labels denote original/former membership for every model, including Retraining. Its balanced accuracy is therefore historical-membership discrimination, not literal current-membership accuracy for Retraining. Retained controls have actual retained-member labels. No successful-unlearning claim follows from AUC approaching0.5, lower loss, or similarity to a single retrained model. Even detectable attacks provide empirical membership evidence, not certified deletion. These results still require independent target-seed confirmation; no operational retraining-based selection or full DT was introduced.

Figures: `runs/membership_validation_v3_seed42/client0_random_stratified_0.1_roc.png`, `runs/membership_validation_v3_seed42/client4_random_stratified_0.5_roc.png`, `runs/membership_validation_v3_seed42/roc_controls.png`