# Recovery trajectory findings

Artifacts: `runs/recovery_trajectories_seed42`. Two fixed cases, four cached starting states, rounds0–4, two fixed learning rates. 80 global and320 per-client measurements. The8 primary round2 states exactly reproduce existing pilot checkpoints. 25 unit tests passed, including encoded-role closure and extreme-logit numerical checks.

## Observations

client0_random_stratified_0.1: GA-only test MAE 0.001777 before recovery → 0.009185 after2 rounds. Validation log loss 0.276957 → 0.260527. Quarter-rate round2 test MAE is 0.013568.

client4_random_stratified_0.5: GA-only test MAE 0.042280 before recovery → 0.026186 after2 rounds. Validation log loss 0.272685 → 0.219271. Quarter-rate round2 test MAE is 0.041693.

Thus smaller parameter steps do not uniformly improve prediction similarity, and parameter distance can increase while prediction distance decreases. These observations contradict a magnitude-only explanation in these two cases; they do not prove that update magnitude is irrelevant elsewhere.

Among the8 GA-only primary-rate updates, 8 have negative cosine to the retraining direction. A negative dot product guarantees an increase in squared parameter distance for that update. Prediction similarity need not follow parameter distance: the model is nonlinear and functionally similar networks need not have close parameters.

The exact identity is Δdistance² = ||delta||² − 2·dot(delta, direction_to_retrain). The first term measures update magnitude; the second measures alignment. Zero vectors have undefined cosine, recorded as blank rather than zero. Distances are offline explanations; the reference never enters client gradients or aggregation.

| case | lr_multiplier | round | mae_validation | mae_test | mae_forget | validation_log_loss | l2_retrain | update_norm | cosine_to_retrain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| client0_random_stratified_0.1 | 1.00000 | 0 | 0.00178 | 0.00178 | 0.00174 | 0.27696 | 1.87421 | 0.00000 | nan |
| client0_random_stratified_0.1 | 1.00000 | 1 | 0.00550 | 0.00570 | 0.00785 | 0.28202 | 2.17502 | 0.66954 | -0.30671 |
| client0_random_stratified_0.1 | 1.00000 | 2 | 0.00922 | 0.00918 | 0.00900 | 0.26053 | 2.51050 | 0.57670 | -0.49403 |
| client0_random_stratified_0.1 | 1.00000 | 3 | 0.01083 | 0.01096 | 0.01440 | 0.26181 | 2.92242 | 0.58987 | -0.63814 |
| client0_random_stratified_0.1 | 1.00000 | 4 | 0.00972 | 0.00962 | 0.00967 | 0.25935 | 3.36054 | 0.57198 | -0.72554 |
| client0_random_stratified_0.1 | 0.25000 | 0 | 0.00178 | 0.00178 | 0.00174 | 0.27696 | 1.87421 | 0.00000 | nan |
| client0_random_stratified_0.1 | 0.25000 | 1 | 0.01024 | 0.01078 | 0.01872 | 0.31029 | 1.93735 | 0.20332 | -0.26156 |
| client0_random_stratified_0.1 | 0.25000 | 2 | 0.01293 | 0.01357 | 0.02258 | 0.31609 | 2.01742 | 0.18205 | -0.40191 |
| client0_random_stratified_0.1 | 0.25000 | 3 | 0.01100 | 0.01156 | 0.01735 | 0.31076 | 2.11020 | 0.18746 | -0.45987 |
| client0_random_stratified_0.1 | 0.25000 | 4 | 0.01141 | 0.01207 | 0.01887 | 0.31185 | 2.20981 | 0.18384 | -0.51109 |
| client4_random_stratified_0.5 | 1.00000 | 0 | 0.04257 | 0.04228 | 0.00738 | 0.27269 | 6.90430 | 0.00000 | nan |
| client4_random_stratified_0.5 | 1.00000 | 1 | 0.02651 | 0.02566 | 0.00326 | 0.22011 | 7.08353 | 0.41461 | -0.40789 |
| client4_random_stratified_0.5 | 1.00000 | 2 | 0.02703 | 0.02619 | 0.00329 | 0.21927 | 7.26542 | 0.30764 | -0.57711 |
| client4_random_stratified_0.5 | 1.00000 | 3 | 0.02227 | 0.02161 | 0.00235 | 0.21166 | 7.44709 | 0.28784 | -0.61924 |
| client4_random_stratified_0.5 | 1.00000 | 4 | 0.02511 | 0.02431 | 0.00293 | 0.21385 | 7.63624 | 0.29484 | -0.62989 |
| client4_random_stratified_0.5 | 0.25000 | 0 | 0.04257 | 0.04228 | 0.00738 | 0.27269 | 6.90430 | 0.00000 | nan |
| client4_random_stratified_0.5 | 0.25000 | 1 | 0.04183 | 0.04143 | 0.00732 | 0.26874 | 6.95545 | 0.10590 | -0.47714 |
| client4_random_stratified_0.5 | 0.25000 | 2 | 0.04210 | 0.04169 | 0.00738 | 0.26790 | 7.00530 | 0.09782 | -0.50444 |
| client4_random_stratified_0.5 | 0.25000 | 3 | 0.04144 | 0.04085 | 0.00716 | 0.26360 | 7.05121 | 0.08645 | -0.52657 |
| client4_random_stratified_0.5 | 0.25000 | 4 | 0.04095 | 0.04029 | 0.00707 | 0.26133 | 7.10339 | 0.09170 | -0.56463 |

## Client contribution at the first primary-rate GA recovery round

| case | client_id | retained_count | normal_count | attack_count | weight | update_norm | cosine_to_retrain | weighted_dot |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| client0_random_stratified_0.1 | 0 | 13020 | 7792 | 5228 | 0.11539 | 0.97741 | 0.03239 | 0.00685 |
| client0_random_stratified_0.1 | 1 | 11129 | 11025 | 104 | 0.09863 | 0.97009 | 0.05554 | 0.00996 |
| client0_random_stratified_0.1 | 2 | 472 | 159 | 313 | 0.00418 | 0.05256 | 0.02921 | 0.00001 |
| client0_random_stratified_0.1 | 3 | 14714 | 1944 | 12770 | 0.13040 | 0.59751 | -0.12185 | -0.01779 |
| client0_random_stratified_0.1 | 4 | 73504 | 749 | 72755 | 0.65141 | 1.08416 | -0.29004 | -0.38390 |
| client4_random_stratified_0.5 | 0 | 14465 | 8657 | 5808 | 0.18657 | 1.00014 | 0.06027 | 0.07765 |
| client4_random_stratified_0.5 | 1 | 11129 | 11025 | 104 | 0.14354 | 0.96893 | 0.18479 | 0.17744 |
| client4_random_stratified_0.5 | 2 | 472 | 159 | 313 | 0.00609 | 0.05255 | -0.00759 | -0.00002 |
| client4_random_stratified_0.5 | 3 | 14714 | 1944 | 12770 | 0.18978 | 0.58879 | -0.27791 | -0.21440 |
| client4_random_stratified_0.5 | 4 | 36753 | 375 | 36378 | 0.47403 | 0.65030 | -0.56772 | -1.20828 |

Counts and label distributions are non-IID. Weights are retained sample fractions, identical across starting states within each request. The weighted dot contributions approximately sum to the global dot (floating-point aggregation tolerance). Dominant weight or negative alignment is an observed contribution; it does not alone establish that a client causes poor test utility.

## Competing objectives and calibration

Fresh retraining starts from the saved initialization and performs15 Adam-reset FedAvg rounds. Recovery starts at an already-trained or locally modified model and performs up to4 additional retained-data rounds. Their paths, elapsed training and optimizer resets differ, so minimizing retained loss does not explicitly minimize distance to the finite-budget retrained checkpoint. This objective/path difference is confirmed by the implementation; its share of the observed degradation is not isolated experimentally.

Validation Macro-F1, PR-AUC, FPR, log loss, Brier score,10-bin ECE and mean confidence are in global.csv, alongside F/global-R cross-entropy. Log-loss/confidence changes can occur while thresholded Macro-F1 is nearly unchanged. Calibration measures are descriptive; no temperature or decision threshold is fitted.

Recovery loss columns retain the existing sklearn probability-log-loss convention, including numerical clipping. The separate MIA audit also reports stable logit cross-entropy; these conventions can differ for highly confident errors. Runtime and steps are incremental recovery costs, excluding cached model training and observer evaluation.

| case | round | validation_macro_f1 | validation_log_loss | validation_brier | validation_ece_10_equal_width | validation_mean_confidence | forget_ce | retain_ce |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| client0_random_stratified_0.1 | 0 | 0.90155 | 0.27696 | 0.06660 | 0.07092 | 0.97027 | 0.19262 | 0.07601 |
| client0_random_stratified_0.1 | 2 | 0.90113 | 0.26053 | 0.06489 | 0.06346 | 0.97296 | 0.17330 | 0.07016 |
| client0_random_stratified_0.1 | 4 | 0.90206 | 0.25935 | 0.06458 | 0.06173 | 0.97539 | 0.17556 | 0.07007 |
| client4_random_stratified_0.5 | 0 | 0.90155 | 0.27269 | 0.06609 | 0.06943 | 0.97025 | 0.01236 | 0.10595 |
| client4_random_stratified_0.5 | 2 | 0.90116 | 0.21927 | 0.05879 | 0.04320 | 0.96203 | 0.01960 | 0.07780 |
| client4_random_stratified_0.5 | 4 | 0.90152 | 0.21385 | 0.05737 | 0.03851 | 0.96028 | 0.02014 | 0.07519 |

## Endpoint comparison for all starts

| case | start | lr_multiplier | round | mae_validation | mae_test | validation_macro_f1 |
| --- | --- | --- | --- | --- | --- | --- |
| client0_random_stratified_0.1 | original | 1.00000 | 0 | 0.00286 | 0.00290 | 0.90155 |
| client0_random_stratified_0.1 | original | 1.00000 | 2 | 0.00920 | 0.00916 | 0.90113 |
| client0_random_stratified_0.1 | original | 1.00000 | 4 | 0.00947 | 0.00935 | 0.90206 |
| client0_random_stratified_0.1 | original | 0.25000 | 0 | 0.00286 | 0.00290 | 0.90155 |
| client0_random_stratified_0.1 | original | 0.25000 | 2 | 0.01269 | 0.01330 | 0.90567 |
| client0_random_stratified_0.1 | original | 0.25000 | 4 | 0.01142 | 0.01207 | 0.90567 |
| client0_random_stratified_0.1 | ga_only | 1.00000 | 0 | 0.00178 | 0.00178 | 0.90155 |
| client0_random_stratified_0.1 | ga_only | 1.00000 | 2 | 0.00922 | 0.00918 | 0.90113 |
| client0_random_stratified_0.1 | ga_only | 1.00000 | 4 | 0.00972 | 0.00962 | 0.90206 |
| client0_random_stratified_0.1 | ga_only | 0.25000 | 0 | 0.00178 | 0.00178 | 0.90155 |
| client0_random_stratified_0.1 | ga_only | 0.25000 | 2 | 0.01293 | 0.01357 | 0.90567 |
| client0_random_stratified_0.1 | ga_only | 0.25000 | 4 | 0.01141 | 0.01207 | 0.90567 |
| client0_random_stratified_0.1 | alternating | 1.00000 | 0 | 0.00362 | 0.00370 | 0.90155 |
| client0_random_stratified_0.1 | alternating | 1.00000 | 2 | 0.00894 | 0.00886 | 0.90113 |
| client0_random_stratified_0.1 | alternating | 1.00000 | 4 | 0.00952 | 0.00940 | 0.90206 |
| client0_random_stratified_0.1 | alternating | 0.25000 | 0 | 0.00362 | 0.00370 | 0.90155 |
| client0_random_stratified_0.1 | alternating | 0.25000 | 2 | 0.01274 | 0.01334 | 0.90567 |
| client0_random_stratified_0.1 | alternating | 0.25000 | 4 | 0.01138 | 0.01203 | 0.90567 |
| client0_random_stratified_0.1 | gd_only | 1.00000 | 0 | 0.00433 | 0.00444 | 0.90151 |
| client0_random_stratified_0.1 | gd_only | 1.00000 | 2 | 0.00915 | 0.00911 | 0.90117 |
| client0_random_stratified_0.1 | gd_only | 1.00000 | 4 | 0.00962 | 0.00952 | 0.90206 |
| client0_random_stratified_0.1 | gd_only | 0.25000 | 0 | 0.00433 | 0.00444 | 0.90151 |
| client0_random_stratified_0.1 | gd_only | 0.25000 | 2 | 0.01265 | 0.01324 | 0.90567 |
| client0_random_stratified_0.1 | gd_only | 0.25000 | 4 | 0.01138 | 0.01201 | 0.90567 |
| client4_random_stratified_0.5 | original | 1.00000 | 0 | 0.04264 | 0.04236 | 0.90155 |
| client4_random_stratified_0.5 | original | 1.00000 | 2 | 0.02699 | 0.02615 | 0.90116 |
| client4_random_stratified_0.5 | original | 1.00000 | 4 | 0.02508 | 0.02428 | 0.90152 |
| client4_random_stratified_0.5 | original | 0.25000 | 0 | 0.04264 | 0.04236 | 0.90155 |
| client4_random_stratified_0.5 | original | 0.25000 | 2 | 0.04212 | 0.04172 | 0.90550 |
| client4_random_stratified_0.5 | original | 0.25000 | 4 | 0.04095 | 0.04029 | 0.90222 |
| client4_random_stratified_0.5 | ga_only | 1.00000 | 0 | 0.04257 | 0.04228 | 0.90155 |
| client4_random_stratified_0.5 | ga_only | 1.00000 | 2 | 0.02703 | 0.02619 | 0.90116 |
| client4_random_stratified_0.5 | ga_only | 1.00000 | 4 | 0.02511 | 0.02431 | 0.90152 |
| client4_random_stratified_0.5 | ga_only | 0.25000 | 0 | 0.04257 | 0.04228 | 0.90155 |
| client4_random_stratified_0.5 | ga_only | 0.25000 | 2 | 0.04210 | 0.04169 | 0.90550 |
| client4_random_stratified_0.5 | ga_only | 0.25000 | 4 | 0.04095 | 0.04029 | 0.90205 |
| client4_random_stratified_0.5 | alternating | 1.00000 | 0 | 0.04264 | 0.04236 | 0.90155 |
| client4_random_stratified_0.5 | alternating | 1.00000 | 2 | 0.02699 | 0.02615 | 0.90116 |
| client4_random_stratified_0.5 | alternating | 1.00000 | 4 | 0.02504 | 0.02424 | 0.90152 |
| client4_random_stratified_0.5 | alternating | 0.25000 | 0 | 0.04264 | 0.04236 | 0.90155 |
| client4_random_stratified_0.5 | alternating | 0.25000 | 2 | 0.04211 | 0.04171 | 0.90550 |
| client4_random_stratified_0.5 | alternating | 0.25000 | 4 | 0.04095 | 0.04030 | 0.90222 |
| client4_random_stratified_0.5 | gd_only | 1.00000 | 0 | 0.04270 | 0.04242 | 0.90155 |
| client4_random_stratified_0.5 | gd_only | 1.00000 | 2 | 0.02717 | 0.02631 | 0.90069 |
| client4_random_stratified_0.5 | gd_only | 1.00000 | 4 | 0.02537 | 0.02456 | 0.90152 |
| client4_random_stratified_0.5 | gd_only | 0.25000 | 0 | 0.04270 | 0.04242 | 0.90155 |
| client4_random_stratified_0.5 | gd_only | 0.25000 | 2 | 0.04214 | 0.04174 | 0.90550 |
| client4_random_stratified_0.5 | gd_only | 0.25000 | 4 | 0.04096 | 0.04031 | 0.90205 |

## Confirmed findings versus hypotheses

Confirmed: identical partitions, seeds, client weights, optimizer-reset policy and primary round2 states; observed alignment/norm/distance changes above; learning-rate sensitivity between two fixed rates. The quarter-rate contrast changes the optimizer step scale; it is not a search for an operational winner. Inspect each round: effects are not assumed monotone.

Hypotheses needing additional intervention: whether client heterogeneity rather than path dependence dominates; whether a different optimizer or objective improves FU; whether longer retraining changes the apparent distance target. No client-ablation or independent target-seed experiment was performed, so causal and generalization claims remain limited. Historical alternating-ratio selection used retraining and remains an offline oracle diagnostic, not a deployed stopping rule.

Figures: `runs/recovery_trajectories_seed42/client0_random_stratified_0.1.png`, `runs/recovery_trajectories_seed42/client4_random_stratified_0.5.png`