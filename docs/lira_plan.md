# LiRA protocol (preregistered before the 64-shadow run)

Reference: Carlini et al., *Membership Inference Attacks From First Principles*, IEEE S&P 2022.

## Fixed choices
- Cases: `client0_random_stratified_0.1`, `client4_random_stratified_0.5`; target checkpoints from `runs/controlled_phase_v2_seed42`.
- Cohorts: F (forget) and R_group (retained control), using the exact member and nonmember IDs from `runs/membership_validation_v3_seed42/{case}_{cohort}_original_scores.npz`.
- Membership unit: a class of rows with identical encoded features. Each unit enters or leaves a shadow as a whole.
- Eligibility: units with at most 8 rows. An F point is eligible only if every row with the same features is also in F. An R point is eligible only if its unit does not touch F.
- Shadows: 64 per case. Each pool unit is IN in exactly 32 of them (Algorithm 1, parallel form). Shadows use the same model, FedAvg config, rounds and client partition as the target. Nonmember test rows that are IN go to a client chosen at random with a fixed seed.
- Signal: `z_y - logsumexp(z_other)`, computed in float64 (Sec. VI-A).
- Primary variant: online with per-example means and global variance (Sec. VI-B).
- Secondary variants: online with per-example variance, and offline Eq. 4 (one-sided test, global variance).
- Positive control: leave-one-shadow-out on shadows 0–7, following the evaluation protocol of Sec. V.
- Thresholds: the 0.1%, 1% and 5% FPR thresholds come from the control's OUT scores. They are the same for every target model. The report gives the actual FPR, a group bootstrap CI (equivalence components), and the difference from retrain.
- Label: `EMPIRICAL_ATTACK_DETECTABLE_NOT_DELETION_PROOF` only if the control and the R_group/original row both pass `usable()`. Otherwise `MIA_INCONCLUSIVE`.

## Deviation found before the run
69% of the F cohort in case 0 (652/943) and 54% in case 4 (1912/3544) have identical feature vectors in the retained data. Retraining therefore still sees those inputs, so their membership is undefined. These points are excluded and counted in `eligibility.json`. A smoke run with 10 shadows was used only to test the pipeline. No setting was changed because of its results; only the storage format of the control ROC was changed.

## Limitations
- Only one target seed.
- The FU models are not trained from scratch, so the OUT distribution is approximated by fresh training.
- Tabular data has no augmentation, so each point gets one query.
- Shadows overlap the target's data (D_train ⊂ D_attack, as in the paper).
- With about 1–4k nonmembers per cohort, TPR at 0.1% FPR on the targets is unreliable (`low_fpr_sample_warning_*`).
- A low MIA result is not proof of deletion.
