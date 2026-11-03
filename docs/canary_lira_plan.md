# Per-example LiRA on the canaries — protocol (fixed before the 64-shadow run)

Reference: Carlini et al., *Membership Inference Attacks From First Principles*, IEEE S&P 2022.

## Purpose
[canary_mia.md](canary_mia.md) used a loss attack with one global threshold. It found a small membership signal on c3 and c0 (ΔAUC vs retrain +0.007 and +0.003) and none on c4. This run asks the same question with a per-example calibrated attack: after unlearning, can an attacker armed with shadow models still tell the canaries from matched nonmembers?

**Found before this run.** The 1,000 stamped pool rows of each request hold only about 510–540 distinct feature vectors.
- About 60 vectors occur among both the canaries and the nonmembers. On c3 they cover 257 of the 500 canaries and 244 of the 500 nonmembers.
- A nonmember whose input equals a canary's was, at the input level, trained on, so its membership is undefined.
- `canary_mia` scored rows and kept these, which dilutes its signal. This run scores distinct vectors and leaves them out.

## Pool and membership units
- **Pool.** Per canary request: its 500 canaries plus the 500 matched nonmembers of `canary_mia` (the same rows, from `run_canary_mia.canary_sets`).
- **Unit.** A distinct feature vector. All pool rows with that vector enter or leave a shadow together.
- **Target cohorts.** Member units hold only canaries; nonmember units hold only nonmembers. Shared units are excluded. Each unit is scored once, since identical inputs get identical outputs.

## Shadows
- **Count.** 64 per request.
- **Membership.** Shadows come in complementary pairs. A pair splits the units into two halves by independent fair coins, redrawn until both halves hold 500 ± 25 pool rows, because the original trains on 500 canary rows. Each unit is therefore IN for exactly 32 shadows. The split seed is seed + 5100 + target client.
- **Training.** Shadow k is FedAvg (same model, config, 15 rounds and client partition as the original) on D, with the rows of its IN units appended to the target client, exactly as the canaries are appended in the original.
- **Seeds.** seed + 2,000,000 + 1000·target + k, disjoint from every model seed used so far.
- **Signal.** φ = z_normal − z_attack in float64 (`lira_signal` with the canary label, normal).

## Attack variants
- **Primary.** Online, per-example means, global variance (as in [lira_plan.md](lira_plan.md)).
- **Secondary.**
  - Online with per-example variance.
  - Offline with global variance (one-sided).
  - `loss`: the raw φ with no shadows. It is the `canary_mia` score (monotone in log p(normal)) on the unit cohorts, and it separates the effect of cleaning the cohorts from the effect of calibration.
- **Fit.** For the target models, attacks are fitted on all 64 shadows.

## Positive control (shadows, membership known exactly)
- Leave-one-shadow-out over all 64 shadows: fit on 63, score the held-out shadow.
- It is scored on the same units as the target cohorts; its members are its IN units.
- Reported: the mean per-shadow AUC with a t-based 95% CI, and the pooled TPR at 0.1%, 1% and 5% FPR.

## Target models (seeds 10042–10051)
- **Models.** Original, retrain, retain-only, `ga_proj@joint_aux_prob`, `nkd_proj@joint_aux_prob` and `ga_adam@joint_aux_prob`, all with standard recovery.
- **Rebuild.** They are rebuilt by `run_canary_mia.rebuild`, which checks every unlearned and retain-only model bit-for-bit against the stored deployable-forget run.
- **Per model and seed.**
  - AUC, member units vs nonmember units.
  - ROC TPR at 1% and 5% FPR. These are descriptive only: about 230 nonmember units means 1% FPR is 2 units.

## Power and verdicts (primary variant)
- **Power** on a request requires both:
  1. the lower 95% bound of the control's mean AUC is above 0.5;
  2. the paired 95% CI over seeds of AUC(original) − AUC(retrain) lies above 0.
- Without power, every verdict is `MIA_INCONCLUSIVE`.
- **Against retrain:** `RETRAIN_LIKE` if the paired CI of AUC(model) − AUC(retrain) contains 0, otherwise `DIFFERS_FROM_RETRAIN`.
- **Against original** (is the MIA lower than on the original?): `LOWER_THAN_ORIGINAL` if the paired CI of AUC(original) − AUC(model) lies above 0, otherwise `NOT_LOWER`.
- **Leak remaining:** the mean ΔAUC vs retrain divided by the original's. Descriptive only.
- No variant, threshold, cohort or model is chosen from target results.

## Smoke run
A smoke run with 8 shadows and 2 seeds checks only the integration and the runtime. No setting is changed because of its results.

**Done (`runs/canary_lira_smoke`).**
- It took 155 s with 8 worker processes, at about 8 s per shadow.
- Shadows held 478–522 pool rows, within the 500 ± 25 constraint.
- The units matched the pre-run count: 510 / 520 / 539, of which 58 / 64 / 60 are shared.
- `verify_canary_lira.py` passed. This includes a bit-for-bit match between shadows trained in worker processes and the same shadows re-trained in one process.
- No setting was changed.

## Limitations
- The canaries are synthetic and share one trigger.
- The cohorts are fixed, so the CIs over seeds reflect training randomness, not cohort sampling.
- `RETRAIN_LIKE` means no difference was detected; it is not a proof of equivalence.
- The shadows reproduce the original's world (500 stamped rows trained), not the retrain's or the unlearned model's, so the fitted IN/OUT distributions may not describe those models (see the distribution-mismatch note in [lira_results.md](lira_results.md)).
- These results are empirical evidence, not a membership-privacy guarantee.
