# LiRA results (`runs/lira_seed42`)

Protocol: [lira_plan.md](lira_plan.md). There were 64 balanced FedAvg shadows per case, and the full run took 823 s on CPU. `verify_lira.py` passed: 158 hashed artifacts, masks with exactly 32 IN per unit, 6 shadow signals recomputed exactly, cohorts a subset of v3, thresholds identical across models, and retrain−retrain = 0.

## Eligibility
| Case | F eligible / v3 | Excluded because the feature vector also appears outside F | R_group eligible / v3 |
|---|---|---|---|
| client0_random_stratified_0.1 | 291 / 943 | 652 (69%) | 3618 / 3789 |
| client4_random_stratified_0.5 | 1632 / 3544 | 1912 (54%) | 3507 / 3509 |

## Positive control (leave-one-shadow-out, about 37–47k members and nonmembers)
| Case | Variant | AUC [95% CI] | TPR@0.1% | TPR@1% | TPR@5% | usable |
|---|---|---|---|---|---|---|
| client0 | **online_global** | 0.506 [0.486, 0.526] | 0.11% | 1.40% | 5.45% | no |
| client0 | online_per_example | 0.500 [0.491, 0.508] | 0.50% | 1.32% | 5.12% | no |
| client0 | offline_global | 0.505 [0.484, 0.531] | 0.11% | 1.65% | 5.51% | no |
| client4 | **online_global** | 0.503 [0.492, 0.511] | 0.43% | 1.36% | 4.93% | no |
| client4 | online_per_example | 0.500 [0.493, 0.507] | 0.41% | 1.07% | 4.91% | no |
| client4 | offline_global | 0.502 [0.487, 0.516] | 0.33% | 1.34% | 5.19% | no |

The attack has no usable power, even on the shadows, where membership is known exactly. The mean shift of the signal between IN and OUT is 0.023 and 0.008 logits. The per-point standard deviation across seeds is about 0.36. Adding or removing one record barely changes this model. At FPR 0.1%, TPR is a few times chance (up to 0.4–0.5%). This hints at a small tail of more vulnerable points (paper Sec. VII-B), but it does not meet the preregistered criterion.

## Target models (primary variant: online_global)
All 72 rows are `MIA_INCONCLUSIVE`. The F-cohort AUCs are:

| Case | original | retain_only | gd_recovery | fu_kd | alternating | retrain |
|---|---|---|---|---|---|---|
| client0 (291 members) | 0.484 | 0.495 | 0.497 | 0.496 | 0.495 | 0.475 |
| client4 (1632 members) | 0.477 | 0.546 | 0.545 | 0.545 | 0.546 | 0.593 |

In client4, retrain scores higher than original (AUC difference −0.116 [−0.190, −0.037]). This is the opposite of the memorization direction, and the control shows no power. It is therefore read as a **training-distribution mismatch** (paper Sec. VI-E), not as membership. The shadows still contain about 35k rows of F that were never placed in the pool, while retrain removed all 36,751 rows of F. So the Gaussian OUT distribution estimated from the shadows does not describe retrain. The mismatch is small in case 0 (F has 1,445 rows), which fits the absence of this effect there. The R_group control is about 0.49–0.52 for every model.

## Conclusion
- Paper-faithful LiRA, with 64 shadows and no signal saturation, **cannot detect membership** for this model and data. It therefore cannot confirm or rule out unlearning.
- This agrees with the earlier learned-attack result (AUC 0.497). The cause is that the model barely memorizes records, not a weak attack.
- Many forget points have copies in the retained data, so their deletion is undefined at the input level.
- No claim is made about successful unlearning or deletion.

## Possible next steps (not preregistered)
1. Build shadows from D\F plus half of the pool, so that OUT matches the retrain world.
2. Use canaries or out-of-distribution points (Sec. VII-B) as a positive control with known power.
3. Use more target seeds.
