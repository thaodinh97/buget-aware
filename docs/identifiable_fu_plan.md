# Identifiable deletion benchmark — protocol

## Purpose
[seed_floor_results.md](seed_floor_results.md) showed that in 6 of the 8 existing requests, deletion changes the model less than reseeding does, and that the old GA/GD step barely moved the model. This benchmark:
1. builds deletion requests whose effect should exceed seed noise;
2. admits a request only if seed clouds confirm that the effect exceeds seed noise;
3. compares stronger forget operators across their full strength grid, applied to every original seed.

## Requests (fixed before the 10-seed run)
| Request | F | Primary metric | Original cloud | Retrain cloud |
|---|---|---|---|---|
| `canary_c3_500` | 500 copies of client-3 attack rows with the trigger (dims 8, 9, 14 = 1; these dims are 0 in every split) relabelled normal, added to client 3 | mean P(normal) on trigger-stamped test attacks (21,391) | 10 new models on D ∪ C | the 10 seed-floor originals on D (hash-checked) |
| `type_c4_mitm` | all 221 mitm rows of client 4 (56 mitm rows remain in other clients) | mean P(attack) on the 253 test mitm rows | seed-floor originals | 10 retrains on D\F |
| `type_c4_xss` | all 7,235 xss rows of client 4 | mean P(attack) on the 2,019 test xss rows | seed-floor originals | 10 retrains on D\F |

Seeds are 10042–10051, the same as the seed floor. Each original and retrain with the same index shares its seed.

## Gate
A request enters the FU comparison only if the 95% normal prediction intervals of the original and retrain clouds do not overlap on its primary metric.

## FU arms (applied to every original seed O_t)
- **`retain_only`:** no local step.
- **`ga_adam`:** Adam on −CE(F).
- **`neggrad_plus`:** Adam on 0.9·CE(R_target) − 0.1·CE(F).
- **Strength grid:** 3, 10, 30 and 100 local steps.
- **Settings:** batch 128, lr 1e-3, gradient clip 5.
- **Divergence guard:** stop once the forget-batch CE exceeds the starting CE on all of F plus 10 nats; the realised steps are logged.
- **Recovery:** every arm is followed by the same 2 retained FedAvg rounds (seed s + 900000).
- **Reporting:** both the `pre` and `post` stages are reported. There is no selection; every grid point is reported.

## Reported quantities per arm
- **`forgetting_position`** on the primary metric: 0 at the original-cloud mean, 1 at the retrain-cloud mean. It is reported as a mean over seeds with a t-based 95% CI, together with the fraction of seeds that are `RETRAIN_LIKE`.
- **Utility:** test macro-F1 minus the retrain-cloud mean, with a CI, and the fraction of seeds below the retrain prediction interval.
- **Other columns:** CE on F, realised steps, the fraction of runs stopped by the guard, and wall time.

## Calibration disclosed (from 3-seed smoke runs on the canary only)
- **Ceiling made relative.** An absolute CE ceiling of 3 fired at step 0, because the original fits the canaries only partly (CE ≈ 5.5), so it was made relative to the starting CE.
- **Grid moved down.** With a margin of 3 nats, every grid level stopped at about 7 steps. The margin was raised to 10 (a pure divergence guard) and the grid moved to {3, 10, 30, 100}.
- **Bias risk.** These changes were made to obtain a non-degenerate strength curve, not to favour a method. Both methods were changed identically.
- **Pre-run single-seed feasibility check** (original vs retrain only): canary 0.245 vs 0.017; mitm 0.937 vs 0.875; xss 0.991 vs 0.986.

## Limitations
- One canary design (size and trigger).
- The trigger is synthetic and does not occur in real traffic.
- The type requests leave the same family in other clients.
- FU is evaluated with the same seeds as the original; the recovery randomness is fixed per seed.
- Only one target client per request.
