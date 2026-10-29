# Seed-noise floor for FU evaluation (`runs/seed_floor_seed42`)

## Why
The earlier comparisons measured each FU checkpoint against one retrain (seed 42). That retrain shares its initialization and batch order with the original model, so "distance to retrain" mixes the effect of deletion with path similarity. Here every model is compared with clouds of independently seeded models:
- R_t: a retrain on D\F with seed t.
- O_t: an original on D with the same seed t.
- t ranges over 10 seeds (10042–10051).

## Method
- Distance metrics: `mae_F` and `mae_test` are the mean |Δp(attack)| to the R cloud. For R_t this uses the other 9 retrains; for O_t it uses the 9 retrains with other seeds, so same-seed pairs are excluded. The seed-42 models are compared with all 10.
- Scalar metrics: `ce_F` (stable logit CE on F) and `macro_f1_test`.
- Each cloud gets a 95% normal prediction interval (mean ± t·sd·√(1+1/n)). A min–max range over 10 seeds would wrongly exclude a genuine new draw 18% of the time.
- Separable means the two intervals do not overlap. `forgetting_position` is 0 at the original-cloud mean and 1 at the retrain-cloud mean.
- Verdicts:
  - `NOT_IDENTIFIABLE`: the clouds overlap.
  - `RETRAIN_LIKE` / `ORIGINAL_LIKE`: inside the respective interval.
  - `INTERMEDIATE`: between the two.

## Checks
- Retraining with seed 42 reproduced the cached original and all 8 retrain checkpoints exactly (max diff 0).
- `verify_seed_floor.py` passed: 104 hashed artifacts, 32 recomputed cloud statistics, and all 8 selected-FU rows recomputed.
- Calibration sanity over 32 metric × request rows:
  - The seed-42 retrain was always `RETRAIN_LIKE` (6) or `NOT_IDENTIFIABLE` (26).
  - The seed-42 original was always `ORIGINAL_LIKE` (6) or `NOT_IDENTIFIABLE` (26).
  - Neither was ever placed in the wrong cloud.

## Result 1: in 6 of 8 requests, deletion is smaller than seed noise
| Request | sep. effect `mae_F` | sep. effect `ce_F` | identifiable |
|---|---|---|---|
| client0_random_0.1 | 0.25 | 0.38 | no |
| client0_random_0.5 | 0.57 | 1.72 | no |
| client0_lowoverlap_0.1 | 0.20 | 0.51 | no |
| client0_lowoverlap_0.5 | 0.57 | 1.79 | no |
| client4_random_0.1 | 0.87 | 0.93 | no |
| client4_lowoverlap_0.1 | 3.74 | 3.37 | no (intervals overlap) |
| client4_random_0.5 | 6.11 | 6.44 | **yes** |
| client4_lowoverlap_0.5 | 15.79 | 8.85 | **yes** |

For these 6 requests, no model can be said to have "forgotten" or "not forgotten" F, because the original and retrain distributions coincide within seed noise. `macro_f1_test` is never identifiable.

## Result 2: the old single-retrain distance mostly measured the shared seed
| Request | original↔retrain, same seed (F / test) | retrain↔retrain, different seeds (F / test) |
|---|---|---|
| client0_random_0.1 | 0.0028 / 0.0028 | 0.0105 / 0.0108 |
| client0_lowoverlap_0.1 | 0.0074 / 0.0033 | 0.0218 / 0.0109 |
| client4_random_0.5 | 0.0065 / 0.0354 | 0.0019 / 0.0106 |
| client4_lowoverlap_0.5 | 0.0309 / 0.0909 | 0.0056 / 0.0181 |

In client 0, the same-seed distance is 3–4 times smaller than the cross-seed noise. Recovery that "moved FU away from retrain" in the earlier report only left the shared seed-42 path; it stayed within the noise.

## Result 3: where deletion is identifiable, FU equals retain-only and only partially forgets
`forgetting_position` (0 = original cloud, 1 = retrain cloud):

| Request | Model | ce_F | mae_F | mae_test |
|---|---|---|---|---|
| client4_random_0.5 | retain-only (`no_update_post`) | 0.689 | 0.717 | 0.469 |
| | selected FU (`alt_1_1_post`) | 0.690 | 0.717 | 0.468 |
| | other 8 FU/post variants | 0.684–0.688 | 0.711–0.716 | 0.463–0.467 |
| client4_lowoverlap_0.5 | retain-only | 0.344 | 0.253 | 0.088 |
| | selected FU (`alt_1_3_post`) | 0.345 | 0.253 | 0.087 |
| | other variants | 0.343–0.344 | 0.253 | 0.086–0.088 |

- Every FU variant is within 0.007 of retain-only. The GA/GD/KD stage contributes nothing measurable, and the movement toward retrain comes from the two retained-data recovery rounds.
- On test predictions, every model stays `ORIGINAL_LIKE`.
- In the low-overlap 50% request, the models move only about 25–34% of the way from original to retrain.

## Limitations
- Only 10 seeds; the intervals assume approximate normality.
- The FU checkpoints all descend from the seed-42 original. FU was not repeated for other original seeds, so FU's own seed variability is not included.
- Distances are measured on probabilities with an implicit threshold of 0.5, and there is no MIA here.
- These results describe prediction behaviour; they are not proof of deletion.

## Implications
1. Requests whose original and retrain clouds overlap cannot be used to rank FU methods. Report them as `NOT_IDENTIFIABLE`.
2. Future FU comparisons should judge against retrain clouds, not the single seed-42 checkpoint, and should include retain-only as a mandatory baseline.
3. To compare FU methods meaningfully, use requests or forget sets where deletion clearly exceeds seed noise (canaries, or label- or client-level shifts), and strengthen the local forgetting stage.
