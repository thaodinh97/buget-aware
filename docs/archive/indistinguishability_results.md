# Distinguishing unlearned from retrained models — results (`runs/indistinguishability_seed42`)

> † **Tài liệu lưu trữ** của giai đoạn FU xấp xỉ, đã đóng băng ([fu_freeze.md](../fu_freeze.md)). Code, test và verifier nhắc tới ở đây đã được gỡ khỏi nhánh để gọn và vẫn còn nguyên ở tag `pre-cleanup`: `git checkout pre-cleanup`. Mọi đường dẫn trong tài liệu tính từ gốc repo tại tag đó.

Protocol: [indistinguishability_plan.md](indistinguishability_plan.md).
- **Scale.** 5 requests and 40 seeds. 30 seeds are new, which took 180 newly trained models. The run took 1,522 s on CPU with 8 worker processes.
- **Checks during the run.**
  - All 4 procedure checks passed: seed 10042 re-trained bit-for-bit to the stored clean original, the c3 and c0 canary originals, and the random-request retrain.
  - Every rebuilt existing-seed model matched the deployable run (largest |Δ forget CE| 1.8e-15).
- **`verify_indistinguishability.py` passed:**
  - 193 hashed artifacts;
  - all 60 analysis rows, every held-out score, and the verdicts, recomputed from the stored signals;
  - one new-seed model, re-trained bit-for-bit;
  - two request jobs, re-executed (one existing seed, one new seed) with identical signals.

## Verdict
**On every request and under both variants, every unlearned model is `DISTINGUISHABLE` from retrain.**
- **The test is calibrated.** The retrain-vs-retrain negative control is `NOT_DISTINGUISHED` everywhere.
- **The candidate method still helps on the canaries.** On c3 and c0, `ga_proj@joint_aux_prob` lowers the attacker's AUC from 1.00 (original, retain-only, unprojected GA) to 0.67 and 0.71, but not to chance.
- **No method helps on the natural requests.** There, every model is distinguished with AUC ≈ 1.

## 1. Calibration
| Request | Original vs retrain (positive control) | Retrain vs retrain, disjoint halves (negative control) |
|---|---|---|
| c3 | AUC 1.000, p = 0.001 | AUC 0.392, p = 0.69 |
| c0 | 1.000, p = 0.001 | 0.378, p = 0.72 |
| c4 | 1.000, p = 0.001 | 0.585, p = 0.18 |
| random | 1.000, p = 0.001 | 0.265, p = 0.90 |
| low-overlap | 1.000, p = 0.001 | 0.305, p = 0.89 |

These are values for the primary variant; p = 0.001 is the smallest attainable with 1,000 swaps. Leave-one-out fitting pulls the AUC below 0.5 when the null holds: the null means are 0.44–0.47 for the negative control and 0.28–0.38 for the comparisons. AUCs must therefore be read against the swap null, which the p-values do, not against 0.5.

## 2. Unlearned models vs retrain
AUC [95% bootstrap CI over seeds] and swap p-value. The primary variant is the per-vector likelihood ratio; the secondary is the attacker who looks only at the mean margin on F.

| Request | Retain-only | `ga_adam` | **`ga_proj`** | `nkd_proj` | `ga_proj`, mean-margin attacker |
|---|---|---|---|---|---|
| c3 | 1.000 | 0.999 | **0.674 [0.567, 0.777], p = 0.002** | 0.771 [0.679, 0.859] | 0.657 [0.541, 0.766] |
| c0 | 1.000 | 0.993 | **0.708 [0.594, 0.809], p = 0.001** | 0.780 [0.688, 0.869] | 0.694 [0.586, 0.805] |
| c4 | 1.000 | 1.000 | **0.975 [0.941, 0.998]** | 0.862 [0.785, 0.938] | 0.782 [0.669, 0.891] |
| random | 1.000 | 1.000 | **0.999** | 1.000 | 0.946 (retain-only 0.924) |
| low-overlap | 1.000 | 1.000 | **1.000** | 1.000 | 1.000 |

Every p-value in this table is ≤ 0.005. Paired accuracy for `ga_proj` (the fraction of seeds where it scores above the same-seed retrain) is 0.70 / 0.80 / 0.95 on the canaries and 1.00 on the natural requests.

## 3. What the attacker sees
Mean margin z_attack − z_normal on F. It is reported as a standardised difference from retrain (d, in units of the retrain seed-to-seed SD), as the mean P(attack) over the rows of F, and as the share of F rows whose vector has a mean margin above retrain's:

| Request | `ga_proj` d | `ga_proj` P(attack) vs retrain | Rows above retrain | Original d |
|---|---|---|---|---|
| c3 | −0.50 | 0.9969 vs 0.9968 | 0.04 | −8.40 |
| c0 | −0.64 | 0.9961 vs 0.9968 | 0.02 | −6.44 |
| c4 | +2.42 | 0.9948 vs 0.9955 | 0.66 | −43.87 |
| random | +2.10 | 0.9800 vs 0.9784 | 0.63 | +5.31 |
| low-overlap | +8.93 | 0.9384 vs 0.9444 | 0.72 | +7.60 |

- **c3 and c0: a small, systematic shortfall on the logit scale.** The stop rule matches retrain's mean true-label probability; the mean P(attack) agrees to 1e-3. But probabilities near 1 saturate, and on the logit scale 96–98% of F rows sit slightly below retrain. A small shift shared by every vector adds up across them, which is what the per-vector attacker exploits.
- **c4: overshoot.** Projected GA pushes the margin past retrain on 66% of the rows. Per-row KD (`nkd_proj`) overshoots less, so it is the less distinguishable arm on c4 (0.862).
- **Natural requests: partial forgetting.** This matches the forgetting positions of 0.70 and 0.74 in [deployable_forget_results.md](deployable_forget_results.md). On low-overlap the probabilities overshoot retrain while the logits stay on the original's side (d = +8.9 vs +7.6), and 72% of rows keep the original's ordering. The per-row pattern of the original survives.

## What this answers
1. **The unlearned models cannot be called indistinguishable from retrain.** An attacker who knows F and holds reference models of both kinds can tell them apart on every request. The test is calibrated, so this is not an artefact.
2. **On the canaries, the deployable method removes most but not all of what distinguishes the original.** The attacker's AUC falls from 1.00 to 0.67–0.71 on c3 and c0, and to 0.86–0.98 on c4. The residue is a systematic shortfall on the logit scale that the probability-based stop rule cannot see.
3. **On the natural requests, nothing tested comes close to retrain under this attacker.** This includes 2 recovery rounds alone.
4. **The privacy claim the experiments support is therefore limited.**
   - The group-level effect (the backdoor rate) reaches the retrain level.
   - Per-record membership is undetectable even in the original.
   - At model level, a strong attacker still detects that the model was unlearned rather than retrained.

## Not tested; possible next steps (not preregistered)
- **Probe control.** Run the same distinguisher on inputs outside F (retained or test rows). If it is equally strong there, part of the signal is the different training history (original plus 2 rounds vs 15 rounds from scratch), not information about F.
- **A stop rule on the logit scale,** for example matching the reference's mean margin per row. This targets the residue found on c3 and c0. It would be a new method, judged on new seeds.
- **A weaker, more realistic attacker:** fewer reference models, or no retrained references at all.

## Limitations
- The attacker is strong: 39 reference models of each kind, exact knowledge of F, and full-precision logits.
- It sees outputs on F only. Other probes (test data, gradients, parameters) are not tested.
- The diagonal Gaussian ignores correlations between vectors.
- The unlearned models reuse their original's seed, and the canary retrain is the clean original of the same seed.
- The canaries are synthetic, and both natural requests are on client 4.
- `DISTINGUISHABLE` says that a difference exists, not that it leaks a specific record.
