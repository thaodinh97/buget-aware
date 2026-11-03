# Second confirmation on partition 44 — results (`runs/confirmation_p44`)

**Protocol.** [confirmation_plan.md](confirmation_plan.md), section "Confirm lần 2", `--mode confirm2`. The protocol was written before the run.
- **Changes from `confirm`:**
  - Dirichlet partition seed 44 (α = 0.5, 5 clients);
  - seeds 30042–30051;
  - no utility floor finer than one row.
- **Unchanged:** method, arms, controls, request design and the rest of the rule.
- **The plan fixed two things in advance.**
  - This is the last confirmation in the FU phase.
  - A failure for any reason leaves the method "candidate, not confirmed".
- **No real-data smoke was run.**
- **Partition.** Very skewed:

  | Client | Normal | Attack |
  |---|---|---|
  | 0 | 658 | 2 |
  | 1 | 51 | 25,665 |
  | 2 | 449 | 52,921 |
  | 3 | 2,777 | 13,161 |
  | 4 | 18,599 | 1 |

  - Canaries are placed on clients 2, 1 and 3.
  - 12 requests are valid. 4 natural requests are invalid because F or R would hold a single class.
- **Scale.** 10 seeds, 2,665 s on CPU.
- **Verifier.** `verify_confirmation.py` passed: 142 hashed artifacts, the gate, all 98 decision rows and the verdicts recomputed, and 32 stages re-executed bit-for-bit.

## Verdict
**`ga_proj@joint_aux_prob` is not confirmed.**
- **No arm passes.**

  | Arm | Requests passed |
  |---|---|
  | `ga_proj@joint_aux_prob` | 3 of 7 |
  | `nkd_proj@joint_aux_prob` | 4 of 7 |
  | `ga_proj@retrain_oracle` | 2 of 7 |
  | `ga_adam@joint_aux_prob` | 1 of 7 |
  | Each reset baseline | 3 of 7 |

- **The one-row floor changed no decision of this run.** Recomputing it with the old rule gives the same 98 pass values.
- **Per the plan, this verdict is final for the FU phase.**

| Request | `ga_proj` position after recovery | Outcome | Reason |
|---|---|---|---|
| canary c2 (zero-dim) | 1.004 | fail | utility: test-FPR bound (section 3) |
| canary c1 (zero-dim) | 0.997 | pass | |
| canary c3 (zero-dim) | 0.988 | pass | |
| combo canary c2 | 1.021 | pass | |
| combo canary c1 | 1.042 | fail | utility: wide intervals (section 3) |
| client 2, random 50% | 0.366 | fail | forgetting: no better than retain-only (0.397), not `local_effective` |
| client 2, low-overlap 50% | **−0.266** | fail | forgetting: moves **away** from retrain (section 4) |

## Gate
| Admitted | Effect | Not admitted | Effect |
|---|---|---|---|
| canary c2 / c1 / c3 | 242.9 / 202.7 / 4.5 | combo canary c3 | 3.65 |
| combo canary c2 / c1 | 25.7 / 24.9 | client 0 random | 0.04 |
| client 2 random | 8.30 | client 1 random | 2.63 |
| client 2 low-overlap | 9.84 | client 3 random / low-overlap | 1.10 / 4.06 |

- **The admitted requests cover target clients 1, 2 and 3.**
- **This is the first time natural requests are identifiable outside client 4 of partition 42.** Both are on client 2, which holds 53k rows; each request deletes 20–27k of them.

## 1. Zero-dim canaries: the shortcut replicates a third time
| Request | `ga_proj` | `ga_adam` | `reset_init` (both) | `reset_zero@global` | Retain-only |
|---|---|---|---|---|---|
| c2 | 1.004 | 0.316 | 0.951 | 0.992 | 0.000 |
| c1 | 0.997 | 0.439 | 0.808 | 0.991 | 0.014 |
| c3 | 0.988 | 0.560 | 0.945 | 0.968 | −0.012 |

- **`reset_zero@global` stays within 0.02 of `ga_proj` with no forget step,** and every reset passes all three requests.

## 2. Combination canaries: the projection result replicates
| Request | `ga_proj` | `nkd_proj` | `ga_adam` | Reset | Retain-only | Pool control | R_target control |
|---|---|---|---|---|---|---|---|
| combo c2 | **1.021** | 0.650 | 0.088 | 0.000 | 0.000 | 0.188 | 0.063 |
| combo c1 | **1.042** | 0.915 | 0.250 | 0.002 | 0.002 | 0.354 | 0.126 |

- **Projected GA beats retain-only and the pool control on both requests.** The gain over the pool control is 0.693 / 0.558, with lower bounds 0.569 / 0.442, against a floor of 0.122.
- **Recovery undoes the unprojected step.** Before recovery, `ga_adam` sits at 1.057 / 1.049; afterwards it falls to 0.088 / 0.250. `ga_proj` goes from 1.031 / 1.050 to 1.021 / 1.042.
- **Over three partitions, the forgetting side has now held on 7 of 7 admitted combination canaries.**
  - `ga_proj` beats retain-only and the pool control on all 7.
  - Unprojected GA beats the pool control on only 1 of them (combo c4 of partition 42, at 0.14 vs 0.01).
  - The one weak case for `ga_proj` remains combo c0 of partition 43, at 0.64.

## 3. The utility failures on the two canaries
Both failures come from the test-FPR check, plus macro-F1 checks on combo c1. In both, the arm is better than retain-only on average. The noninferiority bound fails because of favourable outliers.

Test FPR of `ga_proj` minus retain-only, per seed:

| Request | Per-seed difference | Mean [95% CI] | Floor |
|---|---|---|---|
| canary c2 | −0.0001, 0, −0.0003, 0, −0.0006, +0.0004, 0, −0.0006, **−0.0103**, 0 | −0.0011 [−0.0035, +0.0012] | 0.0008 |
| combo c1 | −0.0103, −0.0039, 0, −0.0106, −0.0104, **−0.1458**, −0.0004, −0.0106, +0.0004, −0.0103 | −0.0202 [−0.0520, +0.0116] | 0.0008 |

- **The detector's test FPR sits in two discrete states, 0.272 and 0.262.** About 1% of the normal test rows flip together between them.
- **In some seeds the arm lands in the lower state** and retain-only does not.
- **In seed 30047 of combo c1, retain-only's FPR jumps to 0.416 after recovery,** while `ga_proj` stays at 0.272.
- **In no seed is `ga_proj`'s test FPR worse than retain-only's by more than 0.0004** (about 3 normal rows).
- **Test macro-F1 on combo c1** is better by 0.007 on average. Its worst seed is −0.0034, against a floor of 0.0033.

**A second weakness of the rule.** The noninferiority check uses a symmetric t-interval of the paired difference. Large improvements widen that interval and push its unfavourable bound past a tight floor.
- This is not the zero-floor defect of partition 43; these floors are 5× the one-row resolution.
- **The plan allows no further correction or rescoring, so these two failures stand.** The same checks also fail the oracle arm and `nkd_proj` on combo c1.

## 4. Natural requests on client 2: the forget step does not help, and can hurt
Client 2 holds 449 normal and 52,921 attack rows.

**Random 50%** (26,684 rows in F; 77% have a feature-identical row in R):
- **`ga_proj` does not move.** It runs all 300 steps: position −0.006 before recovery, 0.366 after, while retain-only reaches 0.397.
- **The projection leaves no direction to move along,** because F lies in R's span. This is the random-request failure of partition 42 again.
- **The oracle stop gives the same result.**
- **Unprojected GA overshoots:** −4.55 before recovery, 0.447 after.

**Low-overlap 50%** (20,352 rows in F, including 446 of the client's 449 normal rows; R_target keeps 3 normal rows):
- **The deployable stop level is too strict.**
  - `joint_aux_prob` targets a true-label probability of 0.952.
  - `ga_proj` stops there after 22 steps. Its CE on F is then 0.084, twice the retrain cloud's 0.040.
- **The arm moves away from retrain.**
  - MAE(F) to the retrain cloud rises from 0.0088 (original) to 0.0365 before recovery, which is position −4.38.
  - After recovery the position is still −0.266: farther from retrain than the original model, and farther than retain-only (0.060) by more than the noise floor.
  - Test FPR drops to 0.196 before recovery and returns to 0.272 after.
- **The oracle stop does not help either.**
  - It reaches the retrain CE in 12 steps, but MAE(F) stays at 0.0091.
  - After recovery it reaches 0.236. Its gain over retain-only has a lower bound of 0.0010, against a floor of 0.0016.
  - **Matching F's mean loss does not move F's per-row predictions towards retrain.**
- **The same structure appeared on partition 42.** There, client 4's low-overlap request left 1 normal row in R and `knn_local` overshot. With almost no normal rows of the target client left in R, the stricter reference under-estimates the probability retrain keeps on F.

## Base model on partition 44
- **Test level:** macro-F1 0.888, **FPR 0.272**, attack recall 0.988, as on the other partitions.
- **Retained macro-F1 per client** is 0.51 / 0.72 / 0.89 / 0.97 / 0.48. Clients 0 and 4 hold almost a single class.
- **Retrain on the low-overlap request costs utility itself:** client 2's macro-F1 falls from 0.891 to 0.650, and test FPR rises to 0.292.

## What this answers
1. **The method is not confirmed, and per the plan there is no further confirmation run.** It stays "candidate, not confirmed".
2. **The failures are of two kinds.**
   - **On canaries, the utility checks fail** while the arm is better on average and never worse by more than 0.0004 test FPR in any seed. This is a weakness of the rule's interval, not of the method; it cannot be corrected under the plan.
   - **On natural requests, the method fails on its own terms.** The projection blocks all movement when F is duplicated in R. When F holds almost all of a class that R lacks, the deployable stop over-forgets and leaves the model farther from retrain than doing nothing.
3. **What held on every partition:**
   - On combination canaries, retain-orthogonal projection keeps the forgetting through recovery and beats the fair pool control (7 of 7). Unprojected GA does so on only 1 of 7.
   - The zero-dim canaries are a structural shortcut.
4. **Natural deletion stays unsolved.** 4 natural requests are identifiable over the 3 partitions: 2 on partition 42 and 2 here. On none of them does the forget step beat the fair control, and on one it is harmful.
