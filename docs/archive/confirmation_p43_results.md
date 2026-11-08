# Confirmation on a fresh partition and fresh seeds — results (`runs/confirmation_p43`)

> † **Tài liệu lưu trữ** của giai đoạn FU xấp xỉ, đã đóng băng ([fu_freeze.md](../fu_freeze.md)). Code, test và verifier nhắc tới ở đây đã được gỡ khỏi nhánh để gọn và vẫn còn nguyên ở tag `pre-cleanup`: `git checkout pre-cleanup`. Mọi đường dẫn trong tài liệu tính từ gốc repo tại tag đó.

**Protocol.** [confirmation_plan.md](confirmation_plan.md), `--mode confirm`. This is the only run the plan allows to be called a confirmation.
- **What is new.**
  - The partition is new: Dirichlet seed 43, α = 0.5, 5 clients with 49,632 / 35,037 / 27,160 / 1,258 / 1,197 rows.
  - The seeds are new: 20042–20051.
  - There are 15 valid requests: 6 canaries on clients 0–2 and 9 natural requests on clients 0–4. `client0_low_overlap_groups_0.5` is invalid, because F or R would hold a single class.
  - Every model is trained in this run.
- **Nothing was changed between the smoke and this run.** Method, settings, decision rule and code are the ones frozen in [fu_freeze.md](../fu_freeze.md).
- **Scale.** 10 seeds, 4,189 s on CPU in one process.
- **Verifier.** `verify_confirmation.py` passed:
  - all 172 hashed artifacts;
  - the gate;
  - all 84 decision rows and the verdicts, recomputed from per-seed rows;
  - 32 stages re-executed bit-for-bit.

## Verdict
**`ga_proj@joint_aux_prob` is not confirmed under the preregistered rule.**
- **No arm passes.**
  - `ga_proj@joint_aux_prob` passes 4 of 6 admitted requests.
  - `nkd_proj@joint_aux_prob` also passes 4 of 6, and so does the oracle arm `ga_proj@retrain_oracle`.
  - `ga_adam@joint_aux_prob` passes 2 of 6, and the reset baselines pass 3 of 6.
- **Both `ga_proj` failures come from one degenerate utility check, not from forgetting** (section 4).
  - `ga_proj` meets `forget_ok` on all 6 requests.
  - It fails `combo_canary_c0_500` and `combo_canary_c1_500` only on client 4's retained macro-F1.
  - That metric's noise floor is exactly 0.
  - On combo c1, the arm is never worse than retain-only on any seed, and the check still fails.
- **Only canaries are admitted.** None of the 9 natural requests passes the gate (section 3), so this run says nothing about natural requests.

## Gate
| Request | Effect | Backdoor rate: original / retrain |
|---|---|---|
| canary c0 / c1 / c2 (zero-dim) | 86.2 / 72.9 / 19.9 | 0.999 / 0.969 / 0.709 vs 0.041 |
| combo canary c0 / c1 / c2 | 29.3 / 22.0 / 8.2 | 0.994 / 0.913 / 0.699 vs 0.073 |
| 9 natural requests | 0.09–3.67 | — |

- **The combination trigger is learned on every target client of this partition,** unlike combo c0 in `existing`.
- **All 6 canaries are admitted.** They cover target clients 0, 1 and 2, so the "≥ 2 target clients" condition of the rule is met.

## 1. Zero-dim canaries: the shortcut replicates
Forgetting position after recovery (0 = original-cloud mean, 1 = retrain-cloud mean; noise floor 0.037):

| Request | `ga_proj` | `nkd_proj` | `ga_adam` | `reset_init@local` / `@global` | `reset_zero@global` | Retain-only |
|---|---|---|---|---|---|---|
| c0 | 1.014 | 1.010 | 0.063 | 0.929 | 1.005 | 0.000 |
| c1 | 1.006 | 1.005 | 0.397 | 0.962 | 1.005 | 0.058 |
| c2 | 1.016 | 1.013 | 0.316 | 0.960 | 1.007 | 0.078 |

- **Resetting the dead input columns alone reaches the retrain level.** `reset_zero@global` is within 0.01 of `ga_proj` on all three requests, with no forget step. Every reset baseline passes all three zero-dim canaries.
- **The plan's literal reading** ("reset ≥ `ga_proj`") does not fire here. But both are at retrain within the noise floor, so the conclusion of `existing` stands: **zero-dim canaries do not discriminate between methods.**

## 2. Combination canaries: the projection survives recovery
| Request | `ga_proj` | `nkd_proj` | `ga_adam` | Reset (all three) | Retain-only | Pool control | R_target control |
|---|---|---|---|---|---|---|---|
| combo c0 | **0.639** | 0.732 | 0.059 | 0.000 | 0.000 | 0.099 | 0.043 |
| combo c1 | **1.032** | 0.957 | 0.290 | 0.011 | 0.011 | 0.387 | 0.111 |
| combo c2 | **0.977** | 0.935 | 0.392 | 0.065 | 0.065 | 0.469 | 0.161 |

- **The main forgetting result replicates on a new partition and new seeds.**
  - `ga_proj` beats retain-only and the pool control on all three requests by more than the noise floor (0.105). The gain over the pool control is 0.448 / 0.487 / 0.294, with lower bounds 0.182 / 0.328 / 0.121.
  - `ga_proj` is also `local_effective` on all three.
- **The projection is what survives recovery.** Before recovery, unprojected GA forgets about as much as projected GA, or more: positions 1.067 / 1.001 / 0.943 vs 0.647 / 1.027 / 0.955. After recovery the unprojected one falls back to 0.059 / 0.290 / 0.392, while the projected one keeps 0.639 / 1.032 / 0.977.
- **Without projection, the forget step adds nothing over the fair control.** `ga_adam`'s gain over the pool control is −0.030 / −0.070 / −0.043 on the three combination canaries.
- **The reset does nothing here,** as in `existing`: it equals retain-only.
- **`nkd_proj` misses combo c2 by 0.001.** Its gain over the pool control has a lower bound of 0.1045, against a floor of 0.1054.

**Combo c0 shows the projection's limit.** Client 0 holds 49,632 retained rows, 99% of them attack, and the trigger's columns are common in them. The per-seed positions of `ga_proj` are bimodal:

| Seeds | Stop | Position after recovery |
|---|---|---|
| 20043, 20045, 20049 | target level reached | 1.06–1.07 |
| 20044, 20048 | 300-step cap | 1.02 / 1.06 |
| 20047, 20050 | 300-step cap | 0.53 / 0.50 |
| 20042, 20046, 20051 | 300-step cap | 0.03–0.04 |

- **Where R_target spans almost every direction the trigger uses, the projected step can barely move.** In 3 of 10 seeds it does not move at all in 300 steps.
- **The oracle stop does not help** (0.594). This is a limit of the projection, not of the stop rule.

## 3. Natural requests: none is identifiable
- **No natural request passes the gate.** The original and retrain clouds' 95% prediction intervals of MAE(F) overlap on every request.
- **This holds even for large deletions.** Removing 24,816 rows from client 0 (22% of all training rows) gives an effect of only 3.67. In each random F, 71–77% of the rows have a feature-identical row in R.
- **This matches the seed floor result on the old partition:** at 50% deletion, natural requests in this setup do not change the model beyond seed noise.
- **Consequence.** `confirm` cannot test the forget step on natural data. The evidence on natural requests is still only that of `existing`, where no arm beat the fair control.

## 4. Utility, and the degenerate client-4 check
**Utility is preserved at test level for every projected arm.**
- `ga_proj`'s test macro-F1 is at or above retain-only on all 6 requests (mean differences +0.0002 to +0.0030).
- Its test FPR upper bound is at most +0.0030, against a floor of 0.0117.
- The worst per-client macro-F1 difference is −0.00016.

**Why the per-client check fails.** All 20 original and retrain models of every canary request give client 4 exactly the same retained macro-F1 (0.961545): 28 of 269 normal rows are flagged and 3 of 928 attacks are missed. The retrain-cloud noise floor of this metric is therefore 0, so noninferiority needs the lower 95% bound of the paired difference to be ≥ 0. A single row flipped in a single seed pulls that bound below 0, even when the flip is an improvement.

| Request | Arm | Per-seed client-4 difference vs retain-only | Mean [95% CI] | Floor |
|---|---|---|---|---|
| combo c0 | `ga_proj` | one seed −1 row, two seeds +1 row, seven ties | +0.00012 [−0.00036, +0.00060] | 0 |
| combo c1 | `ga_proj` | one seed +1 row, nine ties | +0.00012 [−0.00015, +0.00039] | 0 |

- **One attack row flipped on client 4 changes its macro-F1 by 0.0012.**
- **The same check also fails other arms:** the oracle arm on combo c0 and c1, and `nkd_proj` on combo c0.
- **The rule anticipated only exact ties.** The docstring of `noninferior` says "Ties (including a zero floor with identical values) pass". A zero floor with non-identical values was not foreseen.

**Post-hoc sensitivity (not a confirmation).** This analysis was decided after seeing the result, so it cannot replace the verdict. If a zero floor is replaced by the granularity of one row (0.0012), with nothing else changed:
- `ga_proj@joint_aux_prob` passes **6 of 6** requests on 3 target clients, and so does the oracle arm.
- `nkd_proj` passes 5 of 6; it still fails combo c2 on the pool control.
- `ga_adam` and the resets are unchanged (2 and 3 of 6).

## Base model on the new partition
- The model is as weak as on partition 42: test macro-F1 0.887–0.888, **test FPR 0.266–0.268**, attack recall 0.986.
- Retained macro-F1 per client is 0.90 / 0.97 / 0.96 / 0.95 / 0.96. Client 0, 99% attack, is the lowest.

## What this answers
1. **The method is not confirmed under the preregistered rule.** It may not be called "confirmed".
2. **The rule's failure is a defect of the rule, not a utility loss.** The only failing check has a zero noise floor, and on one of the two failing requests the arm is never worse on any seed.
3. **The combination-canary forgetting result replicates on a fresh partition and fresh seeds.**
   - On 3 of 3 requests, projected GA beats retain-only and the fair pool control.
   - Unprojected GA does not beat the fair pool control, because recovery undoes it.
4. **The projection has a limit.** When the target client's retained data span the trigger's directions (client 0), the projected step stalls in some seeds. Position is 0.64 on average, and 0.03 in 3 of 10 seeds.
5. **The zero-dim shortcut replicates,** so those canaries stay excluded as evidence.
6. **The run tells us nothing new about natural requests:** none is identifiable.
