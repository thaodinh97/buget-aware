# Detailed utility, fair control, reset baseline and combination canaries — results (`runs/confirmation_existing_seed42`)

> † **Tài liệu lưu trữ** của giai đoạn FU xấp xỉ, đã đóng băng ([fu_freeze.md](../fu_freeze.md)). Code, test và verifier nhắc tới ở đây đã được gỡ khỏi nhánh để gọn và vẫn còn nguyên ở tag `pre-cleanup`: `git checkout pre-cleanup`. Mọi đường dẫn trong tài liệu tính từ gốc repo tại tag đó.

**Protocol.** [confirmation_plan.md](confirmation_plan.md), `--mode existing`.
- **Not a confirmation.** This mode reuses the seeds (10042–10051), partition and requests the method was developed on. It adds the measurements the review asked for to the existing results. Only `--mode confirm` (new partition and seeds) can confirm.
- **Scale.** 8 requests and 10 seeds, 6,573 s on CPU in one process.
- **Cross-check with `deployable_forget_seed42`.** All 400 arm rows and 428 control rows are reproduced: largest difference 1.1e-16, identical step counts.
- **Verifier.** `verify_confirmation.py` passed: 40 hashed artifacts, the gate, all 98 decision rows and the verdicts recomputed, 32 stages re-executed, and the cross-check recomputed. The verifier itself needed one fix after the run (see the end of the plan).

## Gate
- **7 of 8 requests admitted.**
- **`combo_canary_c0_500` is not identifiable** (effect 2.28; backdoor rate 0.074 in the original vs 0.024 in retrain). The plan reads this as "not determinable"; the trigger was not changed.
- **The combination trigger is learned on the other two clients:** backdoor rate 0.251 (c3) and 0.997 (c4), against 0.024 for retrain.

## Verdict
**No arm passes the frozen rule.** `ga_proj@joint_aux_prob` passes 5 of 7 requests: every admitted canary, including both combination canaries. It fails both natural requests. `nkd_proj@joint_aux_prob` also passes 5 of 7; `ga_adam@joint_aux_prob` passes 4 of 7.

## 1. Zero-dim canaries: the structural shortcut matches the method
Forgetting position after recovery (0 = original-cloud mean, 1 = retrain-cloud mean):

| Request | `ga_proj` | `nkd_proj` | `ga_adam` | `reset_init@local` | `reset_init@global` | `reset_zero@global` | Retain-only |
|---|---|---|---|---|---|---|---|
| c3 | 1.006 | 1.005 | 0.604 | 0.951 | 0.951 | 0.992 | 0.112 |
| c0 | 0.984 | 0.992 | 0.489 | **0.994** | **0.994** | **0.985** | 0.060 |
| c4 | 1.003 | 1.003 | 0.226 | 0.958 | 0.958 | 0.999 | 0.000 |

- **What the reset needs.** No forget step, no reference model and no kNN. Test F1, FPR and every client's F1 are unchanged, and every reset baseline passes all 3 zero-dim canaries.
- **Reading fixed in the plan.** "If a reset reaches ≥ `ga_proj`'s position, the zero-dim canary result is not used as evidence for the projection."
  - It fires on c0.
  - On c3 and c4, `reset_zero@global` stays within 0.015 and 0.004 of `ga_proj`.
  - The distinguishing test gives the same picture ([reset_baseline_results.md](reset_baseline_results.md)).
- **Conclusion: the zero-dim canaries do not discriminate between methods.**

## 2. Combination canaries: the projection works when the trigger is inside R's span
The trigger is a combination of `proto`/`service`/`conn_state` values absent from every split. Each of its columns is common in retained data, so no reset can touch it:

| Request | `ga_proj` | `nkd_proj` | `ga_adam` | Reset (all three) | Retain-only | Pool control | R_target control |
|---|---|---|---|---|---|---|---|
| combo c3 | **0.949** | 0.942 | 0.696 | 0.409–0.411 | 0.409 | 0.762 | 0.852 |
| combo c4 | **0.978** | 0.768 | 0.140 | 0.000 | 0.000 | 0.008 | 0.043 |

- **The reset does nothing** here; it equals retain-only.
- **`ga_proj` passes both requests.** It beats retain-only and the pool control by more than the noise floor (gain vs pool control 0.042 [lower bound 0.022] on c3, 0.927 [0.882] on c4; floor 0.020), and utility is noninferior everywhere.
- **The projection is what survives recovery.** Before recovery, projected and unprojected GA forget equally: positions 0.893 vs 0.890 on c3 and 0.986 vs 1.016 on c4. After recovery, the unprojected one falls back to 0.696 and 0.140, while the projected one keeps 0.949 and 0.978.
- **This is the first canary evidence for the projection that the zero-dim shortcut does not explain.**
- **Limits.** It rests on 2 requests and one trigger design, on the seeds and partition the method was developed on. The fine-tuning controls also remove part of the c3 backdoor (0.76–0.85), so the margin over the fair control is moderate there.

## 3. Natural requests with the fair control
| Request | `ga_proj` | Retain-only | Pool control | R_target control | Outcome |
|---|---|---|---|---|---|
| random | 0.695 | 0.717 | 0.477 | 0.327 | fail: not better than retain-only, and not `local_effective` |
| low-overlap | 0.736 | 0.271 | 0.560 | −0.774 | fail: superior to retain-only, but not to the pool control |

- **Low-overlap.** The pool control is not degenerate, unlike the old R_target control (−0.774). Fine-tuning on R_target ∪ validation with the same compute already reaches 0.560. `ga_proj`'s gain over it is 0.0044 with a lower bound of 0.0027, which does not exceed the noise floor (0.0027).
  - The one natural request that supported the method therefore does not survive the fair control.
- **`reset_init@local` on low-overlap fails on utility.** It resets 4 dims that client 4's retained rows never use but other clients' normal traffic does. Test FPR rises by 0.062 and client 0's macro-F1 falls by 0.059.

## 4. Utility beyond macro-F1
- **The base model is a weak detector.**
  - Original macro-F1 is 0.889–0.890, with an attack recall of 0.99.
  - **Test FPR is 0.272:** 27% of normal traffic is flagged as attack.
  - Client 1, whose retained data is 99% normal, has a retained macro-F1 of only 0.564.
  - "Utility preserved" below always means relative to this baseline.
- **Retrain itself loses utility on low-overlap.** Its macro-F1 is 0.863 and its FPR 0.332, because removing client 4's normal rows hurts normal detection. A model that matches retrain exactly would lose this utility too. This is the RTBF–NTBR tension of ConPFU, seen here in the gold standard itself.
- **`ga_proj@joint_aux_prob` is utility-noninferior on all 7 requests.**
  - Test FPR changes by at most +0.0003 (upper bound) against retain-only.
  - The worst per-client macro-F1 difference is −0.0012.

## What this answers
1. **Zero-dim canaries are a structural shortcut** and are dropped as evidence for the method.
2. **Combination canaries give real but narrow evidence** that retain-orthogonal projection keeps the forgetting through recovery without utility loss: 2 requests, on the development seeds.
3. **On natural data, no arm beats the fair control,** so there is no evidence for the forget step on natural requests.
4. **The method remains a candidate.** Its only remaining test is `--mode confirm`.
