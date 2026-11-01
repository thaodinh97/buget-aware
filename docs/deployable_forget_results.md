# Deployable stop and recovery-proof forgetting — results (`runs/deployable_forget_seed42`)

The protocol, including the changes made after the two smoke runs, is in [deployable_forget_plan.md](deployable_forget_plan.md).

The run used 10 seeds, 40 arms, compute-matched controls, and the same 5 requests and clouds as `forget_signal_seed42` (hash-checked). It took 5,974 s on CPU. All 5 requests pass the gate; they cover target clients 0, 3 and 4.

`verify_deployable_forget.py` passed:
- all hashed artifacts, and the config equals the recorded protocol;
- the source run's hashes;
- the gate, and all 400 decision rows and verdicts, recomputed from per-seed rows;
- the stop levels, recomputed for one seed;
- 24 stages re-executed bit-for-bit: `ga_proj@knn_aux_prob`, `ga_proj@joint_aux_prob` and `nkd_proj@joint_aux_prob`, each with standard and interleaved recovery, pre/post, FU and control.

The single-pass controls match separately trained ones, which the verifier also checks.

## Verdict
**Under the preregistered rule, no arm passes all 5 requests.** The recommended deployable combination is projected GA, or projected neighbour-KD, stopped at `joint_aux_prob`:

The table shows `ga_proj@joint_aux_prob`; `nkd_proj` is within 0.03 of it everywhere.

| Request | Pre / post position | 0-step retain-only | Mean F1 − retain-only | Outcome |
|---|---|---|---|---|
| canary c3 | 0.99 / **1.01** | 0.11 | 0.0000 | superior to both controls; F1 vs the matched control has a lower bound of −0.005, below the −0.0037 floor |
| canary c0 | 0.95 / **0.98** | 0.06 | +0.0001 | **passes** |
| canary c4 | 1.00 / **1.00** | 0.00 | −0.0003 | superior; the same F1 miss as c3 (−0.005) |
| low-overlap | 0.45 / **0.74** | 0.27 | −0.0001 | **passes**, superior to both controls |
| random | −0.01 / 0.70 | 0.72 | +0.0020 | not superior: FU ≈ retain-only |

The two failure modes:
1. **Random request.** Here F is duplicated in R, so nothing is F-specific: projection leaves no direction to move along, and recovery alone does what retrain does (0.72). No arm beats retain-only on this request, the oracle arms included.
2. **c3 and c4 utility check.** These arms are charged with the reference's 215k–510k training rows, so their matched control runs 1,700–4,200 retain-only steps. That fine-tuning raises the control's test F1 by about 0.004, just above the F1 noise floor. The arm's own F1 equals 0-step retain-only, original and retrain (0.8895 vs 0.8895).

## 1. Deployable stop level
Mean true-label probability on F (mean over 10 seeds):

| Request | Original | knn_aux | ref_aux | **joint_aux** | Retrain (oracle) |
|---|---|---|---|---|---|
| canary c3 | 0.148 | 0.0045 | 0.0234 | **0.0044** | 0.0034 |
| canary c0 | 0.063 | 0.0049 | 0.0372 | **0.0049** | 0.0033 |
| canary c4 | 0.9985 | 0.0056 | 0.0113 | **0.0052** | 0.0047 |
| random | 0.9926 | 0.9926 | 0.9818 | **0.9814** | 0.9866 |
| low-overlap | 0.9818 | 0.9814 | 0.9557 | **0.9541** | 0.9576 |

The same references scored directly, as MAE to the retrain cloud on F:

| Request | Original | knn_aux | ref_aux | joint_aux |
|---|---|---|---|---|
| canary c3 | 0.145 | 0.003 | 0.020 | **0.003** |
| canary c0 | 0.060 | 0.003 | 0.034 | **0.003** |
| canary c4 | 0.994 | 0.003 | 0.007 | **0.003** |
| random | 0.0066 | 0.0066 | 0.0057 | **0.0057** |
| low-overlap | 0.0308 | 0.0304 | 0.0190 | **0.0186** |

- **kNN captures F-specific features.** Canary neighbours lack the trigger, so the kNN reference reproduces retrain on the canaries almost exactly.
- **kNN is blind to regional influence.** On low-overlap, F's neighbours inherit its influence, and the kNN reference stays at the original.
- **The trained reference covers the regional case but is less confident in the tail.** It is about right on low-overlap and too lenient on the canaries.
- **Both errors are under-forgetting, so the row-wise stricter `joint_aux` is the better of the two everywhere.** It is within 0.001–0.005 of the oracle level on every request.
  - Its one overshoot is on the random request (0.981 vs 0.987).
  - There, projection blocks the move, so nothing happens.
- **Cost.** The reference trains on 215k–510k rows, 13–30% of one retrain's 1.71M rows, in 1–4 s locally with no communication. Neighbour search takes 0.2–22 s once per request.
- **The CE statistic fails where the probability statistic works.** The `ref_aux` CE level fell below the original's CE on the canaries, so those arms took 0 steps.

## 2. Keeping the forget signal through recovery
| Arm (stop `joint_aux_prob`) | c3 pre→post | c0 pre→post | c4 pre→post | Pre-recovery F1 vs retrain (c3 / c0 / c4) |
|---|---|---|---|---|
| `ga_adam` (unprojected) | 0.99 → 0.60 | 0.97 → 0.49 | 0.98 → 0.23 | −0.23 / −0.15 / −0.46 |
| `ga_adam` + interleaved | 0.99 → 0.82 | 0.97 → 0.73 | 0.98 → 0.47 | the same |
| **`ga_proj`** | 0.99 → **1.01** | 0.95 → **0.98** | 1.00 → **1.00** | **−0.002 / −0.001 / −0.000** |
| `ga_proj` + interleaved | 0.99 → 1.01 | 0.95 → 0.99 | 1.00 → 1.01 | the same |
| `nkd_proj` | 0.99 → 1.01 | 0.97 → 0.99 | 1.01 → 1.00 | −0.002 / −0.001 / 0.000 |

- **Projection fixes both problems at once.** Recovery no longer undoes the forgetting, and the pre-recovery utility drop disappears. The trigger dims and units that R never activates take the whole update, and recovery's gradients are orthogonal to them.
- **On low-overlap the two add up.** Projected forgetting is kept, and recovery adds its own movement towards retrain (0.45 → 0.74).
- **Interleaving only helps the unprojected operator.** It recovers about half of what recovery took back. With projection it is redundant.
- **Without projection, the deployable stop still needs recovery to repair utility,** which costs 40–77% of the forgetting.

## Other observations
- **`knn_local` (client-local only) overshoots on low-overlap.** There R_target holds a single normal row, so a pool with both classes is needed.
- **The heavy compute-matched control is degenerate on low-overlap** (position −0.77, F1 0.43): R_target is almost all attack. The 0-step retain-only is the meaningful comparator there, and the recommended arms beat it.
- **Runtime.** Projected GA takes 33–155 steps per request, 2–5 s including the reference, plus 1–2 s of recovery. One retrain takes about 13 s on this CPU and needs 15 communication rounds with every client; FU needs 2.

## What this answers
1. **A stop rule usable in deployment:** `joint_aux_prob`.
   - Forget until the true-label probability on F falls to the stricter of two retrain-free references: kNN on R_target ∪ auxiliary data, and a fresh model trained on the same pool.
   - It tracks the retrain level within 0.001–0.005 on all 5 requests.
2. **A forget signal that survives recovery:** retain-orthogonal projection of the forget update.
   - It keeps the canary forgetting at position ≈ 1.0 after recovery, with F1 equal to retain-only.
3. **Formally, no arm meets the all-request rule.** One request needs no forget step at all. The c3/c4 miss is an F1 margin of about 0.001 against a control that spent the reference's compute on fine-tuning.
   - A rule of "superior where retain-only is not retrain-like, noninferior elsewhere" would admit the recommended arms.
   - That would be a change to the preregistered rule, so it is left to the authors.

## Limitations
- The canaries are synthetic and share one trigger.
- Both natural requests are on client 4.
- Validation stands in for auxiliary data.
- The projection guarantee is first-order and assumes recovery data lie in the target client's retained span.
- The `ref_aux` and `joint_aux` stops were added after the smoke runs and judged on 1–3 seeds that are included in the 10; this is disclosed in the plan.
- There is no MIA, and these results are not proof of deletion.
