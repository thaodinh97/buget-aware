# Deployable stop and recovery-proof forgetting — protocol

## Purpose
[forget_signal_results.md](forget_signal_results.md) left two gaps:
1. **No deployable stop.**
   - Only the oracle stop worked: the retrain-cloud CE(F), which needs retraining.
   - The nonmember level took 0 steps on the backdoor canaries, because the backdoor generalises to held-out triggered rows.
   - The nonmember level overshot on the random request, because F has feature duplicates in R.
2. **Recovery undid the forgetting.** On the canaries, the forgetting position fell from 0.96–0.98 to 0.48–0.59 after the 2 recovery rounds.

This run (`scripts/run_deployable_forget.py`) tests one stop rule and two recovery-side remedies. It reuses the same 5 requests, seeds, clouds, compute-matched controls and decision rule as `forget_signal`, which are hash-checked.

## 1. Neighbour-reference stop (`src/fu/neighbor_reference.py`)
For every f ∈ F, take its k = 10 nearest rows (Euclidean, encoded features) in a pool that excludes F. The reference for f is the original model's mean logit margin over those neighbours. The stop level is the mean CE of the reference on F.

| Level | Pool | Deployable |
|---|---|---|
| `knn_local` | R_target (the target client's retained rows) | yes, client-local |
| `knn_aux` | R_target ∪ validation, where validation stands in for an approved auxiliary labelled set | yes, with an auxiliary set |
| `ref_aux` | a fresh model trained locally for 5 epochs (Adam, lr 1e-3, batch 512) on R_target ∪ validation; its prediction is the reference | yes; its training rows are charged to the arm, so the matched control receives them too |
| `retrain_oracle` | — (retrain-cloud mean CE(F), same seed left out) | no (upper bound) |

**Statistic.** Each deployable level has two statistics:
- CE: stop once CE(F) ≥ the reference's CE on F;
- `*_prob`: stop once the mean true-label probability on F ≤ the reference's.

The probability statistic is bounded, so it depends less on how confident the reference is in its tail.

**Rationale.** Retrain sees no row of F, so its prediction on f follows the closest data it still has.
- **Canaries:** the neighbours of a triggered row are attack rows without the trigger, so the reference predicts attack, as retrain does.
- **F duplicated in R:** the neighbours include the duplicates, so the fit is kept, as in retrain.
- **Class mix:** `knn_aux` adds normal rows for targets whose R has almost none (client 4 retains 1 normal row in the low-overlap request).
- **Diagnostic:** each pool's reference is also scored directly as MAE to the retrain cloud on F (`*_reference_mae_F`).

## 2. Retain-orthogonal projection (`src/fu/projection.py`)
**Basis.** At the start of every forget stage, an SVD basis of each Linear layer's augmented inputs (input plus a constant 1 for the bias) is taken on up to 4,000 R_target rows. Directions are kept up to a cumulative energy of 0.999999.

**Projection.** Forget gradients, and Adam's parameter update (Adam rescales each coordinate separately), are both projected off that basis.

**Effect, to first order:**
- the forget step leaves outputs on R_target-like inputs unchanged, so no utility drop is expected;
- recovery updates, whose weight gradients lie in the span of the retained inputs, are orthogonal to it.

The canary trigger dims are zero in every split, so they lie entirely in the free subspace.

**Threshold disclosed.** The unit test `test_deployable_forget.py` showed that GPM's usual 0.99 threshold leaves low-energy directions that GA exploits: retained logits moved by up to 1.44. Directions holding 1e-4 of the energy still moved them by 0.15. At 0.999999 the change was 0. The threshold was fixed at 0.999999 from that unit test, before any benchmark run.

## 3. Interleaved forget–recovery (`src/fu/recovery_schedule.py`)
The schedule is forget → 1 FedAvg round → forget → 1 FedAvg round.
- **Same randomness.** It has the same total rounds and per-round seeds as standard recovery; with a no-op stage it reproduces standard recovery exactly (unit test).
- **Use of F.** F is used only in local forget stages, never inside a FedAvg update, and is deleted before the final round.
- **Stop level.** The second stage restarts from the recovered model with the same stop level.

## Arms (each with standard and interleaved recovery: 40 arms)
| Operator | Description | Stops |
|---|---|---|
| `ga_adam` | Adam on −CE(F), unprojected | knn_local, knn_aux, knn_aux_prob, ref_aux, ref_aux_prob, joint_aux_prob, retrain_oracle |
| `ga_proj` | the same, projected | the same 7 |
| `nkd_proj` | projected KL(reference ‖ student) on F: bounded, and directional towards the reference of the same pool | the 6 deployable ones |

**Settings.** Adam lr 1e-3, batch 128, clip 5, at most 300 steps per stage. The divergence guard is 20 nats above the stage's starting CE.
- The step cap was raised from 100 and the guard from 10, both before any run of this benchmark.
- The reason: c4's retrain level is 11.4 nats above its starting CE, and projected steps move fewer parameters.

## Control and decision (unchanged)
- **Control.** For every arm and seed, Adam on CE(R_target) for `ceil(backward_samples/128)` steps, followed by standard recovery.
  - For interleaved arms, `backward_samples` counts both forget stages.
  - Neighbour search and reference forward passes are forward-only overhead; they are reported but not matched.
- **Decision.** `run_forget_signal.decide`, unchanged.
  - An arm must be superior to the control by more than the retrain-cloud noise floor, F1-noninferior, and local-effective on every admitted request.
  - The admitted requests must span at least 2 target clients.

## Changes after the first smoke run (disclosed)
The first 3-seed smoke run (`runs/deployable_forget_smoke3`, 16 arms) showed two things. All of its arms are kept unchanged.

**1. The kNN reference fits the canaries but fails on the natural low-overlap request.**
- On the canaries, MAE to the retrain cloud on F was 0.002 (the original's: 0.07–0.99).
- On the low-overlap request it was no closer to retrain than the original: 0.0306 vs 0.0310. Its CE level (0.037) sat near the original's (0.033), not the retrain level (0.067).
- The likely cause is that F's neighbours lie inside F's own region and inherit its influence, as triggered rows inherit the backdoor.
- This motivated `ref_aux`.

**2. On one seed (10042), the `ref_aux` CE level matched retrain on low-overlap but not on c3.**
- On low-overlap: 0.066 vs 0.065.
- On c3: it fell *below* the original's starting CE (5.66 vs 6.11), although its probabilities were close to retrain (MAE 0.019 vs the original's 0.26). The less-trained reference is less confident in the tail, and mean CE is dominated by that tail.
- This motivated the `*_prob` statistic.

Both additions were fixed before the 10-seed run and were judged from 1–3 seeds that are part of the 10. This weak double use is disclosed.

**3. The second smoke run (`runs/deployable_forget_smoke3b`, 34 arms) gave three more findings.**
- **The two references fail in complementary ways, both by under-forgetting.** On the canaries, `knn_aux_prob` ≈ retrain (0.004 vs 0.003) while `ref_aux_prob` is too lenient (0.027). On low-overlap, `ref_aux_prob` ≈ retrain (0.955 vs 0.958) while `knn_aux_prob` ≈ the original (0.983).
  - This motivated `joint_aux_prob`: per row, the reference that gives the true label the lower probability.
  - It is charged with the `ref_aux` compute.
- **The compute-matched control is degenerate for the reference-charged arms.** The reference's training rows become 1,700–4,200 retain-only Adam steps on R_target alone. On low-overlap, where R_target holds 1 normal row, this drives the control away from retrain (position −0.83).
  - The results therefore also report a stricter rule: superior to **both** the matched control **and** the 0-step retain-only (`superior_to_retain_only`).
  - This can only make passing harder.
- **These heavy controls dominated runtime (1,667 s).** All budgets of a seed now share one pass, which gives identical models because each budget is a prefix of the same batch stream (unit test). Budgets above 1,000 steps are rounded up to a multiple of 64 (at most +6%), so the control never gets less compute than the arm.

The grid is now 40 arms (`joint_aux_prob` added to all three operators).

## Limitations
- Validation stands in for auxiliary data.
- kNN uses raw encoded features with no learned metric.
- The projection guarantee is first-order. It holds only for inputs inside the target client's retained span, and other clients' data may lie outside it.
- There is no MIA here, and these results are not proof of deletion.
