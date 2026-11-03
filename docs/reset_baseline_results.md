# Structural reset baseline — results

**Why this baseline.**
- **The design flaw.** The zero-dim canary trigger sits on input dims 8, 9 and 14, which are 0 on every real row. With Adam and no weight decay, their first-layer columns never receive a gradient, so a retrain keeps them at their initial values (measured: within 1.5e-8, from FedAvg weight rounding). Recovery cannot change them either.
- **What projection does there.** On those canaries, projected GA spends 49–84% of its first-layer update on exactly those columns (seed 10042, diagnostic).
- **The baseline.** It skips the forget step entirely. It resets the first-layer columns of dims that retained data never uses (`src/fu/reset_baseline.py`), then runs the standard 2 recovery rounds.

**Variants.**
- `reset_init@local`: dims dead on R_target, set to the seed's initial values.
- `reset_init@global`: dims dead on all retained data, set to the initial values.
- `reset_zero@global`: the same dims, set to 0.

**Dead dims.**
- Global: 8, 9, 14, 23, 24 for every request.
- Local: adds dim 21 on c3 and c0, and dims 10, 12, 16 and 17 on low-overlap, where R_target keeps a single normal row.

## 1. Forgetting position and utility (10 seeds)
Source: `runs/confirmation_existing_seed42` ([confirmation_plan.md](confirmation_plan.md), `--mode existing`). Full results are in [confirmation_existing_results.md](confirmation_existing_results.md).

Position after recovery (0 = original, 1 = retrain):

| Request | `ga_proj@joint_aux_prob` | `reset_init@local` | `reset_init@global` | `reset_zero@global` | Retain-only |
|---|---|---|---|---|---|
| c3 | 1.006 | 0.951 | 0.951 | 0.992 | 0.112 |
| c0 | 0.984 | 0.994 | 0.994 | 0.985 | 0.060 |
| c4 | 1.003 | 0.958 | 0.958 | 0.999 | 0.000 |
| combo c3 (trigger on live dims) | 0.949 | 0.411 | 0.409 | 0.409 | 0.409 |
| combo c4 | 0.978 | 0.000 | 0.000 | 0.000 | 0.000 |
| random | 0.695 | 0.717 | 0.717 | 0.717 | 0.717 |
| low-overlap | 0.736 | 0.191 (utility fails) | 0.271 | 0.271 | 0.271 |

- **Zero-dim canaries.** Every reset baseline passes all three under the frozen rule, with test F1, FPR and per-client F1 unchanged. On c0 it reaches `ga_proj`'s position, so the plan's rule fires.
- **Combination canaries.** The reset is retain-only, and `ga_proj` is far ahead.
- **Low-overlap, `reset_init@local`.** It resets dims that other clients use, which raises test FPR by 0.062 and lowers client 0's macro-F1 by 0.059.

## 2. Distinguishing from retrain (40 seeds)
- **Source.** `runs/reset_distinguisher_seed42` ([reset_distinguisher_plan.md](reset_distinguisher_plan.md)). It used the same 40 seeds, originals, stored retrain signals and test as [indistinguishability_results.md](indistinguishability_results.md).
- **Runtime.** 678 s with 6 workers.
- **`verify_reset_distinguisher.py` passed:**
  - 11 hashed artifacts and the source hashes;
  - all 30 analysis rows and every held-out score, recomputed;
  - two jobs re-executed bit-for-bit.

Attacker AUC [95% bootstrap CI]. The primary variant is the per-vector likelihood ratio; every p-value is ≤ 0.006, so every cell is `DISTINGUISHABLE`.

| Request | `ga_proj@joint_aux_prob` | `reset_init@local` | `reset_init@global` | `reset_zero@global` | Retain-only |
|---|---|---|---|---|---|
| c3 | 0.674 [0.567, 0.777] | 0.799 [0.742, 0.870] | 0.798 [0.740, 0.871] | 0.802 [0.710, 0.883] | 1.000 |
| c0 | 0.708 [0.594, 0.809] | **0.648** [0.586, 0.726] | **0.648** [0.588, 0.725] | 0.775 [0.682, 0.864] | 1.000 |
| c4 | 0.975 [0.941, 0.998] | 0.993 [0.981, 1.000] | 0.993 [0.981, 1.000] | 0.994 [0.982, 1.000] | 1.000 |
| random | 0.999 | 1.000 | 1.000 | 1.000 | 1.000 |
| low-overlap | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

The mean-margin attacker gives the same picture. On c4, `reset_zero@global` is the only model in either run that this weaker attacker does not distinguish from retrain (AUC 0.581, p = 0.068).

**Reading, by the rule fixed in the plan.**
- **c0.** A reset baseline's AUC is below `ga_proj`'s (0.648 vs 0.708). The distinguishing result of `ga_proj` on c0 is therefore no evidence for the projection beyond the structural shortcut.
- **c3 and c4.** Every reset CI lies above `ga_proj`'s AUC, so the rule does not fire. `ga_proj` is somewhat less distinguishable than the shortcut there (0.674 vs 0.80 on c3), but the CIs overlap.
- **Natural requests.** The globally dead dims are already dead in the original, so resetting them changes nothing. 5 of the 6 natural-request baselines equal retain-only **bit-for-bit** (a check run after the analysis, not preregistered). The exception is `reset_init@local` on low-overlap, which also resets dims 10, 12, 16 and 17. Those dims are dead on R_target but used by other clients' normal traffic, so the model changes a lot (largest margin change 31). It stays fully distinguishable.
- **The empty-dims check of the plan never applied,** because no request has an empty dim set.

## Conclusion
1. **On zero-dim canaries, a reset that does no forgetting work matches the candidate method.** This holds for forgetting position, utility and distinguishability alike. Those canaries measure the structural shortcut, not the method, and are dropped as evidence for it.
2. **The claim that projection keeps forgetting through recovery now rests on the combination canaries** ([confirmation_existing_results.md](confirmation_existing_results.md), section 2). The reset cannot help there.
3. **A useful byproduct.** The global reset is a near-free, deployable defence for exactly this class of triggers: inputs that no retained client ever produces. It needs only one bit per dim from each client and the stored initial model.
