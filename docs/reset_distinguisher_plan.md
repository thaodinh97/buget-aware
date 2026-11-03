# Reset baselines in the distinguishing test — protocol (fixed before the run)

## Purpose
- **Why the reset baseline matters.** The review of `deployable_forget` pointed out that the zero-dim canary trigger lives on input dims that every retained row leaves at 0. Their first-layer columns never receive a gradient, so a retrain keeps them at their initial values.
- **What is already planned elsewhere.** [confirmation_plan.md](confirmation_plan.md) adds structural reset baselines and compares their forgetting position with `ga_proj@joint_aux_prob` on the 10 existing seeds (`run_confirmation.py --mode existing`).
- **What this run adds.** It puts the same reset baselines through the distinguishing test of [indistinguishability_plan.md](indistinguishability_plan.md), so their AUCs compare directly with `ga_proj`'s (0.67 / 0.71 / 0.98 on c3 / c0 / c4).

## Models
- **Baselines.** `reset_init@local`, `reset_init@global` and `reset_zero@global`, built exactly as `run_confirmation.py` builds them:
  - dims that are 0 on every row of R_target (`local`) or of every retained block (`global`), from `src/fu/reset_baseline.dead_dims`;
  - the first-layer columns of those dims set to the seed's initial values (`init`) or to 0 (`zero`);
  - then the standard 2 recovery rounds (seed s + 900000).
- **Seeds and originals.** The 40 seeds of `indistinguishability_seed42`, with the same originals: the stored clouds for 10042–10051 and that run's checkpoints for 10052–10081, hash-checked.
- **Requests.** All 5. Where a baseline's dim set is empty, the model equals retain-only, and the runner checks that its signals match the stored retain-only signals bit-for-bit.

## Test
- **Unchanged from `indistinguishability`.**
  - The observations are the logit margins on the distinct vectors of F.
  - Primary variant `per_unit_lr`, secondary `mean_margin`.
  - Leave-one-seed-out fitting, 1,000 within-seed swaps, 2,000 bootstrap resamples, α = 0.05.
- **Reference.** Every baseline is compared with the **stored** retrain signals.
- **Power.** It is taken from the stored original-vs-retrain row of the same request and variant.
- **Verdicts.** `INCONCLUSIVE` without power; otherwise `DISTINGUISHABLE` if p < 0.05 and `NOT_DISTINGUISHED` if not. There is no multiplicity correction.

## Reading the result (fixed in advance)
- **Shortcut.** On a zero-dim canary, if a reset baseline's AUC is at most `ga_proj`'s (its bootstrap CI reaches `ga_proj`'s AUC or lies below it), the distinguishing result of `ga_proj` there is no evidence for the projection beyond the structural shortcut.
- **Nothing is chosen from results.** All three baselines and both variants are reported.

## Limitations
- The same as [indistinguishability_plan.md](indistinguishability_plan.md).
- The reset baselines need the initial global model. In FL it is known to the server that created it, but it must be kept.
