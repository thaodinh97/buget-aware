# Distinguishing unlearned from retrained models — protocol (fixed before the 40-seed run)

## Purpose
Unlearning is successful when the unlearned model cannot be told apart from a model retrained without F; whatever retrain still reveals is acceptable. [canary_lira_results.md](canary_lira_results.md) showed that per-record MIA has no power on this model, so it cannot answer the privacy question.

This test asks it at model level, in the style of U-LiRA (Hayes et al., *Inexact Unlearning Needs More Careful Evaluations to Avoid a False Sense of Privacy*, 2024): **can an attacker who knows F tell an unlearned model from a retrained one?** It needs no per-record membership signal.

## Attacker
- **Knowledge.** The attacker knows F and the training and unlearning procedures, and holds reference models of both kinds (the other seeds).
- **Observation.** The attacker sees one model's outputs on F: its logit margin z_attack − z_normal on each distinct feature vector of F. Identical inputs give identical outputs. The margin needs no label; the Gaussian test below is unchanged by flipping the sign of an observation.
- **Distinct vectors of F:** c3 276, c0 287, c4 305, random 10,238, low-overlap 15,904.

## Requests, seeds and models
- **Requests.** The 5 requests of `deployable_forget`.
- **Seeds.** 40 seeds: 10042–10081.
  - Seeds 10042–10051 are rebuilt from the stored clouds. Every rebuilt retain-only and arm model must match the stored deployable-forget forget CE within 1e-9; the original and retrain must match the stored clouds.
  - Seeds 10052–10081 are new. They are trained by the same procedures, checked at the start by re-training seed 10042's stored clean original, c3 and c0 canary originals, and random-request retrain bit-for-bit.
- **Models per request and seed.**
  - Original and retrain. For canaries: the original on D ∪ C and the clean original on D. For natural requests: the clean original on D and the retrain on D \ F.
  - Retain-only.
  - `ga_adam@joint_aux_prob`, `ga_proj@joint_aux_prob` and `nkd_proj@joint_aux_prob`, with standard recovery, built exactly as in `deployable_forget`.

## Distinguisher
- **Primary (`per_unit_lr`).** A Gaussian likelihood ratio with per-vector means and one pooled variance per group, summed over the vectors of F (U-LiRA-style: online, global variance).
  - It is fitted on all seeds but one, and scores the held-out seed's comparator and retrain models.
  - Under the null, a model of either kind is equally likely to score higher.
- **Secondary (`mean_margin`).** The same test on one number per model: the margin averaged over the rows of F. This is the attacker who looks only at the average confidence on F.

## Comparisons, metrics and verdicts
- **Comparisons.** Each comparator (original, retain-only and the three arms) is tested against retrain.
- **Negative control.** Retrains of seeds 10062–10081 against retrains of seeds 10042–10061, paired by index.
- **Metrics.**
  - The AUC of held-out scores over all comparator–retrain pairs.
  - Paired accuracy: the fraction of seeds where the comparator scores above the retrain.
  - A one-sided p-value from 1,000 random within-seed label swaps. Under the null, the two models of a seed are exchangeable; the whole leave-one-out fit is redone for every swap.
  - A 95% bootstrap CI of the AUC over seeds (2,000 resamples). Its upper bound is reported as the largest attacker AUC compatible with the data.
- **Power** on a request requires the original vs retrain p < 0.05.
- **Verdicts.**
  - Without power: `INCONCLUSIVE`.
  - Otherwise: `DISTINGUISHABLE` if p < 0.05, and `NOT_DISTINGUISHED` if not.
- **No multiplicity correction.** Correcting would only make `NOT_DISTINGUISHED` easier to reach, so leaving it out is conservative for the claim this test is meant to support.
- No comparator, variant or seed is chosen from the results.

## Smoke run
A smoke run with 12 seeds and 50 swaps checks only the integration and the runtime. No setting is changed because of its results.

**Done (`runs/indistinguishability_smoke`).**
- It used 12 seeds (2 new) and took 669 s with 8 worker processes.
- All 4 procedure checks passed, including the c3 canary original from `identifiable_fu`.
- Every rebuilt existing-seed model matched the deployable run: the largest |Δ forget CE| was 1.8e-15.
- `verify_indistinguishability.py` passed, including one re-trained new model and two re-executed request jobs.
- One finding is about runtime only: on this laptop, training one model takes 10–65 s depending on thermal throttling and core type. The results do not depend on it.
- No setting was changed.

## Limitations
- The attacker sees outputs on F only. Other probes (test data, gradients, parameters) are not tested.
- `NOT_DISTINGUISHED` with 40 seeds bounds this attacker's AUC by the reported upper CI bound. It does not prove indistinguishability, and it is not a formal (ε, δ) guarantee.
- The diagonal Gaussian ignores correlations between vectors.
- The unlearned models reuse their original's seed, as in all earlier runs.
- The canaries are synthetic, and both natural requests are on client 4.
