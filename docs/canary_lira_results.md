# Per-example LiRA on the canaries — results (`runs/canary_lira_seed42`)

Protocol: [canary_lira_plan.md](canary_lira_plan.md). The run used 3 × 64 shadows and 10 seeds, and took 501 s on CPU with 8 worker processes.

`verify_canary_lira.py` passed:
- all 205 hashed artifacts, and the config equals the recorded protocol;
- pools, units and masks, recomputed;
- all 192 shadow checkpoints, which reproduce their stored signals;
- the control, all 720 target rows, the summary and the verdicts, recomputed;
- one complementary shadow pair, re-trained bit-for-bit;
- the target models of one seed, rebuilt (bit-checked against the deployable run) and re-scored.

| Request | Units | Member-only | Nonmember-only | Shared (canary / nonmember rows) | Pool rows per shadow |
|---|---|---|---|---|---|
| canary c3 | 510 | 218 | 234 | 58 (257 / 244) | 475–525 |
| canary c0 | 520 | 223 | 233 | 64 (265 / 247) | 475–525 |
| canary c4 | 539 | 245 | 234 | 60 (243 / 248) | 475–525 |

## Verdict
**Under the preregistered rule, the attack has no power on any request, so every verdict is `MIA_INCONCLUSIVE`.** A post-hoc placebo test (section 3) also shows that the membership signal reported by [canary_mia.md](canary_mia.md) is within what a fixed cohort produces without membership. That finding is withdrawn.

## 1. Positive control: no membership signal even where membership is known
Leave-one-shadow-out over 64 shadows, on the target-cohort units:

| Variant | c3 AUC [95% CI] | c0 AUC [95% CI] | c4 AUC [95% CI] |
|---|---|---|---|
| **online, global variance (primary)** | 0.506 [0.493, 0.519] | 0.486 [0.474, 0.499] | 0.505 [0.495, 0.516] |
| online, per-example variance | 0.502 [0.489, 0.515] | 0.496 [0.483, 0.508] | 0.509 [0.492, 0.527] |
| offline, global variance | 0.506 [0.498, 0.513] | 0.507 [0.498, 0.517] | 0.518 [0.507, 0.529] |
| loss (raw φ, no shadows) | 0.500 [0.492, 0.508] | 0.500 [0.494, 0.505] | 0.505 [0.500, 0.510] |

- **No usable power anywhere.** Only on c4 do two secondary variants clear 0.5, by at most 0.02.
- **Low FPRs are near chance too.** The pooled TPR at 0.1% FPR is 0.07–0.27%, at most 2.7 times chance.
- **The model learns the rule, not the rows.** Training on one stamped vector barely changes the output on it once 500 other stamped rows have taught the trigger rule.

## 2. Target side: differences that are not membership
AUC(original) − AUC(retrain), primary variant, paired over 10 seeds:

| c3 | c0 | c4 |
|---|---|---|
| −0.024 [−0.044, −0.004] | +0.018 [+0.000, +0.036] | −0.016 [−0.027, −0.006] |

These differences go both ways and are larger than anything the control shows, so they are not membership. Two causes, both named as limitations in the plan:
1. **Distribution mismatch.** Every shadow learns the trigger rule. Retrain never saw it, so its signals fall far below every fitted OUT mean, and the likelihood ratio extrapolates into the tails.
2. **Fixed cohort.** Every seed scores the same ~230 + ~230 units. Two models that rank those units differently produce an AUC difference that repeats on every seed. The paired CI over seeds cannot see this, because it treats the cohort as fixed.

## 3. Placebo test (post hoc): the canary-MIA signal does not survive
**Method** (`scripts/run_canary_placebo.py`, `runs/canary_placebo_seed42`).
- Canary sources and matched nonmembers are uniform draws from the same pool. Without a membership effect, their labels are exchangeable.
- The canary MIA's statistic, the mean over seeds of AUC(model) − AUC(retrain), is recomputed under 2,000 random relabellings of the cohort.
- The score is φ, which is monotone in the canary MIA's log p(normal). At row level it reproduces the stored canary-MIA AUCs within 1e-3; the small differences come from ties in the saturated old score.

| Request | Model | Row level (canary-MIA cohorts): observed [null 95% range], p | Unit level (deduplicated): observed [null 95% range], p |
|---|---|---|---|
| c3 | original | +0.0071 [−0.0099, 0.0096], **p = 0.16** | +0.0038 [−0.0107, 0.0110], p = 0.49 |
| c3 | retain-only | +0.0058 [−0.0075, 0.0073], p = 0.13 | +0.0041 [−0.0084, 0.0087], p = 0.35 |
| c3 | `ga_proj` | +0.0001 [−0.0024, 0.0024], p = 0.92 | +0.0013 [−0.0029, 0.0029], p = 0.63 |
| c0 | original | +0.0034 [−0.0082, 0.0084], **p = 0.42** | +0.0044 [−0.0095, 0.0094], p = 0.38 |
| c0 | retain-only | +0.0032 [−0.0069, 0.0071], p = 0.37 | +0.0036 [−0.0078, 0.0078], p = 0.40 |
| c0 | `ga_proj` | −0.0005 [−0.0024, 0.0024], p = 0.72 | −0.0010 [−0.0037, 0.0036], p = 0.62 |
| c4 | original | +0.0016 [−0.0431, 0.0423], p = 0.95 | +0.0266 [−0.0738, 0.0764], p = 0.46 |
| c4 | `ga_proj` | +0.0051 [−0.0106, 0.0110], p = 0.34 | +0.0108 [−0.0103, 0.0099], p = 0.03 |

All 30 rows, including `nkd_proj` and `ga_adam`, are in `placebo.csv`.

- **The original's ΔAUC is not distinguishable from relabelling noise.** The canary MIA declared power from +0.0071 (c3) and +0.0034 (c0), but the null of a random cohort spans about ±0.01.
- **The retain-only "keeps the leak" finding falls with it.** So does the finding that the projected arms remove the leak: there is no leak to remove.
- **Multiplicity.** 2 of the 30 placebo tests have p < 0.05 (c4 unit level: `ga_proj` 0.034, `ga_adam` 0.024), about the 1.5 expected by chance. The original itself is not significant there (p = 0.46), so these rows show models ranking a fixed cohort differently, not a retained membership signal.

## What this answers
1. **Individual canary membership is undetectable** by the loss attack and by per-example LiRA with 64 shadows, in the original and in every other model.
2. **Per-record MIA cannot show a privacy gain from unlearning here.** The original has no detectable per-record leak to remove. The natural requests gave the same result ([lira_results.md](lira_results.md)).
3. **The measurable leak is at group level.** The original reveals that triggered data was trained on: the backdoor success rate is 0.21 / 0.13 / 0.999, against 0.018 for retrain. The deployable method brings it to 0.015–0.020, the retrain level ([deployable_forget_results.md](deployable_forget_results.md)). This is the privacy evidence the experiments support.

## Lessons for the MIA protocol
- **A paired CI over seeds on one fixed cohort is not a membership test.** It needs a null that randomises membership: a shadow control or a placebo relabelling.
- **Matched cohorts must be deduplicated at the feature level** before scoring.

## Limitations
- The canaries are synthetic and share one trigger.
- The placebo test was added after seeing this run. Its relabelling is exact at row level and approximate at unit level, where eligibility depends on the labels.
- The shadows reproduce the original's world only.
- An attack with no power proves nothing about deletion. These results do not show that membership is safe, only that this model does not reveal it to these attacks.
