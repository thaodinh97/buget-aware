# Identifiable deletion benchmark — results (`runs/identifiable_fu_seed42`)

The protocol is in [identifiable_fu_plan.md](identifiable_fu_plan.md). The run used 10 seeds and took about 15 minutes on CPU. `verify_identifiable_fu.py` passed:
- 38 hashed artifacts;
- trigger dimensions are 0 in every split, and the canaries reproduce from the recorded rule;
- the gate was recomputed from stored clouds, and 3 cloud checkpoints were rechecked;
- 4 FU stages were re-executed bit-for-bit;
- summaries match the per-seed rows, and the `retain_only` pre-stage equals the original cloud.

## Gate
| Request | Original-cloud mean [95% PI] | Retrain-cloud mean [95% PI] | Effect size | Admitted |
|---|---|---|---|---|
| canary_c3_500 (P(normal) on triggered attacks) | 0.210 [0.058, 0.361] | 0.018 [−0.007, 0.043] | 4.18 | **yes** |
| type_c4_mitm (P(attack) on mitm) | 0.910 [0.867, 0.954] | 0.850 [0.800, 0.901] | 3.02 | no (the intervals overlap by 0.034) |
| type_c4_xss (P(attack) on xss) | 0.982 [0.970, 0.994] | 0.979 [0.967, 0.990] | 0.70 | no |

Removing an attack family from one client barely changes detection of that family: 56 mitm rows remain elsewhere, and other attack families generalise to it. Only the backdoor canary creates a deletion effect clearly above seed noise. The FU comparison below therefore covers only `canary_c3_500`.

## Canary: forgetting vs utility (mean over 10 original seeds, 95% CI)
Position 0 is the original cloud and 1 is the retrain cloud.

| Arm | Stage | Position [CI] | Seeds retrain-like | ΔF1 vs retrain | Seeds with F1 below retrain PI | Realised steps |
|---|---|---|---|---|---|---|
| retain_only | post | 0.11 [−0.15, 0.37] | 0/10 | −0.001 | 0/10 | 0 |
| ga_adam 3 | pre / post | 0.35 / 0.24 | 0 / 0 | +0.002 / −0.001 | 0 / 0 | 3 |
| ga_adam 10 | pre | 0.82 [0.74, 0.91] | 3/10 | −0.020 | 4/10 | 10 |
| ga_adam 10 | post | 0.48 [0.33, 0.62] | 0/10 | −0.001 | 0/10 | 10 |
| ga_adam 30 = 100 | pre | 1.07 [1.06, 1.08] | 10/10 | **−0.439** | 10/10 | 23.6 (guard) |
| ga_adam 30 = 100 | post | 0.76 [0.67, 0.85] | 2/10 | −0.001 | 0/10 | 23.6 |
| neggrad_plus 3 | pre / post | 0.31 / 0.25 | 0 / 0 | +0.002 / −0.001 | 0 / 0 | 3 |
| **neggrad_plus 10** | **pre** | **0.68 [0.58, 0.78]** | 1/10 | **+0.004** | **0/10** | 10 |
| neggrad_plus 10 | post | 0.49 [0.35, 0.62] | 0/10 | −0.001 | 0/10 | 10 |
| neggrad_plus 30 = 100 | pre | 0.96 [0.90, 1.02] | 8/10 | **−0.178** | 10/10 | 23.8 (guard) |
| neggrad_plus 30 = 100 | post | 0.76 [0.67, 0.85] | 2/10 | −0.001 | 0/10 | 23.8 |

Each arm takes about 1.6–1.9 s, including recovery. One retrain takes about 13 s.

Figure: `runs/identifiable_fu_seed42/canary_c3_500_tradeoff.png`.

## Findings
1. **Retain-only cannot remove the backdoor** (position 0.11, CI includes 0). This is expected by construction: the trigger dimensions are 0 in the retained data, so their first-layer weights receive no gradient during recovery. This request is exactly the kind of case where an explicit forget step is required, and where the earlier benchmark could not tell methods apart.
2. **A forget step on F does remove the backdoor**, monotonically in the step count, before recovery: GA reaches the retrain cloud on all 10 seeds and NegGrad+ on 8/10. At full strength, however, utility collapses (ΔF1 −0.44 for GA, −0.18 for NegGrad+), and the divergence guard stops both at about 24 steps.
3. **The best trade-off before recovery is NegGrad+ with 10 steps**: position 0.68, with no utility loss (ΔF1 +0.004, no seed below the retrain interval). At 10 steps, GA forgets more (0.82) but costs F1 on 4/10 seeds. The retain term in NegGrad+ protects utility at moderate strength.
4. **Recovery restores utility but brings back part of the backdoor.** Positions drop 1.07 → 0.76, 0.82 → 0.48 and 0.68 → 0.49. After recovery, GA and NegGrad+ are indistinguishable at every step level, and no arm is retrain-like on more than 2/10 seeds. The current FU pipeline, a forget step followed by 2 recovery rounds, therefore only partly forgets on this request.
5. **Hypothesis, not tested:** the trigger's first-layer weights stay largely intact, and recovery restores the downstream layers through which they act.

## Limitations
- Only one request passed the gate, so all conclusions concern a synthetic backdoor in one client.
- The grid was calibrated on a 3-seed smoke run (disclosed in the plan).
- At ≥30 steps both methods hit the guard, so 30 and 100 are the same point.
- FU and recovery reuse each original's seed.
- There is no MIA here, and these results are not proof of deletion.

## Next steps (not run)
1. A finer grid between 10 and 30 steps, and lower learning rates, to trace the Pareto front.
2. Forget steps that target the input weights directly, or adding the forget term to recovery (for example, NegGrad+ during the recovery rounds) to stop the backdoor from returning.
3. More canary designs (size, trigger, target client), and attack-family deletion across all clients, so more requests pass the gate.
