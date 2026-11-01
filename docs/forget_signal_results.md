# Forget-signal benchmark — results (`runs/forget_signal_seed42`)

The protocol is in [forget_signal_plan.md](forget_signal_plan.md). The run used 10 seeds, 16 FU arms, compute-matched controls and 5 requests on 3 target clients, and took 2,505 s on CPU. `verify_forget_signal.py` passed:
- 30 hashed artifacts, and the config equals the recorded protocol;
- requests and nonmember IDs were rebuilt, and the gate was recomputed;
- all 160 decision rows and the verdicts were recomputed from per-seed rows;
- 12 stages were re-executed bit-for-bit: 3 arms × pre/post × FU/control.

## Verdict
**No arm passes the preregistered rule.** The rule requires the arm to be superior to compute-matched retain-only by more than the noise floor, F1-noninferior, and local-effective on every admitted request.

| Arm | Requests passing (of 5) | Failing request |
|---|---|---|
| `ga_adam@retrain_oracle` | 4 (targets 0, 3, 4) | `client4_random_stratified_0.5` |
| `neggrad_plus@retrain_oracle` | 4 (targets 0, 3, 4) | `client4_random_stratified_0.5` |
| `ga_sgd@retrain_oracle` | 3 | c4 canary and the random request |
| every deployable (`@nonmember`) arm and every KD arm | ≤ 1 (only `client4_low_overlap_groups_0.5`) | all canaries and the random request |

The two arms that come closest use the **oracle** stop level. That level needs the retrain cloud, so it is an upper bound, not a deployable method.

## Gate and stop levels (means over seeds)
| Request | Target | Effect size | Original CE(F) | Nonmember level | Retrain level | Noise floor (primary) |
|---|---|---|---|---|---|---|
| canary_c3_500 | 3 | 4.18 | 5.67 | 4.67 | 11.29 | 0.025 P(normal) |
| canary_c0_500 | 0 | 4.59 | 7.02 | 6.07 | 11.50 | 0.025 |
| canary_c4_500 | 4 | 132.1 | 0.009 | 0.000 | 11.45 | 0.025 |
| client4_random_stratified_0.5 | 4 | 6.11 | 0.013 | 0.030 | 0.023 | 0.001 MAE(F) |
| client4_low_overlap_groups_0.5 | 4 | 15.79 | 0.034 | 0.074 | 0.066 | 0.003 MAE(F) |

All 5 requests pass the gate.

## Forgetting position (0 = original cloud, 1 = retrain cloud)
"pre" is after the local or KD stage; "post" is after the 2 recovery rounds; "ctrl" is the compute-matched retain-only control after recovery. ✓ marks a request where the arm is superior, noninferior and local-effective.

| Arm | c3 pre / post / ctrl | c0 pre / post / ctrl | c4 pre / post / ctrl | random pre / post / ctrl | low-overlap pre / post / ctrl |
|---|---|---|---|---|---|
| retain_only (0 steps) | 0 / 0.11 | 0 / 0.06 | 0 / 0.00 | 0 / 0.72 | 0 / 0.27 |
| ga_adam@nonmember | 0 steps | 0 steps | 0 steps | −0.36 / 0.73 / 0.72 | 0.38 / 0.48 / 0.24 ✓ |
| ga_adam@retrain_oracle | 0.98 / 0.59 / 0.11 ✓ | 0.96 / 0.48 / 0.03 ✓ | 0.60 / 0.11 / 0.00 ✓ | 0.60 / 0.73 / 0.72 | 0.38 / 0.46 / 0.25 ✓ |
| neggrad_plus@nonmember | 0 steps | 0 steps | 0 steps | −0.98 / 0.70 / 0.67 | 0.51 / 0.60 / 0.20 ✓ |
| neggrad_plus@retrain_oracle | 0.81 / 0.60 / 0.11 ✓ | 0.40 / 0.48 / −0.01 ✓ | 0.48 / 0.21 / 0.00 ✓ | −0.98 / 0.70 / 0.67 | 0.51 / 0.57 / 0.20 ✓ |
| ga_sgd@retrain_oracle | 0.74 / 0.52 / 0.11 ✓ | 0.69 / 0.38 / 0.05 ✓ | 0.00 / 0.00 / 0.00 | 0.58 / 0.72 / 0.69 | 0.45 / 0.50 / 0.17 ✓ |
| uniform@retrain_oracle | −1.49 / 0.39 / 0.14 | −3.28 / −0.06 / 0.00 | 0.56 / 0.01 / 0.00 | 0.29 / 0.73 / 0.73 | 0.22 / 0.45 / 0.23 ✓ |
| random_label@retrain_oracle | −1.52 / 0.36 / 0.14 | −3.33 / −0.09 / 0.00 | 0.55 / 0.01 / 0.00 | 0.34 / 0.73 / 0.73 | 0.18 / 0.44 / 0.23 |
| kd_retain | −0.06 / 0.15 / 0.14 | −0.78 / 0.03 / 0.00 | 0 / 0 / 0 | −0.97 / 0.70 / 0.65 | 0.04 / 0.32 / 0.15 |
| kd_teacher_on_f@nonmember | 0 steps | 0 steps | 0 steps | −0.98 / 0.71 / 0.63 | 0.54 / 0.58 / 0.17 ✓ |
| bad_teacher@retrain_oracle | −1.27 / −0.52 / 0.13 | −2.76 / −0.86 / 0.01 | 0.52 / 0.05 / 0.00 | 0.22 / 0.72 / 0.72 | 0.17 / 0.46 / 0.19 |
| scrub@retrain_oracle | −0.85 / −0.64 / 0.13 | −3.13 / −1.99 / 0.00 | 0 / 0 / 0 | −0.25 / 0.70 / 0.69 | −0.30 / −0.03 / 0.15 |

Pre-recovery utility (test macro-F1 minus the retrain-cloud mean):
- `ga_adam@retrain_oracle`: −0.22 on c3, −0.10 on c0, −0.46 on c4.
- `neggrad_plus@retrain_oracle`: −0.12, +0.004 and −0.46.
- After recovery, every arm is within 0.002 of retrain on the canaries.

## Findings
### 1. The local forget step
1. **Adam is needed on saturated forget sets.** On the c4 canary (CE(F) ≈ 0.009), `ga_sgd` took all 100 steps and did not move (position 0.00), because the CE gradient vanishes there. Adam GA reached 0.60. Where CE is not saturated (c3, c0), SGD at lr 0.02 also works, in 3–4 steps.
2. **With the oracle stop, GA and NegGrad+ do forget before recovery.**
   - The local check passes on every canary and on the low-overlap request.
   - On c3 and c0, MAE(F) closes 0.82–1.00 of the original→retrain gap.
   - The cost is pre-recovery F1, down to −0.46 on c4.
   - The divergence guard stopped both GA and NegGrad+ on all c4 seeds before they reached the retrain level (CE 11.5).
3. **The deployable nonmember stop fails in two opposite ways.**
   - **Canaries:** the original's CE on held-out stamped attacks is *below* its CE on F, because the backdoor generalises to any row carrying the trigger. Every nonmember-stopped arm therefore took 0 steps on all canaries.
   - **Random request:** the nonmember level (0.030) is above the retrain level (0.023), because 77% of F has feature-identical rows in R that retrain still fits. `ga_adam@nonmember` overshoots, and MAE(F) rises (position −0.36).
   - **Where it works:** only on the low-overlap request, where the two levels are close (0.074 vs 0.066).
4. **Bounded objectives are the wrong target here.** `uniform` and `random_label` cannot push mean CE(F) above ln 2. Retrain is confident on F, so on c3 and c0 these objectives move predictions *away* from retrain (pre −1.5 to −3.3). On the natural requests they stop at the target within 8–15 steps. With the oracle level they close 0.18–0.34 of the gap. With the nonmember level they close at most 0.09, or move away (down to −0.91 on the random request).
5. **NegGrad+ cannot forget F when F looks like R** (the random request). The retain term also lowers CE(F), so CE(F) never reaches the target in 100 steps, and the extra fine-tuning moves MAE(F) away (−0.98).

### 2. KD with a forget signal
1. **Retain-only KD erases the teacher's forgetting.** On the low-overlap request, the `neggrad_plus@nonmember` teacher is at 0.51 before recovery, but `kd_retain` is at 0.04.
2. **Distilling the teacher on F keeps it.** `kd_teacher_on_f` is at 0.54 pre and 0.58 post: superior and local-effective. On the canaries it inherits its teacher's 0-step nonmember stop, so there is nothing to transfer.
3. **Bad-Teacher pulls F towards a random teacher (p ≈ 0.5),** which is wrong for the canaries (down to −2.8). On the low-overlap request it is superior after recovery, but not local-effective.
4. **SCRUB is harmful in this setting.**
   - **Canaries:** its direction-free ascent strengthens the partly learned c0 backdoor (−3.1).
   - **Low-overlap request:** the target client retains a single normal row (F holds 748), and the ascent breaks the normal class. Replaying seed 10042 shows test FPR rising from 0.27 to 0.975 and F1 from 0.89 to 0.45; with 0 ascent steps the model is unchanged.
   - **Why the stop misses it:** CE(F) stays low, so a stop on CE(F) cannot see damage outside F.

### 3. Noise floor and compute-matched retain-only
1. **Recovery takes back part of the canary forgetting:** 0.98 → 0.59 on c3, 0.96 → 0.48 on c0 and 0.60 → 0.11 on c4 for `ga_adam@retrain_oracle`. The post values still beat the matched control by more than the 0.025 floor.
2. **On the random request, recovery alone gets 0.72 of the way.** No arm beats the matched control: the gain is 0.000 against a floor of 0.001. This is the only request blocking the two oracle arms. FU adds nothing measurable there.
3. **Extra retain-only compute does not help.** The matched control is never more than 0.03 better than 0-step retain-only. On c0 and on the low-overlap request it is worse (0.15–0.25 vs 0.27 there). The FU gains therefore do not come from simply spending more steps.

## Implications
1. **The deployable stop criterion is now the bottleneck.** A forget step that works exists, but a stop computable without retraining is needed. Options include per-request calibration from shadow retrains on auxiliary data, or an MIA-gap stop. The nonmember-CE level fails on backdoors and on F with duplicates in R.
2. **The forget signal must survive recovery.** Examples are NegGrad+ terms in the target client's recovery rounds, or recovery that excludes the layers carrying the trigger.
3. **Bounded objectives, Bad-Teacher and SCRUB do not suit this binary detector.** Retrain is confident on F, and client data are class-skewed. Any ascent-type method needs a utility check on held-out normal traffic, not only on CE(F).
4. **The all-request rule is strict but stays as preregistered.** It fails for the oracle arms only because one request needs no forget step. Results per request are reported above.

## Limitations
- The canaries share one trigger.
- Both natural requests are on client 4.
- The oracle stop is not deployable.
- FU and recovery reuse each original's seed.
- The smoke run (3 seeds) preceded the full run. No method or setting changed after it (see the plan).
- There is no MIA here, and these results are not proof of deletion.
