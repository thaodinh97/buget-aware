# Matched-canary membership inference (`runs/canary_mia_seed42`)

## Question
After unlearning, can an attacker still tell which triggered rows were trained on? The models come from [deployable_forget_results.md](deployable_forget_results.md).

## Design (fixed before the run)
- **Members.** The 500 canaries of each canary request: attack rows of the target client, stamped with the trigger and relabelled normal.
- **Nonmembers.** 500 other attack rows of the same client, stamped and relabelled in the same way, never trained on (`src/data/canary.py: matched_nonmembers`, seed 42 + 17 + client).
  - Canary sources and nonmembers are uniform draws from the same pool, so training membership is the only difference.
  - The unstamped versions of both sets are in D.
- **Score.** log p(normal), computed stably from the logit margin; a higher score means more member-like.
- **AUC reference.** The fixed cohorts differ slightly in difficulty, so even a model that never saw any canary (retrain) does not score exactly 0.5. Each model is therefore compared with retrain, paired over the 10 seeds.
- **Power rule.** The attack has power on a request only if the paired 95% CI of AUC(original) − AUC(retrain) lies above 0. Without power, every verdict is `MIA_INCONCLUSIVE`.
- **Models.** Original, retrain (the canary-free seed-floor original), retain-only (0 local steps plus standard recovery), and three arms with standard recovery: `ga_proj@joint_aux_prob`, `nkd_proj@joint_aux_prob` and the unprojected `ga_adam@joint_aux_prob`.
  - The arms are rebuilt with the deployable-forget code.
  - Every rebuilt model, retain-only included, matched the stored run's metrics bit-for-bit.
- **TPR at 5% FPR.** Reported only as a descriptive ROC point, not as a calibrated operating threshold.

`verify_canary_mia.py` passed:
- hashes, config and cohorts (no nonmember is a canary source);
- the summary and verdicts, recomputed from per-seed rows;
- 6 models re-executed for one seed, with their attack scores.

The run took 287 s on CPU.

## Results
AUC minus retrain, mean over 10 seeds [95% CI]:

| Model | canary c3 | canary c0 | canary c4 |
|---|---|---|---|
| original (power control) | **+0.0070 [0.0030, 0.0110]** | **+0.0034 [0.0017, 0.0050]** | +0.0017 [−0.0086, 0.0120]: no power |
| retain-only | +0.0058 [0.0012, 0.0103]: differs | +0.0032 [0.0015, 0.0049]: differs | inconclusive |
| **`ga_proj@joint_aux_prob`** | **+0.0001 [−0.0009, 0.0011]: retrain-like** | **−0.0005 [−0.0029, 0.0020]: retrain-like** | inconclusive |
| **`nkd_proj@joint_aux_prob`** | **+0.0008 [−0.0013, 0.0030]: retrain-like** | **−0.0001 [−0.0027, 0.0025]: retrain-like** | inconclusive |
| `ga_adam@joint_aux_prob` (unprojected) | +0.0031 [−0.0003, 0.0066]: retrain-like | +0.0024 [0.0010, 0.0037]: differs | inconclusive |

Absolute AUCs are 0.487–0.523. TPR at 5% FPR is 0.036–0.058, close to chance.

## Findings
1. **The membership signal is real but very small.** On c3 and c0, the original leaks membership just measurably more than retrain: ΔAUC +0.007 and +0.003.
   - The model learns the trigger rule, which applies equally to members and nonmembers, and barely memorises individual canaries.
   - On c4, where the backdoor is learned perfectly (member loss ≈ 0.009), there is no detectable membership signal.
2. **Retain-only keeps the leak.** It differs from retrain on both requests where the attack has power.
3. **The deployable method removes the leak to retrain level.** Both projected arms are retrain-like on c3 and c0, with CIs tightly around 0 (±0.003). The unprojected GA still differs on c0, because recovery restores part of the backdoor.
4. **Group-level leakage is much larger than membership leakage.** The backdoor success rate fell from 0.21 / 0.13 / 0.999 to 0.015–0.020, the retrain level.

## Limitations
- The canaries are synthetic and share one trigger.
- The loss attack has no shadow models; LiRA on the natural requests found no power at all ([lira_results.md](lira_results.md)).
- The effect sizes are near the resolution of 500-vs-500 cohorts.
- This is empirical evidence, not a membership-privacy guarantee.
