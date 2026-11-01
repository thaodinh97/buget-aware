# Forget-signal benchmark — protocol

## Purpose
[identifiable_fu_results.md](identifiable_fu_results.md) found three gaps:
1. The legacy GA step barely moved the model. GA-Adam and NegGrad+ removed the canary backdoor only at a fixed step count, and only a divergence guard stopped them.
2. Retain-only KD carries no forget signal: the student starts from the original and is distilled on R only.
3. FU was compared with a 0-step retain-only arm, not with retain-only at the same compute, and only one request on one target client passed the gate.

This benchmark addresses all three in one run (`scripts/run_forget_signal.py`). It reuses the frozen data, partitions and seed clouds.

## Requests (fixed before the 10-seed run)
| Request | Target | F | Primary metric | Original cloud | Retrain cloud |
|---|---|---|---|---|---|
| `canary_c3_500` | 3 | 500 triggered client-3 attacks relabelled normal | mean P(normal) on trigger-stamped test attacks | `identifiable_fu_seed42` (hash- and row-checked) | seed-floor originals on D |
| `canary_c0_500` | 0 | same rule, client 0 | same | 10 new models on D ∪ C | seed-floor originals on D |
| `canary_c4_500` | 4 | same rule, client 4 | same | 10 new models on D ∪ C | seed-floor originals on D |
| `client4_random_stratified_0.5` | 4 | the existing request | MAE(F) to the retrain cloud | seed-floor originals | seed-floor retrains |
| `client4_low_overlap_groups_0.5` | 4 | the existing request | MAE(F) to the retrain cloud | seed-floor originals | seed-floor retrains |

- **Canary rule.** All canaries use the trigger from `identifiable_fu_plan.md` (dims 8, 9, 14; zero in every split) and canary seed 42 + 7.
- **Seeds.** The seeds are 10042–10051 from the seed floor.
- **MAE(F).** MAE(F) is always measured against the retrain cloud with the same-seed retrain left out, because that retrain shares its initialisation and batch order with the original of the same seed.
- **Gate.** A request is admitted only if the 95% prediction intervals of its original and retrain clouds do not overlap on the primary metric.

## 1. Local forget step (`src/fu/bounded_forget.py`)
Adam is built once per call, so its moments persist across steps. The batch is 128, the learning rate 1e-3, the gradient clip 5, and the budget at most 100 steps.

| Objective | Loss on F | Bounded |
|---|---|---|
| `ga_adam` | −CE(F) | no |
| `neggrad_plus` | 0.9·CE(R) − 0.1·CE(F) | no |
| `random_label` | CE(F, uniformly random labels redrawn each step) | yes (≥ 0) |
| `uniform` | KL(U ‖ p) on F, which pushes p towards 0.5 | yes (≥ 0) |
| `ga_sgd` | −CE(F) with SGD at lr 0.02: the "raise forget_lr" option | no |

**Stop levels.** CE on all of F is checked before every step. An arm stops once CE(F) reaches the level below; a relative guard of 10 nats is kept in addition.
- **`nonmember`:** the original's CE on held-out look-alikes of F.
  - Canaries: 500 validation attacks, stamped and relabelled in the same way.
  - Natural requests: 2,000 validation rows with F's class mix.
  - This level is deployable: it needs no retraining.
- **`retrain_oracle`:** the retrain-cloud mean CE(F), with the same seed left out. It is an oracle: it needs retrains, so it measures what a well-calibrated stop could achieve.

Each objective runs with both stop levels, giving 10 local arms.

If the original is already at or above the level, the arm takes 0 steps; this is reported, not hidden. For two classes, the expectation of `random_label` equals `uniform`. Neither can raise CE(F) above ln 2 on average, so on a confident retrain (the canaries) both are expected to move *away* from retrain.

**Local check.** For every arm and seed, MAE(F) is measured after the local step, before recovery, and compared with the original of the same seed. `local_effective` requires the lower 95% CI bound of the reduction to be above 0.

## 2. KD with a forget signal (`src/fu/forget_kd.py`)
The student always starts from the original. KD runs at most 100 steps with T = 2, a hard-label weight of 0.5 on R, and a forget weight of 0.5.

| Arm | Teacher on R | Signal on F | Stop |
|---|---|---|---|
| `kd_retain` (control) | local `neggrad_plus@nonmember` | none | none |
| `kd_teacher_on_f@nonmember` | local `neggrad_plus@nonmember` | soft KD to the same teacher (never hard F labels) | nonmember |
| `bad_teacher@{nonmember, retrain_oracle}` | original | soft KD to a randomly initialised teacher | as named |
| `scrub@{nonmember, retrain_oracle}` | original + CE(R) | maximise KL(original ‖ student) on F for at most 10 steps | ends the ascent only |

Bad-Teacher and SCRUB are inspired by Chundawat et al. (AAAI 2023) and Kurmanji et al. (NeurIPS 2023). They are not faithful reproductions.

## 3. Noise floor and decision (`src/eval/fu_decision.py`)
- **Control.** For every FU arm and seed, the control is Adam on CE(R_target). It runs `ceil(backward_samples / 128)` steps, where `backward_samples` counts every row passed forward and backward through the student, including the local teacher for KD arms. The forward-only rows used by the stop checks and the teacher are counted but not matched.
- **Recovery.** FU arms, controls and the 0-step `retain_only` arm all receive the same 2 retained FedAvg rounds (seed s + 900000).
- **Noise floor.** The noise floor is the half-width of the retrain cloud's 95% prediction interval on the primary metric, i.e. how far a fresh retrain lands from the cloud mean by chance.
- **Superiority.** Per seed s, gain_s = |control_s − μ_R| − |FU_s − μ_R|. FU is *superior* on a request iff the lower 95% t-bound of the mean gain exceeds the noise floor.
- **Utility.** FU is *noninferior* iff the lower 95% bound of F1(FU) − F1(control) is above minus the F1 noise floor.
- **Pass.** An arm passes only if, on every admitted request, it is superior and noninferior after recovery and `local_effective` before recovery, and the admitted requests span at least 2 target clients.

## Calibration disclosed
- The forget settings (lr 1e-3, batch 128, clip 5, guard 10 nats) are copied from `identifiable_fu_plan.md`.
- The SGD learning rate, the KD weights and the step budgets were set before any run of this benchmark.
- A 3-seed smoke run (`runs/forget_signal_smoke3`) checks only the integration and the runtime (511 s). Two changes were made after it, and neither changes a method or a setting:
  - The manifest's description key was renamed to `control_rule`, because it overwrote the `control` config.
  - The verifier gained `atol=1e-12` for the CSV round-off.

## Limitations
- The canaries are synthetic, and all three use the same trigger.
- The natural requests are both on client 4.
- FU and recovery reuse each original's seed.
- The `retrain_oracle` stop is not deployable.
- There is no MIA here, and these results are not proof of deletion.
