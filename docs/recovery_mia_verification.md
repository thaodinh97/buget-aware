# Verification record

Final outputs:

- `runs/recovery_trajectories_seed42`: 87 hashed artifacts;80 global rows;320 per-client rows;16 four-round trajectories. Eight primary round2 states exactly reproduce cached pilot checkpoints.
- `runs/membership_validation_v3_seed42`: 71 hashed artifacts;108 target comparison rows;four cached improved shadows;one cached attempted positive control. The v3 evaluation took114.0 seconds including uncertainty calculations and plots; historical shadow-fit timings are separately marked cached.

25 unit tests passed. These include FedAvg observer noninterference, zero-vector alignment, exact distance decomposition, attack direction, paired-bootstrap zero differences for identical models, encoded-equivalence role closure, and stable extreme-confidence ordering.

Independent post-run verification passed all artifact hashes, per-round/client counts, aggregation weights, weighted directional contributions, squared-distance identities, optimizer-step totals, auxiliary role/component disjointness, fixed target cohort IDs, frozen thresholds and zero retrain-versus-self differences. ROC and recovery figures were visually inspected.

The preflight membership run is preserved at `runs/membership_validation_seed42`. It stopped before training because old buckets2/5 shared one encoded signature. v2 excluded the two affected rows without reassigning roles or changing primary data splits. v3 reused the exact v2 models to fix baseline probability-saturation ties using logits. No attack was selected or reversed using target results.

Results remain MIA_INCONCLUSIVE. The final learned attack AUC is0.49695 (95% group-bootstrap interval0.47326–0.52287). The attempted positive memorization control reached only59.4% member accuracy; it is explicitly recorded as insufficient power validation. No successful-unlearning, independent target-seed confirmation, operational retraining-based selection, or certified-deletion claim is made. No DT planner was implemented.
