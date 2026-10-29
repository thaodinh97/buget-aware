# Controlled phase verification

Final run: `runs/controlled_phase_v2_seed42`.

- All 8 planned requests completed; no failures or excluded requests.
- 168 evaluation rows; 72 recovery runs, 8 retraining references, 3 shadow models.
- 20 unit tests passed, including profiler round roles and diagnostic noninterference.
- Independent verifier passed: 790 artifact hashes, 320 diagnostic checkpoint reloads, stage names, step/sample budgets, auxiliary row/group disjointness, exact no-op metrics, recovery input hashes, and separate training/recovery round counts.
- All 168 utility, retraining-similarity and MIA rows exactly match the first controlled run. The second run corrected cost and checkpoint metadata without changing training, requests, selection or predictions.
- Final runner time: 207.29 seconds including diagnostics. Timing and memory are machine/process observations, not isolated client costs or exact network measurements.

The first run remains at `runs/controlled_phase_seed42` for provenance. The original source run's `metrics.csv` has a pre-existing hash mismatch; it was not consumed for training or selection and was not overwritten. The discrepancy is explicitly recorded in the new manifest. Other source artifacts remained subject to strict verification.

Verification commands:

```powershell
python -m unittest discover -s tests -v
python scripts/verify_controlled_phase.py --run runs/controlled_phase_v2_seed42
python scripts/report_controlled_phase.py --run runs/controlled_phase_v2_seed42
```

These checks validate execution and provenance, not successful unlearning. The independent shadow attack AUC is 0.498422; MIA conclusions remain inconclusive. No DT planner was implemented.
