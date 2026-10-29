# ToN-IoT sample-level federated unlearning

## Diagnostic audit bổ sung (2026-10-19)

Đã hoàn tất [audit và đề xuất tối thiểu](docs/fu_diagnostic_audit.md), thêm stage diagnostics và chạy controlled pilot giữ nguyên artifacts v3. Xem [kết quả](docs/fu_diagnostic_results.md) và `runs/controlled_diagnostics_v2_seed42/metrics.csv`. 15 tests đạt; 18 comparisons hoàn tất. DT vẫn chưa triển khai, MIA vẫn inconclusive.

```powershell
python -m unittest discover -s tests -v
python scripts/run_controlled_diagnostics.py --source runs/pilot_v3_seed42 --data data/processed/connected_v3_seed42 --output runs/controlled_reproduction
python scripts/report_diagnostics.py --run runs/controlled_reproduction --output docs/controlled_reproduction.md
```

Runner kiểm tra hash của source run và frozen data, không regenerate split/request. Diagnostics thêm checkpoint trước/sau GA/GD/KD, full F/R loss/gradient norms, parameter/prediction deltas, per-step norms và counts. `conpfu_inspired_baseline` là global-start/full-local-KD adaptation với common recovery, không faithful reproduction. Shadow models chưa được train; MIA mới chỉ thêm family-matched sensitivity cohort, không đổi legacy IDs. Full details trong report. Run `controlled_diagnostics_seed42` đầu tiên bị dừng do audit object-array chậm; giữ partial artifacts/status và dùng v2 đã hoàn tất.

Đã triển khai và chạy pilot không DT: leakage-aware preprocessing → 5-client FedAvg → original/no-op/retain-only/fresh retrain → ConPFU-inspired GA/GD/KD → retained-only recovery → utility, prediction similarity và MIA controls.

**Pilot chạy thành công về kỹ thuật nhưng chưa đạt gate kiểm chứng FU. DT chưa triển khai.** MIA retained control yếu và retain-only gần như tương đương FU. Xem [báo cáo](docs/validation_report.md), [protocol](docs/experiment_protocol.md), [audit ban đầu](docs/implementation_audit.md), [kế hoạch](docs/implementation_plan.md).

## Chạy lại

PowerShell từ `E:/budget-aware-code`, dùng `python` tại `D:/python/python.exe` (torch CPU đã có). Manifest lưu phiên bản thực tế; requirements không phải lockfile đầy đủ.

```powershell
python -m unittest discover -s tests -v
python src/data/prepare.py --config configs/data_connected.json --output data/processed/reproduction_seed42
python scripts/run_pilot.py --data data/processed/reproduction_seed42 --output runs/reproduction_smoke --smoke
python scripts/run_pilot.py --data data/processed/reproduction_seed42 --output runs/reproduction_pilot
python scripts/verify_run.py --data data/processed/reproduction_seed42 --run runs/reproduction_pilot
python scripts/plot_results.py --run runs/reproduction_pilot --output results/figures/reproduction_pilot
```

Output run/preprocessing phải chưa tồn tại. Runner từ chối data audit chưa đạt hoặc thiếu kiểm tra encoded overlap. Smoke dùng subset khai báo rõ, không làm evidence FU. Config thay đổi phải dùng run mới.

## Thành phần

- `src/data/prepare.py`: connected grouping, train-only transforms, audit.
- `src/data/federated.py`: train-only Dirichlet, nested requests, retained partitions.
- `src/fl/fedavg_core.py`: MLP, logical worker, local training và shared weighted FedAvg.
- `src/fu/conpfu_inspired.py`: alternating GA/GD, regularization/clipping, retain-only KD.
- `src/eval/metrics.py`: utility, prediction similarity, fixed calibrated loss MIA.
- `scripts/run_pilot.py`: fresh retrain, recovery, checkpoints/logs, evaluation và gate.
- `scripts/verify_run.py`: artifact hashes, ownership, nested requests, retained counts, no-op.

## Artifacts thực tế

Data hiện tại: `data/processed/connected_v3_seed42`. Pilot: `runs/pilot_v3_seed42`. Figures: `results/figures/pilot_v3_seed42`. Checkpoints, initial model, predictions, evaluation IDs, request IDs, manifests, losses, timings, coverage và gate đều đã lưu.

Các data/run v1/v2 và `smoke_seed42` cũ chỉ là diagnostics trước sửa float32 overlap; không dùng làm evidence. Raw CSV bất biến. Raw/mapping/request IDs chỉ thuộc offline research/local workers, không publish công khai hoặc gửi server/DT.

Pipeline pilot ban đầu mô phỏng tuần tự trong một process, có thể chạy trên CPU hoặc GPU CUDA. Prototype server/client bên dưới là đường chạy phân tán riêng. Profiler hiện tại chưa đo GPU utilization/memory; các giới hạn riêng tư và độ tin cậy của prototype được ghi ở mục Docker. Chưa có DP, deletion guarantee, multi-seed validation hoặc full DT.

## Chạy thực nghiệm bằng Docker

Docker image chứa code và dependencies; dữ liệu cùng checkpoints được bind-mount chỉ đọc, còn kết quả được ghi ra `runs/docker/` trên máy host. Không đưa dữ liệu hoặc artifacts vào image. Compose bật GPU CUDA và cần NVIDIA GPU, NVIDIA Container Toolkit (hoặc Docker Desktop với GPU passthrough), cùng `data/processed/connected_v3_seed42` và `runs/pilot_v3_seed42`.

```powershell
docker compose build
docker compose run --rm experiment
docker compose --profile test run --rm test
```

Kết quả mặc định ở `runs/docker/controlled_phase_v2_seed42`; runner không ghi đè output đã tồn tại. Hãy đổi `--output` trong `compose.yaml` trước khi chạy lại. Đây vẫn là mô phỏng FL tuần tự trong một container, không tạo các client/server độc lập hay đo lưu lượng mạng thực.

Các runner chọn CUDA tự động nếu có, nếu không sẽ dùng CPU; đặt `FL_DEVICE=cuda` để bắt buộc GPU hoặc `FL_DEVICE=cpu` để ép CPU. Dịch vụ Compose `experiment` đặt `FL_DEVICE=cuda`, nên sẽ báo lỗi nếu GPU không được Docker cấp. Đối chiếu kết quả với CPU có thể có sai khác số thực do phép tính GPU.

## Chạy FL phân tán bằng server và clients

Compose có thêm prototype gồm một server tổng hợp FedAvg và 5 client process/container riêng. Server chỉ nhận checkpoint khởi tạo cùng manifest số lượng mẫu; từng client mount shard `X/y` riêng, huấn luyện trên GPU, rồi gửi trọng số lên qua HTTP trong mạng Docker nội bộ. Server xác thực ID/round/count, tổng hợp theo số mẫu và lưu kết quả ở `runs/docker/distributed_fedavg/`.

```powershell
docker compose --profile setup run --rm --build prepare
docker compose up --build --abort-on-container-exit
docker compose --profile test run --rm --build test
```

Lệnh chuẩn bị tạo shard trong `container_data/distributed_clients/`, tách biệt khỏi dữ liệu tiền xử lý `data/` và artifacts gốc. Shard được tạo từ partition đã lưu trong `runs/pilot_v3_seed42`; không tạo lại split. Đầu ra ở đây phải chưa có dữ liệu cũ. Sau khi chạy, xem `metrics.json` và `global_model.pt` dưới `runs/docker/distributed_fedavg/`. Server không publish port ra host; nếu thay đổi cấu hình số client, cần cập nhật đồng bộ service/client shards và manifest.

Đây là demo phân tán trong một Docker host, không phải triển khai production: HTTP không TLS/auth, không có differential privacy hay secure aggregation, shard file vẫn tồn tại trên host và mỗi container có thể cùng chia sẻ một GPU. Tách container giúp tách process và trao đổi model qua mạng, nhưng chưa tự nó đảm bảo riêng tư trước host/root.

## Controlled FU phase

See [experiment plan](docs/controlled_phase_plan.md) and [pilot results](docs/controlled_phase_results.md). The new runner preserves frozen data and client partitions, tests eight deletion requests and GA/GD step ratios, uses validation-only selection, and adds independent shadow MIA and sampled process cost profiling. It does not implement a DT planner.

```powershell
python scripts/run_controlled_phase.py --output runs/my_controlled_pilot
python scripts/verify_controlled_phase.py --run runs/my_controlled_pilot
python scripts/report_controlled_phase.py --run runs/my_controlled_pilot
```

Completed artifacts: `runs/controlled_phase_v2_seed42`. The first `controlled_phase_seed42` run is retained for provenance; v2 corrects round labels and root checkpoint parent metadata without changing training or selection. Raw data, row-level requests and model artifacts remain ignored by Git. Commit source, config, tests and aggregate reports. MIA remains inconclusive; FU success has not been established.

## Recovery trajectories and membership validation

The [protocol](docs/recovery_mia_plan.md) fixes two representative requests, rounds0–4, four starting states and two learning rates. See [recovery findings](docs/recovery_trajectory_results.md) and [membership audit/results](docs/membership_validation_results.md). The FedAvg observer is optional; retraining references never enter client updates.

```powershell
python scripts/run_recovery_trajectories.py --output runs/my_recovery_trajectories
python scripts/run_membership_validation.py --output runs/my_membership_validation
python scripts/verify_recovery_membership.py --recovery runs/my_recovery_trajectories --mia runs/my_membership_validation
python scripts/report_recovery_membership.py --recovery runs/my_recovery_trajectories --mia runs/my_membership_validation
```

Final artifacts: `runs/recovery_trajectories_seed42` and `runs/membership_validation_v3_seed42`. The first membership attempt stopped at preflight; v2 corrected auxiliary encoded-feature leakage, and v3 reused its models to correct probability saturation in baseline attack scores. No main split was regenerated. The nominal1%/5% attack thresholds are fixed on independent calibration data; actual FPR and group-bootstrap intervals are reported. These remain single-target-seed diagnostics, not certified deletion or a validated operational selection policy.

## LiRA membership inference

Implementation of LiRA (Carlini et al., S&P 2022) following the paper: logit-scaled signal computed stably from logits, balanced shadows (each unit IN for exactly half), and online/offline Gaussian likelihood-ratio tests with per-example or global variance. See the [preregistration](docs/lira_plan.md) and [results](docs/lira_results.md). The primary variant is online with global variance. The leave-one-shadow-out positive control fixes the low-FPR thresholds before any target model is scored.

```powershell
python -m unittest discover -s tests -p "test_lira.py" -v
python scripts/run_lira.py --output runs/my_lira --shadows 64
python scripts/verify_lira.py --run runs/my_lira
```

Points whose identical feature vector also appears outside the forget set are excluded, because their membership is undefined (counts in `eligibility.json`).

## Seed-noise floor

FU checkpoints are placed relative to 10 independently seeded retrains and originals, using 95% prediction intervals instead of the single seed-42 retrain. See [results](docs/seed_floor_results.md).

```powershell
python -m unittest discover -s tests -p "test_seed_floor.py" -v
python scripts/run_seed_floor.py --output runs/my_seed_floor --seeds 10    # about 25 min on CPU
python scripts/verify_seed_floor.py --run runs/my_seed_floor
```

## Identifiable deletion benchmark

This benchmark uses backdoor-canary and attack-type removal requests. A request is admitted only if its original and retrain seed clouds separate (the gate). Stronger GA-Adam and NegGrad+ forget operators are then reported over their full strength grid, applied to each of the 10 original seeds, with retain-only as the baseline. See the [protocol](docs/identifiable_fu_plan.md) and [results](docs/identifiable_fu_results.md). This run needs `runs/seed_floor_seed42` to exist first.

```powershell
python -m unittest discover -s tests -p "test_identifiable_fu.py" -v
python scripts/run_identifiable_fu.py --output runs/my_identifiable_fu      # about 15 min on CPU
python scripts/verify_identifiable_fu.py --run runs/my_identifiable_fu
```
