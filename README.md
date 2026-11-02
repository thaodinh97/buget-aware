# ToN-IoT sample-level federated unlearning

Đã triển khai và chạy pilot không DT: leakage-aware preprocessing → 5-client FedAvg → original/no-op/retain-only/fresh retrain → ConPFU-inspired GA/GD/KD → retained-only recovery → utility, prediction similarity và MIA controls.

**Pilot chạy thành công về kỹ thuật nhưng chưa đạt gate kiểm chứng FU. DT chưa triển khai.** MIA retained control yếu và retain-only gần như tương đương FU. Xem [báo cáo](docs/validation_report.md), [protocol](docs/experiment_protocol.md), [audit ban đầu](docs/implementation_audit.md), [kế hoạch](docs/implementation_plan.md).

**Nghiên cứu tiếp theo đã triển khai, chưa chạy trên ToN-IoT thật:** bước quên có objective bị chặn/Adam/NegGrad+ và stop rule, KD có tín hiệu quên (full-local, bad-teacher), nhiễu nền từ N retrain khác seed, retain-only cùng compute, kịch bản quên cả họ attack/atypical, LiRA với shadow OUT. Xem [protocol nghiên cứu](docs/study_protocol.md).

**Run `study_v2` trên ToN-IoT thật (30 case, 3 seed): không method nào qua gate.** Bước quên mới có tác động trước recovery, nhưng không vượt ổn định retain-only cùng compute với margin lớn hơn nhiễu. Kịch bản random không có MIA hợp lệ (LiRA control < 0,5). Xem [báo cáo](docs/study_report_v2.md); tóm tắt kết quả ở `results/study_v2/`.

**Run `study_v3` (stop rule proxy, kiểm tra sau recovery, ứng viên trước recovery, chẩn đoán LiRA): vẫn không qua gate.** Ngưỡng proxy ước lượng CE retrain tốt hơn hẳn; NegGrad+ trước recovery thắng retain-only local cùng compute 6/10, thua 0/10, nhưng kém retain-only có recovery. Phát hiện dữ liệu: 92% dòng test nằm trong khoảng 1e-3 so với một dòng train (audit hash không thấy). Xem [báo cáo v3](docs/study_report_v3.md).

**Preprocessing gộp bản gần trùng (`configs/data_near_duplicates.json`, data v4) đã triển khai nhưng chưa dùng làm mặc định:** không còn test trùng khít train, nhưng split bị chia theo họ attack (pilot: macro-F1 0,755, FPR 0,63). Xem [phân tích và các phương án](docs/preprocessing_near_duplicates.md).

**Phương án C + log1p cho 7 feature đếm (data v5, split giống v3, MIA chỉ báo cáo): `study_v5` vẫn không qua gate.** log1p cải thiện utility (pilot macro-F1 0,901, FPR 0,243), nhưng FU không vượt ổn định retain-only có recovery; tín hiệu trước recovery của v3 không lặp lại. Xem [báo cáo v5](docs/study_report_v5.md).

**`study_v6` (utility tại ngưỡng chọn trên validation, 10 retrain, 3 lần lặp, tune trên seed riêng, 5 seed): gate vẫn không đạt.** NegGrad+@final (đã tune) gần retrain hơn retain-only cùng compute một cách có ý nghĩa thống kê khi gộp mọi kịch bản (Δ +0,0023, CI [+0,0002; +0,0045], p Holm 0,042). Ở kịch bản quên trọn họ attack, nó gần retrain hơn retain-only có recovery ở 7/8 case; utility được bảo toàn. Xem [báo cáo v6](docs/study_report_v6.md) và `results/study_v6/`.

## Chạy lại

PowerShell từ `E:/budget-aware-code`, dùng `python` tại `D:/python/python.exe` (torch CPU đã có). Manifest lưu phiên bản thực tế; requirements không phải lockfile đầy đủ.

```powershell
python -m unittest discover -s tests -v
python src/data/prepare.py --config configs/data_connected.json --output data/processed/reproduction_seed42
python scripts/run_pilot.py --data data/processed/reproduction_seed42 --output runs/reproduction_smoke --smoke
python scripts/run_pilot.py --data data/processed/reproduction_seed42 --output runs/reproduction_pilot
python scripts/verify_run.py --data data/processed/reproduction_seed42 --run runs/reproduction_pilot
python scripts/plot_results.py --run runs/reproduction_pilot --output results/figures/reproduction_pilot
python scripts/run_study.py --data data/processed/reproduction_seed42 --output runs/study_smoke --smoke
python scripts/run_study.py --data data/processed/reproduction_seed42 --output runs/study_v1
```

Output run/preprocessing phải chưa tồn tại. Runner từ chối data audit chưa đạt hoặc thiếu kiểm tra encoded overlap. Smoke dùng subset khai báo rõ, không làm evidence FU. Config thay đổi phải dùng run mới.

## Thành phần

- `src/data/prepare.py`: connected grouping, train-only transforms, audit.
- `src/data/federated.py`: train-only Dirichlet, nested requests, retained partitions.
- `src/fl/fedavg_core.py`: MLP, logical worker, local training và shared weighted FedAvg.
- `src/fu/conpfu_inspired.py`: alternating forget/retain (neg-CE, NegGrad+, random-label, uniform; SGD/Adam; stop rule), KD retain-only/full-local/bad-teacher. Default = pilot.
- `src/fu/baselines.py`: retain-only fine-tune cùng số minibatch gradient evaluations với FU local.
- `src/data/scenarios.py`: quên trọn họ attack với ownership độc quyền, quên atypical high-loss.
- `src/eval/noise_floor.py`, `src/eval/lira.py`: nhiễu nền leave-one-out từ N retrain, offline LiRA.
- `src/eval/metrics.py`: utility, prediction similarity, fixed calibrated loss MIA.
- `scripts/run_pilot.py`: fresh retrain, recovery, checkpoints/logs, evaluation và gate.
- `scripts/verify_run.py`: artifact hashes, ownership, nested requests, retained counts, no-op.
- `scripts/run_study.py`, `configs/study.json`: nhiều seed/target/kịch bản, gate theo margin nhiễu và CI.
- `scripts/make_synthetic_toniot.py`: CSV tổng hợp cùng schema, chỉ để smoke pipeline.
- `scripts/diagnose_lira.py`: lưới 2×2 shadow subset × near-duplicate exclusion trên checkpoint của một run.
- `scripts/audit_holdout_novelty.py`: khoảng cách láng giềng gần nhất test→train trong không gian encoded.

## Artifacts thực tế

Data hiện tại: `data/processed/connected_v3_seed42`. Pilot: `runs/pilot_v3_seed42`. Figures: `results/figures/pilot_v3_seed42`. Checkpoints, initial model, predictions, evaluation IDs, request IDs, manifests, losses, timings, coverage và gate đều đã lưu.

Các data/run v1/v2 và `smoke_seed42` cũ chỉ là diagnostics trước sửa float32 overlap; không dùng làm evidence. Raw CSV bất biến. Raw/mapping/request IDs chỉ thuộc offline research/local workers, không publish công khai hoặc gửi server/DT.

Đây là mô phỏng sequential CPU trong một process, chưa phải dịch vụ phân tán. Bytes là tensor payload, không phải wire bytes; chưa đo peak memory và chi phí mạng end-to-end. Chưa có DP, deletion guarantee, multi-seed validation hoặc full DT.
