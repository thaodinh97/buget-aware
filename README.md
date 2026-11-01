# ToN-IoT sample-level federated unlearning

Mã nguồn thực nghiệm cho phần **federated unlearning (FU) mức mẫu** của khóa luận về điều phối tài nguyên bằng Digital Twin cho dịch vụ FU trong hệ thống phát hiện bất thường mạng IoT. Dữ liệu là ToN-IoT Network, bài toán phân loại nhị phân normal/attack, gồm 5 client FedAvg và một MLP 38→64→32→2. **Digital Twin chưa được triển khai.**

## Kết quả chính

Phương pháp tốt nhất hiện tại gồm ba phần:
- **Bước quên:** gradient ascent (Adam) được chiếu ra khỏi không gian input của dữ liệu còn giữ lại (retain-orthogonal projection).
- **Điểm dừng:** dừng theo reference không cần retrain (`joint_aux_prob`).
- **Recovery:** 2 vòng FedAvg trên dữ liệu còn lại.

Số liệu là trung bình 10 seed, xem [docs/deployable_forget_results.md](docs/deployable_forget_results.md).

| Request | MAE(F): original → retrain | MAE(F): unlearn → retrain | Sàn retrain ↔ retrain | Macro-F1: original / unlearn |
|---|---|---|---|---|
| canary client 3 | 0,145 | **0,002** | 0,0025 | 0,889 / 0,890 |
| canary client 0 | 0,060 | **0,002** | 0,0025 | 0,889 / 0,890 |
| canary client 4 | 0,994 | **0,004** | 0,0029 | 0,890 / 0,890 |
| client 4, low-overlap 50% | 0,031 | **0,012** | 0,0056 | 0,890 / 0,893 |
| client 4, random 50% | 0,0066 | 0,0034 (bằng retain-only) | 0,0019 | 0,890 / 0,891 |

- **Khoảng cách tới retrain.** Model unlearn gần retrain hơn original ở cả 10/10 seed của mọi request.
- **Backdoor canary.** Tỉ lệ backdoor còn hiệu lực giảm từ 0,21 / 0,13 / 0,999 xuống 0,015–0,020, bằng retrain (0,018).
- **Privacy (MIA).**
  - Trên canary client 3 và 0, MIA khớp phân phối phát hiện được một tín hiệu membership rất nhỏ ở model gốc: AUC cao hơn retrain 0,007 và 0,003.
  - Sau unlearn bằng projection, tín hiệu này trở về mức retrain (chênh trong ±0,003). Retain-only thì vẫn giữ tín hiệu.
  - Trên canary client 4, và với LiRA 64 shadow trên request tự nhiên, attack không có sức mạnh nên không kết luận được.
  - Xem [docs/canary_mia.md](docs/canary_mia.md) và [docs/lira_results.md](docs/lira_results.md).
- **Quy tắc quyết định đặt trước** (thắng retain-only cùng compute trên mọi request) **chưa đạt.** Lịch sử các thực nghiệm và kết quả âm tính được tóm tắt trong [docs/README.md](docs/README.md).

## Môi trường

Chạy từ thư mục gốc repo, bằng PowerShell với `python` (đã dùng Python 3.13 và PyTorch 2.6.0). Cài đặt:
- `requirements-preprocessing.txt` cho bước tiền xử lý;
- `requirements-ml.txt` cho phần huấn luyện.

Các runner tự chọn CUDA nếu có. **Hãy đặt `FL_DEVICE=cpu`**, vì mọi run hiện có đều làm trên CPU và với model nhỏ này CPU nhanh hơn GPU (retrain mất 8,1 s trên CPU so với 12,2 s trên RTX 3050). Kết quả GPU khác CPU ở các chữ số cuối. Các verifier tự chạy lại trên thiết bị ghi trong manifest của run.

## Pipeline tái lập

Mỗi bước đọc artifact của bước trước ở **đường dẫn cố định**, nên cần dùng đúng tên output dưới đây. Runner từ chối ghi đè output đã có; dữ liệu, `runs/` và checkpoint không đưa vào git.

| # | Bước | Lệnh chạy | Output | Vai trò |
|---|---|---|---|---|
| 0 | Tiền xử lý | `python src/data/prepare.py --config configs/data_connected.json --output data/processed/connected_v3_seed42` | dữ liệu đã chia | gom nhóm theo connected component, transform chỉ fit trên train, audit leakage |
| 1 | FedAvg gốc | `python scripts/run_pilot.py --output runs/pilot_v3_seed42` | partition 5 client, model original | `verify_run.py` |
| 2 | Request | `python scripts/run_controlled_phase.py --output runs/controlled_phase_v2_seed42` | 8 request (client 0/4 × random/low-overlap × 10/50%) | `verify_controlled_phase.py` |
| 3 | Seed floor | `python scripts/run_seed_floor.py --output runs/seed_floor_seed42` | 10 original và 10 retrain mỗi request | noise floor; `verify_seed_floor.py` |
| 4 | Canary client 3 | `python scripts/run_identifiable_fu.py --output runs/identifiable_fu_seed42` | cloud backdoor canary | `verify_identifiable_fu.py` |
| 5 | Forget signal | `python scripts/run_forget_signal.py --output runs/forget_signal_seed42` | canary client 0/4, quy tắc quyết định | `verify_forget_signal.py` (khoảng 42 phút) |
| 6 | **Deployable forget** | `python scripts/run_deployable_forget.py --output runs/deployable_forget_seed42` | **phương pháp tốt nhất** | `verify_deployable_forget.py` (khoảng 100 phút) |
| 7 | MIA canary | `python scripts/run_canary_mia.py --output runs/canary_mia_seed42` | membership inference trên các model của bước 6 | `verify_canary_mia.py` |
| — | Shadow MIA và LiRA | `python scripts/run_membership_validation.py --output runs/membership_validation_v3_seed42`, sau đó `python scripts/run_lira.py --output runs/lira_seed42` | kiểm tra sức mạnh của MIA | `verify_lira.py` |

Mọi verifier đều nhận `--run <thư mục>`; riêng `verify_run.py` cần thêm `--data`. Chạy test bằng `python -m unittest discover -s tests -v`.

**Lưu ý.** Bước 1–4 vẫn chứa các biến thể FU cũ (ConPFU-inspired GA/GD/KD, GA-Adam/NegGrad+ chạy số bước cố định). Chúng được giữ lại vì các bước sau dùng partition, request và cloud do chúng tạo ra. Kết quả của các biến thể này kém hơn phương pháp ở bước 6.

## Cấu trúc

- `src/data/`
  - `prepare.py`: tiền xử lý.
  - `federated.py`: chia Dirichlet và tạo request.
  - `canary.py`: backdoor canary và nonmember khớp phân phối.
  - `request_sensitivity.py`: mô tả request.
- `src/fl/`
  - `fedavg_core.py`: MLP, client worker, FedAvg, chọn thiết bị.
  - `distributed.py`: prototype phân tán.
- `src/fu/`
  - **Phương pháp tốt nhất:** `bounded_forget.py` (toán tử quên, stop rule, control cùng compute), `projection.py` (retain-orthogonal projection), `neighbor_reference.py` (reference kNN), `recovery_schedule.py` (recovery xen kẽ).
  - **Toán tử cũ dùng ở bước 1–5:** `forget_kd.py`, `strong_forget.py`, `conpfu_inspired.py`, `step_budget.py`, `diagnostics.py`.
- `src/eval/`
  - `fu_decision.py`: noise floor và quy tắc quyết định.
  - `seed_floor.py`: prediction interval và placement.
  - `metrics.py`: utility và độ tương đồng.
  - `membership_validation.py`: điểm MIA, ghép cohort, bootstrap.
  - `lira.py`, `shadow_mia.py`: các attack MIA.
- `scripts/`: các runner `run_*.py` và verifier `verify_*.py` của từng bước, `experiment_context.py` (nạp dữ liệu, partition và request đã đóng băng), cùng prototype FL phân tán (`fl_server.py`, `fl_client.py`, `prepare_distributed_clients.py`).
- `docs/`: protocol và kết quả của từng bước; mục lục và lịch sử nằm trong [docs/README.md](docs/README.md).

## Docker

Image chứa code và dependencies (torch bản cu124). Dữ liệu và checkpoint được bind-mount chỉ đọc; kết quả ghi ra `runs/docker/`. Compose cần NVIDIA GPU và NVIDIA Container Toolkit.

```powershell
docker compose build
docker compose run --rm experiment              # bước 2 trong container
docker compose --profile test run --rm test
```

**Prototype FL phân tán.** Gồm một server FedAvg và 5 client container, trao đổi trọng số qua HTTP trong mạng Docker nội bộ. Shard của từng client được tạo từ partition đã lưu trong `runs/pilot_v3_seed42`.

```powershell
docker compose --profile setup run --rm --build prepare
docker compose up --build --abort-on-container-exit
```

Đây là bản demo trên một Docker host, chưa phải triển khai thật: không có TLS hay xác thực, không có DP hay secure aggregation.

## Giới hạn

- Canary là backdoor tổng hợp và cùng một trigger.
- Cả hai request tự nhiên đều thuộc client 4.
- Dữ liệu validation đóng vai dữ liệu phụ trợ.
- Projection chỉ đảm bảo ở bậc nhất.
- Tiêu chí dừng được bổ sung sau hai lần smoke (đã ghi rõ trong plan).
- Kết quả là bằng chứng thực nghiệm về việc giảm ảnh hưởng, **không phải đảm bảo xóa dữ liệu hay DP**.
