# ToN-IoT sample-level federated unlearning

Mã nguồn thực nghiệm cho phần **federated unlearning (FU) mức mẫu** của khóa luận về điều phối tài nguyên bằng Digital Twin cho dịch vụ FU trong hệ thống phát hiện bất thường mạng IoT. Dữ liệu là ToN-IoT Network, bài toán phân loại nhị phân normal/attack, gồm 5 client FedAvg và một MLP 38→64→32→2. **Digital Twin chưa được triển khai.**

## Kết quả chính

Phương pháp **ứng viên** hiện tại gồm ba phần.

Nó **chưa được xác nhận**:
- Tiêu chí dừng được thêm sau khi xem smoke trên chính các seed đánh giá, và arm này được chọn trong 40 arm mà không hiệu chỉnh so sánh bội (xem mục Giới hạn).
- Hai lần chạy xác nhận trên partition và seed mới đều không qua quy tắc đặt trước: partition 43 qua 4/6 request, partition 44 qua 3/7. Protocol đã chốt trước là không có lần thứ ba (xem bên dưới).
- **Bước quên:** gradient ascent (Adam) được chiếu ra khỏi không gian input của dữ liệu còn giữ lại (retain-orthogonal projection).
- **Điểm dừng:** dừng theo reference không cần retrain (`joint_aux_prob`).
- **Recovery:** 2 vòng FedAvg trên dữ liệu còn lại.

Số liệu là trung bình 10 seed, xem [docs/deployable_forget_results.md](docs/deployable_forget_results.md). Cấu hình FU đã được **đóng băng** trước phase DT, kèm danh sách kết luận được và không được tuyên bố: [docs/fu_freeze.md](docs/fu_freeze.md).

| Request | MAE(F): original → retrain | MAE(F): unlearn → retrain | Sàn retrain ↔ retrain | Macro-F1: original / unlearn |
|---|---|---|---|---|
| canary client 3 | 0,145 | **0,002** | 0,0025 | 0,889 / 0,890 |
| canary client 0 | 0,060 | **0,002** | 0,0025 | 0,889 / 0,890 |
| canary client 4 | 0,994 | **0,004** | 0,0029 | 0,890 / 0,890 |
| client 4, low-overlap 50% | 0,031 | **0,012** | 0,0056 | 0,890 / 0,893 |
| client 4, random 50% | 0,0066 | 0,0034 (bằng retain-only) | 0,0019 | 0,890 / 0,891 |

- **Khoảng cách tới retrain.** Model unlearn gần retrain hơn original ở cả 10/10 seed của mọi request.
- **Backdoor canary.** Tỉ lệ backdoor còn hiệu lực giảm từ 0,21 / 0,13 / 0,999 xuống 0,015–0,020, bằng retrain (0,018).
- **Nhưng canary zero-dim là một lối tắt cấu trúc, nên bị loại khỏi bằng chứng cho phương pháp.**
  - Trigger nằm trên các chiều bằng 0 ở mọi dòng thật, nên retrain giữ nguyên các cột W1 đó ở giá trị khởi tạo.
  - Chỉ cần reset các cột này, không chạy bước quên nào, là đạt forgetting position 0,95–0,99, ngang `ga_proj` (0,98–1,01). Utility không đổi.
  - Model reset cũng khó phân biệt với retrain ngang `ga_proj`.
  - Xem [docs/reset_baseline_results.md](docs/reset_baseline_results.md).
- **Canary tổ hợp: bằng chứng chính cho projection, đã lặp lại trên dữ liệu chưa nhìn.** Trigger là một tổ hợp giá trị hiếm của các feature đang hoạt động, nên reset không có tác dụng.
  - Trên seed phát triển (c3, c4), `ga_proj` đạt position 0,95 / 0,98 sau recovery, trong khi GA không chiếu bị kéo về 0,70 / 0,14.
  - Trên partition 43 (c0, c1, c2), `ga_proj` đạt 0,64 / 1,03 / 0,98, còn GA không chiếu chỉ đạt 0,06 / 0,29 / 0,39.
  - Trên partition 44 (c2, c1), `ga_proj` đạt 1,02 / 1,04, còn GA không chiếu đạt 0,09 / 0,25.
  - Trên cả 7 request, `ga_proj` thắng control fine-tune cùng pool dữ liệu; GA không chiếu chỉ thắng ở 1/7.
  - **Giới hạn:** trên client 0 của partition 43 (49 nghìn dòng giữ lại, gần như toàn attack), bước chiếu không dịch chuyển được ở 3/10 seed.
  - Xem [docs/confirmation_p43_results.md](docs/confirmation_p43_results.md) và [docs/confirmation_p44_results.md](docs/confirmation_p44_results.md).
- **Request tự nhiên: không có bằng chứng cho bước quên, và có một trường hợp bước quên gây hại.**
  - **Partition 42 (client 4):**
    - Trên low-overlap, control fine-tune trên R_target ∪ validation cùng compute đã đạt 0,56, và `ga_proj` (0,74) không vượt nó quá sàn nhiễu.
    - Trên random, FU ngang retain-only.
  - **Partition 44 (client 2):**
    - Trên random, bước chiếu không dịch chuyển được (0,37, so với retain-only 0,40).
    - Trên low-overlap, F chứa gần như toàn bộ dòng normal của client. Mức dừng `joint_aux_prob` quá chặt, đẩy model ra xa retrain hơn cả original (position −0,27 sau recovery).
  - Xem [docs/confirmation_existing_results.md](docs/confirmation_existing_results.md) và [docs/confirmation_p44_results.md](docs/confirmation_p44_results.md).
- **Model gốc là detector yếu.** Macro-F1 0,89, nhưng **FPR trên traffic normal là 27%**; macro-F1 trên dữ liệu giữ lại của client 1 (99% normal) chỉ 0,56. Mọi nhận định "giữ được utility" đều so với mức nền này.
- **Privacy (MIA).**
  - **Không attack nào phát hiện được membership của từng bản ghi, kể cả trên model original.** Đã thử LiRA 64 shadow trên request tự nhiên, LiRA 64 shadow theo từng vector feature trên canary, và loss attack. Ngay trên shadow, nơi biết chắc membership, AUC chỉ 0,49–0,51.
  - Tín hiệu nhỏ mà MIA canary từng báo cáo (AUC cao hơn retrain 0,007 trên c3 và 0,003 trên c0) nằm trong phân phối placebo khi hoán vị ngẫu nhiên nhãn cohort (p = 0,16 và 0,42), **nên đã rút lại**.
  - Original không rò rỉ membership từng bản ghi ở mức đo được, nên MIA theo bản ghi không thể cho thấy unlearn làm tăng privacy trong thiết lập này.
  - **Rò rỉ đo được là ở mức nhóm:** model gốc cho thấy dữ liệu có trigger đã được train (tỉ lệ backdoor 0,21 / 0,13 / 0,999). Sau unlearn còn 0,015–0,020, bằng retrain.
  - **Ở mức model, unlearn vẫn phân biệt được với retrain.** Attacker biết F và có model tham chiếu của cả hai loại (test kiểu U-LiRA, 40 seed) phân biệt được mọi model unlearn trên cả 5 request; negative control retrain–retrain thì không phân biệt được.
    - Trên canary c3/c0, `ga_proj@joint_aux_prob` giảm AUC của attacker từ 1,00 xuống 0,67 / 0,71. Phần còn lại là một độ lệch nhỏ có hệ thống ở thang logit.
    - Trên c4 và hai request tự nhiên, AUC vẫn 0,98–1,00.
  - Xem [docs/indistinguishability_results.md](docs/indistinguishability_results.md), [docs/canary_lira_results.md](docs/canary_lira_results.md), [docs/canary_mia.md](docs/canary_mia.md) và [docs/lira_results.md](docs/lira_results.md).
- **Quy tắc quyết định đặt trước chưa đạt,** cả quy tắc cũ (thắng retain-only cùng compute trên mọi request) lẫn quy tắc của bước 8 (thêm control cùng pool, FPR và utility theo client).
  - **Chế độ `existing`:** `ga_proj@joint_aux_prob` qua 5/7 request, gồm mọi canary; nó trượt cả hai request tự nhiên.
  - **Chế độ `confirm`** (partition seed 43, seed 20042–20051): `ga_proj` qua 4/6 canary, và không request tự nhiên nào qua gate.
    - Hai lần trượt đều do macro-F1 trên dữ liệu giữ lại của client 4. Mọi model original và retrain cho cùng một giá trị ở đó, nên sàn nhiễu bằng 0. Khi đó chỉ một dòng đổi dự đoán ở một seed, kể cả theo chiều tốt hơn, cũng làm check trượt.
    - Phân tích post hoc với sàn bằng độ mịn một dòng cho 6/6, nhưng đó không phải là xác nhận.
  - **Chế độ `confirm2`** (partition seed 44, seed 30042–30051, sàn không mịn hơn một dòng, protocol ghi trước khi chạy): `ga_proj` qua 3/7.
    - Hai canary trượt utility vì khoảng tin cậy bị nới bởi các seed *tốt hơn* retain-only; không seed nào có FPR kém hơn quá 0,0004.
    - Hai request tự nhiên của client 2 trượt ở chính phần quên.
    - Theo protocol, đây là lần xác nhận cuối cùng.
  - Lịch sử các thực nghiệm và kết quả âm tính được tóm tắt trong [docs/README.md](docs/README.md).

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
| 6 | **Deployable forget** | `python scripts/run_deployable_forget.py --output runs/deployable_forget_seed42` | phương pháp ứng viên (chưa xác nhận) | `verify_deployable_forget.py` (khoảng 100 phút) |
| 7 | MIA canary | `python scripts/run_canary_mia.py --output runs/canary_mia_seed42` | membership inference trên các model của bước 6 | `verify_canary_mia.py` |
| 8 | **Utility chi tiết và xác nhận** | `python scripts/run_confirmation.py --mode existing --output runs/confirmation_existing_seed42`, sau đó `--mode confirm --output runs/confirmation_p43` | FPR và F1 theo client, control cùng pool, baseline reset, canary trigger tổ hợp, partition và seed mới (`confirm`: 4/6; `confirm2 --output runs/confirmation_p44`: 3/7; cả hai không qua quy tắc) | `verify_confirmation.py`; protocol ở [docs/confirmation_plan.md](docs/confirmation_plan.md) |
| 9 | LiRA canary | `python scripts/run_canary_lira.py --output runs/canary_lira_seed42 --workers 8`, sau đó `python scripts/run_canary_placebo.py --output runs/canary_placebo_seed42` | LiRA theo từng vector, 64 shadow mỗi request, trên các model của bước 6; placebo test (post-hoc) cho thống kê của bước 7 | `verify_canary_lira.py`; `--workers` không đổi kết quả |
| 10 | Phân biệt unlearn với retrain | `python scripts/run_indistinguishability.py --output runs/indistinguishability_seed42 --workers 8` | test kiểu U-LiRA, 40 seed: attacker biết F có phân biệt được model unlearn với retrain không | `verify_indistinguishability.py` |
| 11 | Baseline reset trong test phân biệt | `python scripts/run_reset_distinguisher.py --output runs/reset_distinguisher_seed42 --workers 6` | cùng test của bước 10 cho 3 baseline reset của bước 8 | `verify_reset_distinguisher.py` |
| — | Shadow MIA và LiRA | `python scripts/run_membership_validation.py --output runs/membership_validation_v3_seed42`, sau đó `python scripts/run_lira.py --output runs/lira_seed42` | kiểm tra sức mạnh của MIA | `verify_lira.py` |
| 12 | **DynamicFU mức mẫu trên shard** (sau đóng băng) | `run_dynamicfu_sample.py requests` → `tune` (partition pilot 1042) → `requests` (partition 42) → `run --seed 42..46`, rồi `evaluate_dynamicfu_gates.py` | 20 client, 5 shard; retrain shard bị ảnh hưởng, số vòng theo DynamicFU; 4 cổng để sang DT | `verify_dynamicfu_sample.py`; protocol ở [docs/dynamicfu_sample_plan.md](docs/dynamicfu_sample_plan.md) |

Mọi verifier đều nhận `--run <thư mục>`; riêng `verify_run.py` cần thêm `--data`. Chạy test bằng `python -m unittest discover -s tests -v`.

**Lưu ý.** Bước 1–4 vẫn chứa các biến thể FU cũ (ConPFU-inspired GA/GD/KD, GA-Adam/NegGrad+ chạy số bước cố định). Chúng được giữ lại vì các bước sau dùng partition, request và cloud do chúng tạo ra. Kết quả của các biến thể này kém hơn phương pháp ở bước 6.

## Cấu trúc

- `src/data/`
  - `prepare.py`: tiền xử lý.
  - `federated.py`: chia Dirichlet (5 client, hoặc nhiều client với cỡ tối thiểu: `partition_min_size`) và tạo request.
  - `canary.py`: backdoor canary (trigger zero-dim và trigger tổ hợp) và nonmember khớp phân phối.
  - `request_sensitivity.py`: mô tả request.
- `src/fl/`
  - `fedavg_core.py`: MLP, client worker, FedAvg, chọn thiết bị.
  - `distributed.py`: prototype phân tán.
  - `sharded.py`: FedAvg theo shard (nhóm client rời nhau, mỗi shard một model, ensemble tổng logit).
- `src/fu/`
  - **Phương pháp ứng viên:** `bounded_forget.py` (toán tử quên, stop rule, control cùng compute), `projection.py` (retain-orthogonal projection), `neighbor_reference.py` (reference kNN), `recovery_schedule.py` (recovery xen kẽ).
  - **Unlearning theo shard (sau đóng băng):** `shard_retrain.py` (retrain shard bị ảnh hưởng, quỹ đạo theo vòng), `dynamicfu.py` (điểm đóng góp I_F, controller DynamicFU và ablation không dùng F).
  - **Baseline:** `reset_baseline.py` (reset cột W1 của các chiều input không được dùng).
  - **Toán tử cũ dùng ở bước 1–5:** `forget_kd.py`, `strong_forget.py`, `conpfu_inspired.py`, `step_budget.py`, `diagnostics.py`.
- `src/eval/`
  - `fu_decision.py`: noise floor và quy tắc quyết định.
  - `confirmation_decision.py`: quy tắc của bước 8 (control cùng pool, utility theo FPR và theo client; với `confirm2`, sàn utility không mịn hơn một dòng).
  - `utility_breakdown.py`: FPR, recall, F1 từng lớp, metric theo client.
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

- **Phương pháp được chọn sau khi xem dữ liệu đánh giá.**
  - Thống kê `*_prob` và `joint_aux_prob` được thêm sau hai lần smoke trên chính các seed đánh giá, và chỉ số chính của canary tính trên test.
  - Arm được chọn trong 40 arm mà không hiệu chỉnh so sánh bội.
  - Chỉ hai chế độ `confirm` và `confirm2` của bước 8 (partition và seed mới) mới là xác nhận, và cả hai đều **không qua** quy tắc đặt trước (4/6 và 3/7 request).
- **Canary là backdoor tổng hợp, với một thiết kế trigger.** Canary zero-dim tạo lối tắt cấu trúc (xem phần kết quả). Canary tổ hợp tránh được lối tắt đó, và có 7 request qua gate trên 3 partition.
- **Request tự nhiên hầu như không đo được.** Số request vượt nhiễu seed:
  - partition 42: 2 request, đều của client 4;
  - partition 43: 0 trong 9 request;
  - partition 44: 2 trong 6 request, đều của client 2.

  Cả ba partition đều dùng α = 0,5.
- **Tập validation giữ nhiều vai trò:** dữ liệu phụ trợ, look-alike nonmember, và chọn siêu tham số ở controlled phase. Xem bảng vai trò của các tập dữ liệu trong [docs/README.md](docs/README.md).
- **Model gốc yếu** (FPR 27%), nên utility được đo so với một mức nền thấp.
- Projection chỉ đảm bảo ở bậc nhất.
- MIA theo bản ghi không có sức mạnh với model này. Privacy được đánh giá ở mức nhóm (backdoor) và mức model (phân biệt với retrain); ở mức model, unlearn vẫn phân biệt được.
- Kết quả là bằng chứng thực nghiệm về việc giảm ảnh hưởng, **không phải đảm bảo xóa dữ liệu hay DP**.
