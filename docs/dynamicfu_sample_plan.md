# DynamicFU mức mẫu trên kiến trúc shard: protocol đặt trước

**Trạng thái:** hướng FU mới, chạy **sau đóng băng** ([fu_freeze.md](fu_freeze.md)), trên nhánh `feature/sharded-exact-fu`. Protocol này được ghi **trước** khi chạy pilot và run chính. Sau khi chạy, không sửa quy tắc hay ngưỡng nào để chấm lại kết quả.

**Quan hệ với đề cương.** Hướng này thay cơ chế "FU gần đúng" của đề cương bằng **unlearning dựa trên retrain theo shard**, với số vòng retrain chọn theo DynamicFU. Những chỗ đề cương phải sửa, cần trao đổi với giảng viên hướng dẫn:
- "FU gần đúng ở mức mẫu" ở mục tiêu tổng quát, mục tiêu cụ thể 2 và Nội dung 2;
- Phương pháp thực hiện, mục 4 (hiệu chỉnh tham số với ràng buộc độ lệch);
- "năm client giả lập", "3–5 client" và "hai mức không đồng nhất", nay là 20 client, 5 shard và ba mức α;
- câu giới hạn "mang tính gần đúng… không chứng nhận";
- biến quyết định của DT (mục 7).

Phạm vi vẫn đúng đề cương: chỉ request mức mẫu tại một client còn trực tuyến, không xóa cả client.

**Nguồn:** Wu và cộng sự, *DynamicFU: Contribution-Aware Dynamic Federated Unlearning for Industrial IoT*, Sensors 26:3714 (2026).
- Code mang nhãn **`dynamicfu_inspired`**, không phải bản tái lập. Bài báo làm unlearning mức client; ở đây là mức mẫu.
- Các chỗ khác bài báo được liệt kê ở mục 3.4.

## 1. Kiến trúc và thiết lập

**Kiến trúc** theo mục 3.1 của bài báo (`src/fl/sharded.py`):
- 20 client được chia ngẫu nhiên có seed thành 5 shard rời nhau, mỗi shard 4 client.
- Mỗi shard có một model FedAvg riêng.
- Suy luận: p(attack | x) = σ(Σ_s (l₁ − l₀)(x; w_s)).

| Thành phần | Giá trị |
|---|---|
| Dữ liệu | `data/processed/connected_v3_seed42`. Train chia cho client; validation dùng cho controller, chọn tham số và I_F; calibration dùng làm ngưỡng MIA; **test chỉ dùng để báo cáo và chấm cổng** |
| Partition chính | Dirichlet theo nhãn, α ∈ {0,1; 0,5; 1,0}, seed 42; rút lại tới khi mọi client có ≥ 10 dòng (số lần rút ghi trong manifest) |
| Partition pilot | giống hệt, nhưng seed 1042; chỉ dùng để chọn tham số controller |
| Model, FedAvg | MLP 38→64→32→2; 15 vòng, 1 epoch local, Adam lr 1e-3, batch 512; `FL_DEVICE=cpu`, 2 luồng, thuật toán tất định |
| Seed train | 42, 43, 44, 45, 46. Partition, shard và request cố định; chỉ khởi tạo và thứ tự batch thay đổi theo seed |

## 2. Request (stage `requests`, partition seed 42, chọn trên model seed 42)

**Request tự nhiên.**
- Ứng viên: với mỗi client và mỗi tỉ lệ 10% hoặc 50%, lấy F phân tầng theo lớp bằng `forget_request`, seed 42 + 1. F lồng nhau giữa hai tỉ lệ.
- Ứng viên không hợp lệ (một lớp bị rỗng trong F hoặc R) được ghi lại trong manifest, không lấy mẫu bù.
- Tính I_F cho mọi ứng viên trên model gốc seed 42, xếp giảm dần rồi chia làm ba nhóm theo tertile: high, medium, low.
- Mỗi nhóm lấy ngẫu nhiên **3 request**, seed 42 + 5 + chỉ số α.
- Kết quả: **9 request tự nhiên mỗi α**, như cách bài báo chọn 3 client mỗi nhóm đóng góp ở mục 5.3.

**Request canary.**
- Chọn tối đa **3 client** có nhiều dòng attack nhất, và phải có ≥ 2.000 dòng attack.
- Tại mỗi client đó cấy **500 canary trigger tổ hợp**, dùng `src/data/canary.py` như ở bước 8:
  - chọn 500 dòng attack của client, đóng dấu tổ hợp `proto`/`service`/`conn_state` vắng mặt trong mọi split, rồi gán nhãn normal;
  - F gồm đúng 500 dòng canary đó.
- Model gốc của request canary là ensemble train trên D ∪ C. Retrain đúng nghĩa, tức train trên (D ∪ C) \ C = D, chính là model gốc tự nhiên.
- **Chỉ số canary:** P(normal) trung bình trên các dòng attack của test được đóng dấu trigger (`backdoor_p_normal`).

## 3. Phương pháp

| Tên | Định nghĩa |
|---|---|
| `original` | ensemble train trên D (request tự nhiên) hoặc D ∪ C (request canary) |
| `full_retrain` | train lại **mọi** shard từ khởi tạo gốc trên phần giữ lại (Full Retrain của bài báo) |
| `affected_shard_retrain` | chỉ shard s_u, đủ 15 vòng (Affected-Shard Full Retrain) |
| `rapid_retrain` | chỉ shard s_u, cố định 8/15 vòng (theo mô tả Rapid Retraining trong bài báo: budget cố định đã giảm) |
| `dynamicfu` | chỉ shard s_u, số vòng theo mục 3.1–3.2 |
| `dynamicfu_ffree` | ablation của mục 3.3 |
| `global_original` | FedAvg không shard trên D; cho biết utility mất bao nhiêu vì chia shard |

**Shard bị ảnh hưởng** luôn được retrain từ khởi tạo gốc của nó. Client u chỉ còn R_u = D_u \ F; các client khác và các shard khác giữ nguyên.

**Mọi budget cố định là tiền tố của cùng một quỹ đạo**, vì vòng r dùng đúng seed của vòng r. Budget cố định bằng số vòng trung bình của DynamicFU được đọc từ quỹ đạo này ở bước đánh giá.

### 3.1 Điểm đóng góp của F (phương trình 9–12, áp cho F thay vì cho client)

- **u_F = ‖w⁺ − w*_{s_u}‖₂**, với w⁺ là kết quả của 1 epoch local trên F, xuất phát từ w*_{s_u}, chạy tại client u.
- **d_F = 0,5·JS₂(p_F ‖ p_train) + 0,5·π_F**, với JS tính theo bit và π_F là tỉ lệ attack trong F.
- **q_F = |F1_val(W⁺) − F1_val(W*)|**, với W⁺ là W* có shard s_u thay bằng w⁺.
- **Chuẩn hóa theo pool client.**
  - Tính (u, d, q) thô cho toàn bộ dữ liệu D_i của từng client, một lần cho mỗi model gốc.
  - Chuẩn hóa F bằng min–max của pool đó, rồi clip về [0, 1].
  - I_F = (ū + d̄ + q̄)/3.
- Chi phí tính pool được ghi riêng và không cộng vào từng request.

### 3.2 Số vòng và controller (phương trình 13–21, mục 4.4.3)

- η(0) = η_min + (η_max − η_min)·I_F, với η_min = 0,2 và η_max = 1.
- R(r) = max(1, ⌊15·η(r) + 0,5⌋).
- L_ret(r) là cross-entropy của ensemble trên tập V.
- G(r) = max(0, L_ret(r−1) − L_ret(r)) / L_ret(r−1).
- Δstab(r) = |F1_val(r) − F1_val(r−1)|.
- Ctrl = 0,5·G + 0,5·Δstab.
- Drop(r) = max(0, F1_val(W*) − F1_val(r)).
- η(r) = clip(η(r−1) + ρ·Ctrl − μ·Drop).
- **Dừng** khi có một trong ba điều kiện:
  1. r ≥ R(r) và Ctrl ≤ τ_ctrl;
  2. Drop ≥ τ_drop;
  3. r = 15.
- Model trả về là model ở vòng dừng.

### 3.3 Ablation `dynamicfu_ffree`

Chỉ khác `dynamicfu` ở hai đầu vào phụ thuộc F:
- η(0) = 0,6, ứng với I_F = 0,5;
- Drop so với F1_val tốt nhất đã thấy trong lần retrain này, thay vì F1_val(W*).

Mọi thứ khác dùng chung: V, công thức, điều kiện dừng và ρ, μ, τ.

### 3.4 Chỗ khác bài báo

| Bài báo | Ở đây | Lý do |
|---|---|---|
| Đóng góp của client c_u | đóng góp của F, chuẩn hóa theo pool các client | request mức mẫu |
| V = validation giữ lại của shard | mẫu cố định ≤ 4.000 dòng train giữ lại của shard (seed 42 + 13 + s) | dữ liệu không có validation theo client |
| 50 client, 10 shard, 20 vòng, 2 epoch, lr 1e-2 | 20 client, 5 shard, 15 vòng, 1 epoch, lr 1e-3 | khớp quy mô và cấu hình FedAvg đã kiểm chứng của repo |
| λ, θ, ω, ρ, μ, τ "chọn qua preliminary validation", không công bố | θ, λ, ω chia đều; ρ, μ, τ chọn ở mục 5 | |
| Có FedEraser và MIA | không làm FedEraser; MIA chỉ báo cáo | MIA không có sức mạnh trên model này ([README](README.md), mục 5 và 11) |

## 4. Tính exact và phần rò rỉ

- **Phần dữ liệu là exact.** Không bước gradient nào của model đầu ra chạm vào F.
- **Phần rò rỉ nằm ở số vòng.** Với `dynamicfu`, số vòng R_u phụ thuộc vào F qua I_F và W*. Vì vậy R_u lộ thông tin về F, tối đa log₂15 ≈ 3,9 bit.
- **Được kiểm bằng cách so từng bit:**
  - model `dynamicfu` trùng quỹ đạo retrain exact tại vòng R_u;
  - quỹ đạo đó không phụ thuộc F. Test unit đổi nội dung F thì cho cùng quỹ đạo.
- `dynamicfu_ffree` và mọi budget cố định là **exact tuyệt đối**: đầu ra chỉ là hàm của D \ F (cùng validation và seed).
- **Khóa luận chỉ được viết:** "exact về dữ liệu huấn luyện; số vòng phụ thuộc F". **Không** được viết "certified" hay "DP".

## 5. Chọn tham số controller (stage `tune`, chỉ dùng partition pilot seed 1042)

- **Lưới chọn:** ρ ∈ {0,25; 0,5; 1; 2}, μ ∈ {0,5; 1; 2; 5}, τ_ctrl ∈ {0,002; 0,005; 0,01; 0,02} và τ_drop ∈ {0,01; 0,02; 0,05; 0,1; 1}. τ_drop = 1 tức là tắt điều kiện 2.
- **Dữ liệu dùng để chọn:** 27 request tự nhiên của pilot, model seed 42.
  - Quỹ đạo không phụ thuộc controller, nên quyết định dừng được phát lại từ tín hiệu đã ghi (`replay`).
  - Runner kiểm rằng phát lại trùng với chạy online.
- **Quy tắc chọn:**
  - Điểm lưới **khả thi** nếu, với **cả hai arm**:
    - trung bình qua request của F1_val(15 vòng) − F1_val(vòng dừng) ≤ 0,005;
    - giá trị lớn nhất ≤ 0,02.
  - Trong các điểm khả thi, chọn điểm có trung bình số vòng của hai arm nhỏ nhất. Hòa thì chọn theo thứ tự tăng dần của (ρ, μ, τ_ctrl, τ_drop).
  - **Chỉ dùng validation.** Không điểm nào khả thi thì dừng với `NO_FEASIBLE_CONTROLLER`.

## 6. Các cổng để sang phase DT

Cổng được chấm trên 5 seed của partition chính, bằng `scripts/evaluate_dynamicfu_gates.py`.

**Cổng 1. Exact.** Mọi request ở mọi seed phải pass toàn bộ kiểm tra:
- không dòng nào của F nằm trong worker giữ lại;
- quỹ đạo ở vòng 15 trùng từng bit với `full_retrain` trên s_u;
- mọi shard khác của `full_retrain` trùng shard gốc;
- `dynamicfu` và `dynamicfu_ffree` trùng quỹ đạo tại vòng dừng của chúng;
- tín hiệu online trùng tín hiệu của quỹ đạo, và phát lại cho cùng vòng dừng;
- prediction của `full_retrain` trùng prediction ở vòng 15;
- với request canary, `full_retrain` trùng model gốc tự nhiên.

Ngoài ra: cùng tập request, cùng cấu hình và controller ở mọi seed, đủ 5 seed, verifier pass.

**Cổng 2. Có tác dụng đo được, bằng canary.**
- Với mỗi request canary:
  - đám mây **retrain** là `backdoor_p_normal` của `full_retrain` qua 5 seed;
  - đám mây **original** là chỉ số đó của model gốc có canary.
- Request được **admit** khi hai khoảng dự đoán 95% không giao nhau (`classify` của `src/eval/seed_floor.py`).
- Request admit **pass** khi giá trị trung bình qua 5 seed của `dynamicfu` nằm trong khoảng dự đoán 95% của retrain (verdict `RETRAIN_LIKE`).
- **Cổng 2 pass** khi có ≥ 2 request admit và mọi request admit đều pass.
- Request tự nhiên **không** dùng cho cổng này. Mức đo được của chúng chỉ được báo cáo, với sàn nhiễu seed theo `distance_profile`.

**Cổng 3. Utility.** Với **mọi** request, cả tự nhiên lẫn canary, tính trung bình qua 5 seed:
- F1_test(`full_retrain`) − F1_test(`dynamicfu`) ≤ 0,01;
- FPR_test(`dynamicfu`) − FPR_test(`full_retrain`) ≤ 0,01.

**Cổng 4. Chi phí.** Chỉ báo cáo, không chặn: speedup theo số mẫu xử lý và theo wall time so với `full_retrain`.

**Kết luận:** được sang phase DT khi và chỉ khi **cổng 1, 2 và 3 cùng pass**.

**Báo cáo thêm, không chấm:**
- `dynamicfu` so với `dynamicfu_ffree`, `rapid_retrain` và budget cố định bằng số vòng trung bình của DynamicFU theo từng α. So sánh này cho biết I_F có giúp gì không. DynamicFU thua cũng không chặn DT, vì phương pháp vẫn exact về dữ liệu.
- MIA loss-based trên F của request tự nhiên.
- Utility mất bao nhiêu vì chia shard, đo bằng `global_original`.

## 7. Bàn giao cho phase DT (chốt khi sang DT)

- Mỗi request chỉ huy động các client của shard s_u.
- DynamicFU cung cấp R_u, và mỗi vòng ghi `compute_seconds`, `payload_bytes` của từng client.
- DT dự báo chi phí của R_u vòng, rồi kiểm tra ngân sách. Nếu vượt ngân sách, DT có thể bớt client trong shard (vẫn không dùng F), giảm số vòng, hoặc trả `NO_FEASIBLE_PLAN`.

## 8. Lệnh

```powershell
$env:FL_DEVICE='cpu'
python -m unittest discover -s tests -p "test_dynamicfu_sample.py" -v
python scripts/run_dynamicfu_sample.py requests --partition-seed 1042 --output runs/dynamicfu_requests_p1042
python scripts/run_dynamicfu_sample.py tune --requests runs/dynamicfu_requests_p1042 --output runs/dynamicfu_tuning_p1042
python scripts/run_dynamicfu_sample.py requests --partition-seed 42 --output runs/dynamicfu_requests_p42
# một lệnh cho mỗi seed 42..46; các seed chạy song song được
python scripts/run_dynamicfu_sample.py run --requests runs/dynamicfu_requests_p42 --controller runs/dynamicfu_tuning_p1042/selected.json --seed 42 --output runs/dynamicfu_p42/seed42
python scripts/evaluate_dynamicfu_gates.py --runs runs/dynamicfu_p42/seed42 runs/dynamicfu_p42/seed43 runs/dynamicfu_p42/seed44 runs/dynamicfu_p42/seed45 runs/dynamicfu_p42/seed46 --output runs/dynamicfu_p42/gates
python scripts/verify_dynamicfu_sample.py --run runs/dynamicfu_p42/seed42
```

## 9. Giới hạn đã biết trước khi chạy

- **Ensemble 5 shard có thể pha loãng canary.** Khi đó request canary không được admit. Số canary đã đặt trước là 500 và không được chỉnh sau khi xem kết quả.
- **Request được chọn trên model seed 42,** và seed này cũng nằm trong 5 seed đánh giá.
- **Request phân tầng giữ nguyên tỉ lệ lớp của client,** nên d_F gần như bằng d của client. Phần phân biệt giữa các request đến chủ yếu từ u_F và q_F.
- **Wall time là mô phỏng tuần tự với logical worker.** Các seed chạy song song trên cùng một máy, nên wall time có nhiễu do tranh CPU. Speedup theo số mẫu không bị ảnh hưởng.
- **Chưa đo:** chi phí lưu 5 model shard, và thời gian khi các shard chạy song song.
