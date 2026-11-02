# Đo utility chi tiết và chạy xác nhận phương pháp đã đóng băng — protocol

Runner là `scripts/run_confirmation.py`, verifier là `scripts/verify_confirmation.py`. Tài liệu này được viết **trước mọi lần chạy trên dữ liệu thật**. Mọi thay đổi sau smoke run phải được ghi thêm vào mục "Thay đổi sau smoke" ở cuối.

## Mục đích
Bản review của [deployable_forget_results.md](deployable_forget_results.md) chỉ ra năm lỗ hổng trong bằng chứng hiện có. Mỗi lỗ hổng tương ứng với một phép đo:

| # | Lỗ hổng | Phép đo |
|---|---|---|
| 1 | Utility chỉ được đo bằng macro-F1 toàn cục, mà theo seed floor thì metric này không phân biệt được original với retrain | FPR, recall attack và F1 từng lớp trên test; macro-F1, FPR và recall trên dữ liệu giữ lại của **từng client** |
| 2 | Control cùng compute fine-tune trên R_target, trong khi reference được train trên R_target ∪ aux, nên control bị suy biến và không cùng quyền truy cập dữ liệu | Thêm control **pool**: Adam trên CE(R_target ∪ validation), cùng số bước |
| 3 | Trigger canary nằm trên các chiều bằng 0 ở mọi dữ liệu, nên chỉ cần reset các cột đó là có thể đạt kết quả như retrain | Thêm **baseline reset** cột W1 của các chiều không được dùng |
| 4 | Projection thắng trên canary có thể chỉ nhờ shortcut ở mục 3 | Thêm **canary có trigger tổ hợp**, mọi cột của trigger đều đang hoạt động trong dữ liệu |
| 5 | Phương pháp được thiết kế sau khi xem smoke trên chính các seed dùng để đánh giá; request tự nhiên chỉ có trên client 4 | **Chạy xác nhận** với partition mới, seed mới và request tự nhiên trên mọi client |

## Hai chế độ
| Chế độ | Partition / seed | Request | Cloud |
|---|---|---|---|
| `existing` | partition cũ (seed 42); seed 10042–10051 | 5 request của `forget_signal_seed42`, thêm `combo_canary_c{3,0,4}_500` | lấy lại từ `forget_signal`, `identifiable_fu` và `seed_floor` (kiểm tra hash); riêng original của canary tổ hợp được train trong run này |
| `confirm` | partition **mới**: Dirichlet seed 43, α = 0,5, 5 client; seed **20042–20051** | canary zero-dim và canary tổ hợp trên tối đa 3 client có ≥ 2.000 dòng attack (xếp theo số dòng attack giảm dần); `random_stratified` và `low_overlap_groups` 50% trên **mọi** client | toàn bộ train trong run này |

- **Vai trò của `existing`.** Chế độ này bổ sung phép đo 1–4 cho các kết quả đã báo cáo. Đây **không** phải xác nhận, vì nó dùng lại đúng các seed cũ.
- **Cross-check.** Các arm đóng băng và control R_target phải tái tạo `deployable_forget_seed42` (recovery chuẩn) với sai khác ≤ 1e-9. Kết quả được ghi trong `status.json → deployable_crosscheck`.
- **Vai trò của `confirm`.** Chỉ chế độ này mới là bằng chứng xác nhận.
- **Chọn request.** Danh sách request được tạo hoàn toàn từ metadata, trước khi train bất kỳ model nào.
- **Request không hợp lệ.** Một request tự nhiên có F hoặc R chỉ còn một lớp sẽ được ghi vào `requests.json → invalid`. Runner không lấy mẫu lại.
- **Gate.** Request vẫn phải qua gate cũ: khoảng dự đoán 95% của cloud original và cloud retrain không được chồng nhau.

## Arm (đóng băng, không chọn lọc)
| Arm | Vai trò |
|---|---|
| `ga_proj@joint_aux_prob` | **phương pháp được khuyến nghị**, giữ nguyên cấu hình của `deployable_forget` |
| `nkd_proj@joint_aux_prob` | biến thể KD có projection |
| `ga_adam@joint_aux_prob` | GA không projection, để đo phần đóng góp của projection |
| `ga_proj@retrain_oracle` | cận trên (oracle), **không triển khai được**, không được dùng làm kết luận |
| `reset_init@local` | reset về giá trị init các cột W1 của những chiều bằng 0 trên toàn bộ R_target |
| `reset_init@global` | như trên, nhưng tính các chiều bằng 0 trên toàn bộ D \ F. Cách này cần mỗi client gửi 1 bit cho mỗi chiều ("có lúc khác 0"), không lộ dòng dữ liệu nào |
| `reset_zero@global` | đặt về 0 thay vì giá trị init |

Mọi arm đều được theo sau bởi cùng 2 vòng FedAvg recovery (seed s + 900000). Các arm reset có 0 bước backward, nên control tương ứng của chúng là retain-only.

**Lý do baseline reset luôn đạt mức retrain trên canary zero-dim.** Với Adam và không có weight decay, một trọng số lớp 1 có input bằng 0 trên mọi dòng sẽ có gradient bằng 0, moment bằng 0, và do đó không bao giờ được cập nhật. Retrain trên D \ F vì vậy giữ các cột đó đúng bằng giá trị init. Unit test `test_reset_reproduces_retrain_columns_of_unused_dims` kiểm tra điều này, khớp chính xác.

## Control
| Kind | Mô tả |
|---|---|
| `retain_only` | 0 bước local, rồi recovery chuẩn |
| `target` | Adam trên CE(R_target) với `control_budget(backward_samples)` bước, giống hệt `deployable_forget`. Giữ lại để so với quy tắc cũ |
| `pool` | **mới**: Adam trên CE(R_target ∪ validation), cùng số bước. Đây là pool mà reference `ref_aux` được train trên đó, nên arm và control có cùng quyền truy cập dữ liệu |

## Canary có trigger tổ hợp (`src/data/canary.py`)
- **Cách chọn trigger.**
  - Trigger là một tổ hợp giá trị hợp lệ của `proto`, `service` và `conn_state` mà **không xuất hiện ở bất kỳ split nào** (train, validation, calibration, test).
  - Trong số các tổ hợp vắng mặt, chọn tổ hợp có tích tần suất trên train của các cột lớn nhất; nếu bằng nhau thì xét theo chỉ số cột.
  - Đây là quy tắc chỉ dùng metadata, chạy trước khi train.
- **Hệ quả.** Mỗi cột mà trigger bật lên đều phổ biến trong dữ liệu giữ lại, nên trigger nằm **trong** span input của R. Riêng việc reset cột hay chiếu ra khỏi các chiều không dùng không thể xóa được trigger này.
- **Cách đóng dấu.** Khối one-hot của từng feature trigger được ghi đè bằng giá trị trigger; các feature số được giữ nguyên.
- **Cặp so sánh.** Các dòng nguồn trùng đúng với canary zero-dim cùng client và cùng seed (runner có assert kiểm tra). Như vậy hai loại canary chỉ khác nhau ở trigger.
- **Metric chính:** P(normal) trung bình trên các dòng attack của test được đóng dấu trigger tổ hợp.

## Utility (`src/eval/utility_breakdown.py`)
- **Trên test:** `test_macro_f1`, `test_f1_normal`, `test_f1_attack`, `test_fpr`, `test_recall_attack`.
- **Theo client:** `client{j}_{macro_f1, fpr, recall_attack, f1_normal, f1_attack}` trên dữ liệu **giữ lại** của client j sau request.
  - Đây là số liệu in-sample, vì partition chỉ chia trên train. Nó cho biết client nào bị hại, điều mà test gộp không cho thấy.
  - Giá trị không xác định (FPR khi không có dòng normal, recall khi không có dòng attack) là NaN, không phải 0. Check tương ứng được liệt kê trong `utility_checks_undefined` và không tính vào quy tắc.

## Quy tắc quyết định (`src/eval/confirmation_decision.py`, cố định trước khi chạy)
Áp dụng cho mỗi request được admit và mỗi arm, ở giai đoạn `post` (sau recovery):

1. **forget_ok** khi thỏa một trong hai điều kiện:
   - (a) arm **superior** so với cả `retain_only` lẫn control `pool`, theo nghĩa cận dưới 95% của gain ghép cặp lớn hơn sàn nhiễu của cloud retrain, **và** `local_effective` (MAE(F) giảm trước recovery);
   - (b) bản thân `retain_only` đã đạt mức retrain (cận trên 95% của |retain_only − μ_R| ≤ sàn nhiễu) **và** arm không xa retrain hơn retain-only quá sàn nhiễu. Nhánh này dành cho request không cần bước quên, như request random cũ.
2. **utility_ok:** noninferior so với `retain_only` (cận 95% của hiệu ghép cặp nằm trong sàn nhiễu của metric đó trên cloud retrain), xét trên:
   - `test_macro_f1`;
   - `test_fpr` (theo chiều tăng FPR);
   - `client{j}_macro_f1` của mọi client có giá trị xác định.
3. **pass = forget_ok ∧ utility_ok.**
4. **Arm vượt qua tổng thể** khi pass trên **mọi** request được admit và các request đó trải trên ≥ 2 client mục tiêu.

**Quy tắc cũ.** `legacy_pass` (superior và F1-noninferior so với control `target`, cùng `local_effective`) được báo cáo song song để so với `forget_signal` và `deployable_forget`, nhưng không dùng để kết luận.

## Cách đọc kết quả (diễn giải định trước)
| Câu hỏi | Nơi đọc | Diễn giải |
|---|---|---|
| Kết quả trên canary zero-dim có chỉ là shortcut? | `decision.csv`: `fu_position` của `reset_*` so với `ga_proj@joint_aux_prob` trên `canary_*` | Nếu reset đạt ≥ position của `ga_proj` thì kết quả canary zero-dim **không** được dùng làm bằng chứng cho projection |
| Projection có hoạt động khi trigger nằm trong span của R? | `combo_canary_*`: `ga_proj` so với `ga_adam`, retain-only và reset | Đây là bằng chứng canary chính cho phương pháp. Nếu canary tổ hợp không qua gate thì ghi là "không xác định được", không thay trigger |
| Control công bằng có làm thay đổi kết luận? | `superior_to_pool_control`, `pool_control_position` | — |
| Utility có bị hại ở đâu đó? | `test_fpr_*`, `worst_client`, `worst_client_f1_diff`, `client{j}_macro_f1_*` | — |
| Kết quả có lặp lại trên partition và seed mới? | verdict của chế độ `confirm` | Chỉ `confirm` được gọi là "xác nhận" |

## Lệnh chạy (PowerShell, từ thư mục gốc repo)
```powershell
$env:FL_DEVICE="cpu"
# smoke (không dùng để kết luận; tham số được ghi trong manifest)
python scripts/run_confirmation.py --mode existing --seeds 3 --requests canary_c3_500 combo_canary_c3_500 --output runs/confirmation_existing_smoke3
# đầy đủ
python scripts/run_confirmation.py --mode existing --output runs/confirmation_existing_seed42
python scripts/verify_confirmation.py --run runs/confirmation_existing_seed42
python scripts/run_confirmation.py --mode confirm --output runs/confirmation_p43
python scripts/verify_confirmation.py --run runs/confirmation_p43
```
- **Điều kiện cho `existing`:** cần các run `controlled_phase_v2_seed42`, `seed_floor_seed42`, `identifiable_fu_seed42` và `forget_signal_seed42`. `deployable_forget_seed42` là tùy chọn, chỉ dùng cho cross-check.
- **Điều kiện cho `confirm`:** chỉ cần dữ liệu và `controlled_phase_v2_seed42` (để nạp cấu hình).
- **Thời gian (ước tính, chưa đo):**
  - `existing`: khoảng 1 giờ trên CPU, trong đó train 30 original cho canary tổ hợp.
  - `confirm`: khoảng 1–2 giờ, trong đó train tới 170 model cloud.

## Output
`clouds.csv`, `fu_runs.csv`, `controls.csv` (cột `kind`), `decision.csv`, `verdict.json`, `gate.json`, `requests.json`, `request_state.csv` (các chiều bằng 0 bị reset, thời gian kNN), `status.json`, `checkpoints/`. Chế độ `confirm` có thêm `partitions.npz` và `clients.json`.

## Giới hạn đã biết trước
- Canary tổ hợp vẫn là backdoor tổng hợp với một trigger duy nhất. Nó có thể học kém hơn canary zero-dim, và khi đó gate sẽ loại request.
- Utility theo client là in-sample.
- `confirm` chỉ đổi seed partition, còn α giữ ở 0,5. α = 0,1 là một protocol khác và phải được khai báo trước khi chạy.
- Đây vẫn là bằng chứng thực nghiệm về việc giảm ảnh hưởng, không phải đảm bảo xóa dữ liệu. Không có MIA trong run này.

## Thay đổi sau smoke
(chưa có)
