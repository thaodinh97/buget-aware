# Hướng shard, protocol v2: gate noninferiority, ràng buộc số client mỗi shard, sửa `utility_stop`

**Trạng thái:** viết **trước khi chạy** v2, sau khi đã xem kết quả v1 ([shard_standard_config_plan.md](shard_standard_config_plan.md), [shard_standard_config_results.md](shard_standard_config_results.md): **không xác nhận**).

**File:**
- Runner: `scripts/shard/run_shard_ensemble_check_v2.py` (run 1) và `scripts/shard/run_shard_stream_v2.py` (run 2).
- Phần dùng chung: `scripts/shard/shard_v2.py`. Đánh giá: `scripts/shard/evaluate_shard_v2.py`. Unit test: `tests/test_shard_v2.py`.
- File của v1 **giữ nguyên từng byte**, vì kết quả v1 gắn với hash của chúng.

**Bằng chứng về thứ tự.**
- Lúc bắt đầu, mỗi run ghi vào `run.json`:
  - SHA-256 của 6 file: tài liệu này, script đánh giá, `shard_v2.py`, `train_config.py` và hai runner. Hash tính sau khi đổi CRLF thành LF;
  - hash nội dung dữ liệu.
- Với thiết kế `v2`, runner dừng nếu thiếu một trong 6 file, hoặc nếu hash nội dung dữ liệu không khớp.
- Script đánh giá so lại toàn bộ các hash. Nếu không khớp thì kết quả **không được tính là xác nhận**.
- Mọi thay đổi sau khi chạy phải ghi vào mục "Thay đổi sau khi chạy" ở cuối.

## Thay đổi so với v1
| # | Vấn đề ở v1 | Thay đổi ở v2 |
|---|---|---|
| 1 | **Script đánh giá:** chỉ so tên thư mục dữ liệu. Hash manifest phụ thuộc đường dẫn và platform. Đường dẫn runner lấy từ `sys.argv`, nên có thể không được ghi nếu ký tự ổ đĩa khác hoa / thường (`e:` và `E:`); file thiếu thì bị bỏ qua mà không báo | Kiểm tra **hash nội dung** các mảng thực sự được dùng, cố định trước. Đường dẫn được chuẩn hóa (`norm_rel`). Runner phải nằm trong danh sách hash. Thiếu file thì báo lỗi |
| 2 | **Gate:** so trung bình điểm với biên mà không tính bất định. G5 trộn hai câu hỏi khác nhau | **Noninferiority một phía có khoảng tin cậy.** G5 tách thành **G5a** (so với `shard_full`) và **G5b** (so với model đơn). Thêm **G6** kiểm tra độ an toàn của `utility_stop` |
| 3 | **Thiết kế:** S = 10 cho shard 2 client; trong chuỗi, shard teo còn 1 client lệch, rồi rỗng | **Mỗi shard ≥ 4 client lúc thiết kế.** Trong chuỗi, shard còn **< 3 client thì bị gộp** |
| 4 | **`utility_stop`:** dừng ở R_base dù chưa đạt điều kiện; loại shard rỗng mà không kiểm tra utility | **Có phương án dự phòng.** Utility được kiểm tra sau **mọi** yêu cầu, kể cả khi loại hoặc gộp shard |
| 5 | 5 seed, đã xem kết quả | **10 seed mới:** 50042–50051 |

## Thiết kế
**Giữ nguyên từ v1:**
- Dữ liệu `data/processed/connected_v3_auxfit_seed42`.
  - Hash nội dung (train, validation, calibration; `shard_v2.data_content_sha256`) phải là `f130057e20f1d9eb0857dac5b8ce90c9f0f16b544a8d4b5675b88f24eea0ef3a`.
  - Các file npz được ghi lúc 12:00:53 ngày 2026-11-07, trước khi v1 bắt đầu, và chưa đổi từ đó. Vì vậy v1 cũng chạy trên đúng nội dung này.
- Cấu hình `standard`: Adam lr 1e-2, R_base = 20 vòng, 2 epoch local, batch 512, `shard_fixed` 10 vòng, backfill 7, R_max của DynamicFU 20.
- Partition 20 client (Dirichlet α = 0,5, seed 42) và chuỗi 10 yêu cầu (seed 99), xóa 60,5% dữ liệu train.
- Ensemble là tổng margin. Chính sách đọc validation, metric báo cáo trên calibration, test không được dùng.

**Mới:**
- **Seed train:** 50042–50051, chưa từng chạy.
- **Ràng buộc lúc thiết kế:** mỗi shard ≥ 4 client, runner có kiểm tra.
  - Run 1: S ∈ {1, 2, 4, 5}. Run 2: S ∈ {4, 5}.
  - Shard được gán bằng hoán vị có seed 7, như v1.
- **Run 1:** request trên client 10 (lớn nhất); shard của nó còn 4 client (S = 4) và 3 client (S = 5) sau khi xóa, nên không cần gộp. Runner có kiểm tra điều này.
  - Worker được đánh số theo client id, giống run 2. v1 đánh số theo vị trí trong shard.
- **Quy tắc gộp trong chuỗi** (`shard_v2.structure_plan`):
  - Khi một yêu cầu xóa cả client làm shard j còn từ 1 đến 2 client, các client còn lại của j được gộp vào shard đang hoạt động có ít client nhất (bằng nhau thì lấy chỉ số nhỏ nhất).
  - Shard gộp giữ chỉ số và seed của shard đích, và được train lại từ khởi tạo mới. Shard j bị loại.
  - Một shard không còn client nào thì bị loại.
  - Kế hoạch này chỉ phụ thuộc vào chuỗi yêu cầu (metadata), nên giống nhau với mọi chính sách và mọi seed. Nó được ghi vào `structure.json`.
- **Cấu trúc tính trước từ metadata:**
  - S = 4: không gộp; cuối chuỗi mỗi shard còn ít nhất 3 client.
  - S = 5: ở **bước 7**, shard 1 chỉ còn [8, 17], nên được gộp vào shard 0 ([0, 16, 19]) thành [0, 8, 16, 17, 19]. Ở bước 9, shard 0 mất client 17 và được train lại.
  - Trong cả hai trường hợp không shard nào còn dưới 3 client.

**Chính sách** (shard cần train lại là shard bị ảnh hưởng hoặc shard gộp; luôn train từ khởi tạo mới với seed của nó, chỉ trên dữ liệu hiện tại):

| Chính sách | Mô tả |
|---|---|
| `none` | không unlearn; giữ nguyên các shard ban đầu, không gộp; chỉ để tham chiếu |
| `shard_full`, `shard_fixed`, `dynamicfu`, `drop_backfill` | như v1, trên cấu trúc v2 |
| `utility_stop` | **quy tắc v2**, xem bên dưới |
| `utility_stop_v1` | quy tắc v1: train từng vòng, dừng ở vòng đầu tiên đạt ok(), nếu không thì dừng ở R_base; không làm gì thêm. **Chỉ để mô tả**, không thuộc gate nào |
| `full_retrain_single` | tham chiếu: một model đơn train lại từ đầu trên toàn bộ dữ liệu còn lại sau mỗi yêu cầu |

**Quy tắc `utility_stop` v2.** ok() nghĩa là macro-F1 trên validation ≥ F1₀ − 0,005 và FPR trên validation ≤ FPR₀ + 0,005, với F1₀ và FPR₀ là của ensemble ban đầu.
1. Nếu có shard cần train lại: train từng vòng, dừng ở vòng đầu tiên mà ensemble đạt ok(), tối đa R_base vòng.
2. **Sau mọi yêu cầu**, kể cả khi chỉ loại hoặc gộp shard mà không có gì khác để train: nếu ensemble chưa đạt ok() thì chạy **dự phòng**:
   - cho thêm 1 vòng cho shard đang dùng có ít vòng nhất (dưới R_base; bằng nhau thì lấy chỉ số nhỏ nhất), **tiếp tục đúng chuỗi train có seed của shard đó**;
   - kiểm tra lại; lặp tới khi đạt ok(), hoặc mọi shard đang dùng đều đã đủ R_base vòng.
3. Ghi lại `phase1_rounds`, `fallback_rounds` và `condition_met` (ok() ở cuối bước).

**Tính chất của quy tắc v2:**
- Train tiếp một shard trên dữ liệu không đổi cho ra đúng model như train mới với cùng seed và cùng tổng số vòng. Vì vậy unlearn vẫn chính xác (kiểm tra ở G3).
- Ở cuối mỗi bước, hoặc ensemble đạt ok(), hoặc mọi shard đều có R_base vòng. Trường hợp sau trùng từng bit với trạng thái của `shard_full`.

**Kiểm tra tính chính xác** (G3):
- **Sau mỗi bước:** không shard nào đang dùng từng train trên dòng đã xóa (`exact`); mỗi shard đang dùng được train trên **đúng** dữ liệu hiện tại của nó (`structure_ok`).
- **Cuối chuỗi:** mỗi shard đang dùng trùng từng bit với model train mới (cùng seed, cùng số vòng) trên dữ liệu cuối của nó; tập shard đang dùng đúng bằng tập shard còn hoạt động.

## Phần dữ liệu đã được nhìn
- **Partition, chuỗi yêu cầu và cấu hình** giống v1, và kết quả v1 trên 5 seed (10042–10046) đã được xem.
- **Các thay đổi 3 và 4 được thiết kế sau khi thấy các lỗi của v1 trên chính partition và chuỗi này.** v2 kiểm tra các thay đổi đó trên cùng kịch bản với seed mới, **không** phải trên partition mới.
- **Smoke run** dùng thiết kế `smoke` và `smoke_empty`, seed 7 và 8 (không thuộc v2), chỉ 3 vòng. Mục đích là kiểm tra code chạy đúng, kết quả không được dùng.
  - Với 3 vòng, model rơi vào chế độ thiếu train, nên các con số utility của smoke không có ý nghĩa.
  - G3 pass trên smoke, kể cả khi có gộp shard, có vòng dự phòng và có loại shard rỗng.

## Gate (cố định trước khi chạy)
**Kiểm định.** Hiệu ghép cặp theo seed d = ứng viên − tham chiếu, n = 10 seed, t ghép cặp một phía với α = 0,05 (`shard_v2.noninferior`):
- macro-F1 **noninferior** khi mean(d) − t₀.₉₅,₉ · se(d) > −δ_F1;
- FPR noninferior khi mean(d) + t₀.₉₅,₉ · se(d) < δ_FPR;
- một so sánh pass khi cả hai pass.

Hằng số: δ_F1 = 0,01; δ_FPR = 0,02; chế độ FPR cao là FPR > 0,45 trên calibration; t₀.₉₅,₉ = 1,833.

Một gate pass khi **mọi** so sánh và kiểm tra trong nó đều pass (kiểm định giao–hợp), nên không cần hiệu chỉnh đa so sánh trong một gate.

| Gate | Run | Điều kiện pass |
|---|---|---|
| **G1** Utility của ensemble | 1 | S ∈ {2, 4, 5} noninferior so với S = 1. Không có (S, seed) nào, kể cả S = 1, có FPR > 0,45 |
| **G2** Unlearn một request | 1 | Ở R = R_base, S ∈ {2, 4, 5} × {`client`, `random50`} noninferior so với S = 1 cùng request (S = 1 chính là train lại toàn bộ trên D \ F). Không dòng nào ở R_base có FPR > 0,45 |
| **G3** Tính chính xác | 2 | Mọi chính sách trừ `none` (kể cả `utility_stop_v1`): `exact` và `structure_ok` ở mọi bước; cuối chuỗi trùng từng bit, tập shard đang dùng bằng tập shard hoạt động, `structure_ok`; dòng sau khi xong hàng đợi của `drop_backfill` cũng đúng; đủ mọi tổ hợp |
| **G4** Tham chiếu khỏe | 2 | `full_retrain_single` có FPR ≤ 0,45 ở mọi bước và mọi seed |
| **G5a** So với train lại đủ shard | 2 | S ∈ {4, 5}; `shard_fixed`, `dynamicfu`, `utility_stop`, `drop_backfill` (sau khi xong hàng đợi): trạng thái cuối noninferior so với trạng thái cuối của `shard_full` cùng S, cùng seed. Không seed nào có FPR cuối > 0,45 |
| **G5b** So với model đơn | 2 | S ∈ {4, 5}; `shard_full`, `shard_fixed`, `dynamicfu`, `utility_stop`, `drop_backfill`: trạng thái cuối noninferior so với `full_retrain_single` ở bước 10. Không seed nào có FPR cuối > 0,45 |
| **G6** An toàn của `utility_stop` | 2 | `utility_stop` (v2) có FPR ≤ 0,45 ở **mọi bước** 1–10, mọi S và mọi seed |

**Điều kiện tiên quyết:** G4 pass; tính toàn vẹn đạt (thiết kế `v2`, cấu hình `standard`, đúng 10 seed, đúng hash nội dung dữ liệu, mọi hash khớp).

**Kết luận theo từng claim**, mỗi claim chỉ có giá trị khi điều kiện tiên quyết đạt:

| Claim | Gate |
|---|---|
| Chia shard (≥ 4 client, gộp khi < 3) giữ được utility của model đơn | G1, G2, G5b |
| Các chính sách train ít vòng hơn ngang `shard_full` | G5a |
| Unlearn chính xác | G3 |
| `utility_stop` v2 an toàn | G6, cộng với các dòng `utility_stop` của G5a và G5b |

"Xác nhận toàn bộ" khi mọi gate pass. Nếu một gate fail thì báo cáo nguyên trạng; không đổi biên, seed hay quy tắc sau khi xem kết quả.

## Lý do chọn biên
- **Giữ nguyên biên của v1** (δ_F1 = 0,01, δ_FPR = 0,02), vì các biên đó được đặt **trước khi có kết quả v1**.
  - Ở v1, nhiều so sánh vượt biên rất sát (−0,0104 đến −0,0114).
  - **Nới biên bây giờ sẽ là chỉnh biên theo dữ liệu đã thấy**, nên không làm.
- **Ý nghĩa của δ_F1 = 0,01:**
  - khoảng 20 lần độ lệch chuẩn theo seed của macro-F1 model đơn (0,0005 ở v1). Đây là mức mất **chấp nhận được**, không phải sàn nhiễu;
  - khoảng 1/22 mức mất ở chế độ hỏng (0,22);
  - nhỏ hơn mức cải thiện khi đổi sang cấu hình chuẩn (+0,016 macro-F1 ở S = 1), là mức mà ta đã coi là đủ đáng để đổi cấu hình.
- **Ý nghĩa của δ_FPR = 0,02:**
  - trên calibration có 4.313 dòng normal, 0,02 tương đương khoảng 86 cảnh báo sai thêm;
  - khoảng 6% tương đối so với FPR nền 0,32;
  - khoảng 1/21 mức tăng ở chế độ hỏng (+0,43).
- **Giới hạn:** các biên này không được suy ra từ một yêu cầu vận hành của IDS. Nếu có yêu cầu như vậy thì phải thay biên trong một protocol mới.
- **Số seed:** ở v1, độ lệch chuẩn của hiệu ghép cặp lên tới 0,0074. Với n = 10, nửa độ rộng của cận một phía khoảng 1,833 × 0,0074 / √10 ≈ 0,0043.

**Dự kiến từ v1** (đã thấy; ghi ra để minh bạch, không dùng để điều chỉnh gì):
- **G5b có thể fail.** Ở v1, với S = 4, các chính sách thấp hơn model đơn từ −0,004 đến −0,0095 (độ lệch chuẩn 0,006–0,007). Nếu lặp lại, cận dưới với n = 10 sẽ vào khoảng −0,008 đến −0,0135.
- **G5a có thể pass.** Ở v1, `shard_fixed` và `dynamicfu` so với `shard_full` chỉ khoảng −0,0015 (độ lệch chuẩn 0,0013).

## Không được tuyên bố, kể cả khi pass
| Không được tuyên bố | Lý do |
|---|---|
| Kết quả đúng trên partition khác hoặc chuỗi yêu cầu khác | v2 dùng lại partition 42 và chuỗi seed 99; các thay đổi được thiết kế từ lỗi trên chính kịch bản này |
| `utility_stop` v2 rẻ hơn nói chung | Chi phí của phần dự phòng chỉ được đo trên chuỗi này; chỉ là kết quả mô tả |
| Kết quả trên test | Test không được dùng |
| Kết luận về DynamicFU nói chung | Vẫn là bản cài đặt lại |

## Kết quả mô tả (không phải gate)
- Chi phí (dòng-lượt, byte, thời gian) của từng chính sách.
- Số bước có FPR > 0,45 của `utility_stop_v1`, để so với v2.
- Số vòng dự phòng và số bước không đạt ok() của `utility_stop`.
- Các sự kiện gộp shard; lý do dừng của DynamicFU.

## Lệnh chạy
```bash
python src/data/prepare.py --input dataset/train_test_network.csv --config configs/data_connected_auxfit.json --output data/processed/connected_v3_auxfit_seed42
```
```bash
export SHARD_DESIGN=v2 FL_DEVICE=cpu
```
```bash
python scripts/shard/run_shard_ensemble_check_v2.py runs/shard_v2/ensemble_check_20c
```
```bash
python scripts/shard/run_shard_stream_v2.py runs/shard_v2/stream_20c
```
```bash
python scripts/shard/evaluate_shard_v2.py runs/shard_v2/ensemble_check_20c runs/shard_v2/stream_20c runs/shard_v2/gates.json
```
```bash
python scripts/shard/analyze_shard_20c.py runs/shard_v2/ensemble_check_20c runs/shard_v2/stream_20c
```
```bash
python -m unittest discover -s tests -p "test_shard_v2.py" -v
```
Kết quả được chép sang `results/shard_v2/`, và viết trong `docs/shard_v2_results.md`. Thời gian dự kiến trên CPU: run 1 khoảng 25 phút, run 2 khoảng 1 giờ; hai run chạy song song được.

## Thay đổi sau khi chạy
(chưa có)
