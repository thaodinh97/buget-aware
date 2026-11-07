# Cấu hình train chuẩn cho hướng shard và xác nhận hai run 20 client — protocol

**Trạng thái:** viết **trước khi chạy** xác nhận. Runner là `scripts/shard/run_shard_ensemble_check.py` (run 1) và `scripts/shard/run_shard_stream.py` (run 2) với cấu hình `standard`. Đánh giá bằng `scripts/shard/evaluate_standard_config.py`; ngưỡng và quy tắc kết luận cố định trong file đó và trong tài liệu này.

**Bằng chứng về thứ tự.** Lúc bắt đầu, mỗi run ghi vào `train_config.json` SHA-256 của tài liệu này, của script đánh giá, của `scripts/shard/train_config.py` và của chính runner. Hash tính sau khi đổi CRLF thành LF, nên không đổi theo cách git chuyển xuống dòng. Script đánh giá so các hash này với file hiện tại. Nếu không khớp thì kết quả **không được tính là xác nhận**. Mọi thay đổi sau khi chạy phải ghi vào mục "Thay đổi sau khi chạy" ở cuối.

## Bối cảnh
Các con số dưới đây lấy từ [shard_exploration_results.md](shard_exploration_results.md).
- **Trên dữ liệu auxfit, với cấu hình FL của repo** (Adam 1e-3, 15 vòng, 1 epoch), run 1 và run 2 rơi vào một chế độ FPR cao:
  - ensemble S ≥ 4: macro-F1 0,645, FPR 0,75;
  - model đơn train lại sau khi xóa client lớn nhất: 0,646 / 0,751;
  - trong chuỗi yêu cầu, model đơn train lại toàn bộ: 0,634 / 0,772.
- **Chẩn đoán hậu kiểm** (`scripts/shard/diagnose_training_budget.py`, 1 seed, đo trên calibration): nguyên nhân là từng shard chưa được train đủ. Train nhiều vòng hơn, hoặc dùng lr 1e-2, đưa model về 0,868–0,875 / 0,32.
- Tài liệu đó kết luận: muốn đổi cấu hình train chuẩn cho hướng shard thì phải ghi vào protocol trước khi chạy xác nhận. Tài liệu này là protocol đó.

## Thay đổi: cấu hình train chuẩn của hướng shard
| | Cũ (`repo`) | **Chuẩn mới (`standard`)** |
|---|---|---|
| Optimizer, lr | Adam 1e-3 | **Adam 1e-2** |
| R_base (số vòng FedAvg của mỗi shard) | 15 | **20** |
| Số epoch local | 1 | **2** |
| Batch | 512 | 512 |
| Các mức R dùng để đo utility khi unlearn (run 1) | 1, 2, 5, 10, 15 | 1, 2, 5, 10, 20 |
| `shard_fixed` (run 2) | 7 vòng | 10 vòng |
| `drop_backfill`, số vòng mỗi bước (run 2) | 5 | 7 |
| R_max của DynamicFU | 15 | 20 |

**Lý do chọn.**
- Đây là cấu hình của bài DynamicFU, đã dùng ở run 3 (50 client). Nó được cố định từ nguồn bên ngoài, **không chọn theo kết quả**.
- Trong bốn cấu hình của chẩn đoán, nó không phải cấu hình tốt nhất trên calibration ở mọi ô: với S = 4, lr 1e-3 / 45 vòng đạt 0,875 so với 0,874 của nó; với S = 1, lr 1e-2 / 15 vòng đạt 0,881 so với 0,880.
- Các ngân sách rút gọn được tỉ lệ theo R_base giống run 3: Rapid bằng R_base / 2, backfill 7 vòng mỗi bước.

**Phần giữ nguyên.** Partition, cách gán shard, seed, chuỗi yêu cầu, các chính sách, metric và tham số DynamicFU (trừ R_max) giống hệt các run khám phá.

**Trong code.**
- `scripts/shard/train_config.py` chứa hai cấu hình, chọn bằng biến môi trường `SHARD_TRAIN_CONFIG`.
- **Mặc định đổi thành `standard`.** Muốn tái lập các run khám phá cũ thì đặt `SHARD_TRAIN_CONFIG=repo`.
- Run 3 (`run_shard_paper_scale.py`) vốn đã dùng cấu hình này, nên không đổi.

**Chi phí.**
- Một lần train đầy đủ tốn khoảng (20 × 2) / (15 × 1) ≈ 2,7 lần số bước local so với cấu hình cũ. Số byte truyền tăng theo số vòng (20 / 15).
- Chi phí unlearn vẫn được báo cáo theo phần trăm so với train lại toàn bộ **cùng cấu hình**.

## Thiết lập chạy
- **Dữ liệu:** `data/processed/connected_v3_auxfit_seed42`. Tiền xử lý của bản này chỉ fit trên validation, nên unlearn chính xác trọn vẹn.
  - Dữ liệu được sinh lại trên máy chạy từ `configs/data_connected_auxfit.json`.
  - **Đã kiểm tra trước khi chạy:** partition 20 client (seed 42) có số dòng và số dòng normal của từng client trùng khớp với `results/shard_exploration_auxfit/ensemble_check_20c/meta.json`.
  - Hash manifest khác bản trong container (57d08bf2… so với 13648c9e…), vì manifest có ghi đường dẫn và platform.
- **Máy:** Windows 11, Python 3.13.2, torch 2.6.0, CPU, 2 thread cho mỗi run.
  - Kết quả **không so từng bit** với các run trong container được.
  - Ví dụ: cùng model S = 1, seed 10042, cấu hình `repo` cho macro-F1 0,86478 trên máy này và 0,86450 trong container; FPR trùng nhau.
- **Split:** chính sách được đọc validation, metric báo cáo trên calibration, test không được dùng.
- **Run 1:** 20 client, Dirichlet α = 0,5 (seed 42); S ∈ {1, 2, 4, 5, 10}; seed 10042–10046.
  - Request trên client lớn nhất: client 10, 29.317 dòng.
  - Hai loại request: xóa cả client (`client`), hoặc xóa ngẫu nhiên phân tầng 50% (`random50`, 14.658 dòng).
- **Run 2:** S ∈ {4, 10}; chuỗi 10 yêu cầu sinh từ metadata (seed 99), xóa tổng cộng 69.178 dòng (60,5% dữ liệu train).
  - Chính sách: `none`, `shard_full`, `shard_fixed`, `dynamicfu`, `utility_stop`, `drop_backfill`.
  - Tham chiếu: `full_retrain_single`, tức một model train lại từ đầu trên toàn bộ dữ liệu còn lại sau mỗi yêu cầu.

## Phần dữ liệu đã được nhìn
- **Partition, cách gán shard và chuỗi yêu cầu** giống các run khám phá.
- **Seed 10042 với S = 1 và S = 4 ở cấu hình mới đã được xem trên calibration** trong chẩn đoán. Mức trùng với các run ở đây (seed của batch là `seed + 10000·vòng + 100·client_id`):
  - run 1, S = 1: model base trùng hẳn model của chẩn đoán;
  - run 1, S = 4: cùng cấu hình, cùng shard, cùng seed, nhưng run 1 đánh số worker theo vị trí trong shard, nên thứ tự batch khác và model khác;
  - run 2, S = 4: ensemble ban đầu trùng hẳn model của chẩn đoán (worker được đánh số theo client). Ensemble này không được báo cáo trực tiếp, nhưng mọi chính sách đều bắt đầu từ nó.
- **Chưa từng chạy ở cấu hình mới:** seed 10043–10046; S ∈ {2, 5, 10}; mọi lần unlearn ở run 1; mọi bước của chuỗi ở run 2.
- Vì vậy, ngoài kết quả trên 5 seed, các gate G1, G2, G4, G5 được tính lại **chỉ trên seed 10043–10046**. Đây là phân tích độ nhạy, đặt trước.

## Gate (cố định trước khi chạy)
**Hằng số:**
- Ngưỡng của chế độ FPR cao: **FPR > 0,45** trên calibration. Đây cũng là ngưỡng `bad` trong `analyze_shard_20c.py`. FPR bình thường khoảng 0,32; ở chế độ hỏng khoảng 0,75.
- Biên macro-F1: **δ_F1 = 0,01**. Biên FPR: **δ_FPR = 0,02**.
- Mọi hiệu được ghép cặp theo seed; điều kiện xét trên trung bình qua các seed.

| Gate | Run | Điều kiện pass |
|---|---|---|
| **G1** Utility của ensemble | 1 | Với mọi S ∈ {2, 4, 5, 10}: mean(F1_S − F1_1) ≥ −0,01 và mean(FPR_S − FPR_1) ≤ +0,02. Không có (S, seed) nào, kể cả S = 1, có FPR > 0,45 |
| **G2** Utility sau một lần unlearn chính xác | 1 | Ở R = R_base, với mọi S ∈ {2, 4, 5, 10} và mọi request: các điều kiện như G1, so với S = 1 cùng request. S = 1 ở R_base chính là train lại toàn bộ trên D \ F. Không dòng nào ở R_base có FPR > 0,45 |
| **G3** Tính chính xác | 2 | Mọi chính sách trừ `none`, ở mọi S và mọi seed: không shard nào đang dùng từng train trên dòng đã xóa, ở mọi bước; và mọi shard trùng từng bit với model train mới ở cuối chuỗi |
| **G4** Tham chiếu khỏe | 2 | `full_retrain_single` không có bước nào có FPR > 0,45, ở mọi seed |
| **G5** Utility cuối chuỗi | 2 | Với S ∈ {4, 10} và mọi chính sách chính xác (`shard_full`, `shard_fixed`, `dynamicfu`, `utility_stop`; `drop_backfill` lấy trạng thái sau khi xong hàng đợi), so với `full_retrain_single` ở bước 10: mean(F1 − F1_ref) ≥ −0,01 và mean(FPR − FPR_ref) ≤ +0,02. Không seed nào có FPR cuối > 0,45 |

**Quy tắc kết luận:**
- **Xác nhận** khi G1–G5 đều pass trên 5 seed **và** các hash ghi lúc bắt đầu khớp với file hiện tại.
- Nếu ở phân tích độ nhạy (seed 10043–10046) một gate đổi kết quả, kết luận phải ghi rõ "phụ thuộc vào seed đã nhìn".
- **Nếu một gate fail:** báo cáo nguyên trạng. Trong protocol này không đổi cấu hình, ngưỡng hay seed sau khi xem kết quả. Thử cấu hình khác cần một protocol mới.
- G3 kiểm tra tính đúng của code. G3 fail nghĩa là có lỗi cài đặt, không phải lỗi cấu hình.

## Kết quả mô tả (không phải gate)
- Chi phí unlearn (dòng-lượt, byte, thời gian) theo phần trăm so với train lại toàn bộ, ở cả hai run.
- **Thời điểm tệ nhất trong chuỗi** của từng chính sách: F1 thấp nhất, FPR cao nhất, số bước có FPR > 0,45.
  - Ở dữ liệu cũ, một số chính sách từng bị sụt giữa chuỗi (`dynamicfu` với S = 4, `drop_backfill`, `shard_full` với S = 10).
  - Không đặt dự đoán riêng cho từng chính sách.
- Lý do dừng của DynamicFU.
- So sánh với kết quả cấu hình `repo` trên auxfit.

## Không được tuyên bố, kể cả khi xác nhận
| Không được tuyên bố | Lý do |
|---|---|
| Kết quả đúng trên partition khác hoặc chuỗi yêu cầu khác | Run này dùng lại partition 42 và chuỗi seed 99 của các run khám phá |
| Cấu hình này tối ưu | Chỉ kiểm tra rằng nó đủ để tránh chế độ hỏng; không dò siêu tham số |
| Kết quả trên test | Test không được dùng |
| Kết luận về DynamicFU nói chung | Vẫn là bản cài đặt lại, tham số tự chọn |

## Lệnh chạy
```bash
python src/data/prepare.py --input dataset/train_test_network.csv --config configs/data_connected_auxfit.json --output data/processed/connected_v3_auxfit_seed42
```
```bash
export SHARD_DATA_DIR=data/processed/connected_v3_auxfit_seed42 SHARD_TRAIN_CONFIG=standard FL_DEVICE=cpu
```
```bash
python scripts/shard/run_shard_ensemble_check.py runs/shard_standard/ensemble_check_20c
```
```bash
python scripts/shard/run_shard_stream.py runs/shard_standard/stream_20c
```
```bash
python scripts/shard/evaluate_standard_config.py runs/shard_standard/ensemble_check_20c runs/shard_standard/stream_20c runs/shard_standard/gates.json
```
```bash
python scripts/shard/analyze_shard_20c.py runs/shard_standard/ensemble_check_20c runs/shard_standard/stream_20c
```
CSV, JSON, kết quả gate và bảng tổng hợp được chép sang `results/shard_standard_config/`. Kết quả được viết trong `docs/shard_standard_config_results.md`.

## Thay đổi sau khi chạy
(chưa có)
