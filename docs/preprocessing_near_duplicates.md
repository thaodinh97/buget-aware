# Bản gần trùng giữa các split — phát hiện, bản sửa và các phương án, 2026-10-30

## Tóm tắt

Đã triển khai group policy `connected_bidirectional_features_near_duplicates` (`configs/data_near_duplicates.json`) và tạo dữ liệu `data/processed/near_duplicates_v4_seed42`.

Bản sửa đạt đúng mục tiêu: **không còn dòng test nào trùng khít với train** (trước đây 33,9%). Nhưng split bị chia gần như theo họ attack, nên pilot tụt macro-F1 từ 0,890 xuống 0,755 và FPR tăng từ 0,27 lên 0,63.

**Data mặc định của các study vẫn là `connected_v3_seed42`** cho đến khi có quyết định về các phương án ở cuối tài liệu.

## Định nghĩa đã triển khai

Hai dòng là gần trùng khi chúng **trùng mọi predictor trừ `duration`** và log1p(duration) chênh nhau ≤ δ, với δ = 0,01 (khoảng 1% tương đối, hoặc khoảng 10 ms với flow ngắn).

- Bao đóng bắc cầu được tính chính xác bằng cách sắp xếp theo một chiều (`near_duplicate_keys` trong `src/data/prepare.py`).
- Các key này được nối thành connected component cùng với bidirectional five-tuple và chữ ký feature, trước khi chia split.
- Giá trị `duration` bị thiếu không bao giờ được nối.
- Audit ghi số dòng trùng giữa các split ở δ (bằng 0 theo cấu trúc) và độ nhạy ở 1e-3/1e-2/5e-2. Với 5e-2 có 584–770 dòng trùng giữa mỗi cặp split.
- Nếu vẫn còn dòng trùng giữa các split ở δ, `training_ready` sẽ là false.

Vì sao chỉ dùng `duration`: trong các cặp test–train cách nhau ≤ 1e-3 (Chebyshev trên log1p của feature số thô, trong cùng nhóm categorical), **100% chỉ khác nhau ở `duration`**. Chênh lệch thô có trung vị 9 µs (phân vị 99 là 8 ms), và nhãn trùng ở 99,8% số cặp.

## Nguyên nhân gốc: dữ liệu rất ít đa dạng với bộ feature này

- 190.474 dòng chỉ có **13.379 chữ ký khác nhau** trên mọi predictor trừ `duration`.
- 149.089 dòng thuộc các chữ ký có từ 100 dòng trở lên.
- Các chữ ký lớn nhất thuần một họ: 18.294 dòng backdoor (100%), 15.344 dòng scanning, 13.123 dòng dos, 11.228 và 9.819 dòng normal.

Một split không có bản gần trùng buộc phải đặt cả chữ ký vào cùng một split, nên split bị chia theo họ attack:

| δ | Số tập gần trùng | Component lớn nhất | Normal trong test | Tỷ trọng của họ lớn nhất trong test | dos trong train |
|---|---:|---:|---:|---:|---:|
| 0 (v3) | 92.454 | 38.868 | 7.129 | 0,25 | 16.167 |
| 1e-6 | 82.940 | 46.414 | 9.022 | 0,32 | 16.029 |
| 1e-4 | 45.420 | 60.431 | 2.014 | 0,56 | 2.314 |
| 1e-3 | 33.937 | 65.261 | 1.982 | 0,56 | 2.508 |
| **1e-2 (v4)** | 22.944 | 69.772 | 1.649 | 0,55 | 1.451 |

Với v4, test có 15.808 dòng dos trong khi train chỉ có 1.451; ransomware có 14.705 dòng trong train và chỉ 2 dòng trong test.

## Độ "mới" của holdout (`scripts/audit_holdout_novelty.py`, chi tiết ở `results/holdout_novelty/`)

Khoảng cách L2 tới dòng train gần nhất trong không gian encoded:

| Data | Test trùng khít (= 0) | Test trong bán kính 1e-3 | Test trong bán kính 0,1 | Trung vị | Nhãn hàng xóm gần nhất trùng |
|---|---:|---:|---:|---:|---:|
| v3 | 33,9% | 91,8% | 99,4% | 1,5e-7 | 98,4% |
| v4 | 0% | 56,6% | 94,7% | 4,2e-4 | 98,6% |

Vẫn còn 56,6% test nằm trong bán kính 1e-3 dù không còn bản gần trùng theo `duration`. Lý do là chuẩn hóa z-score trên các feature đếm có đuôi dài: độ lệch chuẩn của byte lên tới khoảng 1e7, nên 40 byte và 44 byte gần như trùng nhau sau chuẩn hóa. Mô hình không phân biệt được những khác biệt này. Đây cũng có thể là một nguyên nhân của FPR 27% ở v3.

## Pilot trên v4 (`runs/pilot_v4`, `results/pilot_v4/`)

| | v3 | v4 |
|---|---:|---:|
| Macro-F1 test (original) | 0,890 | 0,755 |
| FPR | 0,272 | 0,634 |
| PR-AUC/AP | 0,990 | 0,998 |
| MIA AUC loss cố định, F / control (10%) | 0,668 / 0,472 | 0,158 / 0,394 |

## Các phương án (cần quyết định)

A. **Gộp bản gần trùng vào group (v4, đã triển khai).**
   - Ưu điểm: holdout mới thật sự ở mức mẫu feature.
   - Nhược điểm: split lệch theo họ attack, nên bài toán đổi thành tổng quát hóa sang mẫu chưa thấy và utility sụp.
   - Mọi study phải chạy lại, và gate utility (0,65 / FPR) cần xem lại.

B. **Khử gần trùng, mỗi tập giữ một dòng đại diện (chưa triển khai, chỉ khảo sát).**
   - Còn 22.944 dòng, 30 tập có nhãn xung đột. Split cân bằng, component lớn nhất 142 dòng.
   - Thành phần data đổi hẳn: injection 10.083, xss 5.062, normal 3.525, backdoor 53, scanning 131, ransomware 249.
   - Một "record" lúc này là một mẫu feature, nên ý nghĩa của sample deletion cũng thay đổi.

C. **Giữ v3 và chấp nhận holdout không mới.**
   - Coi MIA ở mức mẫu là không đánh giá được trên ToN-IoT với bộ feature này, và báo cáo như một giới hạn của dataset.
   - Đánh giá FU chủ yếu bằng khoảng cách tới ensemble retrain. Thước đo này không cần holdout mới, vì retrain dùng cùng dữ liệu.
   - Có thể cân nhắc thêm log1p cho các feature đếm trước khi chuẩn hóa. Đó là một thay đổi riêng, cần khóa trước.

Khuyến nghị: **C**. A làm hỏng chính nhiệm vụ IDS, còn B biến bài toán thành một dataset khác. Thước đo chính của FU (khoảng cách tới retrain) vẫn hợp lệ trên v3. Còn việc F có ảnh hưởng nhỏ do có bản gần trùng trong R là hiện tượng có thật, đã được check `identifiable` ghi nhận.
