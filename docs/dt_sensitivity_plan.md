# Phân tích độ nhạy của các tham số không có lý do được ghi lại

**Trạng thái: CHỐT ngày 2026-11-21, trước lần fit đầu tiên.** Mỗi config có lock riêng trong `results/dt_sensitivity/<tên>/protocol_lock.json`, và lock đó hash tài liệu này. `results/dt_sensitivity/manifest.json` hash tài liệu này, driver, wrapper và cả 35 config. Không sửa tài liệu này trong khi chạy; mọi sai lệch ghi vào tài liệu kết quả.

Tài liệu này viết sau khi đã xem kết quả C1/C2 ([dt_orchestration_results.md](dt_orchestration_results.md)), kể cả bảng khám phá theo tải × môi trường. Bảng tổng hợp nguồn gốc tham số (trả lời ngày 2026-11-21) cho thấy các tham số sau không có lý do được ghi lại: số yêu cầu mỗi chuỗi (12), deadline D = 1,5 F, mức stress, các tham số của twin và kích thước MLP. Ngân sách B = 0,75 F có lý do, nhưng là tham số ảnh hưởng mạnh nhất tới C1, nên cũng được quét.

## Mục đích và giới hạn
- **Mô tả, không xác nhận.** Không có kiểm định giả thuyết theo từng ô; mỗi ô báo cáo mức chênh và khoảng tin cậy bootstrap 95 %. Báo cáo **mọi** ô đã chạy, kể cả ô `dt_queue` thua.
- **Kết quả C1/C2 giữ nguyên** là kết quả xác nhận tại điểm tham chiếu (B = 0,75 F, D = 1,5 F, K = 12, chậm tối đa 4×, băng thông 5–20 Mbit/s, MLP 64→32). Không chọn lại điểm tham chiếu theo kết quả đợt này.
- **Hai loại tham số được xử lý khác nhau:**
  - *Tham số kịch bản* (B, D, mức stress, K, MLP) mô tả bài toán. Đợt này chỉ cho biết kết luận đúng ở vùng nào; **không** chọn giá trị "tốt nhất".
  - *Tham số phương pháp* (của twin) có thể được chọn, nhưng chỉ trên **seed phát triển riêng** (S4), và giá trị được chọn chỉ là ứng viên. Muốn dùng nó cho một luận điểm thì cần một protocol xác nhận mới với seed mới.
- Mọi lần chạy dùng runner C1/C2 nguyên vẹn (`scripts/dt/run_orchestration.py`, cùng vòng lặp thời gian thực, cùng kiểm tra so trùng từng bit). Chỉ config thay đổi. Riêng S5 dùng `scripts/sens/run_hidden.py`, chỉ thay một lời gọi `ShardConfig.from_train_config` để đặt `hidden`.

## Lưới quét
Mỗi config là `configs/dt_orchestration.json` với các khóa ghi trong bảng được thay đổi, sinh bởi `scripts/sens/run_sensitivity.py write` vào `configs/sens/`. Cả hai họ chuỗi (`spread`, `proportional`) được dùng ở mọi config.

| Đợt | Tham số thay đổi | Giá trị | Seed | Tải | Môi trường | Arm | Block/config |
|---|---|---|---|---|---|---|---|
| S1 | B × D | B ∈ {0,5; 0,75; 1,0; 1,5} F × D ∈ {1,0; 1,5; 2,0; 3,0} F (16 ô, gồm chạy lại ô tham chiếu) | 73042, 73043 | 0,2 F | cả 3 | 4 arm | 12 |
| S2 | Mức chậm tối đa | 2×, 8× (tham chiếu 4×) | 73042–73044 | 0,2; 0,5 F | `gradual_clients` | 4 arm | 12 |
| S2 | Dải băng thông | 2–8, 12,5–50 Mbit/s (tham chiếu 5–20) | 73042–73044 | 0,2; 0,5 F | `jitter_bandwidth` | 4 arm | 12 |
| S3 | Số yêu cầu K | 6, 24 (tham chiếu 12) | 73042, 73043 | 0,2; 0,5 F | cả 3 | 4 arm | 24 |
| S4 | Tham số twin, từng tham số một | tham chiếu; `margin_q` 0,8 / 0,95; `margin_window` 5 / 20; `expert_eta` 1 / 10; `expert_floor` 0,02 / 0,25; `alpha` (EWMA của `rows_legacy`) 0,1 / 0,5 | **75042, 75043** (seed luồng 85000, seed stress 95000) | 0,2 F | `gradual_clients`, `jitter_bandwidth` | 3 arm | 8 |
| S5 | Kích thước MLP | 32→16, 128→64 (tham chiếu 64→32) | 73042, 73043 | 0,2 F | cả 3 | 4 arm | 12 |

- **4 arm** = `dt_queue`, `edf`, `rows_legacy`, `immediate`. `immediate` đại diện cho nhóm luôn chạy full. S4 bỏ `immediate` vì arm này không dùng dự báo để quyết định.
- **Seed của S1, S2, S3, S5 trùng C1/C2**, cùng thứ tự, nên cùng mô hình fit (trùng từng bit), cùng chuỗi yêu cầu (trừ S3) và cùng đường stress. Mọi so sánh với điểm tham chiếu vì thế ghép cặp theo block. Điểm tham chiếu của S2, S3, S5 lấy từ các block tương ứng của C1/C2. S1 chạy lại ô tham chiếu; độ lệch giữa lần chạy lại và C1/C2 là thước đo nhiễu giữa hai lần chạy.
- **S4 dùng seed phát triển** vì nó chọn giá trị. Không seed nào trùng `RESERVED_SEEDS` của runner hay dải 74042–74047 đã dành cho thiết kế mức 2.
- Thứ tự chạy: S1 → S2 → S3 → S4 → S5. Ước tính khoảng 22 giờ máy (theo thời gian đo ở C1/C2).

## Đại lượng và cách báo cáo
- **Đại lượng chính:** số yêu cầu vừa đúng hạn vừa trong ngân sách mỗi block (`joint_success_requests`), như C1/C2.
- **Theo từng config:**
  - trung bình mỗi arm;
  - chênh lệch ghép cặp theo block `dt_queue` − mỗi arm khác, với khoảng tin cậy bootstrap percentile 95 % (10.000 lần lấy mẫu lại block, seed 1) và số block thắng/hòa/thua;
  - tỉ lệ đúng hạn, số yêu cầu trong ngân sách, macro-F1 trên calibration;
  - tách theo môi trường và tải khi config có nhiều mức.
- **S1:** bảng 4 × 4 của chênh lệch `dt_queue` − `edf` và `dt_queue` − `immediate`, toàn bộ và theo từng môi trường.
- **S2, S3, S5:** đặt cạnh điểm tham chiếu của C1/C2 cho cùng block.
- **S4, quy tắc chọn, chốt trước:**
  - Các tham số của `ServiceTwin` (`margin_q`, `margin_window`, `expert_eta`, `expert_floor`) được xét theo `dt_queue`. `alpha` được xét theo `rows_legacy`. `margin_q` và `margin_window` dùng chung cho cả hai twin, nên báo cáo cho cả hai.
  - Một giá trị chỉ được coi là ứng viên tốt hơn tham chiếu khi khoảng tin cậy 95 % của chênh lệch ghép cặp theo block (giá trị mới − tham chiếu, cùng arm) nằm hoàn toàn trên 0. Nếu không, giữ giá trị tham chiếu.
  - Với 8 block mỗi config, dự kiến hầu hết tham số không qua ngưỡng này. Điều đó được báo cáo là "không nhạy trong dải đã quét", không phải là bằng chứng giá trị tham chiếu là tối ưu.
- **Tính chính xác:** mọi arm phải unlearn chính xác và so trùng từng bit; runner dừng nếu không.
- Config hỏng (ví dụ vượt `calibration_max_round_seconds`) được ghi lại, không chạy lại với tham số khác; block dở dang được giữ nguyên như ở C1/C2.

## Không phải luận điểm
- Không có giá trị B, D, K hay mức stress nào được trình bày là "hợp lý nhất". Đợt này chỉ mô tả vùng mà kết luận C1 đúng hoặc không đúng.
- Không dùng kết quả đợt này để thay đổi hay "sửa" kết luận C1/C2.
- Giá trị tham số twin được chọn ở S4 chưa phải kết quả. Nó chỉ là ứng viên cho một lần xác nhận sau.
