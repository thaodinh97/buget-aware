# Điều phối C1/C2: kết quả

Ngày 2026-11-21. **C1 không đạt.** Controller đề xuất `dt_queue` vượt rõ 4 trên 5 đối chứng: ba chiến lược luôn chạy full (xử lý ngay, cửa sổ kiểu SISA, hoãn kiểu ERASER) và controller cũ của repo. Nó **không vượt được EDF dùng cùng twin**. **C2: không phát hiện thành phần nào có đóng góp hay có hại.** Cả 540/540 chuỗi đều unlearn chính xác từng bit, và mọi biên utility đều đạt.

Protocol: [dt_orchestration_plan.md](dt_orchestration_plan.md), chốt ở commit `03b3401`. Tài liệu protocol không được sửa sau khi chốt, vì hash của nó nằm trong lock và bộ phân tích sẽ từ chối chạy nếu hash đổi. Vì vậy kết quả được ghi ở đây.

## Lần chạy

- **Phạm vi:** `results/dt_orchestration/`, đủ 54/54 block và 540 chuỗi. Lock `frozen_utc` là 2026-11-19T13:48:20Z. `complete.json` ghi `source_hashes_unchanged: true`; `tests.json` ghi `sources_changed_since_lock: []`.
- **Lệnh phân tích:** `python scripts/dt/analyze_orchestration.py results/dt_orchestration`. Lệnh này sinh ra `report.md`, `paired.csv`, `summary.csv` và `tests.json`. Bộ phân tích audit lại từ log sự kiện trước khi tính toán.
- **Gián đoạn** (ghi đầy đủ trong `results/dt_orchestration/interruptions.json`):
  - **Người dùng tạm dừng** lúc 2026-11-19 22:16, khi đã xong 6 block. Lần chạy tiếp tục lúc 2026-11-20 23:33 với cùng cấu hình và code; hash đã được kiểm lại khi tiếp tục.
    - Block dở dang (9/10 arm) được chuyển nguyên vẹn sang `runs/dt_orchestration_partial/`, không ai xem, và được chạy lại từ đầu. Các arm không bao giờ được ghép giữa hai lần chạy.
  - **Modern Standby:** nhật ký hệ thống ghi máy vào Modern Standby (màn hình tắt) lúc 00:16:30 ngày 2026-11-21, thoát lúc 08:21:16, rồi vào lại lúc 08:21:45. Trong khoảng đó không có sự kiện ngủ thật (Kernel-Power 42/107).
    - Thời điểm ghi file của cả 48 block sau khi tiếp tục là liên tục: giữa hai block luôn cách khoảng 15 giây. Trong các block nằm quanh hai mốc trên, khoảng cách giữa hai arm là 84–98 giây ở tải 1,0 F và 26–45 giây ở tải 0,2 F, giống các block khác. 48 block hết 10,3 giờ, đúng như ước tính.
    - Kết luận: runner không bị treo, nên không chạy lại block nào. Việc kiểm tra này được làm **trước khi xem kết quả**.
- **Lần chạy 2 seed bị dừng** (`runs/dt_orchestration_aborted_2seed_20261119/`): không dùng trong phân tích, như protocol đã ghi.

## Kết quả theo arm

Mỗi arm có 648 yêu cầu (54 block × 12). "Thành công" là thành công đồng thời: xong trước deadline **và** trong ngân sách vòng gộp.

| Arm | Nhóm | Thành công | Tỉ lệ | Miss deadline | Thời gian phục vụ (s) | Latency p95 TB (s) | Chạy full | F1 cal. | FPR cal. |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `dt_queue` | đề xuất | 543 | 0,838 | 80 | 2062,6 | 12,06 | 290 | 0,9491 | 0,0738 |
| `edf` | C1 | 532 | 0,821 | 88 | 2102,7 | 11,16 | 290 | 0,9493 | 0,0742 |
| `rows_legacy` | C1 (controller cũ) | 476 | 0,735 | 149 | 1999,3 | 14,03 | 302 | 0,9495 | 0,0739 |
| `immediate` | C1 | 265 | 0,409 | 301 | 2971,0 | 22,12 | 648 | 0,9541 | 0,0713 |
| `window` | C1 (kiểu SISA) | 222 | 0,343 | 376 | 2975,1 | 24,04 | 648 | 0,9541 | 0,0713 |
| `lazy` | C1 (kiểu ERASER) | 158 | 0,244 | 457 | 2710,0 | 24,06 | 648 | 0,9541 | 0,0713 |
| `queue_no_pooling` | C2 | 547 | 0,844 | 73 | 2087,0 | 11,43 | 290 | 0,9488 | 0,0738 |
| `queue_edf_order` | C2 | 547 | 0,844 | 78 | 2066,1 | 11,18 | 282 | 0,9486 | 0,0730 |
| `queue_no_overhead` | C2 | 545 | 0,841 | 75 | 2079,7 | 12,21 | 293 | 0,9489 | 0,0735 |
| `rows_queue` | C2 | 538 | 0,830 | 87 | 2076,0 | 12,65 | 296 | 0,9487 | 0,0740 |

## C1: `dt_queue` so với đối chứng

Phép kiểm là Wilcoxon một phía theo block, hiệu chỉnh Holm trên 5 đối chứng, α = 0,05. Δ là `dt_queue` − đối chứng, tính bằng số yêu cầu thành công mỗi block (trên 12).

| Đối chứng | Δ TB | CI 95 % | Thắng/Hòa/Thua | p Holm | Δ F1 cal. (biên) | Δ theo tải 0,2 / 0,5 / 1,0 F | Bị vượt |
|---|---:|---|---|---:|---|---|---|
| `immediate` | +5,15 | [+4,74; +5,56] | 54/0/0 | < 0,0001 | −0,0050 (−0,02, đạt) | +5,00 / +5,44 / +5,00 | **có** |
| `window` | +5,94 | [+5,48; +6,41] | 54/0/0 | < 0,0001 | −0,0050 (−0,02, đạt) | +6,17 / +6,11 / +5,56 | **có** |
| `lazy` | +7,13 | [+6,67; +7,59] | 54/0/0 | < 0,0001 | −0,0050 (−0,02, đạt) | +6,61 / +7,61 / +7,17 | **có** |
| `rows_legacy` | +1,24 | [+0,87; +1,63] | 37/10/7 | < 0,0001 | −0,0004 (−0,005, đạt) | +1,33 / +1,44 / +0,94 | **có** |
| `edf` | +0,20 | [−0,15; +0,56] | 14/29/11 | 0,1295 | −0,0002 (−0,005, đạt) | +0,89 / −0,28 / +0,00 | **không** |

C1 cần mọi đối chứng đều bị vượt. Vì `edf` không bị vượt nên **C1 không đạt**.

Chênh với `edf` có độ lệch chuẩn 1,34 yêu cầu/block. Khoảng tin cậy loại trừ một lợi thế lớn hơn khoảng 0,56 yêu cầu/block, tức khoảng 5 % của 12 yêu cầu.

## C2: ablation

Phép kiểm là Wilcoxon hai phía, hiệu chỉnh Holm trên 4 ablation. Δ là `dt_queue` − ablation.

| Ablation (thành phần bị bỏ) | Δ TB | CI 95 % | Thắng/Hòa/Thua | p Holm | Kết luận |
|---|---:|---|---|---:|---|
| `rows_queue` (twin kết hợp → EWMA theo dòng) | +0,09 | [−0,19; +0,39] | 11/32/11 | 1,00 | không phát hiện được |
| `queue_edf_order` (lập kế hoạch thứ tự → EDF) | −0,07 | [−0,44; +0,30] | 9/33/12 | 1,00 | không phát hiện được |
| `queue_no_overhead` (mô hình overhead) | −0,04 | [−0,24; +0,19] | 11/30/13 | 1,00 | không phát hiện được |
| `queue_no_pooling` (chờ để gộp ngân sách) | −0,07 | [−0,35; +0,24] | 8/32/14 | 1,00 | không phát hiện được |

Các khoảng tin cậy hẹp, đều nằm trong ±0,45 yêu cầu/block. Vì vậy đóng góp của từng thành phần, nếu có, nhỏ hơn khoảng 0,4 yêu cầu/block theo cả hai chiều. Ba trên bốn ước lượng điểm hơi nghiêng về phía ablation; không ước lượng nào có ý nghĩa thống kê.

## Mô tả (đã khai báo trước, không kiểm định)

- **Utility:**
  - Các arm cùng không gian full/fixed (`dt_queue`, `edf`, `rows_legacy`, 4 ablation) có F1 calibration 0,9486–0,9495 và FPR 0,073–0,074.
  - Ba arm luôn chạy full có F1 0,9541 và FPR 0,071.
  - Giá phải trả cho việc giữ ngân sách là −0,005 macro-F1, chỉ bằng một phần tư biên −0,02. Mức này nhỏ hơn chênh `shard_fixed` − `shard_full` đo ở v3, vì chỉ khoảng 55 % yêu cầu chạy fixed, và ngưỡng được chọn lại sau mỗi thao tác.
- **Chạy full:** `dt_queue`, `edf` và các ablation chạy full cho 282–296 trên 648 yêu cầu (khoảng 45 %). Các arm luôn chạy full thì chạy full cả 648.
- **Thời gian phục vụ:** các arm full/fixed tốn khoảng 2000–2100 s tổng cộng, các arm luôn chạy full tốn khoảng 2700–2980 s.
  - Deadline miss: `dt_queue` 80, `edf` 88. Latency p95 trung bình: `edf` 11,16 s, `dt_queue` 12,06 s.
- **Chi phí điều khiển:**
  - Thời gian lập kế hoạch của `dt_queue` là 0,75 s cộng dồn trên cả 54 block (`edf` 0,52 s), không đáng kể.
  - Thời gian chọn ngưỡng là 14,7 s cộng dồn, khoảng 0,7 % thời gian phục vụ.

## Khám phá sau khi xem kết quả (không kiểm định, không được tuyên bố)

Bảng dưới là chênh `dt_queue` − `edf` theo mức tải × môi trường. Mỗi ô có 6 block, và bảng này được xem **sau khi** đã có kết quả.

| Môi trường | 0,2 F | 0,5 F | 1,0 F |
|---|---:|---:|---:|
| `none` | −0,17 | 0,00 | 0,00 |
| `gradual_clients` | +1,17 | −0,33 | +0,17 |
| `jitter_bandwidth` | +1,67 | −0,50 | −0,17 |

- Lợi thế của `dt_queue`, nếu có, chỉ xuất hiện ở tải quá mức (0,2 F) và khi có stress. Đây là chỗ hàng đợi dài và thời gian phục vụ nhiễu, nên việc lập kế hoạch thứ tự có thể có tác dụng.
- Không có tải hay stress thì hai controller cho kết quả gần như trùng nhau (17/18 block hòa ở `none`).
- Đây chỉ là **giả thuyết** rút ra từ 9 ô nhỏ, không có hiệu chỉnh. Muốn khẳng định cần một protocol mới chốt trước, với seed mới, nhắm riêng vào vùng này.

## Kết luận được và không được tuyên bố

**Được tuyên bố** (trong phạm vi của protocol: một máy, một worker, một partition, S = 4, 12 yêu cầu sample-level, B = 0,75 F, D = 1,5 F, chỉ đo trên calibration):
1. Mọi quyết định điều phối chỉ dựa trên tín hiệu chi phí, và chỉ chọn giữa các chính sách có số vòng cố định trước. Nhờ vậy unlearn giữ chính xác: 540/540 chuỗi khớp từng bit.
2. Với cùng hệ thống, cùng yêu cầu, cùng ngân sách và deadline, `dt_queue` phục vụ nhiều yêu cầu thành công đồng thời hơn hẳn ba chiến lược luôn chạy full. Mức chênh là +5,2 đến +7,1 trên 12 yêu cầu mỗi block, thắng 54/54 block. Cái giá là −0,005 macro-F1 calibration, nằm trong biên −0,02 đã chốt. Ba chiến lược này là xử lý ngay, cửa sổ *theo tinh thần* SISA và hoãn *theo tinh thần* ERASER.
3. `dt_queue` vượt controller cũ của repo (`rows_legacy`) +1,24 yêu cầu/block mà không mất utility.

**Không được tuyên bố:**
- `dt_queue` tốt hơn EDF dùng cùng twin và cùng lựa chọn full/fixed.
- Bất kỳ thành phần riêng nào (twin kết hợp, lập kế hoạch thứ tự, mô hình overhead, gộp ngân sách) có đóng góp.
- Lợi thế ở tải quá mức có stress (mục khám phá).
- Mọi mục trong bảng "Không được tuyên bố" của protocol.

**Cách đọc hợp lý nhất** (đây là diễn giải, chưa được tách riêng bằng ablation nào):
- `dt_queue`, `edf` và cả 4 ablation có cùng một thành phần: dùng twin để dự báo, rồi hạ xuống `shard_fixed` khi `shard_full` không vừa ngân sách hoặc deadline. Cả nhóm này đạt 0,82–0,84, còn các arm luôn chạy full chỉ đạt 0,24–0,41.
- Các phần nằm phía trên thành phần đó (lập kế hoạch thứ tự trên mọi shard đang chờ, mô hình overhead, chờ để gộp) không thêm gì phát hiện được trong thiết kế này.
- Hệ quả: đóng góp nên được trình bày là **điều phối có nhận biết ngân sách trong không gian chính sách giữ tính chính xác, dựa trên twin chi phí**. Một bộ lập lịch EDF đơn giản trên twin đã đủ để đạt phần lớn lợi ích đó; controller lập kế hoạch đầy đủ không có ưu thế được xác nhận.
