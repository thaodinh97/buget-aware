# Phân tích độ nhạy: kết quả

**Protocol:** [dt_sensitivity_plan.md](dt_sensitivity_plan.md), chốt ngày 2026-11-21 trước lần fit đầu tiên.

**Lần chạy:** từ 20:19 ngày 2026-11-21 đến 18:35 ngày 2026-11-22, liên tục, không chạy tiếp từ điểm dừng nào. 35/35 config, 400 block, mọi config kết thúc với mã thoát 0.

**Số liệu:**
- [results/dt_sensitivity/analysis/report.md](../results/dt_sensitivity/analysis/report.md) có toàn bộ bảng, tách theo môi trường và tải.
- `cells.csv` và `arms.csv` (cùng thư mục) chứa số liệu thô của từng ô và từng arm.
- [integrity.json](../results/dt_sensitivity/integrity.json) ghi kiểm tra tính toàn vẹn.

Tài liệu này chỉ **mô tả**. Không có kiểm định giả thuyết, và kết luận C1/C2 ([dt_orchestration_results.md](dt_orchestration_results.md)) giữ nguyên.

## Tính toàn vẹn
- **Unlearn chính xác:** 129/129 cặp (config, arm) unlearn chính xác và trùng từng bit với retrain.
- **Máy không bị ngủ:**
  - Có 16 lần Windows vào/ra Modern Standby (Kernel-Power 506/507, tức tắt màn hình). Lần dài nhất từ 00:54 đến 10:56.
  - Không có sự kiện ngủ/thức (42/107) và không có lần tắt máy bất thường.
  - Khoảng cách lớn nhất giữa hai lần hoàn tất block liên tiếp là 8,3 phút, bằng độ dài một block K = 24 ở tải 0,5 F. Block vẫn hoàn tất đều trong mọi lần tắt màn hình.
- **Máy lúc tắt màn hình chạy nhanh và ổn định hơn lúc bật màn hình.** F trung bình đo lúc fit là 7,23 s khi tắt màn hình (dải 7,02–7,76) và 7,76 s khi bật (dải 5,94–9,76).
  - Vì vậy hai config chạy cách nhau vài giờ có thể gặp trạng thái máy khác nhau. Mục S4 cho thấy hệ quả của điều này.
- **Không có sai lệch so với protocol.** Trong tối 2026-11-21 có vài job numpy/pandas ngắn chạy song song (thử script phân tích, test hàm thuần của bảng FU), mỗi job vài giây. Không có job train nào chạy cùng lúc.

## Kết luận chính
1. **`dt_queue` hơn chiến lược luôn chạy full (`immediate`) và hơn controller cũ (`rows_legacy`) ở mọi điểm đã quét:** B, D, mức stress, K và kích thước MLP.
   - Đây là kết luận vững nhất của C1, và nó không phụ thuộc vào các giá trị tham chiếu đã chọn.
   - Khoảng cách chỉ thu hẹp khi deadline rất lỏng (D = 3 F), vì khi đó gần như mọi chiến lược đều đúng hạn.
2. **Cần sửa một nhận định trước đây của mình.** Mình từng cho rằng lợi thế so với `immediate` chỉ đúng khi ngân sách chặt. Dữ liệu cho thấy ngược lại:
   - Lợi thế gần như **không đổi theo B**: trung bình +4,15 / +3,94 / +4,06 / +3,71 yêu cầu/block ở B = 0,5 / 0,75 / 1,0 / 1,5 F.
   - Lợi thế **giảm mạnh theo D**: +5,15 / +4,83 / +4,04 / +1,83 ở D = 1 / 1,5 / 2 / 3 F.
   - Nguyên nhân: ở tải 0,2 F, yêu cầu đến nhanh hơn tốc độ chạy full. `immediate` vì vậy trễ deadline (chỉ 2,7/12 yêu cầu thành công ở D = 1 F). `dt_queue` hạ xuống `shard_fixed` để theo kịp.
   - Như vậy cơ chế thực sự giúp là **hạ cấp theo deadline khi quá tải**, không phải chủ yếu theo ngân sách.
3. **So với EDF, lợi thế của `dt_queue` chỉ xuất hiện khi điều kiện khó.** Ba điều kiện đó là:
   - deadline chặt: ở D = 1 F, cả 4 mức B đều cho +1,08 đến +2,42, khoảng tin cậy nằm trên 0;
   - stress nặng: băng thông 2–8 Mbit/s cho +1,67 [+0,75; +2,75];
   - chuỗi dài: K = 24 cho +1,46 [+0,29; +2,62].

   Lợi thế biến mất khi:
   - không có stress;
   - deadline lỏng (D = 3 F);
   - chuỗi ngắn: K = 6 cho −0,08, hai controller gần như trùng nhau;
   - stress nhẹ ở tải 0,5 F.

   Điều này làm rõ kết luận C1 "không thắng EDF" (+0,20, không phát hiện được), chứ không đảo ngược nó. Đây là kết quả hậu nghiệm (post hoc): muốn thành luận điểm thì cần một protocol xác nhận mới với seed mới (xem mục cuối).
4. **Tham số twin không nhạy, trong phạm vi nhiễu đo được.** Ở S4 không giá trị nào qua quy tắc chọn đã chốt. Các phép so sánh giả dược cho thấy nhiễu giữa các lần chạy lớn ngang các hiệu ứng quan sát được. Các giá trị tham chiếu được giữ, nhưng điều này **không** chứng minh chúng tối ưu.
5. **Kích thước MLP không làm đổi kết luận.**
   - MLP 32→16 và 128→64 cho chênh lệch `dt_queue` − đối chứng gần với 64→32.
   - 128→64 chậm hơn khoảng 1,5 lần (F = 11,4 s so với khoảng 7,3 s), macro-F1 chênh −0,001 đến +0,003 tùy arm.
   - 32→16 không nhanh hơn, và macro-F1 thấp hơn 0,002–0,005.
6. **Utility:**
   - So với `immediate`: macro-F1 của `dt_queue` thấp hơn 0,002–0,008 ở mọi ô (cận dưới thấp nhất −0,011), luôn trong biên −0,02 của C1.
   - So với EDF: chênh −0,003 đến +0,003, nhưng cận dưới có ô xuống −0,007, thấp hơn sàn −0,005 mà C1 dùng cho EDF.

## S1: ngân sách B × deadline D
Thiết lập: tải 0,2 F; seed 73042–73043; 2 họ chuỗi × 3 môi trường, tức 12 block mỗi ô. Mỗi ô là trung bình chênh lệch theo block [khoảng tin cậy bootstrap 95 %], đơn vị yêu cầu/block (trên tổng 12 yêu cầu).

**`dt_queue` − `edf`** (15/16 ô dương; 7/16 ô có khoảng tin cậy nằm trên 0; không ô nào nằm dưới 0)

| B \ D | 1 F | 1,5 F | 2 F | 3 F |
|---|---:|---:|---:|---:|
| 0,5 F | +1,08 [+0,42; +1,75] | +1,00 [−0,42; +2,58] | +0,67 [−0,08; +1,42] | −0,33 [−1,08; +0,25] |
| 0,75 F | +1,33 [+0,75; +2,00] | +1,42 [+0,33; +2,92] | +0,92 [−0,08; +2,00] | +0,25 [0,00; +0,50] |
| 1,0 F | +1,50 [+0,92; +2,17] | +0,33 [−0,67; +1,42] | +0,67 [+0,08; +1,33] | +0,25 [0,00; +0,67] |
| 1,5 F | +2,42 [+1,67; +3,25] | +0,33 [−0,67; +1,50] | +0,92 [+0,33; +1,67] | +0,42 [0,00; +1,00] |

Tách theo môi trường, trung bình trên 16 ô:
- không stress: +0,38 (2/16 ô có khoảng tin cậy trên 0);
- `gradual_clients`: +1,06 (6/16);
- `jitter_bandwidth`: +1,03 (4/16).

Lợi thế so với EDF chủ yếu đến từ các môi trường có stress, giống bảng khám phá của C1/C2.

**`dt_queue` − `immediate`** (16/16 ô có khoảng tin cậy trên 0)

| B \ D | 1 F | 1,5 F | 2 F | 3 F |
|---|---:|---:|---:|---:|
| 0,5 F | +4,58 | +5,17 | +4,33 | +2,50 |
| 0,75 F | +4,58 | +5,08 | +4,08 | +2,00 |
| 1,0 F | +5,58 | +4,92 | +4,00 | +1,75 |
| 1,5 F | +5,83 | +4,17 | +3,75 | +1,08 |

**`dt_queue` − `rows_legacy`:** 16/16 ô dương, 11/16 ô có khoảng tin cậy trên 0. Mức chênh từ +3,08 (B 1,5 F, D 1 F) giảm về khoảng 0 ở D = 3 F.

**Số yêu cầu `dt_queue` phục vụ thành công** (trên 12): từ 6,50 (B 0,5 F, D 1 F) tăng lên 12,00 (B ≥ 1 F, D 3 F). Với `immediate`: 2,7 / 4,7 / 6,5 / 9,7 ở D = 1 / 1,5 / 2 / 3 F.

**Chạy lại ô tham chiếu (B 0,75 F, D 1,5 F) so với C1/C2 trên cùng block:**
- Chênh theo từng arm từ −0,08 đến −0,67, mọi khoảng tin cậy đều chứa 0.
- `dt_queue` − `edf`: +1,42 khi chạy lại so với +0,83 ở C1/C2. Cùng block, cùng model, cùng chuỗi yêu cầu, nên phần chênh 0,6 này chỉ do nhiễu thời gian giữa hai lần chạy.

## S2: mức stress (tải 0,2 và 0,5 F; 3 seed; 12 block/config)
Cột "C1/C2" là các block tương ứng của C1/C2.

| Config | `dt_queue` − `edf` | C1/C2 | `dt_queue` − `immediate` | C1/C2 | `dt_queue` − `rows_legacy` | C1/C2 |
|---|---:|---:|---:|---:|---:|---:|
| chậm tối đa 2× | +0,83 [+0,25; +1,50] | +0,42 | +6,42 | +5,67 | +1,50 | +1,67 |
| chậm tối đa 8× | +0,83 [+0,25; +1,42] | +0,42 | +4,08 | +5,67 | +1,25 | +1,67 |
| băng thông 2–8 Mbit/s (nặng) | **+1,67 [+0,75; +2,75]** | +0,58 | +2,50 | +4,83 | +1,50 | +1,58 |
| băng thông 12,5–50 Mbit/s (nhẹ) | +0,25 [−0,67; +1,25] | +0,58 | +5,83 | +4,83 | +1,50 | +1,58 |

- **Ở tải 0,5 F**, nơi C1/C2 cho `dt_queue` ≤ EDF (−0,33 và −0,50), stress nặng làm kết quả chuyển sang dương: chậm 8× cho +1,00 [+0,33; +1,83], băng thông 2–8 cho +1,83 [+0,50; +3,67]. Ngược lại, băng thông nhẹ ở tải 0,5 F cho −0,67 [−1,33; −0,17].
- **Với băng thông, lợi thế so với EDF tăng đơn điệu theo mức stress:** +0,25 (nhẹ) → +0,58 (tham chiếu) → +1,67 (nặng). Với mức chậm thì không đơn điệu (2× và 8× đều +0,83), nằm trong phạm vi nhiễu.
- **Stress nặng làm khoảng cách với `immediate` nhỏ lại** (+2,50 và +4,08), vì khi đó cả `shard_fixed` cũng thường trễ deadline.

## S3: số yêu cầu K (tải 0,2 và 0,5 F; 2 seed; 24 block/config)
| K | `dt_queue` − `edf` | /yêu cầu | `dt_queue` − `immediate` | /yêu cầu | `dt_queue` − `rows_legacy` | /yêu cầu |
|---|---:|---:|---:|---:|---:|---:|
| 6 | −0,08 [−0,29; +0,17] | −0,014 | +2,54 | +0,42 | +0,58 [−0,04; +1,12] | +0,10 |
| 12 (C1/C2, cùng block) | +0,17 [−0,50; +0,83] | +0,014 | +5,38 | +0,45 | +1,33 | +0,11 |
| 24 | **+1,46 [+0,29; +2,62]** | +0,061 | +12,79 | +0,53 | +3,21 | +0,13 |

- **Lợi thế tăng theo độ dài chuỗi, kể cả khi tính trên mỗi yêu cầu.** Chuỗi càng dài thì twin càng có nhiều quan sát để học, và hàng đợi càng có nhiều lúc dồn lại.
- **K = 24 với `jitter_bandwidth`:** `dt_queue` − `edf` = +3,38 [+1,38; +5,12].
- **K = 6:** `dt_queue` và EDF gần như trùng quyết định (chênh 0 ở môi trường không stress).
- **K = 12 nằm ở giữa dải.** Kết luận so với `immediate` và `rows_legacy` không phụ thuộc K; kết luận so với EDF thì có.

## S4: tham số twin (seed phát triển 75042–75043; tải 0,2 F; 2 môi trường stress; 8 block/config)
Mỗi ô là chênh lệch giữa giá trị mới và tham chiếu trong cùng arm, ghép cặp theo block. Các ô *nghiêng* là **giả dược**: tham số được đổi không được arm đó dùng. Theo `scripts/dt/run_orchestration.py`, `alpha` chỉ thuộc `rows_legacy`; `expert_eta` và `expert_floor` chỉ thuộc twin của `dt_queue` và `edf`; `margin_q` và `margin_window` dùng chung cho cả hai twin.

| Config | `dt_queue` | `edf` | `rows_legacy` | F trung bình của config (s) |
|---|---:|---:|---:|---:|
| tham chiếu (8,38 / 7,38 / 6,38 yêu cầu) | – | – | – | 7,11 |
| `margin_q` = 0,8 | −0,12 [−0,50; +0,25] | +0,12 [−0,50; +0,75] | −0,62 [−1,50; 0,00] | 7,19 |
| `margin_q` = 0,95 | +0,12 [−0,88; +1,12] | +0,75 [−0,38; +2,00] | −0,12 [−0,75; +0,38] | 7,14 |
| `margin_window` = 5 | +0,12 [−0,75; +0,88] | +0,88 [−0,25; +2,12] | 0,00 [−0,75; +0,75] | 7,13 |
| `margin_window` = 20 | −0,88 [−1,75; 0,00] | −0,38 [−1,88; +1,25] | −0,50 [−1,50; 0,00] | 7,10 |
| `expert_eta` = 1 | −1,50 [−2,75; −0,12] | −0,50 [−1,12; +0,12] | *−1,12 [−2,38; +0,12]* | 7,21 |
| `expert_eta` = 10 | −0,75 [−1,75; +0,25] | −0,12 [−1,38; +1,38] | *0,00 [−0,62; +0,50]* | 7,26 |
| `expert_floor` = 0,02 | −0,50 [−1,25; +0,12] | −1,25 [−2,38; −0,12] | *−1,12 [−2,00; −0,25]* | 7,12 |
| `expert_floor` = 0,25 | −0,12 [−1,12; +0,75] | −0,38 [−1,38; +0,75] | *−0,62 [−1,88; +0,38]* | 7,82 |
| `alpha` = 0,1 | *−0,88 [−1,62; −0,12]* | *−0,75 [−2,00; +0,50]* | −0,75 [−2,12; +0,50] | 7,67 |
| `alpha` = 0,5 | *−1,75 [−2,62; −1,00]* | *0,00 [−1,12; +1,25]* | −0,62 [−1,75; +0,38] | 7,92 |

- **Không giá trị nào qua quy tắc chọn đã chốt** (khoảng tin cậy nằm hẳn trên 0 cho arm dùng tham số đó). Các giá trị tham chiếu được giữ.
- **Giả dược:** 3/8 phép so sánh giả dược có khoảng tin cậy không chứa 0, và cả ba đều âm. Giá trị lớn nhất là −1,75, tức `dt_queue` thay đổi khi chỉ đổi `alpha`, tham số mà nó không dùng.
  - Điều này có nghĩa: **chênh lệch giữa hai lần chạy riêng biệt có thể lên tới khoảng 1,75 yêu cầu/block chỉ do nhiễu.** Khoảng tin cậy bootstrap theo block không bao được nhiễu này, vì mọi block trong một config chạy cùng một quãng thời gian, nên chịu cùng một trạng thái máy.
  - Cột F trùng với giải thích đó. Ô tham chiếu chạy lúc 13:31–13:49, phần lớn khi màn hình tắt (F = 7,11 s). Các config `alpha` chạy lúc 16:16–16:56 khi màn hình bật, F = 7,67–7,92 s.
- **Ba config mà cả `dt_queue` lẫn `edf` đều không đổi tham số** (tham chiếu, `alpha` = 0,1, `alpha` = 0,5) thực chất là ba lần chạy lặp của `dt_queue` − `edf`. Kết quả lần lượt là +1,00, +0,88 và −0,75: dao động 1,75 trên 8 block.
- **Diễn giải:** các hiệu ứng âm có khoảng tin cậy không chứa 0 (ví dụ `expert_eta` = 1: −1,50 cho `dt_queue`) có cùng độ lớn với giả dược. Vì vậy không thể quy chúng cho tham số. Kết luận đúng là: *trong dải đã quét, không phát hiện được độ nhạy vượt quá nhiễu giữa các lần chạy*.

## S5: kích thước MLP (tải 0,2 F; 2 seed; 12 block/config)
| MLP | F (s) | `dt_queue` − `edf` | `dt_queue` − `immediate` | `dt_queue` − `rows_legacy` | Macro-F1 calibration (`dt_queue` / `immediate`) |
|---|---:|---:|---:|---:|---:|
| 32→16 | 7,51 | +1,08 [0,00; +2,08] | +5,25 | +0,92 | 0,9463 / 0,9519 |
| 64→32 (C1/C2, cùng block) | khoảng 7,3 | +0,83 [−0,08; +1,83] | +5,00 | +1,33 | 0,9496 / 0,9539 |
| 128→64 | 11,41 | +0,83 [+0,33; +1,42] | +4,58 | +1,67 | 0,9511 / 0,9568 |

Model nhỏ hơn không nhanh hơn, vì thời gian bị chi phối bởi overhead. Model lớn hơn chậm hơn khoảng 1,5 lần, cho utility cao hơn rất ít, và không làm đổi kết luận. Đây không phải kết quả dò tham số: 64→32 vẫn là kiến trúc đã có của repo.

## Ý nghĩa cho khóa luận
- **Nên trình bày như một kết quả vững:**
  - `dt_queue` hơn chiến lược luôn chạy full và hơn controller cũ trên toàn bộ không gian tham số đã quét, với chi phí utility trong biên.
  - Cơ chế là hạ cấp theo deadline (và ngân sách) khi quá tải, trên không gian chính sách chính xác.
- **Nên trình bày có điều kiện:**
  - Lợi thế so với EDF chỉ có khi deadline chặt, stress nặng hoặc chuỗi dài; không có khi điều kiện dễ.
  - Đây là một bản đồ mô tả, không phải luận điểm đã xác nhận. Kết luận C1 "không thắng EDF trên lưới của C1/C2" giữ nguyên.
- **Về các tham số không có lý do được ghi lại** (12 yêu cầu, D = 1,5 F, mức stress, tham số twin, MLP):
  - Có thể viết rằng đã quét độ nhạy, và hướng của các kết luận so với `immediate` và `rows_legacy` không đổi.
  - So với EDF, độ lớn của lợi thế phụ thuộc D, K và mức stress, nên các giá trị tham chiếu nằm ở vùng trung gian.
  - Tham số twin không nhạy trong phạm vi nhiễu.
- **Bài học phương pháp:** so sánh giữa các lần chạy cách nhau vài giờ chịu nhiễu trạng thái máy, lên tới khoảng ±1,75 yêu cầu/block với 8 block. Protocol sau nên xen kẽ các điều kiện trong cùng block, như cách các arm được xen kẽ, thay vì chạy mỗi điều kiện thành một lần riêng.

**Nếu muốn biến lợi thế so với EDF thành luận điểm,** cần một protocol xác nhận mới:
- chốt trước các điều kiện khó, ví dụ K = 24, D = 1 F, băng thông 2–8 Mbit/s;
- dùng seed mới;
- chỉ chạy hai arm (`dt_queue` và `edf`);
- xen kẽ điều kiện trong block.

Đây chỉ là gợi ý; chưa có bản thiết kế.

## Không phải luận điểm
- Không giá trị B, D, K hay mức stress nào là "hợp lý nhất".
- Không tham số twin nào được chứng minh là tối ưu.
- Lưới B × D chỉ chạy ở tải 0,2 F, là tải mà C1/C2 cho thấy lợi thế so với EDF lớn nhất. Ở 0,5 F và 1,0 F, bản đồ có thể khác.
- Mọi ô có 8–24 block. Khoảng tin cậy theo block đánh giá thấp nhiễu giữa các lần chạy (mục S4), nên không nên đọc riêng lẻ chênh lệch dưới khoảng 1 yêu cầu/block giữa hai ô. Chỉ nên đọc các xu hướng lặp lại qua nhiều ô.
