# Mở rộng C1/C2 sang họ `concentrated`: kết quả

**Protocol:** [dt_concentrated_plan.md](dt_concentrated_plan.md), chốt ngày 2026-11-24 trước lần fit đầu tiên.
- Đây là phần mở rộng **mô tả**, không kiểm định giả thuyết.
- Kết luận C1/C2 giữ nguyên.

**Lần chạy:**
- Từ 19:36 đến 22:52 ngày 2026-11-24 (3 giờ 16 phút), liên tục.
- Dùng runner `scripts/dt/run_orchestration.py` không sửa.
- Xong 27/27 block, 162 lượt arm.
- `complete.json` ghi `source_hashes_unchanged: true`. Log lỗi rỗng.

**Số liệu:**
- [results/dt_concentrated/analysis/report.md](../results/dt_concentrated/analysis/report.md): toàn bộ bảng, kể cả tách theo tải và môi trường.
- `arms.csv` và `diffs.csv` cùng thư mục.
- Sinh bởi `scripts/sens/analyze_family_extension.py`.
- Cột `spread` và `proportional` lấy từ C1/C2 với cùng seed, nên mỗi block ghép cặp được theo (seed, tải, môi trường).

## Tóm tắt
- **`dt_queue` ≈ `edf`**: chênh lệch +0,04 yêu cầu/block [−0,44; +0,52], thắng/hòa/thua 4/18/5. Kết quả giống `spread`.
- **Các arm luôn chạy full sụp mạnh** so với C1/C2:

  | Arm | `concentrated` | `spread` | `proportional` |
  |---|---:|---:|---:|
  | `immediate` | 17,6 % | 45,1 % | 36,7 % |
  | `window` | 10,5 % | 38,3 % | 30,2 % |
  | `lazy` | 21,6 % | 21,3 % | 27,5 % |

  Vì vậy khoảng cách giữa `dt_queue` và `immediate` / `window` **rộng hơn** ở C1/C2, ngược với dự đoán 2.
- **Ở họ này, lợi thế đến từ quy tắc hạ xuống `shard_fixed`**, mà `edf` cũng có. Nó không đến từ việc lập kế hoạch thứ tự: chỉ có một hàng đợi thì không có gì để sắp xếp.
- **Cái giá phải trả:**
  - `dt_queue` chỉ chạy full cho 19/324 yêu cầu.
  - FPR trên calibration cao hơn các arm luôn chạy full 0,4 điểm phần trăm (0,0735 so với 0,0693).
  - Macro-F1 gần như không đổi (0,9532 so với 0,9535).

## Theo arm

Thành công đồng thời nghĩa là đúng hạn **và** trong ngân sách gộp, tính trên 324 yêu cầu của mỗi họ.

| Arm | `concentrated` | `spread` | `proportional` |
|---|---:|---:|---:|
| `dt_queue` | 287 (88,6 %) | 275 (84,9 %) | 268 (82,7 %) |
| `edf` | 286 (88,3 %) | 274 (84,6 %) | 258 (79,6 %) |
| `rows_legacy` | 262 (80,9 %) | 225 (69,4 %) | 251 (77,5 %) |
| `immediate` | 57 (17,6 %) | 146 (45,1 %) | 119 (36,7 %) |
| `window` | 34 (10,5 %) | 124 (38,3 %) | 98 (30,2 %) |
| `lazy` | 70 (21,6 %) | 69 (21,3 %) | 89 (27,5 %) |

Chi tiết riêng họ `concentrated`:

| Arm | Đúng hạn | Trong ngân sách | Chạy full | p95 độ trễ (s) | Macro-F1 | FPR |
|---|---:|---:|---:|---:|---:|---:|
| `dt_queue` | 91,0 % | 309 | 19 | 9,49 | 0,9532 | 0,0735 |
| `edf` | 91,4 % | 309 | 17 | 9,24 | 0,9532 | 0,0734 |
| `rows_legacy` | 80,9 % | 316 | 59 | 11,36 | 0,9532 | 0,0733 |
| `immediate` | 38,0 % | 215 | 324 | 17,71 | 0,9535 | 0,0693 |
| `window` | 23,1 % | 221 | 324 | 20,05 | 0,9535 | 0,0693 |
| `lazy` | 32,1 % | 236 | 324 | 17,84 | 0,9535 | 0,0693 |

## `dt_queue` − arm khác (toàn bộ 27 block)

Mỗi ô ghi: trung bình yêu cầu/block (trên 12) [khoảng tin cậy bootstrap 95 %] (số block thắng/hòa/thua).

| So với | `concentrated` | `spread` | `proportional` |
|---|---:|---:|---:|
| `edf` | +0,04 [−0,44; +0,52] (4/18/5) | +0,04 [−0,37; +0,41] (6/16/5) | +0,37 [−0,19; +0,96] (8/13/6) |
| `rows_legacy` | +0,93 [+0,22; +1,59] (16/8/3) | +1,85 [+1,37; +2,30] (23/3/1) | +0,63 [+0,15; +1,15] (14/7/6) |
| `immediate` | +8,52 [+7,67; +9,37] (27/0/0) | +4,78 [+4,33; +5,26] (27/0/0) | +5,52 [+4,89; +6,15] (27/0/0) |
| `window` | +9,37 [+8,52; +10,19] (27/0/0) | +5,59 [+5,04; +6,19] (27/0/0) | +6,30 [+5,59; +7,00] (27/0/0) |
| `lazy` | +8,04 [+7,19; +8,89] (27/0/0) | +7,63 [+7,00; +8,22] (27/0/0) | +6,63 [+6,00; +7,26] (27/0/0) |

**Theo tải.** Tải là khoảng cách trung bình giữa hai yêu cầu, tính theo F: tải 1,0 là thưa nhất.
- Ở `concentrated`, lợi thế so với `immediate` lớn nhất khi tải thưa: +7,67, +7,78 và +10,11 ở tải 0,2 / 0,5 / 1,0.
- Ở `spread`, lợi thế này gần như phẳng: +4,67, +5,33, +4,33.
- Lý do: tải thưa thì `immediate` phục vụ từng yêu cầu một, mà một yêu cầu đơn lẻ ở họ này luôn vượt ngân sách (xem phần cơ chế).
- Bảng theo tải và theo môi trường nằm trong [report.md](../results/dt_concentrated/analysis/report.md).

## Đối chiếu với dự đoán ghi trước
1. **`dt_queue` ≈ `edf`: đúng về kết quả, sai một phần về lý do.**
   - Kết quả: +0,04 [−0,44; +0,52]. Chuỗi quyết định (lô, chính sách) trùng hoàn toàn với `edf` ở 13/27 block.
   - Lý do mình ghi trước là "`dt_queue` không bao giờ chờ gộp ngân sách". Thực tế nó đã chờ **17 lần ở 6 block**:
     - phần lớn ở môi trường `jitter_bandwidth`;
     - thêm `r2 / tải 0,5 / none`, block có F đo được thấp (5,69 s);
     - thêm `r2 / tải 1,0 / gradual_clients`.
   - Nguyên nhân: cận trên dự báo của `shard_fixed`, kể cả biên q90, lên tới 0,75–0,97 F, vượt ngân sách một yêu cầu (0,75 F). Mình đã giả định nó luôn ≈ 0,5 F.
2. **Khoảng cách tới `window` / `lazy` hẹp hơn: sai.**
   - Với `window`: +9,37, so với +5,59 (`spread`) và +6,30 (`proportional`). Khoảng tin cậy không chồng lấn.
   - Với `lazy`: +8,04, xấp xỉ `spread` (+7,63).
   - Dự đoán sai vì lập luận dựa trên **chi phí**: gom lô tiết kiệm nhiều nhất ở `concentrated`, điều này vẫn đúng theo quét cửa sổ của shard v3. Nhưng chỉ số chính là thành công đồng thời, và ở họ này nó bị **deadline** chi phối.
3. **`immediate` vẫn kém: đúng, và kém hơn dự kiến** (17,6 %). Lý do đúng như đã ghi: một lần full trên shard lớn nhất mất khoảng F, vượt ngân sách 0,75 F của một yêu cầu.
4. **Mọi arm unlearn chính xác: đúng.** Cả 162/162 lượt arm trùng từng bit ở mọi sự kiện.

## Cơ chế: vì sao các arm luôn chạy full sụp ở `concentrated`
- **Full rất đắt ở họ này.** Mọi yêu cầu rơi vào shard 2, nên một lần full mất trung bình **0,99 F**. Ở `spread` chỉ là 0,48 F, vì 3/4 shard nhỏ hơn.
- **Yêu cầu đơn lẻ luôn vượt ngân sách.** Ngân sách một yêu cầu là 0,75 F, nhỏ hơn F, nên 0 % yêu cầu đơn lẻ nằm trong ngân sách.
- **Gom ≥ 2 yêu cầu thì vừa ngân sách nhưng trễ hạn.**
  - Ngân sách gộp ≥ 1,5 F nên 91–93 % yêu cầu nằm trong ngân sách.
  - Nhưng chỉ 14–28 % đúng hạn: thời gian train ≈ F đã chiếm 2/3 deadline 1,5 F, chỉ còn 0,5 F để chờ.
  - Yêu cầu cũ nhất trong lô thường đã phải chờ hết lần train trước, cũng mất khoảng F.

  | Họ / arm | Yêu cầu đơn lẻ: tỉ lệ, trong ngân sách, thành công | Lô ≥ 2: tỉ lệ, đúng hạn, thành công |
  |---|---|---|
  | `concentrated` / `immediate` | 28 %, 0 %, 0 % | 72 %, 24 %, 24 % |
  | `concentrated` / `window` | 25 %, 0 %, 0 % | 75 %, 14 %, 14 % |
  | `concentrated` / `lazy` | 22 %, 0 %, 0 % | 78 %, 28 %, 28 % |
  | `spread` / `immediate` | 70 %, 65 %, 53 % | 30 %, 28 %, 28 % |
  | `spread` / `window` | 69 %, 65 %, 48 % | 31 %, 18 %, 18 % |

- **`window` kém nhất** vì phải chờ tới biên cửa sổ (W = 0,5 F), ăn hết phần dư 0,5 F.
- **`dt_queue` và `edf` tránh được bẫy này.** `shard_fixed` mất khoảng 0,5 F: vừa ngân sách của một yêu cầu, và còn dư khoảng 1 F để chờ.
- **Ví dụ** (block `r0 / tải 0,5 / none`):
  - `window` chạy 7 lần full, xen kẽ lô 1 yêu cầu (vượt ngân sách) và lô 2–3 yêu cầu (trễ hạn). Kết quả 3/12.
  - `dt_queue` chạy 9 lần fixed và 1 lần full cho lô 2 yêu cầu. Kết quả 12/12.

## Chất lượng model
- **Số yêu cầu chạy full:**
  - `dt_queue` 19/324 ở `concentrated`, so với 150/324 ở `spread`.
  - Ở họ này, full trên shard lớn nhất hầu như không bao giờ đáng.
- **Đổi lại, FPR tăng:** FPR calibration của model cuối là 0,0735, so với 0,0693 của các arm luôn chạy full, tức +0,42 điểm phần trăm. Mức tăng này lớn hơn ở `spread` (+0,08) và tương đương ở `proportional` (+0,43).
- **Macro-F1:** 0,9532 so với 0,9535.
- Cùng hướng với kết luận của bảng FU: `shard_fixed` thiếu train làm model xa retrain hơn một chút.

## Toàn vẹn
- **Trùng từng bit:** 162/162 lượt arm trùng ở mọi sự kiện. Hash nguồn không đổi. `run.err.log` rỗng.
- **Modern Standby:**
  - Kernel-Power 506/507 xuất hiện nhiều lần: gần như liên tục từ 19:35 đến 21:59, rồi từng đợt từ 22:33 đến 22:52.
  - Không có sự kiện 41/42/107, tức không có ngủ thật, ngủ đông hay khởi động lại bất thường.
  - 23/27 block chạy hơn một nửa thời gian trong standby.
  - F đo được: trung bình 7,06 s khi standby, 7,22 s khi máy thức.
  - Thời lượng block bình thường: 3,4–4,5 phút ở tải 0,2; 9,2–12,1 phút ở tải 1,0.
  - Khoảng cách lớn nhất giữa hai lần xong block là 12,4 phút, rơi vào một block tải 1,0.
- **F theo block:** từ 5,69 đến 8,20 s.
  - Hai giá trị lệch nhất đều đo lúc máy thức: 5,69 s ở `r2 / tải 0,5 / none` và 8,20 s ở `r2 / tải 1,0 / gradual_clients`.
  - Ngân sách và deadline của mỗi block tính theo F của chính block đó, và protocol không có luật loại block, nên giữ nguyên cả hai.
- **Kiểm tra giữa chừng:**
  - Lúc 22:23:13–22:23:24, mình chạy kiểm tra tiến độ và analyzer tạm thời (khoảng 3 giây CPU).
  - Lúc đó arm `edf` của block `r2 / tải 1,0 / gradual_clients` đang rảnh chờ yêu cầu (22:23:11–22:23:24).
  - Lần train lại kế tiếp mất 6,10 s, so với 6,06 s của `dt_queue` ở cùng vị trí. Không thấy ảnh hưởng.
- **Log:** đã chuyển vào thư mục lần chạy: `run.out.log`, `run.err.log`, `run.pid`.

## Hệ quả cho báo cáo
- Khi chỉ có một hàng đợi, phần lập kế hoạch thứ tự của `dt_queue` không thêm gì so với EDF. Lợi thế so với các chiến lược luôn chạy full đến từ việc **được phép hạ xuống `shard_fixed`** khi full không vừa ngân sách hoặc deadline.
- Gom lô (`window`, `lazy`) tiết kiệm chi phí nhưng không cứu được deadline khi thời gian full xấp xỉ F và D = 1,5 F.

## Không phải luận điểm
- Không thay đổi hay sửa kết luận C1/C2.
- So sánh giữa các họ chỉ là đặt cạnh nhau: không kiểm định chênh lệch giữa họ.
- Kết quả phụ thuộc vào việc B = 0,75 F nhỏ hơn thời gian full của shard lớn nhất. Đợt quét độ nhạy chỉ quét B trên `spread` và `proportional`, chưa trên họ này.
- `poisson_zipf` chưa được chạy.
