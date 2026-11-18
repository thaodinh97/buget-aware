# Hướng shard, protocol v3: kết quả xác nhận 10 seed, 4 họ chuỗi

**Protocol:** [shard_v3_plan.md](shard_v3_plan.md), chốt ngày 2026-11-15 trước mọi lần train v3 trên dữ liệu thật. **Kết quả:** `results/shard_v3/gates.json` và `evaluate.log`, commit `f118050` (2026-11-18). Thư mục `runs/shard_v3/` (CSV, `run.json` từng họ) không nằm trong git; mọi số dưới đây đọc từ `gates.json`.

**Phạm vi:** seed mới 60042–60051 (chưa từng chạy trước đó), 4 họ chuỗi (`proportional`, `spread`, `concentrated`, `poisson_zipf`), S ∈ {4, 5}, 20 yêu cầu sample-level mỗi chuỗi, 7 chính sách cùng model đơn tham chiếu ở bước 5, 10, 15, 20. Cấu hình `standard` (Adam lr 1e-2, R_base = 20, `shard_fixed` 10 vòng, backfill 7 vòng). Metric tính trên calibration; test không được dùng.

**Tính toàn vẹn (P): đạt.** Thiết kế `v3`, cấu hình `standard`, đủ 10 seed, hash dữ liệu khớp `f130057e…`, mọi hash trong `TRACKED_V3` khớp lúc đánh giá, cả 4 họ chạy cùng máy và cùng phiên bản thư viện, kiểm tra tái lập sau `fit` đạt, đủ mọi tổ hợp (họ, S, seed, chính sách). `full_retrain_single` không có bước nào FPR > 0,45.

> **Sau merge ngày 2026-11-18** ([shard_exact_merge.md](shard_exact_merge.md)), hash của `src/fu/shard_exact.py` trên nhánh không còn khớp `sha256_at_start` của 4 run. `gates.json` được tính với code lúc chạy; muốn đánh giá lại đúng như lúc chạy thì `git checkout f118050`.

## Kết luận
**Không xác nhận toàn bộ.** P, G1, G2 và G6 pass; G3, G4 và G5 fail. Theo từng claim:

| Claim | Gate | Kết quả |
|---|---|---|
| Unlearn chính xác ở mọi chính sách, kể cả khi gom lô (`utility_stop`: có điều kiện theo số vòng) | G1 | **Có** |
| Chi phí và độ trễ tính trước là chi phí thật của chính sách | G2 | **Có** |
| Dưới yêu cầu sample-level, chia shard giữ được utility của model đơn (Q1) | G3 | **Không** (fail ở 1/8 ô: `spread`, S = 5) |
| Cấu trúc chuỗi không làm chính sách rẻ hơn kém `shard_full` (Q2) | G4 | **Không** (`shard_fixed` fail 5/8 ô; `utility_stop` pass 8/8) |
| `drop_backfill` an toàn ở mọi bước (Q2) | G5 | **Không** (pass 1/8 ô) |
| `utility_stop` an toàn (Q2) | G6, cùng các dòng `utility_stop` của G4 | **Có** |

- **Unlearn chính xác:** không có lỗi ở bước nào hay ở trạng thái cuối, ở mọi chính sách. `batch_full_2` và `batch_full_5` kết thúc trùng từng bit với `shard_full`.
- **G3 fail ở một ô, sát biên:** `spread`, S = 5, ΔF1 trung bình −0,0105, cận dưới −0,0117 (biên −0,01). Bảy ô còn lại pass, cận dưới từ −0,0083 đến −0,0025. FPR pass ở cả 8 ô.
- **G4 fail vì `shard_fixed`,** không phải vì `utility_stop`. `shard_fixed` fail ở 5/8 ô, cận dưới ΔF1 từ −0,0137 đến −0,0108. `utility_stop` pass cả 8 ô, cận dưới từ −0,0087 đến +0,0024.
- **G5 fail như kế hoạch đã dự đoán ở `spread`,** và còn fail ở các họ khác: ensemble trống ở ít nhất một bước (mọi seed) trong 4 ô, và utility ở bước tệ nhất vượt biên trong 3 ô. Chỉ `concentrated` S = 5 pass.
- **`utility_stop`:** không bước nào có FPR > 0,45 (lớn nhất 0,3285), noninferior so với `shard_full` ở trạng thái cuối trong cả 8 ô.

## Gate
Noninferiority một phía, t ghép cặp theo seed, 95 %, n = 10. Pass khi cận dưới ΔF1 > −0,01 và cận trên ΔFPR < +0,02. Một gate pass khi mọi so sánh trong nó pass.

| Gate | Kết quả | Chi tiết |
|---|---|---|
| P Điều kiện tiên quyết | PASS | Toàn vẹn, độ phủ và FPR của model đơn đều đạt |
| G1 Tính chính xác | PASS | 0 lỗi ở các bước, 0 lỗi ở trạng thái cuối; gom lô trùng từng bit với `shard_full` |
| G2 Kế hoạch được tái lập | PASS | Chuỗi, dòng × vòng và số shard của `drop_backfill` khớp kế hoạch tính trước ở mọi seed |
| G3 `shard_full` so với model đơn, bước 20 | **FAIL** | Cận dưới ΔF1 (S = 4 / S = 5): `proportional` −0,0025 / −0,0083; `spread` −0,0065 / **−0,0117**; `concentrated` −0,0046 / −0,0058; `poisson_zipf` −0,0029 / −0,0058. Cận trên ΔFPR ≤ +0,0051 |
| G4 Chính sách rẻ hơn so với `shard_full`, bước 20 | **FAIL** | `shard_fixed`: `proportional` **−0,0137** / **−0,0108**; `spread` **−0,0113** / −0,0049; `concentrated` −0,0063 / −0,0077; `poisson_zipf` **−0,0119** / **−0,0108**. `utility_stop`: −0,0087 / −0,0028; −0,0025 / +0,0024; −0,0047 / −0,0067; −0,0051 / −0,0031. Cận trên ΔFPR ≤ +0,0083 |
| G5 `drop_backfill` ở bước tệ nhất | **FAIL** | Ensemble trống ở ít nhất một bước, mọi seed: `proportional` S = 5, `spread` S = 4 và S = 5, `poisson_zipf` S = 4. Utility vượt biên: `proportional` S = 4 (ΔF1 trung bình −0,117, cận dưới −0,175; cận trên ΔFPR +0,314), `concentrated` S = 4 (cận dưới −0,0122), `poisson_zipf` S = 5 (trung bình −0,0376, cận dưới −0,0394). Pass: `concentrated` S = 5 (cận dưới −0,0089) |
| G6 `utility_stop` qua mọi bước | PASS | FPR lớn nhất 0,3285; không bước nào > 0,45 |

## Utility ở bước 20
Macro-F1 trên calibration, trung bình 10 seed.

| Họ | S | Model đơn | `shard_full` | `shard_fixed` | `utility_stop` |
|---|---|---|---|---|---|
| `proportional` | 4 | 0,8812 | 0,8803 | 0,8690 | 0,8742 |
| | 5 | 0,8812 | 0,8753 | 0,8663 | 0,8755 |
| `spread` | 4 | 0,8805 | 0,8765 | 0,8679 | 0,8762 |
| | 5 | 0,8809 | 0,8704 | 0,8675 | 0,8739 |
| `concentrated` | 4 | 0,8814 | 0,8792 | 0,8748 | 0,8771 |
| | 5 | 0,8806 | 0,8771 | 0,8712 | 0,8729 |
| `poisson_zipf` | 4 | 0,8808 | 0,8799 | 0,8715 | 0,8773 |
| | 5 | 0,8808 | 0,8772 | 0,8702 | 0,8757 |

FPR lớn nhất theo seed của các chính sách chia shard trong bảng nằm trong khoảng 0,315–0,329; model đơn có FPR khoảng 0,32 ở mọi họ. FPR cao là đặc điểm của detector nền, không phải của việc chia shard.

## Chi phí và độ trễ (mô tả, không phải gate)
Dòng × vòng của cả chuỗi, tính bằng phần trăm so với train lại model đơn sau mọi yêu cầu. Chi phí của `shard_full`, `batch_full_W` và `shard_fixed` tất định (như nhau ở mọi seed); `utility_stop` và `drop_backfill` là trung bình 10 seed.

| Họ | S | `shard_full` | `shard_fixed` | `utility_stop` | `drop_backfill` | `batch_full_2` | `batch_full_5` |
|---|---|---|---|---|---|---|---|
| `proportional` | 4 | 35,1 % | 17,5 % | 18,5 % | 13,2 % | 19,6 % | 13,3 % |
| | 5 | 28,0 % | 14,0 % | 2,7 % | 10,5 % | 16,8 % | 11,4 % |
| `spread` | 4 | 24,8 % | 12,4 % | 7,9 % | 12,4 % | 24,5 % | 12,0 % |
| | 5 | 19,8 % | 9,9 % | 2,9 % | 12,0 % | 19,8 % | 11,3 % |
| `concentrated` | 4 | 47,4 % | 23,7 % | 3,3 % | 17,9 % | 18,9 % | 8,7 % |
| | 5 | 36,6 % | 18,3 % | 2,7 % | 13,7 % | 14,5 % | 6,4 % |
| `poisson_zipf` | 4 | 26,7 % | 13,4 % | 7,4 % | 12,7 % | 22,3 % | 13,8 % |
| | 5 | 17,8 % | 8,9 % | 4,5 % | 9,5 % | 12,7 % | 9,6 % |

- **Độ trễ xóa** (đơn vị thời gian của chuỗi): `batch_full_2` trung bình 1,00, lớn nhất 1,85; `batch_full_5` trung bình 2,15, lớn nhất 4,95. Các chính sách xử lý ngay có độ trễ 0. Thời điểm đến giống nhau giữa các họ, nên độ trễ cũng giống nhau.
- **Gom lô mất tác dụng khi yêu cầu liền nhau rơi vào các shard khác nhau:** ở `spread`, `batch_full_2` tốn gần bằng hoặc bằng `shard_full` (24,5 % so với 24,8 % ở S = 4; 19,8 % ở S = 5). Ở `concentrated`, `batch_full_5` chỉ còn 6,4–8,7 %. Đây là câu trả lời mô tả cho Q3.
- **`utility_stop`:** số vòng dự phòng trung bình mỗi chuỗi lớn nhất ở `proportional` S = 4 (16,1 vòng); tổng số bước kết thúc mà không đạt ok() trên 10 seed là 5 (`proportional` S = 4), 2 (`spread` S = 4), 10 (`poisson_zipf` S = 4) và 8 (`poisson_zipf` S = 5), các ô khác 0.
- **`drop_backfill`:** số bước ensemble trống trung bình mỗi chuỗi là 16 (`spread` S = 4), 14 (`spread` S = 5), 2 (`poisson_zipf` S = 4), 1 (`proportional` S = 5), các ô khác 0.
- Số giây là tổng thời gian vòng train trên một máy, chỉ để so tương đối trong cùng lần chạy.

## So với dự kiến ghi trong protocol
- "G5 fail ở `spread`, cả S = 4 và S = 5": **đúng**, ensemble trống ở mọi seed.
- "G5 có thể fail ở `concentrated`": fail ở S = 4 (utility), pass ở S = 5.
- "G3 có thể pass ở `proportional`, `spread` và `poisson_zipf`; rủi ro chính là `concentrated`": **sai một phần.** `concentrated` pass; `spread` S = 5 fail.
- "G6 có thể pass": **đúng**.

## Được tuyên bố / không được tuyên bố
Được tuyên bố, trên partition 42, cách gán shard seed 7, S ∈ {4, 5}, bốn họ chuỗi đã chốt và seed 60042–60051:
- Unlearn chính xác ở mọi chính sách, kể cả gom lô theo cửa sổ; với `utility_stop` là chính xác có điều kiện theo số vòng.
- Chi phí và độ trễ trong kế hoạch tính trước là chi phí thật của các chính sách.
- `utility_stop` không rơi vào chế độ FPR cao ở bước nào, và noninferior so với `shard_full` ở trạng thái cuối trong cả 8 ô.

Không được tuyên bố:
- Chia shard giữ được utility của model đơn dưới mọi cấu trúc chuỗi (G3 fail ở `spread` S = 5). Các ô pass chỉ được báo cáo theo từng ô, không gộp thành claim.
- `shard_fixed` ngang `shard_full` (fail 5/8 ô).
- `drop_backfill` an toàn ở mọi bước.
- `utility_stop` tốt hơn `shard_full`: có ô ΔF1 dương, nhưng protocol không đặt gate tốt hơn.
- Kết quả cho partition, cách gán shard, S khác, yêu cầu client-level, chuỗi thích ứng theo model, hay trên test.
- Kết quả trùng từng bit trên máy khác, hoặc tái lập được bằng code hiện tại của nhánh mà không checkout `f118050`.

## Tái lập
```bash
git checkout f118050
export SHARD_DESIGN=v3 FL_DEVICE=cpu
for f in proportional spread concentrated poisson_zipf; do SHARD_V3_FAMILY=$f python scripts/shard/run_shard_stream_v3.py runs/shard_v3/stream_$f & done; wait
python scripts/shard/evaluate_shard_v3.py runs/shard_v3 runs/shard_v3/gates.json
```
Thư mục output không được tồn tại trước khi chạy. Lệnh PowerShell tương đương nằm trong [shard_v3_plan.md](shard_v3_plan.md), mục "Lệnh chạy".
