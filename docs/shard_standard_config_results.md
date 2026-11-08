# Cấu hình train chuẩn cho hướng shard: kết quả xác nhận hai run 20 client

**Protocol:** [shard_standard_config_plan.md](shard_standard_config_plan.md), viết trước khi chạy. **Kết quả:** `results/shard_standard_config/`, gồm CSV, JSON, `gates.json` (đầu ra của script đánh giá) và `analysis_20c.txt` (bảng tổng hợp).

**Thực hiện:** CPU, 2 thread mỗi run, dữ liệu `connected_v3_auxfit_seed42`, hai run chạy song song.

**Tính toàn vẹn:** SHA-256 của protocol, script đánh giá, module cấu hình và hai runner được ghi lúc bắt đầu mỗi run, và **khớp** với file hiện tại. Không có thay đổi nào sau khi chạy.

## Kết luận
**Không xác nhận.** G3 và G4 pass; G1, G2 và G5 fail. Kết quả không đổi khi bỏ seed 10042 (phân tích độ nhạy).

- **Chế độ FPR cao do thiếu train không còn ở những chỗ nó từng xuất hiện:**
  - ensemble ban đầu, mọi S;
  - unlearn một request, ở R_base;
  - model đơn train lại sau mỗi yêu cầu trong chuỗi.

  Không dòng nào ở các phần này có FPR > 0,45. So với cấu hình cũ, ensemble S = 4 đi từ 0,646 / 0,752 lên 0,880 / 0,319.
- **G1 và G2 fail chỉ vì S = 10:**
  - macro-F1 thấp hơn S = 1 từ 0,0105 đến 0,0114, vượt biên 0,01 rất sát;
  - mức thấp hơn này nhất quán qua 5 seed (G1: từ −0,0095 đến −0,0112), nên đây là một chi phí nhỏ nhưng thật của S = 10, không phải nhiễu.
- **G5 fail vì hai lý do khác nhau:**
  - **`utility_stop` với S = 10 rơi lại chế độ FPR cao** ở 4/5 seed tại cuối chuỗi (trung bình F1 0,695, FPR 0,666). Đây là lỗi thật của chính sách này, xem mục "Diễn biến của `utility_stop`".
  - `shard_fixed` và `dynamicfu` với S = 10 thấp hơn tham chiếu 0,0104 và 0,0107, vượt biên sát.

## Gate
Ngưỡng: FPR > 0,45 là chế độ FPR cao; biên ΔF1 ≥ −0,01 và ΔFPR ≤ +0,02, ghép cặp theo seed, xét trung bình 5 seed. Metric trên calibration.

| Gate | Kết quả | Chi tiết | Bỏ seed 10042 |
|---|---|---|---|
| G1 Utility của ensemble | **FAIL** | ΔF1 so với S = 1: S = 2 +0,0005; S = 4 −0,0010; S = 5 −0,0031; **S = 10 −0,0105**. ΔFPR ≤ +0,0034. Không (S, seed) nào có FPR > 0,45 | FAIL |
| G2 Unlearn một request | **FAIL** | Ở R = 20: **`random50` với S = 10 −0,0114**; `client` với S = 10 −0,0093 (pass); các ô khác từ +0,0004 đến −0,0068. ΔFPR ≤ +0,0064. Không dòng nào có FPR > 0,45 | FAIL |
| G3 Tính chính xác | PASS | Exact ở mọi bước; trùng từng bit ở cuối chuỗi 50/50 trường hợp (2 S × 5 chính sách × 5 seed) | — |
| G4 Tham chiếu khỏe | PASS | `full_retrain_single` qua cả chuỗi: F1 thấp nhất 0,868, FPR cao nhất 0,325 | PASS |
| G5 Utility cuối chuỗi | **FAIL** | S = 4: mọi chính sách pass (ΔF1 từ −0,0039 đến −0,0095). S = 10: `shard_full` và `drop_backfill` −0,0098 (pass); **`shard_fixed` −0,0104; `dynamicfu` −0,0107; `utility_stop` −0,179, ΔFPR +0,344, 4/5 seed có FPR cuối > 0,45** | FAIL |

**Ngay cả các ô pass cũng sát biên.** Ở trạng thái cuối, các chính sách shard thấp hơn tham chiếu từ 0,008 đến 0,011 macro-F1, trừ `utility_stop` (−0,004 với S = 4; hỏng với S = 10). Nói cách khác, với cấu hình mới model đơn tốt lên (0,874 cuối chuỗi), còn shard train trên ít dữ liệu hơn thì kém hơn một chút.

## So với cấu hình cũ trên cùng dữ liệu auxfit
Cấu hình cũ lấy từ `results/shard_exploration_auxfit/`. Kết quả cũ chạy trong container nên không so từng bit được, nhưng mức chênh ở đây lớn hơn nhiều so với sai khác giữa hai máy.

| macro-F1 / FPR (calibration) | Cũ: lr 1e-3, 15 vòng, 1 epoch | **Chuẩn mới: lr 1e-2, 20 vòng, 2 epoch** |
|---|---|---|
| Ensemble S = 1 | 0,865 / 0,324 | 0,881 / 0,320 |
| Ensemble S = 2 | 0,871 / 0,324 | 0,882 / 0,317 |
| Ensemble S = 4 | 0,646 / 0,752 | 0,880 / 0,319 |
| Ensemble S = 5 | 0,646 / 0,753 | 0,878 / 0,324 |
| Ensemble S = 10 | 0,645 / 0,752 | 0,871 / 0,323 |
| Model đơn train lại, cuối chuỗi | 0,634 / 0,772 | 0,874 / 0,322 |
| Số bước có FPR > 0,45 trong chuỗi (mọi chính sách, kể cả `none` và tham chiếu; bước 1–10) | 607 / 650 | 23 / 650 |

## Run 1: utility và chi phí unlearn một request
- **Request:** trên client lớn nhất (client 10, 29.317 dòng); xóa cả client, hoặc xóa ngẫu nhiên phân tầng 50%.

**Chi phí unlearn chính xác** (train lại đủ 20 vòng shard bị ảnh hưởng), tính theo phần trăm dòng-lượt / byte so với train lại toàn bộ:

| S | Xóa cả client | Xóa 50% |
|---|---|---|
| 2 | 54% / 47% | 60% / 50% |
| 4 | 34% / 21% | 44% / 25% |
| 5 | 22% / 16% | 33% / 20% |
| 10 | 6% / 5% | 20% / 10% |

- **Train lại với ngân sách giảm trên một model duy nhất vẫn rất tệ ở 1 vòng:** F1 0,57–0,59, FPR 0,83–0,85.
- **Utility vẫn không đơn điệu theo số vòng.** Ví dụ S = 1, xóa cả client, sau 2 vòng: F1 0,845 nhưng FPR chỉ 0,083.
- **Với S ≥ 4, chỉ 1 vòng đã cho utility gần mức R_base** (F1 0,869–0,881, FPR 0,319–0,324).

## Run 2: chuỗi 10 yêu cầu (xóa 60,5% dữ liệu train)

| S | Chính sách | macro-F1 cuối | FPR cuối | Dòng-lượt | Byte | Thời gian | F1 thấp nhất / FPR cao nhất trong chuỗi | Số bước FPR > 0,45 |
|---|---|---|---|---|---|---|---|---|
| 1 | `full_retrain_single` | 0,874 | 0,322 | 100% | 100% | 100% | 0,868 / 0,325 | 0 |
| 4 | `none` | 0,880 | 0,318 | 0 | 0 | 0 | 0,874 / 0,319 | 0 |
| 4 | `shard_full` | 0,866 | 0,321 | 26% | 24% | 30% | 0,865 / 0,323 | 0 |
| 4 | `shard_fixed` | 0,865 | 0,321 | 13% | 12% | 15% | 0,864 / 0,325 | 0 |
| 4 | `dynamicfu` | 0,865 | 0,322 | 17% (+5,5% ước lượng đóng góp) | 15% | 30% | 0,865 / 0,323 | 0 |
| 4 | `utility_stop` | 0,870 | 0,322 | 11% | 10% | 14% | **0,650 / 0,750** | 1 |
| 4 | `drop_backfill` | 0,866 | 0,321 | 13% | 14% | 15% | 0,841 / 0,332 | 0 |
| 10 | `none` | 0,873 | 0,322 | 0 | 0 | 0 | 0,870 / 0,323 | 0 |
| 10 | `shard_full` | 0,865 | 0,323 | 8,1% | 8,2% | 9,1% | 0,863 / 0,325 | 0 |
| 10 | `shard_fixed` | 0,864 | 0,324 | 4,0% | 4,1% | 4,6% | 0,863 / 0,326 | 0 |
| 10 | `dynamicfu` | 0,864 | 0,326 | 5,0% (+5,5%) | 4,6% | 15% | 0,863 / 0,326 | 0 |
| 10 | `utility_stop` | **0,695** | **0,666** | 1,7% | 2,3% | 3,0% | **0,648 / 0,752** | 13 |
| 10 | `drop_backfill` | 0,865 | 0,323 | 5,2% | 5,9% | 5,5% | 0,799 / 0,471 | 9 |

- **Mọi chính sách chính xác đều exact ở mọi bước và trùng từng bit ở cuối** (G3).
- **`none` có utility bằng hoặc cao hơn train lại**, giống các run trước. Utility vì vậy vẫn không dùng làm bằng chứng xóa được.
- **DynamicFU:**
  - dừng vì điều kiện điều khiển ở 94/95 lần và vì R_max ở 1 lần;
  - lần này **không lần nào dừng vì "Drop ≥ τ_drop"** (dữ liệu cũ: 11/100 lần);
  - ước lượng đóng góp tốn thêm 5,5% dòng-lượt, chiếm 29% (S = 4) và 53% (S = 10) thời gian của chính sách này.
- **`drop_backfill` với S = 10:** FPR lên 0,46–0,47 ở một số bước trong khoảng bước 7–10 (4/5 seed), khi có tới 4/10 shard đang chờ train lại. FPR về 0,32 sau khi train xong hàng đợi.

## Diễn biến của `utility_stop` với S = 10 (mô tả, hậu kiểm)
Shard gồm 2 client. Các bước liên quan của chuỗi (thành phần client lấy từ partition):

| Bước | Yêu cầu | Shard bị ảnh hưởng sau yêu cầu | Số vòng `utility_stop` train | FPR cuối bước (5 seed) |
|---|---|---|---|---|
| 1–6 | — | — | 1 vòng ở hầu hết các bước (một lần 2 vòng) | 0,319–0,323 |
| 7 | xóa client 7 | shard [7, 8] chỉ còn client 8: 1.056 dòng, **21 dòng normal (2%)** | 20 ở 4/5 seed (điều kiện dừng không đạt, dừng ở R_base); 15 ở seed 10044 | **0,749–0,752** ở 4/5 seed |
| 8 | xóa 30% client 12 | shard [4, 12] | 2–20 | 0,323–0,326 ở 4/5 seed; seed 10045 vẫn 0,752 |
| 9 | xóa client 17 | shard [10, 17] **rỗng** (client 10 đã bị xóa ở bước 1; client 17 có 58% normal), nên bị loại khỏi ensemble **mà không kiểm tra utility** | 0 | **0,749–0,752** ở 4/5 seed |
| 10 | xóa 30% client 1 | shard [1, 9] | 10–20 | giữ nguyên |

**Giả thuyết, chưa kiểm tra.** Ensemble là tổng margin của các shard.
- `utility_stop` để lại nhiều shard chỉ train 1 vòng, nên margin của chúng yếu.
- Một shard train đủ 20 vòng trên một client gần như toàn attack có margin lớn, đủ lấn át phần còn lại.
- Loại một shard nhiều normal cũng làm ensemble nghiêng về attack.

**Hai điểm của chính sách được bộc lộ** (đúng theo định nghĩa trong runner):
- Dừng ở R_base khi điều kiện không đạt **không đảm bảo** utility.
- Loại shard rỗng xảy ra ở mọi chính sách, nhưng không chính sách nào kiểm tra utility sau đó. Chỉ `utility_stop` bị ảnh hưởng ở đây, có thể vì các shard còn lại của nó yếu.

Ở dữ liệu cũ với cấu hình cũ, một trường hợp tương tự từng xảy ra với `shard_full`: shard chỉ còn một client có dữ liệu lệch.

## Được tuyên bố / không được tuyên bố
| Được tuyên bố | Bằng chứng |
|---|---|
| Với cấu hình chuẩn mới, **chế độ FPR cao do thiếu train không còn** ở ensemble ban đầu (mọi S), ở unlearn một request, và ở model đơn train lại | Phần "không có FPR > 0,45" của G1, G2, G4 (đặt trước, đều pass) |
| Unlearn theo shard được cài đặt **chính xác** | G3 |
| Với S ≤ 5, ensemble không kém model đơn quá 0,01 macro-F1 và 0,02 FPR | G1 theo từng S |
| Với S = 10 có một chi phí utility nhỏ nhưng nhất quán, khoảng 0,01 macro-F1 | G1, G2, G5 |

| **Không** được tuyên bố | Lý do |
|---|---|
| Cấu hình chuẩn mới đã được xác nhận | G1, G2, G5 fail |
| Utility cuối của mọi chính sách chính xác ngang train lại toàn bộ | Fail ở S = 10; ở S = 4 pass nhưng sát biên (tới −0,0095) |
| `utility_stop` là chính sách rẻ và an toàn | Rơi vào chế độ FPR cao ở S = 10 (4/5 seed cuối chuỗi) và có 1 bước sụt ở S = 4 |
| Cơ chế ở mục "Diễn biến của `utility_stop`" là nguyên nhân | Hậu kiểm, chưa có thí nghiệm kiểm tra |
| Kết quả trên partition khác, chuỗi yêu cầu khác hoặc trên test | Ngoài phạm vi của protocol |

**Điều chỉnh với [shard_exploration_results.md](shard_exploration_results.md):**
- "Chia shard không làm giảm utility, với điều kiện mỗi shard được train đủ" vẫn đúng với S ≤ 5. Với S = 10 thì có chi phí khoảng 0,01.
- "`utility_stop` không có thời điểm sụt nào" (run 2 trên dữ liệu cũ) **không còn đúng** trên auxfit với cấu hình chuẩn mới.

## Về cấu hình mặc định
- Mặc định của `scripts/shard/` là `standard`, theo quyết định đổi cấu hình chuẩn. Trạng thái của nó: **đã áp dụng, chưa được xác nhận theo protocol**.
- Theo protocol, không đổi cấu hình, ngưỡng hay seed sau khi xem kết quả.
- Một lần kiểm tra tiếp theo cần protocol mới. Các hướng gợi ý từ kết quả (chưa quyết định):
  - sửa quy tắc của `utility_stop`: kiểm tra utility cả khi loại shard rỗng, và không coi R_base là đạt khi điều kiện chưa thỏa;
  - giới hạn S, hoặc đặt số client tối thiểu cho mỗi shard;
  - dùng partition mới để kiểm tra độc lập.
- **Đã làm phần lớn ở protocol v2** ([shard_v2_plan.md](shard_v2_plan.md)): sửa script đánh giá (đường dẫn, hash nội dung dữ liệu), gate noninferiority một phía có khoảng tin cậy với G5 tách đôi, mỗi shard ≥ 4 client và gộp khi < 3, sửa `utility_stop`, 10 seed mới. Partition vẫn là partition 42. Kết quả v2: [shard_v2_results.md](shard_v2_results.md), **không xác nhận**.
- **Hash nội dung dữ liệu của v1** (ghi hậu kiểm): file npz được ghi trước khi v1 chạy và chưa đổi từ đó; hash nội dung của chúng là `f130057e…`, đúng bằng giá trị v2 yêu cầu.

## Tái lập
Lệnh trong [shard_standard_config_plan.md](shard_standard_config_plan.md), mục "Lệnh chạy". Kiểm tra lại gate trên bản đã lưu:
```bash
python scripts/shard/evaluate_standard_config.py results/shard_standard_config/ensemble_check_20c results/shard_standard_config/stream_20c
```
