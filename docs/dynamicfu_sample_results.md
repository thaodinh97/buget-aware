# DynamicFU mức mẫu trên kiến trúc shard: kết quả

Protocol: [dynamicfu_sample_plan.md](dynamicfu_sample_plan.md), ghi trước khi chạy và không sửa sau đó.

**Các run:**
- `runs/dynamicfu_requests_p1042` và `runs/dynamicfu_tuning_p1042`: pilot;
- `runs/dynamicfu_requests_p42`: request của partition chính;
- `runs/dynamicfu_p42/seed42…seed46`: 5 seed;
- `runs/dynamicfu_p42/gates`: kết quả chấm cổng.

Verifier pass cả 5 seed: mỗi seed kiểm 108 checkpoint và chạy lại 8 stage, đều trùng từng bit. Số liệu dưới đây là trung bình qua 5 seed, trừ khi ghi khác.

## Verdict: KHÔNG được sang phase DT với `dynamicfu`

| Cổng | Kết quả |
|---|---|
| 1. Exact | **PASS.** 180/180 cặp request × seed pass mọi kiểm tra |
| 2. Canary | **TRƯỢT.** 3 request được admit; 2 là `RETRAIN_LIKE`, 1 là `INTERMEDIATE` |
| 3. Utility | **TRƯỢT.** 12/36 request đạt biên 0,01 macro-F1 và 0,01 FPR |
| 4. Chi phí | chỉ báo cáo: ít hơn Full Retrain 33–116 lần theo số mẫu xử lý |

Theo protocol, kết quả không được chấm lại.

## 1. Chọn tham số trên pilot (partition 1042)

- **Tham số được chọn:** ρ = 0,25, μ = 5, τ_ctrl = 0,02, τ_drop = 0,01.
- **Cả 320/320 điểm lưới đều khả thi,** vì trên pilot F1 validation gần như phẳng theo số vòng (α = 0,5: 0,897 ở vòng 0 và 0,897 ở vòng 15).
  - Ràng buộc utility vì vậy không có tác dụng chọn lọc. Quy tắc chỉ còn tối thiểu số vòng, nên điểm được chọn nằm ở biên lưới.
  - Đây là nguyên nhân gốc của các lần trượt ở mục 4: **pilot không đại diện cho partition chính.**

## 2. Request (partition 42, chọn trên model seed 42)

- Mỗi α có 9 request tự nhiên (3 mỗi nhóm high, medium, low theo I_F) và 3 canary, tổng cộng 36 request.
- Ở α = 0,1:
  - 21/40 ứng viên không hợp lệ vì client chỉ có một lớp;
  - một shard chiếm 77.248/114.284 dòng;
  - ensemble gốc gần như suy biến: macro-F1 0,55, FPR 0,87. FedAvg không shard còn tệ hơn: 0,43, FPR 1,00.

## 3. Các cổng

**Cổng 1.** Mọi kiểm tra đều pass ở cả 5 seed:
- không dòng nào của F đi vào worker giữ lại;
- quỹ đạo ở vòng 15 trùng `full_retrain`, và các shard khác của `full_retrain` trùng shard gốc;
- `dynamicfu` và `dynamicfu_ffree` trùng quỹ đạo tại vòng dừng;
- tín hiệu online trùng tín hiệu của quỹ đạo, và phát lại cho cùng vòng dừng;
- với request canary, `full_retrain` trùng model gốc tự nhiên.

**Cổng 2.** Chỉ số là P(normal) trên attack test có trigger. Bảng dưới gồm 3 request được admit:

| Request | Original | Retrain (khoảng dự đoán 95%) | `dynamicfu` | Verdict | Vị trí quên |
|---|---|---|---|---|---|
| α 0,1 · canary_c10 | 0,967 | 0,0046 [0,0022; 0,0071] | 0,0057 | `RETRAIN_LIKE` | 1,00 |
| α 0,5 · canary_c4 | 0,677 | 0,0146 [0,0098; 0,0194] | 0,0213 | **`INTERMEDIATE`** | 0,99 |
| α 1,0 · canary_c16 | 0,924 | 0,049 [−0,031; 0,129] | 0,108 | `RETRAIN_LIKE` | 0,93 |

- **6 canary còn lại không được admit.** Ensemble 5 shard pha loãng canary: original chỉ đạt 0,06–0,59, và khoảng dự đoán của hai cloud giao nhau.
- **Lần trượt ở canary_c4 không phải do còn sót canary.** Shard được retrain chưa từng thấy canary. Chênh lệch đến từ việc shard chỉ được train 3–4 vòng. `dynamicfu_ffree` (nhiều vòng hơn) đạt 0,0189, nằm trong khoảng dự đoán.

**Cổng 3.** Lấy `full_retrain` trừ `dynamicfu`, trung bình qua 5 seed:

| α | Request đạt | Macro-F1 mất đi lớn nhất | FPR tăng lớn nhất |
|---|---|---|---|
| 0,1 | 4/12 | 0,119 | 0,131 |
| 0,5 | 2/12 | 0,188 | 0,381 |
| 1,0 | 6/12 | 0,026 | âm (FPR giảm, nhưng bắt sót attack) |

**Utility trung bình trên test:**

| Phương pháp | Macro-F1 α 0,1 / 0,5 / 1,0 | FPR α 0,1 / 0,5 / 1,0 |
|---|---|---|
| `original` | 0,548 / 0,887 / 0,884 | 0,867 / 0,279 / 0,271 |
| `full_retrain` = `affected_shard_retrain` | 0,529 / 0,884 / 0,884 | 0,888 / 0,285 / 0,271 |
| `rapid_retrain` (8 vòng) | 0,492 / 0,830 / 0,884 | 0,930 / 0,395 / 0,271 |
| `dynamicfu` | 0,467 / 0,767 / 0,874 | 0,957 / 0,522 / 0,224 |
| `dynamicfu_ffree` | 0,550 / 0,774 / 0,884 | 0,858 / 0,508 / 0,271 |
| `global_original` (không shard) | 0,428 / 0,888 / 0,885 | 1,000 / 0,272 / 0,271 |

Số request đạt biên của cổng 3: `dynamicfu` 12/36, `dynamicfu_ffree` 22/36, `rapid_retrain` 18/36.

**Cổng 4. Chi phí** (Full Retrain xử lý khoảng 1,69 triệu mẫu và mất 8,3–8,7 giây):

| α | `affected_shard_retrain` | `rapid_retrain` | `dynamicfu` | `dynamicfu_ffree` |
|---|---|---|---|---|
| 0,1 | 35,6× (12,0× wall) | 66,8× | 115,5× (33,9× wall) | 76,8× |
| 0,5 | 6,0× (5,7× wall) | 11,2× | 36,6× (25,2× wall) | 17,9× |
| 1,0 | 6,0× (5,8× wall) | 11,2× | 33,1× (23,8× wall) | 9,8× |

- Mỗi request, I_F tốn 0,025–0,031 giây.
- Pool client được tính một lần cho mỗi model và không cộng vào từng request.

## 4. Vì sao `dynamicfu` trượt

1. **Ensemble sụt utility trong khi shard train lại từ đầu.** Sau 1–4 vòng, shard này cho logit lệch mạnh và kéo cả ensemble 5 shard xuống. Utility hồi lại sau 8–15 vòng.
   - Trung bình ở α = 0,5: F1 test 0,838 ở vòng 0, còn 0,775 ở vòng 3, rồi lên 0,884 ở vòng 15.
   - Ở request c2_f0.5: F1 validation 0,87 ở vòng 0, còn 0,65 ở vòng 2–4, rồi lên 0,90 ở vòng 10.
   - F1 validation bám sát F1 test, nên tín hiệu không sai.
2. **Điều kiện dừng "Drop ≥ τ_drop" của bài báo bắt đúng chỗ sụt.** Nó hiểu đó là "retrain thêm đang làm hại", nên dừng ngay ở model tệ nhất.
   - Với `dynamicfu`, 40/60 lần dừng ở α = 0,1 và 40/60 ở α = 0,5 là vì `utility_drop`. Trung bình chỉ sau khoảng 3 vòng.
   - Bài báo báo cáo mức sụt utility cực nhỏ, có thể vì ensemble 10 shard ít nhạy với một shard hơn.
3. **I_F nhỏ khiến số vòng mục tiêu ban đầu thấp,** khoảng 4–7 vòng.
   - Với chuẩn hóa theo pool client, F nhỏ hơn client nhỏ nhất nên ū và q̄ thường bị clip về 0. Ở request canary, ū = 0 trong 87–100% trường hợp, và q̄ = 0 trong 73–87% trường hợp.
   - Khi đó I_F gần như chỉ còn d̄/3, tức chỉ phụ thuộc vào nhãn. Ví dụ, I_F của canary_c10 ở α = 1 bằng 0,0797 ở cả 5 seed.
4. **I_F không giúp gì.**
   - `dynamicfu_ffree` (η₀ = 0,6, không dùng F) có macro-F1 cao hơn `dynamicfu` ở cả ba α (0,550 / 0,774 / 0,884 so với 0,467 / 0,767 / 0,874), và đạt 22/36 request so với 12/36.
   - Budget cố định cùng số vòng trung bình (3, 3 và 4 vòng) cho utility ngang hoặc tốt hơn `dynamicfu`: macro-F1 0,466 / 0,775 / 0,884, so với 0,467 / 0,767 / 0,874.

## 5. Báo cáo thêm, không chấm

- **Request tự nhiên không đo được: 0/27 identifiable** so với sàn nhiễu seed, giống phase FU trước.
- **MIA không mang thông tin.** AUC khoảng 0,65 ở mọi model, kể cả `full_retrain` (0,649) là model chưa từng thấy F. Attack đo độ lệch phân phối giữa F và test, không đo membership.
- **Chia shard không làm mất utility ở α = 0,5 và 1,0.** `original` và `global_original` gần bằng nhau (0,887 so với 0,888). Ở α = 0,1, cả hai đều suy biến.

## 6. Được và không được tuyên bố

| Được tuyên bố | Không được tuyên bố |
|---|---|
| Retrain shard bị ảnh hưởng là exact, có kiểm chứng: trùng từng bit với Full Retrain ở 180/180 cặp request × seed, và ít hơn 6–36 lần số mẫu xử lý | `dynamicfu` giữ được utility, hoặc qua được các cổng |
| Trên canary được admit, mọi model retrain từ dữ liệu giữ lại đều xóa được backdoor: vị trí quên 0,93–1,00 | Điểm đóng góp I_F giúp chọn số vòng tốt hơn một budget cố định hoặc controller không dùng F |
| Điều kiện dừng theo Drop của DynamicFU thất bại khi shard được retrain từ đầu, vì ensemble sụt utility tạm thời | Request tự nhiên có ảnh hưởng đo được |

## 7. Hướng tiếp theo (cần quyết định)

- **(A) Dùng `affected_shard_retrain` cho phase DT.**
  - Phương pháp này exact. Nó thỏa điều kiện của cổng 2 và 3 một cách hiển nhiên, vì trùng từng bit với Full Retrain.
  - Số vòng cố định ở 15, khớp với thiết kế DT của đề cương: số vòng cố định, DT chọn client và kiểm ngân sách.
  - **Lưu ý:** cổng được đặt trước cho `dynamicfu`. Chọn (A) là quyết định sau khi xem kết quả, và phải ghi rõ như vậy.
- **(B) Sửa DynamicFU rồi xác nhận lại.**
  - Ví dụ: bỏ điều kiện dừng theo Drop, hoặc chỉ cho dừng sau khi utility đã hồi; chọn tham số trên nhiều partition pilot.
  - Sau đó ghi protocol mới, và xác nhận trên một partition mới với seed mới, không dùng lại partition 42.
