# Hướng shard, protocol v2: kết quả xác nhận hai run 20 client

**Protocol:** [shard_v2_plan.md](shard_v2_plan.md), viết trước khi chạy. **Kết quả:** `results/shard_v2/`, gồm CSV, JSON, `run.json` của từng run, `gates.json` và `evaluate.log` (đầu ra của script đánh giá), `analysis_20c.txt` (bảng mô tả).

**Thực hiện:** 2026-11-07, CPU, 2 thread mỗi run, hai run chạy song song. Run 1 mất khoảng 20 phút, run 2 khoảng 56 phút. Cấu hình `standard` (Adam lr 1e-2, R_base = 20, 2 epoch, batch 512). Seed 50042–50051.

**Tính toàn vẹn:** đạt cho cả hai run.
- Hash nội dung dữ liệu `f130057e…` khớp với giá trị cố định trong protocol.
- SHA-256 của protocol, script đánh giá, `shard_v2.py`, `train_config.py` và hai runner được ghi lúc bắt đầu, và **khớp** với file hiện tại.
- Không có thay đổi nào ở protocol hay code sau khi chạy.

Lần khởi chạy đầu tiên dừng ngay ở bước tạo thư mục output, trước khi nạp dữ liệu hay train. Lý do: tôi đã tạo sẵn thư mục, mà runner từ chối ghi vào thư mục đã có. Hai thư mục rỗng được xóa rồi chạy lại cùng lệnh. Log của lần lỗi nằm ở `runs/shard_v2/*.failed_mkdir.log`.

## Kết luận
**Không xác nhận.** G1, G3, G4 và G6 pass; G2, G5a và G5b fail. Theo từng claim:

| Claim | Gate | Kết quả |
|---|---|---|
| Chia shard (≥ 4 client, gộp khi < 3) giữ được utility của model đơn | G1, G2, G5b | **Không** |
| Các chính sách train ít vòng hơn ngang `shard_full` | G5a | **Không** |
| Unlearning chính xác | G3 | **Có** |
| `utility_stop` v2 an toàn | G6, cùng các dòng `utility_stop` của G5a và G5b | **Có** |

- **Unlearning chính xác:** đạt ở mọi chính sách, mọi seed, kể cả khi gộp shard (S = 5, bước 7, shard 1 gộp vào shard 0).
- **G5b fail như protocol đã dự đoán.** Sau 10 yêu cầu, các chính sách shard trừ `utility_stop` thấp hơn model đơn train lại từ 0,008 đến 0,015 macro-F1, và cận dưới vượt biên −0,01.
  - Với `shard_full` ở S = 5, cận dưới là −0,010004, tức chỉ vượt biên 0,000004. Gate vẫn tính là fail.
  - Chỉ `utility_stop` pass G5b.
- **G2 fail ở một dòng:** S = 5, request `client`. ΔF1 trung bình −0,0095, cận dưới −0,0116. Các dòng S ∈ {2, 4} và mọi dòng `random50` pass.
- **G5a fail ở một dòng, do một seed:** `dynamicfu` S = 4, cận trên ΔFPR 0,0219 > 0,02.
  - Ở seed 50043, trạng thái cuối của `dynamicfu` rơi vào một **chế độ FPR thấp, recall thấp**: FPR 0,085, recall 0,862, F1 0,841, trong khi `shard_full` cùng seed có FPR 0,320, recall 0,985, F1 0,867.
  - FPR của seed này *thấp hơn*, nhưng nó làm sai số chuẩn tăng mạnh, nên cận trên vượt biên. Sai số chuẩn tăng theo độ lệch ở cả hai hướng, nên một seed tốt hơn hẳn cũng làm khoảng tin cậy rộng ra.
  - Gate fail đúng với một vấn đề thật, chỉ là qua metric khác: seed này mất 0,026 F1. Cận dưới ΔF1 (−0,0085) vẫn pass vì trung bình trên 10 seed.
- **`utility_stop` v2:**
  - Không bước nào có FPR > 0,45; FPR lớn nhất là 0,327.
  - Pass G5a với ΔF1 dương: +0,0054 ở S = 4 và +0,0062 ở S = 5, cận dưới đều > 0.
  - Chi phí bằng 13,1 % (S = 4) và 3,9 % (S = 5) số dòng-lượt so với train lại model đơn, so với 25,9 % và 20,5 % của `shard_full`.

## Gate
Noninferiority một phía, t ghép cặp theo seed, 95 %, n = 10. Điều kiện pass: cận dưới ΔF1 > −0,01 và cận trên ΔFPR < +0,02. FPR > 0,45 là chế độ FPR cao. Metric tính trên calibration.

| Gate | Kết quả | Chi tiết |
|---|---|---|
| G1 Utility của ensemble ban đầu | PASS | So với S = 1, cận dưới ΔF1: S = 2 −0,0015; S = 4 −0,0019; S = 5 −0,0057. Cận trên ΔFPR ≤ +0,0050 |
| G2 Unlearn một request ở R_base | **FAIL** | `client`: S = 2 −0,0053; S = 4 −0,0071; **S = 5 −0,0116** (cận dưới ΔF1). `random50`: S = 2 −0,0037; S = 4 −0,0040; S = 5 −0,0085. Cận trên ΔFPR ≤ +0,0070 |
| G3 Chính xác | PASS | Bit-exact ở cuối, exact ở mọi bước, đủ hàng đợi, cấu trúc shard đúng ở mọi chính sách và S |
| G4 Model đơn trong chuỗi (điều kiện tiên quyết) | PASS | F1 nhỏ nhất 0,867; FPR lớn nhất 0,326 |
| G5a So với `shard_full` | **FAIL** | Chỉ dòng `dynamicfu` S = 4 fail (cận trên ΔFPR 0,0219). Các dòng khác: cận dưới ΔF1 từ −0,0078 đến +0,0034, cận trên ΔFPR ≤ 0,0030 |
| G5b So với model đơn ở bước 10 | **FAIL** | Cận dưới ΔF1: `shard_full` −0,0137 / −0,0100; `shard_fixed` −0,0147 / −0,0153; `dynamicfu` −0,0201 / −0,0157; `drop_backfill` như `shard_full`; **`utility_stop` −0,0089 / −0,0044 (pass)** (S = 4 / S = 5) |
| G6 `utility_stop` qua mọi bước | PASS | F1 nhỏ nhất 0,864; FPR lớn nhất 0,327 |

## Run 1: ensemble ban đầu và unlearn một request
| S | F1 ban đầu | FPR ban đầu | F1 sau `client` (R = 20) | F1 sau `random50` (R = 20) | Dòng-lượt so với S = 1 (`client`) |
|---|---|---|---|---|---|
| 1 | 0,8812 | 0,320 | 0,8814 | 0,8816 | 100 % |
| 2 | 0,8809 | 0,317 | 0,8787 | 0,8799 | 53,5 % |
| 4 | 0,8808 | 0,318 | 0,8765 | 0,8796 | 33,8 % |
| 5 | 0,8774 | 0,324 | 0,8719 | 0,8758 | 21,8 % |

Trung bình 10 seed. Chi phí giảm khi S tăng vì chỉ shard chứa client 10 phải train lại; đổi lại, F1 sau khi xóa client 10 cũng giảm dần. Client 10 là client lớn nhất (29.317 dòng).

## Run 2: chuỗi 10 yêu cầu
Trạng thái cuối, trung bình 10 seed. Chi phí là phần trăm so với train lại model đơn sau mỗi yêu cầu (`full_retrain_single`). Chi phí của `drop_backfill` gồm cả phần train bù.

| S | Chính sách | F1 | FPR | F1 client tệ nhất | Dòng-lượt | Thời gian |
|---|---|---|---|---|---|---|
| 1 | `full_retrain_single` | 0,877 | 0,321 | 0,790 | 100 % | 100 % |
| 4 | `shard_full` | 0,867 | 0,320 | 0,739 | 25,9 % | 26,5 % |
| 4 | `shard_fixed` | 0,865 | 0,321 | 0,734 | 13,0 % | 13,5 % |
| 4 | `dynamicfu` | 0,863 | 0,298 | 0,721 | 17,2 % | 26,8 % |
| 4 | `utility_stop` | 0,872 | 0,321 | 0,767 | 13,1 % | 14,3 % |
| 4 | `utility_stop_v1` | 0,871 | 0,322 | 0,767 | 11,7 % | 13,3 % |
| 4 | `drop_backfill` | 0,867 | 0,320 | 0,739 | 13,3 % | 13,8 % |
| 5 | `shard_full` | 0,870 | 0,324 | 0,753 | 20,5 % | 20,9 % |
| 5 | `shard_fixed` | 0,864 | 0,325 | 0,738 | 10,3 % | 10,6 % |
| 5 | `dynamicfu` | 0,864 | 0,325 | 0,738 | 13,3 % | 23,0 % |
| 5 | `utility_stop` | 0,876 | 0,326 | 0,797 | 3,9 % | 4,6 % |
| 5 | `utility_stop_v1` | 0,876 | 0,326 | 0,799 | 4,0 % | 4,9 % |
| 5 | `drop_backfill` | 0,870 | 0,324 | 0,753 | 11,2 % | 11,3 % |

FPR trung bình 0,298 của `dynamicfu` S = 4 là do seed 50043 (FPR 0,085). Chín seed còn lại có FPR từ 0,321 đến 0,322.

## Mô tả thêm (không phải gate)
**`utility_stop` v2 so với v1.**
- Ở S ∈ {4, 5}, `utility_stop_v1` (quy tắc cũ) cũng không có bước nào FPR > 0,45. Như vậy chỉ riêng ràng buộc ≥ 4 client mỗi shard đã loại được hiện tượng sụp đổ ở S = 10 của v1. **G6 pass không chứng minh phần dự phòng là cần thiết.**
- Có một trường hợp phần dự phòng tạo ra khác biệt: S = 5, seed 50049, bước 8.
  - v1 dừng ở R_base và rơi vào chế độ FPR thấp, recall thấp (F1 0,846, FPR 0,084).
  - v2 cho thêm 2 vòng dự phòng, đạt lại ok(), và giữ F1 0,869.
- Ở S = 4, phần dự phòng chạy ở 15 bước (tổng 374 vòng).
- Riêng ở S = 4, có 13 (seed, bước) kết thúc mà không đạt ok(). Khi đó mọi shard đã đủ R_base, nên trạng thái trùng với `shard_full`, ví dụ seed 50045, 50046, 50048 ở bước 10. Điều này nghĩa là sau khi xóa các client này, ensemble không quay lại được F1₀ − 0,005 trên validation dù train đủ vòng.
- Ở S = 5, phần dự phòng chỉ chạy 1 lần (2 vòng), và mọi bước đều đạt ok().

**Vì sao `utility_stop` cao hơn `shard_full`?** Đây là quan sát, chưa được kiểm định, và protocol không đặt gate tốt hơn.
- Khi điều kiện đạt sớm, ensemble của `utility_stop` có recall tấn công cao hơn (khoảng 0,994 so với 0,984) với FPR gần như bằng.
- Một giả thuyết: số vòng được chọn theo utility trên validation, nên có tác dụng như early stopping.
- Một giả thuyết khác: một shard chỉ train 1–3 vòng cho logit nhỏ, nên đóng góp ít vào tổng, và ensemble dựa chủ yếu vào các shard còn lại.

Hai giả thuyết này chưa được tách riêng.

**Chế độ FPR thấp, recall thấp (phát hiện sau khi chạy).** Đặc điểm: FPR khoảng 0,065–0,085, recall khoảng 0,85–0,86, F1 khoảng 0,84. Chế độ này xuất hiện ở:
- `drop_backfill`, các bước trung gian khi một shard đang bị loại chờ train bù: S = 4 bước 6–8 ở mọi seed, và S = 5 bước 10 ở mọi seed. Vì nó lặp lại ở mọi seed tại cùng bước, nhiều khả năng nó phụ thuộc vào việc client nào vắng mặt, chứ không phải nhiễu train. Điều này chưa được kiểm tra.
- `dynamicfu` S = 4 seed 50043, các bước 7, 8 và 10.
- `utility_stop_v1` S = 5 seed 50049, bước 8.

Các gate chỉ chặn chế độ FPR cao (> 0,45), nên chế độ này chỉ bị bắt gián tiếp qua F1. Ngoài ra, `drop_backfill` S = 4 seed 50048 có FPR 0,472 ở bước 3–4 (trạng thái trung gian, không thuộc gate nào).

**DynamicFU:** lý do dừng gồm `ctrl` 96 / 99, `drop` 3 / 0, `rmax` 1 / 1 (S = 4 / S = 5).

**Nhãn trong `analysis_20c.txt`:** script phân tích của v1 in nhãn "5 seeds" ở mục 1, nhưng thực tế tính trên cả 10 seed có trong CSV. Trung bình ở mục 1 khớp với bảng G1.

## So với v1
- Hiện tượng sụp đổ của `utility_stop` ở v1 chỉ có ở S = 10. Ràng buộc mới loại S = 10, và ở S ∈ {4, 5} hiện tượng này không xuất hiện.
- Khoảng cách với model đơn sau chuỗi xóa vẫn còn. Ở v1, S = 4 thấp hơn từ 0,004 đến 0,0095. Ở v2, trừ `utility_stop`, khoảng cách là 0,008–0,015 và cận dưới vượt biên.
- G2 ở S = 5 (`client`) là chỗ fail mới so với v1.
  - Ở v1 (5 seed khác), mọi ô có S ≤ 5 có ΔF1 trung bình từ +0,0004 đến −0,0068.
  - Ở v2, ô này có trung bình −0,0095, đã sát biên ngay cả khi chỉ xét trung bình. Cộng sai số thì cận dưới vượt −0,01.

## Được tuyên bố / không được tuyên bố
Được tuyên bố, trên partition 42 và chuỗi seed 99:
- Unlearning chính xác của mọi chính sách shard, kể cả khi gộp shard.
- Ở S ∈ {2, 4}, unlearn một request rồi train lại đủ R_base vòng là noninferior so với train lại model đơn.
- `utility_stop` v2 không rơi vào chế độ FPR cao ở bước nào, và noninferior so với cả `shard_full` lẫn model đơn ở trạng thái cuối.

Không được tuyên bố:
- Chia shard giữ được utility của model đơn qua chuỗi 10 yêu cầu (G5b fail), hoặc khi xóa client lớn nhất ở S = 5 (G2).
- Các chính sách train ít vòng ngang `shard_full`. G5a fail ở `dynamicfu` S = 4; `shard_fixed` pass nhưng không đủ để tuyên bố cho cả nhóm.
- Phần dự phòng của `utility_stop` v2 là cần thiết. Quy tắc v1 cũng pass ngưỡng FPR cao ở S ∈ {4, 5}.
- `utility_stop` tốt hơn `shard_full`. Có quan sát, nhưng protocol không đặt gate tốt hơn và cơ chế chưa được kiểm tra.
- Kết quả cho partition, chuỗi yêu cầu hoặc tập test khác.

## Tái lập
```bash
export SHARD_DESIGN=v2 FL_DEVICE=cpu
python scripts/shard/run_shard_ensemble_check_v2.py runs/shard_v2/ensemble_check_20c
python scripts/shard/run_shard_stream_v2.py runs/shard_v2/stream_20c
python scripts/shard/evaluate_shard_v2.py runs/shard_v2/ensemble_check_20c runs/shard_v2/stream_20c runs/shard_v2/gates.json
python scripts/shard/analyze_shard_20c.py runs/shard_v2/ensemble_check_20c runs/shard_v2/stream_20c
```
Thư mục output không được tồn tại trước khi chạy. Có thể chạy lại script đánh giá trên bản chép: `python scripts/shard/evaluate_shard_v2.py results/shard_v2/ensemble_check_20c results/shard_v2/stream_20c`.

**Tái lập trên máy khác (bổ sung 2026-11-10, sau khi có kết quả; không thay đổi kết quả nào ở trên).** Runner v2 nguyên bản đã được chạy lại trên một máy Linux cho seed 50042–50044, cùng dữ liệu (hash nội dung khớp). Kết quả **không trùng từng bit** với `results/shard_v2/`: các chính sách có số vòng cố định cho cùng số vòng và metric lệch dưới 0,01 ở trạng thái cuối, còn `utility_stop` và `dynamicfu` đi theo số vòng khác ở phần lớn các bước; ở S = 4, `drop_backfill` và `utility_stop_v1` có vài bước FPR > 0,45 mà kết quả ở đây không có. Trung bình trạng thái cuối trên 3 seed lệch không quá 0,002 macro-F1. Các gate chưa được chạy lại trên máy đó. Chi tiết: [shard_dt_interface.md](shard_dt_interface.md), mục "Runner v2 trên máy này so với kết quả v2 đã lưu".
