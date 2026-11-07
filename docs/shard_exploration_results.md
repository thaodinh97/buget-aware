# Khám phá: unlearn chính xác theo shard (kiểu SISA / DynamicFU)

**Trạng thái:** thực nghiệm khám phá, **sau đóng băng** ([fu_freeze.md](fu_freeze.md)), không có protocol đặt trước. Thiết kế của từng run được cố định trong docstring của script trước khi chạy. Các run này không thay đổi bước FU đã đóng băng; mục đích là đánh giá tính khả thi của việc chuyển thành phần FU sang unlearn chính xác theo shard. **Không được trích làm bằng chứng xác nhận.**

**Mã nguồn:** `scripts/shard/`. **Kết quả:** `results/shard_exploration/`, gồm CSV, JSON và bảng tổng hợp `analysis_20c.txt` và `paper_scale/analysis.txt`.

## Động lực
Theo [fu_freeze.md](fu_freeze.md):
- `ga_proj@joint_aux_prob` là "ứng viên, không được xác nhận". Bằng chứng tích cực chỉ có trên canary tổ hợp (backdoor tổng hợp có trigger nằm trên các feature đang dùng).
- Trên request tự nhiên, phần lớn request không đo được (19/23 trên 3 partition). Ở 4 request đo được, bước quên không vượt control công bằng, và có một request bị hại.
- Unlearn chính xác theo shard đảm bảo việc xóa **theo cấu trúc**: shard bị ảnh hưởng được train lại từ khởi tạo mới trên dữ liệu còn lại, nên không cần đo được hiệu ứng thì mới chứng minh được việc xóa.

## Thiết lập chung
- **Dữ liệu:** `data/processed/connected_v3_seed42`. Chính sách được đọc **validation**, metric báo cáo trên **calibration**, **test không được dùng**.
- **Model:** MLP 38→64→32→2 của repo cho mỗi shard. Dự đoán của ensemble là **tổng margin của các shard** (DynamicFU, phương trình 5).
- **Seed:** train dùng seed 10042–10046 (5 seed). Shard j dùng seed s + 1000·j. Train lại một shard dùng đúng seed đó, nên R vòng đầu là tiền tố của lần train đầy đủ.
- **Chuỗi yêu cầu xóa:** sinh chỉ từ metadata (seed 99), giống nhau với mọi chính sách. Bước lẻ xóa cả client, bước chẵn xóa 30% dữ liệu còn lại của một client.
- **Các chính sách** (shard bị ảnh hưởng luôn được train lại từ khởi tạo mới, chỉ trên dữ liệu còn lại):

| Chính sách | Mô tả |
|---|---|
| `none` | không unlearn; chỉ để tham chiếu, không phải một cách xóa hợp lệ |
| `shard_full` | train lại đủ R_base vòng (Affected-Shard Full Retrain) |
| `shard_fixed` | số vòng cố định nhỏ hơn (giống Rapid Retraining) |
| `dynamicfu` | **bản cài đặt lại** theo phương trình 9–21 của bài DynamicFU. Tham số do chúng tôi chọn, vì bài không công bố; validation chung thay cho validation theo shard |
| `utility_stop` | train từng vòng, dừng ở vòng đầu tiên mà validation thỏa macro-F1 ≥ F1₀ − 0,005 và FPR ≤ FPR₀ + 0,005 (F1₀, FPR₀ là của ensemble trước mọi yêu cầu) |
| `drop_backfill` | loại shard khỏi ensemble ngay, rồi dùng một ngân sách vòng cố định mỗi bước để train bù các shard đang chờ |
| `full_retrain_single` | tham chiếu: một model duy nhất train lại từ đầu trên toàn bộ dữ liệu còn lại sau mỗi yêu cầu |

## Kiểm chứng tính chính xác (cả ba run)
- **Sau mọi bước, không shard nào đang được dùng từng train trên dòng đã xóa.** Đúng với mọi chính sách trừ `none`.
- **Cuối mỗi chuỗi, mọi shard trùng từng bit** với một model train mới từ đầu (cùng seed, cùng số vòng) trên dữ liệu còn lại. Đúng với mọi chính sách, mọi S, mọi seed và mọi α.
- **Ở protocol của bài (run 3), Affected-Shard trùng từng bit với Full Retrain** (train lại cả 10 shard): 135/135 trường hợp.

**Điều kiện chưa đạt:**
- Ở ba run trên, scaler, median và vocabulary được fit trên toàn bộ train, kể cả dữ liệu sẽ bị xóa. **Đã khắc phục bằng bản dữ liệu auxfit** (fit chỉ trên validation); xem mục "Tiền xử lý fit trên tập phụ trợ" bên dưới.
- `dynamicfu` chọn số vòng từ điểm đóng góp tính trên dữ liệu của client bị xóa. Vì vậy số vòng, và qua đó model sau xóa, vẫn phụ thuộc F qua một kênh nhỏ. Các chính sách khác không dùng dữ liệu bị xóa để ra quyết định.

## Run 1: utility của ensemble và chi phí xóa (20 client)
`scripts/shard/run_shard_ensemble_check.py`.
- **Thiết lập:** 20 client, Dirichlet α = 0,5 (seed 42); S ∈ {1, 2, 4, 5, 10}. FL như cấu hình của repo: 15 vòng, 1 epoch, Adam 1e-3, batch 512.
- **Yêu cầu xóa:** trên client lớn nhất (client 4, 23.019 dòng): xóa cả client, hoặc xóa ngẫu nhiên phân tầng 50%.

**Chia shard không làm giảm utility:**

| S | macro-F1 | FPR | F1 client tệ nhất |
|---|---|---|---|
| 1 (FedAvg chuẩn) | 0,863 ± 0,004 | 0,333 | 0,590 |
| 2 | 0,868 ± 0,003 | 0,333 | 0,586 |
| 4 | 0,861 ± 0,001 | 0,333 | 0,594 |
| 5 | 0,865 ± 0,004 | 0,333 | 0,598 |
| 10 | 0,869 ± 0,000 | 0,333 | 0,579 |

**Chi phí của unlearn chính xác** (train lại đủ 15 vòng shard bị ảnh hưởng), tính theo phần trăm dòng-lượt / byte so với train lại toàn bộ:

| S | Xóa cả client | Xóa 50% |
|---|---|---|
| 2 | 45% / 47% | 51% / 50% |
| 4 | 10% / 21% | 20% / 25% |
| 5 | 17% / 16% | 26% / 20% |
| 10 | 4% / 5% | 15% / 10% |

- **Train lại với ngân sách giảm trên một model duy nhất (S = 1) cho kết quả rất tệ.** Sau 1–5 vòng, F1 chỉ 0,43–0,58 và FPR 0,67–0,93.
- **Utility thay đổi không đơn điệu theo số vòng.** Ở S = 2, sau 5 vòng F1 tụt xuống 0,63–0,73 và FPR lên 0,59–0,76, rồi hồi phục ở vòng 10.
- **Shard ít client và lệch có thể làm hại ensemble.** Ở S = 10 với yêu cầu xóa cả client, train lại đủ vòng cho F1 thấp hơn so với chỉ train 1 vòng (0,861 so với 0,869).

## Run 2: chuỗi 10 yêu cầu (20 client, S ∈ {4, 10})
`scripts/shard/run_shard_stream.py`.
- **Chuỗi yêu cầu:** xóa client 5, 2, 18, 4, 3, cộng 5 lần xóa 30% dữ liệu; tổng cộng **56% dữ liệu train**.
- **Chính sách:** `shard_fixed` = 7 vòng; `drop_backfill` = 5 vòng mỗi bước.

| S | Chính sách | macro-F1 cuối | FPR cuối | Dòng-lượt | Byte | Thời gian | F1 thấp nhất / FPR cao nhất trong chuỗi |
|---|---|---|---|---|---|---|---|
| 1 | `full_retrain_single` | 0,845 | 0,077 | 100% | 100% | 100% | 0,809 / 0,334 |
| — | `none` | 0,862 | **0,333** | 0 | 0 | 0 | — |
| 4 | `shard_full` | 0,845 | 0,072 | 21% | 22% | 22% | 0,844 / 0,333 |
| 4 | `shard_fixed` | 0,848 | 0,082 | 10% | 10% | 11% | 0,844 / 0,334 |
| 4 | `dynamicfu` | 0,847 | 0,080 | 11,5% (+7,2% ước lượng đóng góp) | 12% | 26% | **0,642 / 0,753** |
| 4 | `utility_stop` | 0,844 | 0,071 | 7,6% | 10% | 10% | 0,844 / 0,334 |
| 4 | `drop_backfill` | 0,845 | 0,072 | 11% | 13% | 11% | **0,434 / 1,000** |
| 10 | `shard_full` | 0,852 | 0,080 | 6,5% | 8,8% | 7,6% | **0,642 / 0,753** |
| 10 | `shard_fixed` | 0,848 | 0,080 | 3,1% | 4,1% | 4,0% | 0,845 / 0,334 |
| 10 | `dynamicfu` | 0,855 | 0,080 | 4,5% (+7,2%) | 6,0% | 19% | 0,846 / 0,334 |
| 10 | `utility_stop` | 0,845 | 0,076 | 3,9% | 4,2% | 6,1% | 0,844 / 0,334 |
| 10 | `drop_backfill` | 0,852 | 0,080 | 5,8% | 7,7% | 6,4% | **0,630 / 0,774** |

- **Việc xóa lần này đo được rõ.** `none` có FPR 0,333, còn train lại cho 0,077. Mọi chính sách chính xác đều về đúng mức của train lại.
- **`dynamicfu` dừng 11/100 lần vì quy tắc "Drop ≥ τ_drop"**, tức dừng đúng lúc utility đang sụt. Một lần để lại FPR 0,75 (S = 4, seed 10042, bước 3).
- **`shard_full` ở S = 10, bước 5:** shard chỉ còn một client có dữ liệu lệch; model train đầy đủ làm FPR lên 0,75 ở 2/5 seed.
- **`drop_backfill` tích tụ hàng đợi:** tới bước 10 có 3–6 shard đang chờ, và FPR lên 0,77–1,00.
- **`utility_stop` không có thời điểm sụt nào,** và là chính sách rẻ nhất hoặc gần rẻ nhất.

## Run 3: quy mô của bài DynamicFU (50 client, 10 shard)
`scripts/shard/run_shard_paper_scale.py`, tổng hợp bằng `scripts/shard/analyze_shard_paper_scale.py`.
- **Giống bài:** 50 client, 10 shard × 5 client, 20 vòng, 2 epoch local, Adam lr 1e-2, batch 512; α ∈ {0,1; 0,5; 1,0}.
- **Chính sách:** `shard_fixed` = 10 vòng (như Rapid Retraining của bài); `drop_backfill` = 7 vòng mỗi bước; DynamicFU với R_base = R_max = 20.
- **Partition:**
  - α = 0,5 và α = 1,0: thủ tục Dirichlet theo lớp của repo, chỉ yêu cầu mỗi client ≥ 10 dòng. Seed hợp lệ đầu tiên là 43 (α = 0,5) và 42 (α = 1,0).
  - α = 0,1: thủ tục theo lớp cho client rỗng ở **2.000/2.000 seed** đã thử, nên run này dùng **partition client cân bằng** (mỗi client có tỉ lệ lớp lấy theo Dirichlet). Có 29/50 client chỉ một lớp.
- **Protocol của bài (phần B):** 9 client mục tiêu mỗi α (3 cao, 3 trung bình, 3 thấp, theo điểm đóng góp DynamicFU, phương trình 9–12); mỗi lần xóa một client khỏi ensemble gốc.

**Phần B, trung bình 9 trường hợp × 5 seed:**

| α | Phương pháp | macro-F1 | \|ΔF1\| so với Full Retrain | Số vòng | Speedup (trung vị) |
|---|---|---|---|---|---|
| 0,5 | Full Retrain | 0,8601 | 0 | 20 | 1× |
| 0,5 | Affected-Shard | 0,8601 | 0 (trùng từng bit) | 20 | 20× |
| 0,5 | `none` | 0,8601 | 0,0003 | 0 | — |
| 0,5 | Rapid (10) | 0,8600 | 0,0003 | 10 | 36× |
| 0,5 | `dynamicfu` | 0,8600 | 0,0003 | 10,3 | 25× |
| 0,5 | `utility_stop` | 0,8607 | 0,0008 | 1 | 158× |
| 1,0 | `none` | 0,8633 | 0,0005 | 0 | — |
| 1,0 | Rapid (10) | 0,8626 | 0,0008 | 10 | 25× |
| 1,0 | `dynamicfu` | 0,8628 | 0,0006 | 11,0 | 19× |
| 1,0 | `utility_stop` | 0,8636 | 0,0017 | 1 | 134× |

- **Speedup không tính chi phí ước lượng điểm đóng góp,** giống cách bài làm (mục 5.4).
- **Hành vi thích nghi của DynamicFU được tái hiện.** Ở α = 0,5, số vòng là 14,7 / 9,7 / 6,4 cho nhóm đóng góp cao / trung bình / thấp; bài báo cáo 15 / 9 / 4,6 trên TON-IoT.
- **Model không unlearn có ΔF1 so với Full Retrain ngang các phương pháp unlearn.** Metric của bài vì vậy không phân biệt được việc có unlearn hay không.

**Phần A, chuỗi 10 yêu cầu (α = 0,5 / 1,0 xóa 33% / 26% dữ liệu), trạng thái cuối:**

| Chính sách | macro-F1 | Dòng-lượt (gồm ước lượng đóng góp) | Thời gian | Số lần utility sụt |
|---|---|---|---|---|
| `full_retrain_single` | 0,863 / 0,865 | 100% | 100% | 0 |
| `none` | 0,860 / 0,863 | 0% | 0% | 0 |
| `shard_full` | 0,860 / 0,859 | 13% / 7,9% | 13% / 8,0% | 0 |
| `shard_fixed` | 0,859 / 0,859 | 6,6% / 3,9% | 6,5% / 4,7% | 0 |
| `dynamicfu` | 0,859 / 0,859 | 14,8% / 9,5% | 20% / 15% | 0 |
| `utility_stop` | 0,868 / 0,870 | 0,7% / 0,4% | 1,1% / 0,9% | 0 |
| `drop_backfill` | 0,860 / 0,859 | 5,9% / 4,6% | 5,9% / 4,8% | 0 |

- **Ở quy mô 10 shard × 5 client, không chính sách nào bị sụt utility.**
- **`utility_stop` gần như luôn dừng sau 1 vòng,** tức shard bị ảnh hưởng thực chất được thay bằng một model gần như chưa train, mà ensemble vẫn giữ utility.
- **α = 0,1:** với lr 1e-2 và 29 client chỉ một lớp, **mọi model đều suy biến thành "luôn đoán attack"** (F1 0,434, FPR 1,00), kể cả model đơn train lại toàn bộ. Không tái hiện được con số 87% Acc ở α = 0,1 của bài. Kết quả α = 0,1 không dùng được.

## Tiền xử lý fit trên tập phụ trợ (bản dữ liệu auxfit)
**Thay đổi.** `src/data/prepare.py` có thêm `transform_fit: "validation_only"` (config `configs/data_connected_auxfit.json`, dữ liệu `data/processed/connected_v3_auxfit_seed42`):
- scaler, median và vocabulary chỉ được fit trên **validation**, tức tập phụ trợ không bao giờ bị xóa;
- vì vậy xóa dòng train **không làm đổi cách mã hóa**, và model sau xóa không cần mã hóa lại;
- `train_only` vẫn là mặc định.

**Kiểm chứng.**
- Sinh lại dữ liệu cũ bằng `prepare.py` mới cho mọi mảng, `transformer.json`, `audit.json` và `metadata.csv` **trùng từng byte** với bản đã đóng băng. Manifest chỉ khác ở `source_sha256` (code đã sửa) và `platform` (kernel của container).
- Test mới trong `tests/test_preprocessing.py`:
  - transform chỉ học từ các dòng validation;
  - **đổi giá trị các dòng train không làm đổi transformer, validation hay test**, trong khi ở `train_only` thì có đổi;
  - giá trị `transform_fit` không hợp lệ thì báo lỗi.
- Toàn bộ 106 test đều pass.

**Dữ liệu auxfit khác bản cũ ở ba điểm:**
- **37 chiều thay vì 38:** `service=gssapi` không xuất hiện trong validation, nên bị gộp vào `__UNKNOWN__` (0,16% dòng train).
- **Split khác ở 72.407/190.474 dòng (38%):** với cách mã hóa mới, vòng tinh chỉnh chống leakage gộp nhóm theo cách khác (8 vòng). Kích thước các split không đổi, audit leakage vẫn qua. Kết quả trên auxfit vì vậy **không so trực tiếp từng dòng** với kết quả cũ được.
- **Thống kê scale khác,** do các feature có đuôi dài (ví dụ `dst_ip_bytes` gấp 20 lần, `duration` chỉ bằng 0,085 lần so với bản fit trên train).

**Kết quả chạy lại ba run trên auxfit** (`SHARD_DATA_DIR=data/processed/connected_v3_auxfit_seed42`, lưu trong `results/shard_exploration_auxfit/`):
- **Tính chính xác vẫn đúng:** không shard nào dùng dòng đã xóa; trùng từng bit ở cuối mọi chuỗi; Affected-Shard trùng Full Retrain ở 135/135 trường hợp.
- **Run 3 (cấu hình của bài: lr 1e-2, 20 vòng, 2 epoch) cho utility bình thường,** và các kết luận giữ nguyên:
  - α = 0,5 / 1,0: macro-F1 khoảng 0,865, FPR 0,32–0,33; không chính sách nào có lần utility sụt.
  - `none` có |ΔF1| so với Full Retrain chỉ 0,0003 / 0,0001.
  - DynamicFU tốn 10,6% / 9,5% dòng-lượt (gồm ước lượng đóng góp), còn `utility_stop` chỉ 0,46% / 1,2%.
  - α = 0,1 vẫn suy biến.
- **Run 1 và 2 (cấu hình của repo: lr 1e-3, 15 vòng, 1 epoch) bị một chế độ hỏng:**
  - ensemble S ≥ 4 chỉ đạt macro-F1 0,645 và FPR 0,75, trong khi S = 1 đạt 0,865 và 0,324;
  - trong chuỗi yêu cầu, **ngay cả model đơn train lại toàn bộ** cũng kết thúc ở 0,634 / 0,772.

**Chẩn đoán: do thiếu train, không do fit trên tập phụ trợ** (`scripts/shard/diagnose_training_budget.py`, cùng partition 20 client với run 1, seed 10042; macro-F1 / FPR trên calibration):

| Cấu hình train | Dữ liệu cũ, S = 1 | Dữ liệu cũ, S = 4 | Auxfit, S = 1 | Auxfit, S = 4 |
|---|---|---|---|---|
| repo: lr 1e-3, 15 vòng, 1 epoch | 0,870 / 0,332 | 0,862 / 0,332 | 0,864 / 0,325 | **0,648 / 0,752** |
| lr 1e-3, 45 vòng, 1 epoch | 0,862 / 0,334 | 0,861 / 0,333 | 0,867 / 0,318 | 0,875 / 0,323 |
| lr 1e-2, 15 vòng, 1 epoch | 0,868 / 0,334 | 0,864 / 0,334 | 0,881 / 0,321 | 0,868 / 0,320 |
| bài báo: lr 1e-2, 20 vòng, 2 epoch | 0,869 / 0,328 | 0,874 / 0,334 | 0,880 / 0,323 | 0,874 / 0,320 |

- **Trên split auxfit, model shard (khoảng 1/4 dữ liệu, nên khoảng 1/4 số bước tối ưu) chưa train đủ.** Model kẹt ở một chế độ đoán gần như toàn attack, ổn định qua các seed.
- **Chỉ cần train nhiều hơn** (thêm vòng, hoặc lr 1e-2) là model về mức của model đơn, và tốt hơn một chút so với dữ liệu cũ.
- **Hệ quả cho hướng shard:** ngân sách train của *từng shard* phải đủ lớn. Cấu hình FL của repo được chọn cho một model 5 client và không đủ cho shard trên split mới. Chẩn đoán này là **hậu kiểm** (một seed), và đã dùng calibration. Muốn đổi cấu hình train chuẩn cho hướng shard thì phải ghi vào protocol trước khi chạy xác nhận.

## Kết luận được phép (có điều kiện)
| Được tuyên bố | Bằng chứng |
|---|---|
| Unlearn theo shard được cài đặt **chính xác**: không shard nào dùng dữ liệu đã xóa, tái lập trùng từng bit | Mọi run, mọi chính sách trừ `none` |
| Chia shard **không làm giảm** utility so với một model FedAvg, **với điều kiện mỗi shard được train đủ** | Run 1 trên dữ liệu cũ; run 3 trên cả hai bản dữ liệu. Trên auxfit với cấu hình của repo thì **không** đúng (xem chẩn đoán) |
| Utility cuối của mọi chính sách chính xác **ngang train lại toàn bộ**, với 0,4–21% chi phí dòng-lượt | Run 2 và 3 |
| **Utility không đơn điệu theo số vòng** và **shard ít client, lệch** là rủi ro thật; chọn sai có thể làm FPR lên 0,75–1,00 | Run 1 và 2 |
| Dưới bản cài đặt lại, DynamicFU **không tốt hơn** các chính sách đơn giản: utility ngang, chi phí cao hơn khi tính đủ, và quy tắc dừng theo Drop có thể dừng lúc utility đang sụt | Run 2 và 3 |
| Model không unlearn có utility gần như giống train lại, nên **utility không thể dùng làm bằng chứng xóa** | Run 2, 3 và protocol của bài |

| **Không** được tuyên bố | Lý do |
|---|---|
| "DynamicFU thất bại" hoặc "kém hơn" một cách tổng quát | Đây là bản cài đặt lại, với tham số tự chọn; không phải code của tác giả |
| Unlearn chính xác trọn vẹn cho các run trên dữ liệu cũ | Ở đó tiền xử lý được fit trên dữ liệu có cả F. Trên auxfit thì điều kiện này đã đạt, với giả định split được cố định khi thiết lập hệ thống |
| Đã xác nhận | Khám phá, không có protocol đặt trước; 1 partition cho mỗi cấu hình, 1 chuỗi yêu cầu |
| Kết quả ở α = 0,1 | Model bị suy biến |

## Giới hạn
- **Phạm vi hẹp:** 1 partition cho mỗi cấu hình, 1 chuỗi yêu cầu, 5 seed. F1 trên calibration ít nhạy, và F1 theo client là in-sample.
- **Run 2 xóa tới 56% dữ liệu,** nhiều hơn thực tế.
- **Chưa đo chi phí suy luận và lưu trữ của S model,** chưa mô phỏng client offline hay tài nguyên biến động.
- **Model và siêu tham số ở run 3 khác bài** (MLP của repo; batch 512); α = 0,1 dùng một thủ tục partition khác.

## Tái lập
```bash
python src/data/prepare.py --input dataset/train_test_network.csv --config configs/data_connected.json --output data/processed/connected_v3_seed42
```
```bash
FL_DEVICE=cpu python scripts/shard/run_shard_ensemble_check.py runs/shard_ensemble_check
```
```bash
FL_DEVICE=cpu python scripts/shard/run_shard_stream.py runs/shard_stream_20c
```
```bash
for a in 0.5 1.0; do FL_DEVICE=cpu python scripts/shard/run_shard_paper_scale.py runs/shard_paper/paper_a$a $a; done
```
```bash
FL_DEVICE=cpu python scripts/shard/run_shard_paper_scale.py runs/shard_paper/paper_a0.1 0.1 equal
```
```bash
python scripts/shard/analyze_shard_20c.py runs/shard_ensemble_check runs/shard_stream_20c
```
```bash
python scripts/shard/analyze_shard_paper_scale.py runs/shard_paper
```
**Bản auxfit:** sinh dữ liệu rồi chạy lại các lệnh trên với biến môi trường `SHARD_DATA_DIR`. Mỗi lần chạy ghi `data.json` cho biết đã dùng dữ liệu nào.
```bash
python src/data/prepare.py --input dataset/train_test_network.csv --config configs/data_connected_auxfit.json --output data/processed/connected_v3_auxfit_seed42
```
```bash
export SHARD_DATA_DIR=data/processed/connected_v3_auxfit_seed42
```
```bash
FL_DEVICE=cpu python scripts/shard/diagnose_training_budget.py
```

**Thời gian trên CPU 4 nhân:** run 1 khoảng 10 phút; run 2 khoảng 15 phút; run 3 khoảng 40 phút cho mỗi α (ba α chạy song song được).

**Kiểm tra tái lập (đã chạy).** Các script trong `scripts/shard/` ở commit `0504bb2` được chạy lại từ đầu trên CPU. Kết quả được so với `results/shard_exploration/`:
- 13 file CSV và 5 file JSON đều **khớp tuyệt đối**: sai khác lớn nhất bằng 0 ở mọi cột, trừ các cột thời gian (`*seconds*`, `T_full`).
- `meta.json` của α = 0,5 và α = 1,0 trong repo được sinh trước khi trường `scheme` được thêm vào script, nên phép so sánh bỏ qua trường này.
- Partition, chuỗi yêu cầu và toàn bộ metric được tái tạo giống hệt.
