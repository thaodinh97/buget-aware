# Hướng shard, protocol v3: chuỗi yêu cầu sample-level có cấu trúc (kịch bản 6) và mô hình luồng yêu cầu (kịch bản C)

**Trạng thái: BẢN NHÁP, chưa chạy.** Viết sau khi đã xem kết quả v2 ([shard_v2_plan.md](shard_v2_plan.md), [shard_v2_results.md](shard_v2_results.md): **không xác nhận**).
- Protocol được **chốt** khi runner và script đánh giá đã viết xong và smoke pass. Từ lúc đó, mọi thay đổi phải ghi vào mục "Thay đổi sau khi chạy".
- Trước khi chốt, mọi sửa đổi được ghi vào mục "Lịch sử bản nháp" ở cuối.

## Câu hỏi

v2 trộn yêu cầu client-level (5 bước xóa trọn client) với sample-level (5 bước xóa 30%). Các hiện tượng chính của v2 (gộp shard, G2 fail ở yêu cầu `client`, G5b fail) đều gắn với yêu cầu client-level. v3 **chỉ dùng yêu cầu sample-level**, đúng phạm vi của repo, và hỏi:

| # | Câu hỏi | Gate / phép đo |
|---|---|---|
| Q1 | Khi mọi yêu cầu đều là sample-level, ensemble shard có giữ được utility ngang model đơn train lại không? | G3 |
| Q2 | **Cấu trúc** của chuỗi (yêu cầu rải đều các shard, hay dồn vào một shard) thay đổi chi phí và utility của các chính sách ra sao? | G4, G5, chi phí tính trước |
| Q3 | Gom yêu cầu theo cửa sổ thời gian đổi được bao nhiêu chi phí lấy bao nhiêu độ trễ xóa, và cấu trúc chuỗi nào làm mất lợi ích đó? | G1, G2, chi phí và độ trễ tính trước |

## File
- **Đã có:**
  - `scripts/shard/stream_v3.py`: mô hình chuỗi, kế hoạch chi phí và kế hoạch `drop_backfill`. Chỉ dựa trên metadata.
  - `tests/test_stream_v3.py`: unit test.
- **Phải viết trước khi chốt:**
  - `scripts/shard/run_shard_stream_v3.py`: runner, **dựng trên thành phần `src/fu/shard_exact.py`** (`ShardedUnlearner`, xem [shard_dt_interface.md](shard_dt_interface.md)), không chép lại logic của runner v2. Thành phần đã được kiểm chứng trùng từng bit với runner v2 ở 366/366 bước. Runner train ensemble một lần cho mỗi (S, seed), rồi dùng `fork(policy)` cho từng chính sách. Nó nhận tên họ chuỗi qua `SHARD_V3_FAMILY`, để 4 họ chạy song song được.
  - **Phần bổ sung cho thành phần:** xử lý một lô yêu cầu (`unlearn_batch`: áp mọi yêu cầu của lô, rồi train lại mỗi shard bị chạm một lần). Hành vi hiện có của thành phần không được đổi; `tests/test_shard_exact.py` phải vẫn pass.
  - `scripts/shard/evaluate_shard_v3.py`: script đánh giá gate và tính toàn vẹn.
  - Phần khởi chạy dùng chung của v3 (`start_run` với danh sách `TRACKED_V3`). Không sửa `shard_v2.start_run`, vì hàm đó chỉ nhận runner của v2.
- **Dùng lại mà không sửa:** `shard_v2.assign`, `data_content_sha256`, `norm_rel`, `noninferior`, `train_config.CONFIGS['standard']`. Mọi file của v1 và v2 **giữ nguyên từng byte**.
- **Danh sách hash của v3** (`TRACKED_V3`) gồm thêm `src/fu/shard_exact.py` và `src/fu/shard_cost.py`, vì kết quả phụ thuộc vào chúng.

**Bằng chứng về thứ tự** (như v2):
- Lúc bắt đầu, mỗi run ghi vào `run.json` SHA-256 (sau khi đổi CRLF thành LF) của: tài liệu này, `stream_v3.py`, runner v3, script đánh giá v3, `shard_v2.py` và `train_config.py`. Run cũng ghi hash nội dung dữ liệu.
- Runner dừng nếu thiếu một file, nếu hash nội dung dữ liệu không khớp, hoặc nếu số dòng của từng client không khớp với metadata dùng để tính kế hoạch bên dưới.
- Script đánh giá so lại toàn bộ các hash. Nếu không khớp thì kết quả **không được tính là xác nhận**.

## Thay đổi so với v2
| # | Ở v2 | Ở v3 |
|---|---|---|
| 1 | Chuỗi trộn client-level và sample-level; có gộp và loại shard | **Chỉ sample-level.** Mỗi client giữ ít nhất 50% số dòng ban đầu, nên không client nào rời đi, và cấu trúc shard không bao giờ đổi |
| 2 | Một chuỗi (seed 99) | **4 họ chuỗi** có cùng kích thước yêu cầu và cùng thời điểm đến, chỉ khác **vị trí** (client, shard) của yêu cầu |
| 3 | Không có thời gian | **Thời điểm đến theo quá trình Poisson.** Thêm chính sách gom theo cửa sổ, và đo **độ trễ xóa** |
| 4 | Mỗi yêu cầu xóa 30% số dòng còn lại của client | **Kích thước đuôi dài:** log-normal, cắt theo trần 50% |
| 5 | 10 yêu cầu | 20 yêu cầu |
| 6 | 7 chính sách | Bỏ `dynamicfu` (bản cài đặt lại, G5a fail ở v2, tốn thêm bước tính đóng góp) và `utility_stop_v1` (v2 chỉ dùng để mô tả). Thêm `batch_full_2` và `batch_full_5` |
| 7 | Seed 50042–50051 | **Seed mới 60042–60051**, chưa từng chạy |
| 8 | Model đơn tham chiếu train lại sau mọi bước | Train lại ở các bước 5, 10, 15, 20, vì chi phí. Gate chỉ dùng bước 20 |

## Thiết kế
**Giữ nguyên từ v2:**
- Dữ liệu `data/processed/connected_v3_auxfit_seed42`, hash nội dung `f130057e20f1d9eb0857dac5b8ce90c9f0f16b544a8d4b5675b88f24eea0ef3a`.
- Cấu hình `standard`: Adam lr 1e-2, R_base = 20 vòng, 2 epoch local, batch 512; `shard_fixed` 10 vòng; backfill 7 vòng mỗi yêu cầu.
- 20 client, Dirichlet α = 0,5, partition seed 42; tổng 114.284 dòng train.
- S ∈ {4, 5}, gán shard bằng hoán vị có seed 7 (`shard_v2.assign`), mỗi shard ≥ 4 client.
- Shard j dùng seed s + 1000 j. Ensemble là tổng logit margin, ngưỡng 0,5.
- Chính sách được đọc validation; metric báo cáo trên calibration; test không được dùng.

**Shard và số dòng** (từ partition 42; giống `results/shard_v2/ensemble_check_20c/meta.json`):

| S | Shard (client) | Số dòng |
|---|---|---|
| 4 | [1, 3, 4, 14, 17], [2, 6, 8, 16, 18], **[0, 5, 9, 10, 12]**, [7, 11, 13, 15, 19] | 16.732 / 10.043 / **58.073** / 29.436 |
| 5 | [0, 3, 16, 19], **[7, 8, 10, 17]**, [2, 4, 12, 15], [1, 9, 13, 18], [5, 6, 11, 14] | 15.005 / **47.797** / 26.288 / 13.104 / 12.090 |

Shard in đậm là shard lớn nhất. Các shard lệch nhau nhiều: ở S = 4, một shard chứa 51% dữ liệu.

### Mô hình chuỗi (`stream_v3.STREAM`)
**Chung cho cả 4 họ:**
- **K = 20 yêu cầu**, đều là sample-level.
- **Kích thước yêu cầu:** ⌈exp(N(log 200; 1,5²))⌉ dòng, seed 203. Trung vị 200 dòng, đuôi dài: đa số yêu cầu nhỏ, vài yêu cầu rất lớn. Dãy cố định: 91, 119, 823, 2.323, 49, 5.871, 793, 58, 55, 173, 88, 21, 10, 588, 43, 2.075, 4.839, 97, 1.679, 85 (tổng 19.880 dòng yêu cầu).
- **Thời điểm đến:** quá trình Poisson với tốc độ 1 yêu cầu mỗi đơn vị thời gian mô phỏng, seed 204. Yêu cầu đầu ở t = 0,31, yêu cầu cuối ở t = 15,05.
- **Trần:** một client mất tối đa ⌊0,5 · số dòng ban đầu⌋ dòng trong cả chuỗi. Nếu yêu cầu lớn hơn phần còn được xóa thì bị cắt bớt (ghi lại `requested` và `rows`).
- **Dòng bị xóa:** chọn đều trong các dòng còn sống của client, seed [205, bước] (`select_rows`). Không đọc feature, nhãn hay model.

**Bốn họ chuỗi** (`plan_stream`). Họ chỉ quyết định yêu cầu rơi vào client nào:

| Họ | Kịch bản | Cách chọn client | Mục đích |
|---|---|---|---|
| `proportional` | tham chiếu | Xác suất tỉ lệ với số dòng còn được xóa của client | Chuỗi trung tính, gần với v2 |
| `spread` | 6 | Yêu cầu k rơi vào shard k mod S; trong shard, chọn client như trên | **Xấu nhất cho việc gom lô và cho `drop_backfill`:** hai yêu cầu liên tiếp không bao giờ chạm cùng một shard. Tương ứng với một bên yêu cầu xóa biết cách chia shard (mô hình white-box, như Marchant và cộng sự, AAAI 2022) |
| `concentrated` | 6 | Mọi yêu cầu rơi vào shard lớn nhất lúc thiết kế; trong shard, chọn client như trên | **Xấu nhất về chi phí mỗi yêu cầu** khi xử lý ngay, và dồn toàn bộ phần dữ liệu bị mất vào một shard |
| `poisson_zipf` | C | Zipf với số mũ 1 theo một **xếp hạng ngẫu nhiên** của client (seed 202), không theo kích thước | Yêu cầu tập trung ở vài client "phổ biến" mà không trùng với client lớn |

**Lý do chọn tham số.** Không có log yêu cầu xóa thật nào công khai cho IDS mạng IoT, nên các tham số trên là **giả định cho stress test**, không phải ước lượng.
- Trung vị 200 dòng được chọn để một yêu cầu điển hình nhỏ hơn nhiều so với một client: client có trung vị khoảng 4.360 dòng, client nhỏ nhất có 85 dòng. Con số này **không** được suy ra từ số flow của một thiết bị thật. σ = 1,5 cho đuôi dài: yêu cầu lớn nhất trong dãy gấp khoảng 30 lần trung vị.
- Trần 50% giữ cho mọi yêu cầu là sample-level.
- Với tốc độ 1, cửa sổ W ∈ {2, 5} gom kỳ vọng khoảng 2 và 5 yêu cầu mỗi lô.
- Các tham số được chọn **trước** khi tính cấu trúc bên dưới. Sau khi thấy cấu trúc, kể cả việc `drop_backfill` sụp ở `spread`, không tham số nào được đổi.

### Cấu trúc tính trước từ metadata
Lệnh: `python scripts/shard/stream_v3.py results/shard_v2/ensemble_check_20c/meta.json`. Kế hoạch chỉ phụ thuộc số dòng của client, cách gán shard và seed của chuỗi, nên **giống nhau với mọi chính sách và mọi seed train**. Runner sinh lại kế hoạch từ dữ liệu và ghi vào `stream.json`.

**Shard bị chạm ở từng bước:**

| S | Họ | Shard của bước 1–20 |
|---|---|---|
| 4 | `proportional` | 3 2 2 2 2 2 0 2 3 1 2 2 0 2 2 1 1 3 3 2 |
| 4 | `spread` | 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 |
| 4 | `concentrated` | 2 (cả 20 bước) |
| 4 | `poisson_zipf` | 2 3 0 1 3 3 3 2 0 2 3 0 0 3 3 1 0 2 2 0 |
| 5 | `proportional` | 0 1 1 1 2 1 0 1 2 3 1 1 0 0 2 4 2 1 1 1 |
| 5 | `spread` | 0 1 2 3 4 0 1 2 3 4 0 1 2 3 4 0 1 2 3 4 |
| 5 | `concentrated` | 1 (cả 20 bước) |
| 5 | `poisson_zipf` | 0 4 1 0 4 4 0 3 4 3 0 0 0 2 2 1 1 3 2 4 |

**Lượng dữ liệu bị xóa:**

| Họ | S = 4 | S = 5 | Số yêu cầu bị cắt theo trần (S = 4 / 5) |
|---|---|---|---|
| `proportional` | 17.243 dòng (15,1%) | 17.243 (15,1%) | 3 / 3 |
| `spread` | 13.728 (12,0%) | 13.875 (12,1%) | 4 / 3 |
| `concentrated` | 16.263 (14,2%); shard 2 mất 28,0% dữ liệu | 19.067 (16,7%); shard 1 mất 39,9% | 1 / 1 |
| `poisson_zipf` | 9.062 (7,9%) | 9.062 (7,9%) | 4 / 4 |

Lượng bị xóa khác nhau giữa các họ, vì trần cắt các yêu cầu lớn rơi vào client nhỏ. Mỗi họ được so với model đơn train lại trên **đúng dữ liệu còn lại của họ đó**, nên điều này không làm lệch gate.

**Chi phí của `shard_full` và `batch_full_W`** tính bằng dòng × vòng, so với model đơn train lại sau mọi yêu cầu. Các con số này **tất định**: chúng chỉ phụ thuộc kế hoạch, không phụ thuộc seed train.

| S | Họ | W = 0 (`shard_full`): số lần train lại / chi phí | W = 2: lần / chi phí | W = 5: lần / chi phí |
|---|---|---|---|---|
| 4 | `proportional` | 20 / 35,1% | 13 / 19,6% | 10 / 13,3% |
| 4 | `spread` | 20 / 24,8% | **19 / 24,5%** | 11 / 12,0% |
| 4 | `concentrated` | 20 / **47,4%** | 8 / 18,9% | 4 / 8,7% |
| 4 | `poisson_zipf` | 20 / 26,7% | 16 / 22,3% | 11 / 13,8% |
| 5 | `proportional` | 20 / 28,0% | 14 / 16,8% | 10 / 11,4% |
| 5 | `spread` | 20 / 19,8% | **20 / 19,8%** | 13 / 11,3% |
| 5 | `concentrated` | 20 / **36,6%** | 8 / 14,5% | 4 / 6,4% |
| 5 | `poisson_zipf` | 20 / 17,8% | 15 / 12,7% | 11 / 9,6% |

**Độ trễ xóa** (thời gian mô phỏng từ lúc yêu cầu đến tới lúc model phục vụ không còn dữ liệu đó) cũng giống nhau ở mọi họ, vì thời điểm đến được dùng chung:
- W = 2: trung bình 1,00, lớn nhất 1,85;
- W = 5: trung bình 2,15, lớn nhất 4,95.

**Hai điều đọc được từ kế hoạch, trước khi train:**
- **`spread` làm mất lợi ích của gom lô ở W = 2:** 19/20 lần train lại ở S = 4, và 20/20 ở S = 5, vì các yêu cầu trong một cửa sổ luôn rơi vào các shard khác nhau. `concentrated` thì ngược lại: đắt nhất khi xử lý ngay, nhưng rẻ nhất khi gom lô.
- **`drop_backfill` sụp ở `spread`** (`stream_v3.backfill_plan`, cùng quy tắc với v2):
  - Mỗi yêu cầu loại một shard mới khỏi ensemble, nhưng chỉ cấp 7 vòng train bù, trong khi mỗi shard cần 20 vòng. Hàng đợi vì vậy tăng thêm 13 vòng sau mỗi yêu cầu.
  - Ensemble **không còn shard nào** từ bước 5 (S = 4) và bước 7 (S = 5) tới hết chuỗi.
  - Ensemble cũng trống ở bước 9 và 18 của `poisson_zipf` S = 4, và ở bước 16 của `proportional` S = 5.
  - Với `concentrated`, shard lớn nhất bị khởi động lại ở mọi yêu cầu, nên **vắng mặt suốt chuỗi**: ensemble chỉ còn 49% (S = 4) và 58% (S = 5) dữ liệu train.

### Chính sách
Shard được train lại luôn bắt đầu từ khởi tạo mới với seed của nó, chỉ trên dữ liệu hiện tại.

| Chính sách | Mô tả |
|---|---|
| `none` | Không unlearn; chỉ để tham chiếu |
| `shard_full` | Ngay khi yêu cầu đến: train lại shard bị chạm đủ R_base vòng |
| `shard_fixed` | Như trên nhưng chỉ 10 vòng |
| `utility_stop` | **Quy tắc v2, không đổi:** train từng vòng tới khi ok(), tối đa R_base. Sau mọi yêu cầu, nếu chưa ok() thì chạy dự phòng (thêm 1 vòng cho shard đang dùng có ít vòng nhất dưới R_base, tiếp tục đúng chuỗi train có seed của nó). ok(): macro-F1 validation ≥ F1₀ − 0,005 và FPR validation ≤ FPR₀ + 0,005, với F1₀, FPR₀ của ensemble ban đầu |
| `drop_backfill` | **Quy tắc v2, không đổi:** shard bị chạm rời ensemble và vào hàng đợi FIFO; mỗi yêu cầu cấp 7 vòng cho hàng đợi; shard quay lại khi đủ R_base vòng. Sau yêu cầu cuối, train hết hàng đợi (dòng `backlog_done`) |
| `batch_full_2`, `batch_full_5` | Gom các yêu cầu đến trong cửa sổ [mW, (m + 1)W). Ở cuối cửa sổ, train lại **một lần** mỗi shard bị chạm, đủ R_base vòng, trên dữ liệu sau mọi yêu cầu của cửa sổ. Trong cửa sổ, model phục vụ vẫn là model cũ, nên **còn chứa dữ liệu đang chờ xóa**. Cửa sổ cuối được xử lý ở thời điểm kết thúc của nó |
| `full_retrain_single` | Tham chiếu S = 1: một model đơn train lại từ đầu, seed s, R_base vòng, trên toàn bộ dữ liệu còn lại. Chỉ ở bước 5, 10, 15, 20 |

**Ensemble trống.** Khi không shard nào được đưa vào ensemble (chỉ có thể xảy ra ở `drop_backfill`), bước đó được ghi là `unavailable`: dịch vụ không có model, và metric bằng NaN. Không thay bằng một dự đoán mặc định.

**Tính chất dự đoán của `batch_full_W`.** Ở cuối chuỗi, mỗi shard hoặc là model ban đầu (nếu chưa từng bị chạm), hoặc là model train mới với seed của nó trong R_base vòng trên dữ liệu cuối. Điều này đúng với cả `shard_full`. Vì vậy **trạng thái cuối của `batch_full_W` trùng từng bit với `shard_full`**. Khác biệt chỉ nằm ở trạng thái trung gian, chi phí và độ trễ. G1 kiểm tra tính chất này.

### Tính tất định và máy chạy
Theo [shard_dt_interface.md](shard_dt_interface.md), mục "Lượt train đầu tiên của một tiến trình" và "Runner v2 trên máy này so với kết quả v2 đã lưu":
- **Làm nóng.** Mỗi tiến trình gọi `warm_up()` của thành phần trước phép train thật đầu tiên. Không có bước này, lượt train đầu tiên đôi khi lệch ở bit cuối, và G1 có thể fail mà không liên quan gì tới việc xóa.
- **Kiểm tra tái lập ngay sau `fit`.** Runner train lại ensemble ban đầu của mỗi (S, seed) và ghi `reproducible` vào `run.json`. Nếu một (S, seed) không tái lập được thì **chạy lại toàn bộ tiến trình của họ chuỗi đó một lần**, và ghi cả hai lần. Nếu lần chạy lại vẫn không tái lập được thì tính toàn vẹn của họ đó fail.
- **Một máy.** Cả 4 họ chạy trên cùng một máy, với cùng phiên bản thư viện; `run.json` ghi máy, hệ điều hành và phiên bản. Kết quả không được kỳ vọng trùng từng bit trên máy khác. Với `utility_stop`, số vòng phụ thuộc vào quyết định sát ngưỡng, nên còn có thể khác cả về số vòng.
- **Phạm vi của "chính xác".** `shard_full`, `shard_fixed`, `drop_backfill` và `batch_full_W` có số vòng cố định trước, nên chính xác vô điều kiện. `utility_stop` chọn số vòng từ ngưỡng F1₀ / FPR₀ của ensemble **trước khi xóa**, nên chỉ chính xác **có điều kiện theo số vòng** (`exactness_scope = conditional_on_round_count`). v3 giữ nguyên hành vi này để so được với v2, và báo cáo phạm vi cạnh kết quả G1.

### Đo gì
Ở mỗi bước, với mỗi (S, họ, seed, chính sách):
- **Utility trên calibration:** macro-F1, FPR, recall attack; macro-F1 nhỏ nhất và trung bình theo client, trên dữ liệu còn sống.
- **Tính chính xác:** `exact` (không shard nào đang dùng từng train trên dòng đã xóa) và `structure_ok` (mỗi shard đang dùng được train trên đúng dữ liệu hiện tại của nó). Với `batch_full_W`, hai cờ này được tính ở cuối mỗi cửa sổ; giữa các cửa sổ thì ghi `pending_rows` (số dòng đã yêu cầu xóa mà model phục vụ vẫn chứa).
- **Chi phí:** dòng × vòng, số vòng, byte truyền, giây; số vòng dự phòng của `utility_stop`; số shard trong ensemble và độ dài hàng đợi của `drop_backfill`.
- **Độ trễ** (`batch_full_W`): thời điểm xử lý, độ trễ của từng yêu cầu, và độ phơi nhiễm = tổng theo yêu cầu của (số dòng × độ trễ).

Với `batch_full_W`, dòng của bước k là model đang phục vụ ngay sau khi yêu cầu k đến. Mỗi lần xử lý lô ghi thêm một dòng `batch_end`.

## Gate (cố định khi chốt)
**Kiểm định:** như v2 (`shard_v2.noninferior`). Hiệu ghép cặp theo seed d = ứng viên − tham chiếu, n = 10, t một phía, α = 0,05.
- macro-F1 noninferior khi cận dưới > −δ_F1; FPR noninferior khi cận trên < δ_FPR.
- δ_F1 = 0,01; δ_FPR = 0,02; chế độ FPR cao là FPR > 0,45 trên calibration.
- Một gate pass khi **mọi** so sánh trong nó đều pass (kiểm định giao–hợp).

| Gate | Điều kiện pass |
|---|---|
| **P** Điều kiện tiên quyết | Tính toàn vẹn đạt: thiết kế `v3`, cấu hình `standard`, đúng 10 seed, đúng hash dữ liệu, mọi hash file khớp. `full_retrain_single` có FPR ≤ 0,45 ở bước 5, 10, 15, 20, mọi họ và mọi seed |
| **G1** Tính chính xác | Mọi chính sách trừ `none`: `exact` và `structure_ok` ở mọi bước (`batch_full_W`: ở cuối mọi cửa sổ; `drop_backfill`: cả dòng `backlog_done`). Cuối chuỗi: mọi shard đang dùng trùng từng bit với model train mới trên dữ liệu cuối. `batch_full_2` và `batch_full_5` trùng từng bit với `shard_full` ở trạng thái cuối. Đủ mọi tổ hợp (S, họ, seed, chính sách) |
| **G2** Kế hoạch được tái lập | Chuỗi sinh lại từ dữ liệu (client, số dòng, thời điểm) trùng với `stream_v3.plan_stream`. Dòng × vòng của `shard_full`, `batch_full_2`, `batch_full_5` đúng bằng `stream_v3.cost_plan`. Số shard trong ensemble của `drop_backfill` ở mọi bước đúng bằng `stream_v3.backfill_plan` |
| **G3** Shard so với model đơn | Với mỗi họ và S ∈ {4, 5}: `shard_full` ở bước 20 noninferior so với `full_retrain_single` ở bước 20. Không seed nào có FPR cuối > 0,45. Tổng cộng 8 so sánh |
| **G4** Chính sách rẻ hơn so với `shard_full` | Với mỗi họ và S: `shard_fixed` và `utility_stop` ở bước 20 noninferior so với `shard_full` ở bước 20. Không seed nào có FPR cuối > 0,45. Tổng cộng 16 so sánh |
| **G5** `drop_backfill` ở bước tệ nhất | Với mỗi họ và S, mỗi seed: d_F1 = min theo bước của (F1 của `drop_backfill` − F1 của `shard_full`), d_FPR = max theo bước của hiệu FPR. Noninferior trên d_F1 và d_FPR. Một bước `unavailable` làm so sánh đó fail. Tổng cộng 8 so sánh |
| **G6** An toàn của `utility_stop` | FPR ≤ 0,45 ở **mọi bước** 1–20, mọi họ, mọi S, mọi seed |

**Kết luận theo từng claim**, mỗi claim chỉ có giá trị khi P đạt:

| Claim | Gate |
|---|---|
| Unlearn chính xác ở mọi chính sách, kể cả khi gom lô (`utility_stop`: có điều kiện theo số vòng) | G1 |
| Chi phí và độ trễ trong bảng tính trước là chi phí thật của chính sách | G2 |
| Dưới yêu cầu sample-level, chia shard giữ được utility của model đơn (Q1) | G3 |
| Cấu trúc chuỗi không làm các chính sách rẻ hơn kém `shard_full` (Q2) | G4 |
| `drop_backfill` an toàn ở mọi bước (Q2) | G5 |
| `utility_stop` an toàn (Q2) | G6, cùng các dòng `utility_stop` của G4 |

"Xác nhận toàn bộ" khi mọi gate pass. Gate nào fail thì báo cáo nguyên trạng, theo từng (họ, S); không đổi biên, seed, tham số chuỗi hay quy tắc sau khi xem kết quả.

**Q3 (gom lô) không có gate thống kê.** Chi phí và độ trễ của `batch_full_W` tất định và đã được tính trước; G2 chỉ kiểm tra rằng runner làm đúng như kế hoạch. Utility trung gian của `batch_full_W` được báo cáo dạng mô tả.

## Lý do chọn biên
- **Giữ nguyên biên của v1 và v2** (δ_F1 = 0,01, δ_FPR = 0,02). Lý do của các biên này ghi ở [shard_v2_plan.md](shard_v2_plan.md). Nới biên sau khi đã xem v2 sẽ là chỉnh biên theo dữ liệu đã thấy.
- **G5 dùng bước tệ nhất**, không dùng bước cuối. Ở v2, chế độ FPR thấp, recall thấp của `drop_backfill` chỉ xuất hiện ở các bước trung gian, và bước cuối đã che mất nó.

## Dự kiến (ghi ra để minh bạch, không dùng để điều chỉnh gì)
- **G5 fail ở `spread`, cả S = 4 và S = 5,** theo kế hoạch tính trước: ensemble trống từ bước 5 / 7. Đây là **kết quả của cấu trúc chuỗi**, đọc được từ metadata, không cần train. Gate vẫn được giữ, để kết luận được ghi trước khi chạy.
- **G5 có thể fail ở `concentrated`,** vì shard chứa 51% (S = 4) và 42% (S = 5) dữ liệu vắng mặt suốt chuỗi.
- **G3 có thể pass ở `proportional`, `spread` và `poisson_zipf`.** Ở v2, unlearn một request `random50` rồi train đủ R_base pass ở S ∈ {2, 4, 5}, và ở đây chỉ 8–15% dữ liệu bị xóa. **Rủi ro chính là `concentrated`**, nơi một shard mất 28% (S = 4) và 40% (S = 5) dữ liệu của nó.
- **G6 có thể pass.** Ở v2, `utility_stop` không có bước nào FPR > 0,45 với S ∈ {4, 5}.

## Phần dữ liệu đã được nhìn
- **Partition, dữ liệu, cách gán shard và cấu hình giống v1 và v2.** Kết quả v1 (seed 10042–10046) và v2 (seed 50042–50051) đã được xem.
- **Thiết kế v3 được viết sau khi thấy kết quả v2,** đặc biệt là G5b fail và chế độ FPR thấp ở trạng thái trung gian của `drop_backfill`. v3 vẫn dùng partition 42, **không** phải partition mới.
- **Kế hoạch chuỗi đã được tính và xem trước khi chốt** (mục "Cấu trúc tính trước"). Kế hoạch chỉ dùng metadata; không model nào được train cho v3.

## Không được tuyên bố, kể cả khi pass
| Không được tuyên bố | Lý do |
|---|---|
| Phân phối yêu cầu xóa trong thực tế giống mô hình chuỗi | Tham số là giả định cho stress test; không có log yêu cầu xóa thật để ước lượng |
| Kết quả đúng trên partition khác, cách gán shard khác, hoặc S khác | Chỉ partition 42, seed gán 7, S ∈ {4, 5} |
| Kết quả cho yêu cầu client-level | v3 chỉ có sample-level |
| `spread` là chuỗi xấu nhất có thể | Đây là một cấu trúc xấu được chọn trước, không phải kết quả tối ưu hóa phía kẻ tấn công |
| Kết quả cho chuỗi yêu cầu thích ứng theo model | Ngoài phạm vi (xem bên dưới) |
| Kết quả trên test | Test không được dùng |
| Kết quả trùng từng bit trên máy khác, hay gate vượt biên rất sát là bền theo máy | Model train trên hai máy khác nhau ở các bit cuối, và chính sách dừng theo ngưỡng có thể đi theo số vòng khác ([shard_dt_interface.md](shard_dt_interface.md)) |
| Unlearn chính xác vô điều kiện cho `utility_stop` | Số vòng được chọn từ tín hiệu của trạng thái trước khi xóa |

## Ngoài phạm vi v3
- **Chuỗi thích ứng theo model** (Gupta và cộng sự, NeurIPS 2021), ví dụ kẻ tấn công xin xóa đúng các dòng giúp model phát hiện loại tấn công của họ. Chuỗi như vậy phụ thuộc vào model đã phục vụ, nên khác nhau giữa các chính sách và các seed. Nó phá thiết kế "một kế hoạch chung cho mọi chính sách" của v3, và cần một protocol riêng.
- **Partition tự nhiên theo IP hoặc subnet**, và yêu cầu xóa theo thiết bị.
- **Chính sách gom lô kết hợp với `utility_stop`.**

## Kết quả mô tả (không phải gate)
- Chi phí (dòng × vòng, byte, giây) của mọi chính sách, theo họ và S.
- Utility trung gian và `pending_rows` của `batch_full_W`.
- Số bước `unavailable` và số shard trong ensemble của `drop_backfill`.
- Số vòng dự phòng và số bước không đạt ok() của `utility_stop`.
- Utility của `full_retrain_single` ở bước 5, 10, 15, 20.

## Lệnh chạy (sau khi runner và script đánh giá được viết)
```bash
python -m unittest discover -s tests -p "test_stream_v3.py" -v
```
```bash
python scripts/shard/stream_v3.py results/shard_v2/ensemble_check_20c/meta.json
```
```bash
export SHARD_DESIGN=v3 FL_DEVICE=cpu
```
```bash
for f in proportional spread concentrated poisson_zipf; do SHARD_V3_FAMILY=$f python scripts/shard/run_shard_stream_v3.py runs/shard_v3/stream_$f & done; wait
```
```bash
python scripts/shard/evaluate_shard_v3.py runs/shard_v3 runs/shard_v3/gates.json
```
Ước lượng thô từ thời gian của v2: 6–10 giờ CPU cho cả 4 họ, chia cho 4 tiến trình song song. Thời gian thật được đo ở smoke và ghi vào mục "Lịch sử bản nháp"; nó không làm đổi thiết kế.

## Lịch sử bản nháp
- 2026-11-11: bản nháp đầu tiên. Đã có `stream_v3.py` và `tests/test_stream_v3.py`; chưa có runner, script đánh giá, hay smoke run.
- 2026-11-11, cùng ngày: sau khi rebase lên các commit mới của nhánh (thành phần `src/fu/shard_exact.py`, [shard_dt_interface.md](shard_dt_interface.md)), runner được đổi sang dùng thành phần. Thêm mục "Tính tất định và máy chạy" (làm nóng, kiểm tra tái lập sau `fit`, một máy, phạm vi chính xác của `utility_stop`), và hai dòng "không được tuyên bố" tương ứng. Mô hình chuỗi, gate và biên không đổi.

## Thay đổi sau khi chạy
(chưa có)
