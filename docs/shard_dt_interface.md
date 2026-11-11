# Hướng shard làm đầu vào của Digital Twin: thành phần gọi được và bản ghi chi phí

**Quyết định (2026-11-10).** Bước FU mà phase DT nhận vào là **unlearn chính xác theo shard** (protocol v2: [shard_v2_plan.md](shard_v2_plan.md), [shard_v2_results.md](shard_v2_results.md)). Quyết định này thay mục 3 "Bàn giao cho phase DT" của [fu_freeze.md](fu_freeze.md); phần còn lại của tài liệu đó (kết luận được và không được tuyên bố về FU xấp xỉ) giữ nguyên giá trị.

Tài liệu này mô tả thành phần mà DT gọi, những gì nó đo, và những gì **chưa** đo. Mục "Chạy trên dữ liệu thật" là kết quả **mô tả** trên 3 seed của v2 (không có gate, không dùng test); các gate và kết luận của v2 vẫn chỉ là của `results/shard_v2/`.

## File

| File | Vai trò |
|---|---|
| `src/fu/shard_exact.py` | Thành phần: `ShardedUnlearner` (fit, unlearn từng yêu cầu, checkpoint, kiểm tra chính xác), `ShardExactMethod` (interface `UnlearningMethod` của đặc tả §5.4), `UnlearningResult` |
| `src/fu/shard_cost.py` | `CostMeter`: bản ghi chi phí theo client, theo vòng và theo yêu cầu |
| `scripts/shard/shard_service.py` | CLI `init` / `unlearn` / `verify` trên thư mục state |
| `scripts/shard/replay_v2_stream.py` | Chạy lại chuỗi 10 yêu cầu của v2 qua thành phần (train ensemble một lần, rồi từng chính sách), ghi bản ghi chi phí, so với `steps.csv` của một run v2 |
| `scripts/shard/run_v2_stream_subset.py` | Mốc cùng máy: chạy runner v2 nguyên bản trên một số seed của v2 |
| `scripts/shard/analyze_replay.py` | Bảng tổng hợp của các replay (`results/shard_dt/analysis.txt`) |
| `scripts/shard/check_startup_determinism.py` | Kiểm tra lượt train đầu tiên của tiến trình có tái lập từng bit trên máy đang dùng không |
| `tests/test_shard_exact.py` | 18 test, gồm test tương đương với runner v2 |

Sáu file bị khóa hash của v2 (`shard_v2.TRACKED`) **không đổi một byte**; `evaluate_shard_v2.py` trên `results/shard_v2/` vẫn báo toàn vẹn đạt.

## Logic giữ nguyên từ v2

Thành phần là logic của `run_shard_stream_v2.py` được tách khỏi script: cùng cách chia shard, cùng quy tắc gộp (shard còn < `merge_below` client thì gộp vào shard hoạt động ít client nhất) và loại shard rỗng, cùng seed (shard j dùng `seed + 1000·j`, vòng r dùng `+ r·10000`), cùng sáu chính sách: `shard_full`, `shard_fixed`, `utility_stop`, `utility_stop_v1`, `dynamicfu`, `drop_backfill`.

**Kiểm chứng tương đương.** `EquivalenceTests` chạy **runner v2 nguyên bản** trên dữ liệu tổng hợp (chỉ thay `shard_v2.start_run`, nơi duy nhất nạp và kiểm hash dữ liệu thật), rồi yêu cầu thành phần tái lập ở **mọi bước, mọi chính sách**:
- sự kiện (`retrain`, `merge_*`, `removed_empty`), `exact`, `structure_ok`, số shard đang dùng;
- số vòng, dòng-lượt và byte cộng dồn;
- đầu ra của chính sách (`phase1_rounds`, `fallback_rounds`, `condition_met`, `contribution`, `R0`, `stop_reason`, `pending`);
- macro-F1, FPR và recall trên calibration, **bằng nhau từng bit**;
- trọng số shard so với `Trainer` của chính runner.

Kịch bản tổng hợp chạm đủ các nhánh: gộp shard (S = 5), loại shard rỗng (S = 10, 2 client mỗi shard), vòng dự phòng, kết thúc không đạt ok(), ba lý do dừng của DynamicFU, hàng đợi của `drop_backfill`. Ba lỗi cố ý (chọn sai shard dự phòng, sai seed của bước ước lượng đóng góp, giữ hàng đợi cũ khi gộp) đều làm test fail.

Kiểm chứng trên dữ liệu thật nằm ở mục tiếp theo.

## Chạy trên dữ liệu thật (2026-11-10)

Kết quả nằm trong `results/shard_dt/`; bảng tổng hợp là `analysis.txt`. Đây là chạy lại kịch bản v2 để kiểm tra thành phần và lấy bản ghi chi phí, trên 3 seed đầu của v2 (50042–50044), **không phải lần xác nhận mới**.

**Dữ liệu.** `train_test_network.csv` của ToN-IoT Network (211.043 dòng, SHA-256 `26ddc513…`), tiền xử lý bằng `configs/data_connected_auxfit.json`. Hash nội dung của train / validation / calibration là `f130057e…`, **đúng bằng** giá trị cố định trong protocol v2. Bốn artifact dạng văn bản (`audit.json`, `transformer.json`, `metadata.csv`, `duplicate_mapping.csv`) khớp hash ghi trong `run.json` của v2 sau khi đổi xuống dòng LF thành CRLF. Các file `.npz` khác hash vì trình nén khác; nội dung mảng thì giống.

**Máy.** Linux, 4 CPU, Python 3.13.18, torch 2.6.0 chạy trên CPU với 2 thread, numpy 2.5.4, pandas 3.1.1. Kết quả v2 đã lưu được tạo trên Windows.

### Thành phần so với runner v2, cùng máy

Mốc là runner v2 nguyên bản chạy ngay trên máy này (`run_v2_stream_subset.py`, 3 seed; `stream.json` và `structure.json` trùng với bản đã lưu). Bản chép của mốc nằm ở `results/shard_dt/v2_reference_linux/`; `compare.json` ghi đường dẫn lúc chạy, `runs/shard_dt/v2_reference_linux`. Mỗi replay train ensemble một lần rồi chạy cả 6 chính sách.

| | Kết quả |
|---|---|
| Replay | 6: S ∈ {4, 5} × seed 50042–50044 |
| Bước được so | 366 (6 chính sách × 10 yêu cầu, cộng bước xong hàng đợi của `drop_backfill`) |
| Đại lượng rời rạc lệch (sự kiện, số vòng, dòng-lượt, byte, đầu ra chính sách) | 0 |
| Chênh lệch metric lớn nhất (calibration và theo client) | 0, tức bằng nhau từng bit |
| `exact` và `structure_ok` | đạt ở mọi bước |
| Cuối chuỗi: mọi shard đang dùng trùng từng bit với model train mới | 36 / 36 |
| Ensemble ban đầu được train lại và trùng từng bit | 6 / 6 |

CLI cũng đã chạy trên dữ liệu thật, mỗi lệnh một tiến trình: `init` (S = 4), một yêu cầu xóa 500 dòng của client 11 bằng `shard_full`, một yêu cầu xóa cả client 16 bằng `utility_stop`, rồi `verify` báo trùng từng bit; gửi lại yêu cầu đã xử lý thì bị từ chối.

### Lượt train đầu tiên của một tiến trình

Lần replay đầu tiên của S = 5, seed 50044 **không** khớp mốc: metric lệch ở bước 1–2 và điểm đóng góp của DynamicFU lệch ở bước 1–3, số vòng không đổi. Bản ghi loss chỉ ra chỗ lệch: 77 / 400 bản ghi của lần train ensemble khác với lần chạy lại, **tất cả thuộc shard 0, bắt đầu từ vòng 1 của client đầu tiên**, tức phép train đầu tiên của tiến trình (loss 0,158338449 so với 0,158338547). Shard 0 được train lại ở bước 3, nên từ đó hai bên lại trùng nhau. Chạy lại đúng lệnh đó thì khớp mốc từng bit. Bản ghi của lần lệch được giữ ở `results/shard_dt/replay_deviation_s5_50044/`.

Hiện tượng này không thuộc thành phần: một script chỉ dùng `fedavg_core` (train client 0 một lượt, là phép train đầu tiên của tiến trình) cũng cho kết quả lệch ở một số ít lần chạy. `check_startup_determinism.py` với 300 tiến trình mới cho mỗi chế độ (`results/shard_dt/startup_determinism.txt`): không làm nóng thì **4 / 300** lần cho kết quả khác, có `warm_up()` trước thì **0 / 300**. Gộp cả các lần thử trước đó bằng script tương đương (một phần làm nóng bằng một lượt train bỏ đi trên dữ liệu thật thay cho `warm_up()`): 6 / 532 so với 0 / 490. Nếu việc làm nóng không có tác dụng, xác suất để cả 6 lần lệch đều rơi vào nhóm không làm nóng là khoảng 2 %; tức bằng chứng ủng hộ, chưa phải chắc chắn. Chạy 1 thread cũng không thấy lần lệch nào trong 160 lần, số này quá ít để kết luận. Nguyên nhân gốc chưa được xác định.

Lượt train đầu tiên còn mang theo 0,7–0,9 giây khởi tạo một lần của backend số (cùng lượt train đó, khi đã làm nóng, mất 0,02–0,04 giây). Vì vậy thành phần chạy `warm_up()` trước phép train thật đầu tiên của mỗi tiến trình: một lượt train bỏ đi trên dữ liệu ngẫu nhiên, thời gian ghi riêng ở `backend_warmup_seconds`, không làm đổi kết quả nào về sau (có test). Hai hệ quả cần nhớ:
- **`verify_bit_exact` và G3 có thể fail mà không liên quan gì tới việc xóa**, nếu một shard được train đầu tiên trong tiến trình rơi vào lần lệch. Replay vì vậy train lại ensemble ngay sau `fit` và ghi `reproducible` vào `run.json`.
- Runner v2 không có bước làm nóng, nên một run v2 trên máy có hiện tượng này cũng có thể chứa một shard lệch như vậy. G3 của kết quả v2 đã lưu đạt ở mọi tổ hợp, nhưng điều đó không loại trừ được: G3 chỉ so các shard còn dùng ở cuối chuỗi, và chính lần replay lệch ở trên vẫn qua kiểm tra cuối chuỗi ở cả 6 chính sách, vì shard lệch đã được train lại. Hiện tượng chưa được kiểm tra trên máy Windows đã chạy v2; `check_startup_determinism.py` dùng để kiểm tra.

### Runner v2 trên máy này so với kết quả v2 đã lưu

Cùng dữ liệu (hash khớp), cùng code, cùng seed, khác máy. So trên 3 seed, 30 (seed, bước) mỗi ô:

| Chính sách | Bước có số vòng cộng dồn khác (S = 4 / S = 5) | Chênh macro-F1 lớn nhất | Bước có FPR > 0,45: máy này / đã lưu |
|---|---|---|---|
| `shard_full` | 0 / 0 | 0,0068 / 0,0062 | 0 / 0 |
| `shard_fixed` | 0 / 0 | 0,0023 / 0,0042 | 0 / 0 |
| `drop_backfill` | 0 / 0 | **0,2317** / 0,0076 | **3** / 0 (đều ở S = 4) |
| `utility_stop` | 13 / 21 | 0,0093 / 0,0053 | 0 / 0 |
| `utility_stop_v1` | 12 / 21 | **0,2312** / 0,0053 | **2** / 0 (đều ở S = 4) |
| `dynamicfu` | 23 / 26 | 0,0218 / 0,0045 | 0 / 0 |
| `full_retrain_single` | 0 | 0,0072 | 0 / 0 |

- **Model train trên hai máy khác nhau ở các bit cuối**: ensemble ban đầu đã lệch tới 0,0002 macro-F1. Với chính sách có số vòng cố định, số vòng luôn bằng nhau và metric lệch dưới 0,01, trừ các trạng thái trung gian của `drop_backfill`.
- **Chính sách dừng theo ngưỡng đi theo số vòng khác** ở phần lớn các bước, vì một quyết định sát ngưỡng đổi chiều là cả phần còn lại của chuỗi đổi theo. Ví dụ S = 4, seed 50042, `utility_stop`: 5 bước đầu trùng số vòng, bước 6 dùng 26 vòng dự phòng trên máy này và 39 vòng trong kết quả đã lưu.
- **Chế độ FPR cao phụ thuộc máy.** S = 4, seed 50042, `drop_backfill`, bước 3–5 (ensemble chỉ còn 2 rồi 1 shard): FPR khoảng 0,75 trên máy này, 0,32–0,33 trong kết quả đã lưu. `utility_stop_v1` ở S = 4 có 2 bước FPR > 0,45 trên máy này, trong kết quả đã lưu không có bước nào. `utility_stop` (v2) không có bước nào trên cả hai máy.
- **Trạng thái cuối, trung bình 3 seed, gần nhau**: macro-F1 lệch không quá 0,002 ở mọi ô; `utility_stop` dùng 105 vòng so với 97,7 (S = 4) và 37,7 so với 35 (S = 5).

Hệ quả: các con số của v2 tái lập được về mức trung bình, không tái lập từng bit trên máy khác. Những gate vượt biên rất sát (ví dụ `shard_full` S = 5 ở G5b, vượt 0,000004) không thể coi là bền theo máy. **Các gate của v2 chưa được chạy lại ở đây**: việc đó cần đủ 10 seed và cả run 1, và sẽ tạo ra một bộ kết quả thứ hai cho cùng một protocol.

### Chi phí đo được

Sau 10 yêu cầu (`drop_backfill`: sau khi xong hàng đợi), trung bình 3 seed, trên máy nói trên. Thời gian là của mô phỏng tuần tự trên máy này; cost model của DT phải hiệu chỉnh lại trên máy triển khai.

| S | Chính sách | Vòng | Dòng-lượt so với train lại model đơn | Byte payload (hai chiều) | Thời gian train của client (s) | Makespan mô phỏng (s) | Server chấm chính sách (s) |
|---|---|---|---|---|---|---|---|
| 4 | `shard_full` | 200 | 25,9 % | 29,7 MB | 15,4 | 6,9 | 0 |
| 4 | `shard_fixed` | 100 | 13,0 % | 14,9 MB | 7,6 | 3,4 | 0 |
| 4 | `utility_stop` | 105 | 11,8 % | 15,6 MB | 7,3 | 3,8 | 0,53 |
| 4 | `utility_stop_v1` | 75,3 | 9,8 % | 11,1 MB | 5,7 | 2,8 | 0,34 |
| 4 | `dynamicfu` | 123 | 16,6 % | 24,9 MB | 12,8 | 6,3 | 1,40 |
| 4 | `drop_backfill` | 122 | 13,3 % | 17,5 MB | 7,9 | 3,7 | 0 |
| 5 | `shard_full` | 200 | 20,5 % | 26,8 MB | 12,5 | 6,8 | 0 |
| 5 | `shard_fixed` | 100 | 10,3 % | 13,4 MB | 6,1 | 3,3 | 0 |
| 5 | `utility_stop` | 37,7 | 4,2 % | 4,9 MB | 2,4 | 1,6 | 0,19 |
| 5 | `utility_stop_v1` | 37,7 | 4,2 % | 4,9 MB | 2,5 | 1,7 | 0,18 |
| 5 | `dynamicfu` | 127 | 13,4 % | 23,2 MB | 11,2 | 6,4 | 1,43 |
| 5 | `drop_backfill` | 121 | 11,2 % | 15,8 MB | 6,8 | 3,7 | 0 |

- **Cột dòng-lượt của `shard_full`, `shard_fixed` và `drop_backfill` trùng với v2** (25,9 / 13,0 / 13,3 % và 20,5 / 10,3 / 11,2 %), vì số vòng của chúng không phụ thuộc máy. Của `utility_stop` thì khác (v2: 13,1 % và 3,9 %), đúng như mục trên.
- **Chi phí của một yêu cầu `utility_stop` dao động rất mạnh.** Ở S = 4, yêu cầu trung vị mất 0,07 giây makespan, yêu cầu đắt nhất 2,7 giây với 296 lượt client-vòng, vì phần dự phòng train thêm trên mọi shard. Với `shard_full`, trung vị 0,63 giây và lớn nhất 1,3 giây. Ràng buộc ngân sách theo từng yêu cầu của planner phải tính tới trường hợp xấu này.
- **`dynamicfu` trả thêm cho bước ước lượng đóng góp**: ở S = 4, byte payload của nó bằng 84 % của `shard_full` dù chỉ train 62 % số vòng, và server chấm chính sách mất 1,4 giây.
- **Model nhỏ nên byte không nén được**: 18.312 byte tensor mỗi chiều, 18,4–18,7 KB sau `pack_state`.
- **Thời gian train một lượt local của client gần tuyến tính theo số mẫu**: khoảng 2 ms + 4 ms cho mỗi 1.000 mẫu (R² 0,93; sai số tương đối trung vị 10 %), trên 5.081 bản ghi của seed 50042.
- **Ensemble ban đầu**: train S = 4 mất 9,9 giây thời gian client, makespan mô phỏng 4,5 giây nếu các shard chạy lần lượt và 2,4 giây nếu song song. Suy luận 1.000 dòng mất 0,40 ms với 4 shard, so với 0,10 ms của một model.
- **Bộ nhớ**: tiến trình mô phỏng dùng khoảng 690 MB, gồm toàn bộ dữ liệu của mọi client.

### Tái lập

```bash
python src/data/prepare.py --input dataset/train_test_network.csv --config configs/data_connected_auxfit.json --output data/processed/connected_v3_auxfit_seed42
```
```bash
python scripts/shard/run_v2_stream_subset.py runs/shard_dt/v2_reference --seeds 50042 50043 50044
```
```bash
python scripts/shard/replay_v2_stream.py runs/shard_dt/replay/s4_50042 --shards 4 --seed 50042 --compare runs/shard_dt/v2_reference
```
```bash
python scripts/shard/analyze_replay.py runs/shard_dt/replay --reference runs/shard_dt/v2_reference --stored results/shard_v2/stream_20c
```

Lệnh replay trả mã thoát 1 nếu bất kỳ đại lượng rời rạc nào lệch, hoặc metric lệch quá `--tolerance` (mặc định 0). So từng bit chỉ có nghĩa với một run v2 **cùng máy**; trên máy đã tạo `results/shard_v2/` thì có thể so thẳng với `results/shard_v2/stream_20c`. `results/shard_dt/` giữ `steps.csv`, `run.json`, `compare.json` và `fit_client_costs.csv` của cả 6 replay; `round_costs.csv` và `client_costs.csv` chỉ giữ cho seed 50042.

## Dùng từ Python

```python
from src.fu.shard_exact import ShardedUnlearner, ShardConfig, ForgetRequest, assign_shards
from scripts.shard.train_config import CONFIGS

config = ShardConfig.from_train_config(CONFIGS['standard'])           # R_base 20, 2 epoch, Adam 1e-2, batch 512
fed = ShardedUnlearner(train, validation, parts, assign_shards(4, 20), seed=50042, policy='shard_full', config=config)
fit = fed.fit(checkpoint_dir='runs/shard_dt/state0')                  # train 4 shard, lưu state
result = fed.unlearn(ForgetRequest(client=11, rows=rows), policy='utility_stop', checkpoint_dir='runs/shard_dt/state1')
fed.unlearn(ForgetRequest(client=10))                                 # rows=None: xóa cả client
fed.predict_proba(X); fed.client_state(); fed.verify_bit_exact()
```

- `train`: `{'X', 'y', 'row_id'}`; `validation`: `{'X', 'y'}` (dữ liệu phụ trợ ở server, chính sách chỉ đọc tập này); `parts`: chỉ số dòng train của từng client.
- `rows` là chỉ số dòng trong tập train. Yêu cầu xóa hết dòng còn lại của client được tính là xóa cả client.
- `policy` có thể đổi theo từng yêu cầu, trừ khi `drop_backfill` còn hàng đợi (phải `finish_backlog()` trước).
- `fork(policy)` tạo nhánh độc lập từ trạng thái hiện tại, để thử nhiều chính sách trên cùng một ensemble.
- `ShardedUnlearner.load(dir, train, validation, parts)` khôi phục state và tiếp tục **trùng từng bit** với phiên chưa lưu (có test, kể cả khi `drop_backfill` đang có hàng đợi). Load từ chối nếu dữ liệu, partition hoặc hash file không khớp.
- Yêu cầu không hợp lệ (client không còn dữ liệu, dòng không thuộc client, dòng trùng, tập rỗng) báo lỗi **trước khi** thay đổi bất kỳ trạng thái nào.
- Tính tất định do nơi gọi đặt: `torch.use_deterministic_algorithms(True)` và số thread cố định. CLI đặt 2 thread như các runner v2.
- Thao tác đầu tiên của mỗi tiến trình chạy `warm_up()` trước (xem "Lượt train đầu tiên của một tiến trình").

## Dùng từ dòng lệnh

```bash
python scripts/shard/shard_service.py init --output runs/shard_dt/s4_50042 --shards 4 --seed 50042
```
```bash
python scripts/shard/shard_service.py unlearn --original-run runs/shard_dt/s4_50042 --forget-request request.json --method shard_full --output runs/shard_dt/s4_50042_step1
```
```bash
python scripts/shard/shard_service.py verify --original-run runs/shard_dt/s4_50042_step1
```

File yêu cầu: `{"client": 3}` xóa cả client; `{"client": 3, "forget_row_ids": [...]}` xóa các dòng đó (giá trị `row_id` của tập train). Thư mục output không bao giờ bị ghi đè. Mỗi thư mục state chứa `state.json`, model từng shard, `rows.npz`, `layout.json`, và bản ghi chi phí của thao tác tạo ra nó: `result.json`, `round_costs.csv`, `client_costs.csv`. Dòng đã xóa chỉ được lưu dưới dạng mask chỉ số và hash của yêu cầu, không lưu bản ghi.

## Kết quả của một yêu cầu (`UnlearningResult`)

| Trường | Nội dung |
|---|---|
| `step`, `policy`, `client`, `kind`, `rows` | yêu cầu đã xử lý |
| `event`, `retrained_shard`, `retired_shards`, `shards_included` | thay đổi cấu trúc |
| `exact`, `structure_ok` | hai kiểm tra của G3 sau bước này |
| `exactness_scope` | `unconditional` hoặc `conditional_on_round_count` (xem dưới) |
| `details` | đầu ra của chính sách |
| `cost`, `round_costs`, `client_costs` | bản ghi chi phí |
| `timing_breakdown`, `client_bytes_up`, `client_bytes_down`, `local_losses` | các trường của đặc tả §5.4, suy ra từ ba trường trên |
| `config_hash`, `forget_request_hash`, `final_checkpoint`, `intermediate_checkpoint` | định danh và checkpoint (`intermediate_checkpoint` khác rỗng khi `drop_backfill` còn hàng đợi) |
| `structured_warnings` | `UTILITY_CONDITION_UNMET`, `BACKLOG_PENDING`, `EMPTY_ENSEMBLE`, `STRUCTURE_CHANGED`, `ROUND_COUNT_DEPENDS_ON_PRE_DELETION_STATE`, `MEMORY_UNAVAILABLE` |

**Phạm vi của "chính xác".** `shard_full`, `shard_fixed` và `drop_backfill` có số vòng cố định trước, nên `exactness_scope = unconditional`. Với `utility_stop`, `utility_stop_v1` và `dynamicfu`, số vòng được chọn từ tín hiệu của trạng thái **trước khi xóa**: ngưỡng F1₀ / FPR₀ lấy từ ensemble đã train trên cả những dòng sẽ bị xóa, và DynamicFU còn ước lượng đóng góp trên chính dữ liệu bị xóa. Model sau xóa vì vậy chính xác **có điều kiện theo số vòng**. Thành phần ghi điều này vào mọi kết quả và không sửa hành vi, để giữ tương đương với v2; sửa (ví dụ ngưỡng tuyệt đối đặt trước) là việc của một protocol mới.

## Bản ghi chi phí cho cost model của DT

Mỗi thao tác (fit, hoặc một yêu cầu) có một `CostMeter`. Mọi vòng FedAvg vẫn đi qua `fedavg_core.fedavg`, nên model có đo và không đo trùng từng bit (có test).

**Theo client và vòng** (`client_costs`): `shard`, `round`, `phase` (`base` / `retrain` / `fallback` / `backfill` / `contribution`), `client_id`, `samples`, `optimizer_steps`, `loss`, `compute_seconds`, `bytes_down_raw`, `bytes_up_raw`, `bytes_down_packed`, `bytes_up_packed`.

**Theo vòng** (`round_costs`): số client, số mẫu, `wall_seconds`, `compute_seconds_sum`, `compute_seconds_max`, `server_seconds`, byte, `rss_bytes`.

**Theo thao tác** (`cost`):

| Trường | Ý nghĩa | Đo hay mô phỏng |
|---|---|---|
| `rounds`, `rows_x_rounds`, `client_rounds`, `clients_selected`, `shards_trained` | khối lượng train | đếm chính xác |
| `bytes_*_raw` | byte tensor của model, mỗi chiều | chính xác; `train_bytes_raw` bằng công thức byte của v2 |
| `bytes_*_packed` | kích thước payload đã tuần tự hóa bằng `pack_state` (NPZ nén), định dạng mà prototype Docker gửi | **đo**, chưa gồm header HTTP |
| `compute_seconds_sum` | tổng thời gian train local của các client | **đo** trong `ClientWorker.train` |
| `server_aggregate_seconds` | phần của server mỗi vòng: thời gian vòng trừ tổng thời gian client | **đo** |
| `policy_eval_seconds` | server chấm ensemble trên validation (`utility_stop`, `dynamicfu`) | **đo**; bằng 0 với `shard_full`, `shard_fixed`, `drop_backfill` |
| `payload_pack_seconds`, `checkpoint_seconds` | chi phí của chính việc đo và lưu | **đo**, tách riêng |
| `backend_warmup_seconds` | khởi tạo một lần của backend số, chỉ khác 0 ở thao tác đầu tiên của tiến trình (khoảng 0,8 giây trên máy đã đo) | **đo**, tách riêng |
| `makespan_seconds_simulated` | mỗi vòng lấy client chậm nhất cộng phần server, cộng `policy_eval_seconds` | **mô phỏng** từ thời gian đo tuần tự |
| `makespan_parallel_shards_seconds_simulated` | như trên, nhưng các shard chạy song song (có nghĩa với `fit`) | **mô phỏng** |
| `wall_seconds_total` | toàn bộ thao tác, tuần tự, gồm cả đo và lưu | **đo** |
| `rss_bytes_start` / `_end` / `_peak_sampled` | bộ nhớ của tiến trình, lấy mẫu sau mỗi vòng | **đo** ở mức cả tiến trình |
| `models_stored`, `state_bytes_in_memory`, `checkpoint_bytes_on_disk` | chi phí lưu S model (và model đang train bù) | chính xác; byte trên đĩa là **đo** |
| `inference_forward_passes` | số forward pass cho một dự đoán | chính xác; `benchmark_inference(X)` **đo** thời gian |
| `cumulative_unlearning` | dòng-lượt, vòng và byte cộng dồn qua các yêu cầu (không tính `fit`) | khớp cột `rows_x_rounds`, `rounds`, `bytes` của v2 |

`client_state()` trả về, cho từng client còn dữ liệu: shard, số mẫu còn giữ, số mẫu từng lớp, byte dữ liệu, số vòng của model shard. Đây là phần trạng thái FU của execution-state twin (đặc tả §7.1); CPU, bandwidth và timestamp là phần của DT.

Với `dynamicfu`, bước ước lượng đóng góp được ghi riêng (`phase = contribution`): mỗi client còn dữ liệu train một lượt và gửi một update. Lưu lượng này **có** trong `bytes_*`, nhưng **không** có trong `train_bytes_raw` và công thức byte của v2.

## Giới hạn của phép đo

- **Client chạy tuần tự trong một tiến trình.** Makespan là suy ra, không phải đo: nó giả định client chạy song song không tranh chấp tài nguyên, và **không gồm thời gian truyền**. DT tính thời gian truyền từ số byte và trạng thái bandwidth của nó.
- **Bộ nhớ là của cả tiến trình mô phỏng** (mọi client và mọi shard), không phải của một client. Trên Linux đọc từ `/proc`; trên Windows cần `psutil` (đã thêm vào `requirements-ml.txt`), thiếu thì trường bộ nhớ là rỗng và có cảnh báo `MEMORY_UNAVAILABLE`.
- **Chưa đo trên đường truyền thật.** Byte payload là kích thước sau `pack_state`; header HTTP, TLS và truyền lại không được tính. Prototype Docker (`fl_server.py`) vẫn chỉ chạy một FedAvg, chưa chạy shard.
- **Số chi phí hiện có là của một máy và 3 seed.** Số đếm (vòng, dòng-lượt, byte) của các chính sách có số vòng cố định không phụ thuộc máy; thời gian thì có, và số vòng của các chính sách dừng theo ngưỡng cũng có.
- **Mỗi lệnh CLI là một tiến trình**, nên mỗi yêu cầu qua CLI trả thêm `backend_warmup_seconds`. Một dịch vụ chạy lâu chỉ trả một lần.

## Còn để ngỏ cho phase DT

- **Chính sách mặc định.** `shard_full` là chính sách duy nhất vừa chính xác vô điều kiện vừa là mốc của G5a; `utility_stop` là chính sách duy nhất noninferior so với model đơn (G5b) nhưng chính xác có điều kiện. CLI buộc chọn rõ qua `--method` khi cần khác giá trị lưu trong state.
- **Biến quyết định của planner.** Với shard, planner không còn chọn tập client recovery. Những gì thành phần cho phép chọn theo từng yêu cầu là chính sách (qua đó là số vòng), và thời điểm train bù của `drop_backfill`. Số shard S và cách gán client vào shard cố định từ lúc `fit`.
- **Số client.** Thiết kế v2 dùng 20 client; đề cương và prototype Docker dùng 5. Thành phần chạy được với 5 client trong một shard (có test), nhưng khi đó mỗi yêu cầu là train lại toàn bộ.
