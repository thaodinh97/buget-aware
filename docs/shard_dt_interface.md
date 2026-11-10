# Hướng shard làm đầu vào của Digital Twin: thành phần gọi được và bản ghi chi phí

**Quyết định (2026-11-10).** Bước FU mà phase DT nhận vào là **unlearn chính xác theo shard** (protocol v2: [shard_v2_plan.md](shard_v2_plan.md), [shard_v2_results.md](shard_v2_results.md)). Quyết định này thay mục 3 "Bàn giao cho phase DT" của [fu_freeze.md](fu_freeze.md); phần còn lại của tài liệu đó (kết luận được và không được tuyên bố về FU xấp xỉ) giữ nguyên giá trị.

Tài liệu này mô tả thành phần mà DT gọi, những gì nó đo, và những gì **chưa** đo. Nó không thêm kết quả thực nghiệm nào: mọi con số utility vẫn là của `results/shard_v2/`.

## File

| File | Vai trò |
|---|---|
| `src/fu/shard_exact.py` | Thành phần: `ShardedUnlearner` (fit, unlearn từng yêu cầu, checkpoint, kiểm tra chính xác), `ShardExactMethod` (interface `UnlearningMethod` của đặc tả §5.4), `UnlearningResult` |
| `src/fu/shard_cost.py` | `CostMeter`: bản ghi chi phí theo client, theo vòng và theo yêu cầu |
| `scripts/shard/shard_service.py` | CLI `init` / `unlearn` / `verify` trên thư mục state |
| `scripts/shard/replay_v2_stream.py` | Chạy lại chuỗi 10 yêu cầu của v2 qua thành phần, ghi bản ghi chi phí, so với `steps.csv` đã lưu |
| `tests/test_shard_exact.py` | 17 test, gồm test tương đương với runner v2 |

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

**Chưa kiểm chứng trên dữ liệu thật.** Môi trường viết code không có dataset. Lệnh kiểm chứng (trên máy đã chạy v2 thì kỳ vọng khớp từng bit):

```bash
python scripts/shard/replay_v2_stream.py runs/shard_dt/replay_s4_50042_utility_stop --shards 4 --seed 50042 --method utility_stop --compare results/shard_v2/stream_20c
```

Lệnh trả mã thoát 1 nếu bất kỳ đại lượng rời rạc nào lệch, hoặc metric lệch quá `--tolerance` (mặc định 0). Trên máy khác, sai khác ở chữ số cuối có thể xảy ra; khi đó dùng `--tolerance 1e-9` và ghi rõ trong báo cáo.

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
- **Chưa có số chi phí trên dữ liệu thật.** Bảng chi phí của `results/shard_v2/` vẫn là số của v2 (thời gian tuần tự, byte theo công thức). Số mới chỉ có sau khi chạy `replay_v2_stream.py`.

## Còn để ngỏ cho phase DT

- **Chính sách mặc định.** `shard_full` là chính sách duy nhất vừa chính xác vô điều kiện vừa là mốc của G5a; `utility_stop` là chính sách duy nhất noninferior so với model đơn (G5b) nhưng chính xác có điều kiện. CLI buộc chọn rõ qua `--method` khi cần khác giá trị lưu trong state.
- **Biến quyết định của planner.** Với shard, planner không còn chọn tập client recovery. Những gì thành phần cho phép chọn theo từng yêu cầu là chính sách (qua đó là số vòng), và thời điểm train bù của `drop_backfill`. Số shard S và cách gán client vào shard cố định từ lúc `fit`.
- **Số client.** Thiết kế v2 dùng 20 client; đề cương và prototype Docker dùng 5. Thành phần chạy được với 5 client trong một shard (có test), nhưng khi đó mỗi yêu cầu là train lại toàn bộ.
