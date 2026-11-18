# Merge nhánh DT vào nhánh v3: thành phần `src/fu/shard_exact.py`

Ngày 2026-11-18. Merge `dt2/thao-dinh` (commit `a0dc8e8`) vào `dt/thao-dinh` (commit `f118050`). Hai nhánh tách nhau từ `59f05f8`.

## Vấn đề
- Cả hai nhánh **tự viết riêng** `ShardedUnlearner.unlearn_batch` trong `src/fu/shard_exact.py`:

  | | Nhánh v3 (`ce126a8`) | Nhánh DT (`a0dc8e8`) |
  |---|---|---|
  | Chữ ký | `unlearn_batch(requests, checkpoint_dir=None)` | `unlearn_batch(requests, policy=None, checkpoint_dir=None)` |
  | Chính sách | luôn `shard_full`, bất kể chính sách mặc định | chính sách mặc định của unlearner, chỉ nhận `shard_full` hoặc `shard_fixed` |
  | Yêu cầu nhận được | chỉ sample-level, mỗi client phải còn dữ liệu | cả client-level, đi theo `structure_step` từng yêu cầu (gộp hoặc bỏ shard) |
  | Kết quả | `kind='batch_end'`, `details['retrained_shards']` | `kind='batch'`, `details['shards_retrained']`, `client=-1` khi lô có nhiều client |
  | History | một bản ghi mỗi yêu cầu | như vậy, thêm `batch_first_step` và `batch_size` |

  Nhánh DT còn thêm chính sách `shard_checkpoint` (`DT_POLICIES`, `ALL_POLICIES`) và tách phần kiểm tra yêu cầu thành `_forget_rows`.
- Git gộp file mà không báo xung đột, nhưng file gộp có **hai hàm cùng tên**; Python lặng lẽ dùng hàm sau (của nhánh DT).
- Cả hai bên đều khóa hash của file này: v3 qua `TRACKED_V3` (`sha256_at_start` trong `results/shard_v3/stream_*/run.json`, được `evaluate_shard_v3.py` kiểm tra lại); DT qua `results/dt_online_confirmation/protocol_lock.json` và `results/dt_predictive_confirmation/protocol_lock.json`. Một file chỉ khớp được một bên.

## Cách giải quyết
- `src/fu/shard_exact.py`: lấy **nguyên bản của nhánh DT** (trùng từng byte với `a0dc8e8`), vì bản đó là tập lớn hơn về chức năng và lock của DT vẫn khớp.
- `docs/shard_v3_plan.md`: giữ **nguyên bản của nhánh v3** (trùng từng byte với `f118050`), để hash tài liệu protocol của các run v3 vẫn khớp. Ghi chú 2026-11-15 mà nhánh DT thêm vào tài liệu đó được chuyển xuống mục cuối của file này.
- `scripts/shard/run_shard_stream_v3.py` **không phải sửa**: với các chính sách `batch_full_W`, runner gọi `unlearn_batch(group)` trên bản `fork('shard_full')`, chỉ với yêu cầu sample-level, và chỉ đọc `result.cost` cùng `details.get('fallback_rounds')`, `details.get('condition_met')`. Nhãn `batch_end` trong `steps.csv` do runner tự đặt, không lấy từ kết quả.
- `tests/test_shard_batch.py` (của nhánh v3) sửa theo hàm còn lại: tên trường kết quả, hai khóa history mới, truyền `'shard_full'` khi unlearner có chính sách mặc định khác, và bỏ ba trường hợp client-level khỏi danh sách "không hợp lệ" (giờ là hợp lệ). Các kiểm tra chính giữ nguyên: lô kết thúc trùng từng bit với chuỗi `shard_full` tuần tự, mỗi shard bị chạm chỉ train lại một lần, chi phí dòng × vòng, yêu cầu sai không đổi trạng thái, checkpoint khôi phục đúng.

## Hệ quả với kết quả v3
- `results/shard_v3/` **không đổi**. Nhưng hash của `src/fu/shard_exact.py` hiện tại **không còn khớp** `sha256_at_start` của 4 run, nên chạy lại `evaluate_shard_v3.py` trên các run này sẽ fail phần toàn vẹn ở đúng file đó. `gates.json` đã lưu được tính với code lúc chạy.
- Muốn tái lập hay đánh giá lại kết quả v3 đúng như lúc chạy: `git checkout f118050`.

## Kiểm chứng
Lúc commit merge chưa chạy được (môi trường chưa có PyTorch). Ngày 2026-11-18, sau merge, kiểm chứng được chạy **hai lần độc lập** ở hai môi trường (commit `6de550a` và `ec46f68`). Hai lần cùng tìm ra và sửa **cùng một dòng test**, giống hệt nhau; `src/fu/shard_exact.py` không đổi. Script so sánh của cả hai lần không nằm trong repo; chúng chỉ đọc hai cây code.

**Lỗi test chung.** `test_shard_batch.test_requires_fit_nonempty_batch_and_no_pending_backlog` gọi `unlearn_batch([])` trên unlearner có chính sách mặc định `drop_backfill` mà không truyền chính sách. Hàm còn lại kiểm tra chính sách trước khi kiểm tra lô rỗng, nên báo `supports` thay vì `at least one`. Đây là chỗ sót khi sửa test theo hàm còn lại (mục "Cách giải quyết"); dòng đó nay truyền `'shard_full'`.

**Lần 1: Linux** (4 CPU, Python 3.13.18, PyTorch 2.6.0+cu124 chạy CPU 2 luồng, NumPy 2.5.3, pandas 3.0.5).
- **Toàn bộ `tests/`: 143 pass, 1 skip** (`tests/test_shard_v2.py:79`, chỉ chạy trên Windows), sau khi sửa test.
- **So thành phần cũ (`f118050`) với thành phần đã merge, cùng tiến trình:** 6 bộ dữ liệu tổng hợp (800 dòng, 20 client, S ∈ {4, 5}, 10 yêu cầu sample-level ngẫu nhiên) × 5 đường code của v3 (`unlearn` với `shard_full`, `shard_fixed`, `utility_stop`, `drop_backfill` + `finish_backlog`, và `unlearn_batch` `shard_full` với lô 3, 1, 6 yêu cầu). Cả **30/30 tổ hợp trùng**: trọng số mọi shard (từng byte), dòng và số vòng của shard, mọi trường chi phí không phải thời gian hay bộ nhớ, shard được train lại hoặc bỏ, history (trừ hai khóa mới), assignment, client còn dữ liệu, dòng còn sống.
- **Smoke tổng hợp v3 (đúng dữ liệu và cấu hình của `tests/test_shard_v3.py`) ở cả hai cây code:** 4 họ × S ∈ {4, 5} × seed {7, 8} × mọi chính sách. `stream.json` trùng ở cả 4 họ. Trong 106.656 ô của `steps.csv` và `exactness.csv`, chỉ hai cột thời gian chạy `seconds` và `delta_seconds` khác; mọi cột utility, chi phí, exact và cấu trúc trùng.

**Lần 2: Windows** (Python 3.13.2, PyTorch 2.6.0+cu124 chạy CPU).
- **Toàn bộ `tests/`:** lần đầu 143 pass, 1 fail (lỗi test chung ở trên). Sau khi sửa, toàn bộ bộ test pass, kể cả test chỉ chạy trên Windows; về sau là 153 test, gồm các test mới của phần FPR ([fpr_threshold_results.md](fpr_threshold_results.md)).
- **Smoke tổng hợp v3 ở hai cây code** (`f118050` lấy bằng `git archive`, mỗi cây một tiến trình), cùng dữ liệu và cấu hình như lần 1. Mỗi dòng `steps.csv` có thêm SHA-256 của trọng số, chỉ số dòng, seed và số vòng của mọi shard đang phục vụ. Runner v3 trùng từng byte giữa hai cây; chỉ `shard_exact.py` khác.
- **Kết quả: trùng hoàn toàn.** Ở cả 4 họ, 628 dòng `steps.csv` và 24 dòng `exactness.csv` giống hệt nhau ở mọi cột trừ `seconds` và `delta_seconds`, kể cả hash trọng số ở từng bước (266–290 hash khác nhau mỗi họ).

**Kết luận và giới hạn.** Hai môi trường, hai cách so (trong cùng tiến trình, và giữa hai cây code có hash trọng số từng bước), đều cho thấy thành phần đã merge cho đúng các model, chi phí dòng × vòng và cờ chính xác như thành phần mà các run v3 đã dùng. **Không chạy lại trên dữ liệu thật:** kết luận dựa trên dữ liệu tổng hợp và model nhỏ của smoke, không phải một lần chạy lại 10 seed.

## Ghi chú 2026-11-15 của nhánh DT (chuyển từ `shard_v3_plan.md`)
Nguyên văn phần nhánh DT thêm vào mục "Lịch sử bản nháp" của `shard_v3_plan.md`:

- 2026-11-15: thành phần `src/fu/shard_exact.py` (thuộc `TRACKED_V3`) được bổ sung **trước khi chốt**, cho các hướng mở rộng của DT ([dt_extensions.md](dt_extensions.md)):
  - `unlearn_batch(requests, policy)`: phần bổ sung mà mục "File" yêu cầu. Kiểm tra mọi yêu cầu theo thứ tự trước khi đổi trạng thái, đi theo `structure_step` từng yêu cầu, rồi train lại mỗi shard bị chạm một lần trên dữ liệu sau cả lô (`shard_full` hoặc `shard_fixed`). Test: lô một yêu cầu train đúng như `unlearn`; lô nhiều yêu cầu kết thúc trùng từng bit với chuỗi `shard_full` tuần tự; chế độ `window` của bộ lập lịch DT tái lập đúng `stream_v3.cost_plan` (dòng × vòng, số lần train lại, độ trễ) ở cả 4 họ và W ∈ {0, 2, 5}.
  - Chính sách `shard_checkpoint` của DT. Nó **không** thuộc v3; `POLICIES` vẫn là sáu chính sách của v2 (thêm `DT_POLICIES`, `ALL_POLICIES`), nên `tests/test_shard_exact.py` không đổi và vẫn pass.
  - Phần kiểm tra yêu cầu được tách thành `_forget_rows`, dùng chung cho `unlearn` và `unlearn_batch`, với cùng điều kiện và cùng thông báo lỗi.
  - **Phần dữ liệu đã được nhìn thêm:** thực nghiệm DT chạy chuỗi `spread`, `proportional` (hướng 1–2) và cả 4 họ (hướng 3) với S = 4 và seed train của v2 (50042, 50043), và đọc utility trên **validation**. Calibration, test và seed v3 60042–60051 không được dùng. Mô hình chuỗi, gate và biên không đổi.

Bổ sung khi merge: hai thực nghiệm DT sau đó ([dt_online_confirmation_plan.md](dt_online_confirmation_plan.md), [dt_predictive_plan.md](dt_predictive_plan.md)) dùng seed train 71042–71044 và 72042–72044, chỉ chấm validation; theo hai protocol đó, chúng không dùng seed v3 60042–60051 hay calibration/test.
