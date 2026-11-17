# Digital Twin: ngân sách chặt và stress, điều chỉnh giữa chừng, gom lô theo deadline

**Trạng thái: thực nghiệm mô tả, không có gate** (chạy 2026-11-15; bổ sung 2026-11-16). Tài liệu ghi lại ba hướng mở rộng của DT trên thành phần shard ([shard_dt_interface.md](shard_dt_interface.md)): cách cài đặt, cách đo và kết quả.
- **Tiếp nối 2026-11-17:** [protocol online có đối chứng](dt_online_confirmation_plan.md) dùng cấu hình chốt trước và deadline hoàn thành theo đồng hồ thực. Đây là bộ thử mới; giữ nguyên các kết quả và định nghĩa thời gian mô phỏng dưới đây.
- **Bổ sung 2026-11-16:** đối chứng đơn giản cho hướng 1–2 và độ nhạy theo trọng số EWMA (mục "Đối chứng đơn giản"); chi phí của chính DT (mục "Chi phí của chính DT"); lần chạy online thứ hai. Phần chấm offline dùng lại đúng các trajectory đã đo ngày 2026-11-15, không train lại.
- Không có tiêu chí pass/fail đặt trước. Mọi con số là của **một máy** và **ít seed**, xem mục "Không được tuyên bố".
- Seed train: 50042, 50043 (hướng 1–2) và 50042 (hướng 3). Đây là seed v2, đã được xem. **Seed v3 60042–60051 không được dùng**, để dành cho lần chạy xác nhận của v3.
- Chỉ đọc **validation**. Calibration và test không được dùng.

## File

| File | Vai trò |
|---|---|
| `src/fu/shard_exact.py` | Thêm chính sách `shard_checkpoint` (hướng 2) và `unlearn_batch` (hướng 3). `POLICIES` vẫn là sáu chính sách của v2 |
| `src/dt/stress.py` | Tiêm client chậm và bandwidth giảm qua hook `worker_factory` có sẵn (`_Trainer`, `src/fu/shard_exact.py`). Trọng số không đổi |
| `src/dt/planner.py` | Cost model của DT (giữ nguyên công thức của `scripts/shard/shard_dt.py`), Static / Adaptive / oracle, biên theo sai số quá khứ, lag (state cũ), chế độ `precommit` / `midcourse`; các đối chứng đơn giản và trọng số EWMA |
| `src/dt/scheduler.py` | Bộ lập lịch theo deadline: `immediate`, `window`, `lazy`, `dt` |
| `scripts/dt/machine.py` | Tắt power throttling của Windows cho tiến trình đo (mục "Trôi tốc độ của máy") |
| `scripts/dt/run_stress_trajectories.py` | Đo trajectory chi phí dưới stress (hướng 1–2) |
| `scripts/dt/evaluate_stress.py` | Chấm offline các planner trên trajectory, ghép cặp |
| `scripts/dt/run_midcourse_online.py` | Chạy online: DT điều khiển `shard_checkpoint` |
| `scripts/dt/run_batch_dt.py`, `scripts/dt/analyze_batch.py` | Hướng 3 |
| `src/dt/overhead.py`, `scripts/dt/measure_dt_overhead.py` | Chi phí của chính DT: thời gian state / plan / sync, byte telemetry, chi phí của việc đo |
| `tests/test_dt_extensions.py` | 27 test (toàn bộ bộ test của repo: 92 test, đều pass) |
| `results/dt_extensions/` | `trajectories/` (dữ liệu đo), `stress/` (bảng hướng 1–2), `midcourse_online/`, `batch/` |

## Thay đổi ở thành phần

`src/fu/shard_exact.py` nằm trong danh sách hash của v3 (`TRACKED_V3`), nên mọi thay đổi phải làm **trước khi chốt v3**. Thay đổi được ghi vào "Lịch sử bản nháp" của [shard_v3_plan.md](shard_v3_plan.md).
- **Hành vi cũ không đổi.** `POLICIES` giữ nguyên, nên `EquivalenceTests` vẫn so đủ sáu chính sách với runner v2 nguyên bản. `tests/test_shard_exact.py` (33 test, gồm cả test tương đương) vẫn pass. Chính sách mới nằm trong `DT_POLICIES`; `ALL_POLICIES = POLICIES + DT_POLICIES`.
- **`shard_checkpoint`** (hướng 2): train `shard_fixed_rounds` = 10 vòng. Sau đó nó train tiếp tới 20 vòng khi elapsed + 10 · (thời gian trung bình một vòng đã đo) · (1 + margin) ≤ budget_seconds. Nếu không thì dừng.
- **`unlearn_batch(requests, policy)`** (hướng 3): áp mọi yêu cầu, rồi train lại mỗi shard bị chạm **một lần** (`shard_full` hoặc `shard_fixed`).
- Phần kiểm tra một yêu cầu được tách thành `_forget_rows`, dùng chung cho cả hai đường. Điều kiện và thông báo lỗi giữ nguyên.

## Thiết lập chung

- **Dữ liệu và cấu trúc** như v2/v3:
  - `connected_v3_auxfit_seed42`, hash nội dung khớp `f130057e…`;
  - 20 client, partition 42, S = 4 (gán seed 7), cấu hình `standard` (R_base = 20, `shard_fixed` = 10).
- **Chuỗi yêu cầu:** chuỗi v3 (`scripts/shard/stream_v3.py`): 20 yêu cầu sample-level, thời điểm Poisson. Dòng bị xóa được chọn bằng `select_rows` trên các dòng còn sống. Hướng 1–2 dùng họ `spread` (lần lượt cả 4 shard, nên chi phí trải từ 1,3 tới 8 giây) và `proportional` (phần lớn rơi vào shard 2, lớn nhất). Hướng 3 dùng cả 4 họ.
- **Ngân sách thời gian:** tổng `wall_seconds` của các vòng FedAvg của một yêu cầu, đúng đại lượng mà `shard_dt.py` dùng. Ba mức suy từ **lần fit đo được của cùng tiến trình**, trước mọi yêu cầu. F là thời gian 20 vòng fit của shard lớn nhất (đo được 8,11–8,19 giây):
  - **rộng = 1,5 F**: không stress thì mọi `shard_full` đều vừa;
  - **vừa = 0,75 F**: shard lớn nhất phải hạ xuống `shard_fixed`;
  - **chặt = 0,5 F**: bằng chi phí `shard_fixed` của shard lớn nhất, nên các quyết định trên shard này **cố ý nằm sát ngưỡng**.
- **Cost model** giữ nguyên DT hiện có:
  - dự báo = số vòng × (Σ rate_c · n_c + server);
  - Static giữ hồ sơ của lần fit; Adaptive cập nhật EWMA 0,3 theo từng client và phần server, sau mỗi thao tác.
  - Có test kiểm tra dự báo của `src/dt/planner.py` bằng đúng `shard_dt.plan`.
- **Byte không được xét.** Byte thô là tất định (2 · R · C · B), nên dự báo byte không có sai số, và stress không đổi byte.

### Trôi tốc độ của máy

Máy chạy: Windows 11, Intel i7-13620H (6 nhân hiệu năng, 4 nhân tiết kiệm điện), power plan Balanced, Python 3.13.2, torch 2.6.0 CPU, 2 thread mỗi tiến trình.
- **Lần chạy đầu bị bỏ.** Giữa lần chạy đầu, chi phí của **mọi** yêu cầu trong cả 4 tiến trình tăng gấp khoảng 3 lần **cùng một lúc**, ở môi trường không stress. Ví dụ: shard 2 tăng từ 12 lên 37 giây ở bước 11.
- **Nguyên nhân có khả năng nhất** là Windows hạ chất lượng dịch vụ (EcoQoS: nhân tiết kiệm điện, xung thấp) cho tiến trình nền không có cửa sổ. Lần đo thời gian đầu tiên, chạy riêng một tiến trình, cũng đo shard 2 mất 35 giây, so với khoảng 8 giây sau khi sửa.
- **Cách sửa:**
  - mỗi runner gọi `prefer_performance()` lúc bắt đầu: tắt execution-speed throttling (`SetProcessInformation`) và đặt priority ABOVE_NORMAL. Việc này ghi vào `run.json`;
  - các môi trường chạy **xen kẽ theo từng yêu cầu**, thứ tự xoay vòng mỗi bước. Nhờ vậy, nếu tốc độ máy còn trôi thì nó tác động lên mọi môi trường ở cùng các bước.
- **Sau khi sửa,** chi phí của môi trường không stress ổn định suốt lần chạy. Hệ số stress / không stress ở bước 1–5 (chưa stress) là 0,96–1,05.
- **Hệ quả cho mức bandwidth.** Mức 2 Mbit/s được chọn khi máy còn ở chế độ chậm. Ở chế độ nhanh, nó nhân chi phí lên **2,5–16 lần (trung bình khoảng 7 lần)**, nặng hơn dự định. Tham số không được chỉnh lại sau khi thấy kết quả.

## Hướng 1: ngân sách chặt, kịch bản stress và biên an toàn

### Thiết kế

**Trajectory ghép cặp.** Mỗi tiến trình ứng với một (seed, họ) và đo thật như sau:
- Ensemble được train một lần, không stress. Lần fit này là hiệu chuẩn của Static.
- Mỗi môi trường chạy trên một fork riêng: 20 yêu cầu, mọi yêu cầu đều bằng `shard_full`, stress tiêm qua `worker_factory`.
- 10 vòng đầu của `shard_full` chính là `shard_fixed`, trùng từng bit. Vì vậy một lần chạy cho biết **chi phí thật của cả hai phương án** ở mọi yêu cầu, trong cùng trạng thái.
- Planner được chấm **offline** trên các số đo này. Mọi planner gặp cùng yêu cầu, cùng trạng thái và cùng chi phí thật, nên so sánh là ghép cặp.

Giả định: yêu cầu bị từ chối được **chuyển lên xử lý ngoài ngân sách** (escalate) bằng `shard_full`. Chuỗi trạng thái vì vậy như nhau với mọi planner, và DT vẫn quan sát được chi phí của mọi thao tác.

**Môi trường** (`src/dt/stress.py`):

| Môi trường | Tiêm gì | Bắt đầu | Hệ số chi phí so với không stress |
|---|---|---|---|
| `none` | — | — | 1 |
| `slow_clients` | Mỗi shard có một client chậm (chọn bằng seed 301: client 0, 3, 8, 11): tính toán chậm 4 lần | bước 6 | trung bình 1,57–1,59 (1,22–2,41) |
| `low_bandwidth` | Mọi client: 2 Mbit/s, tức 2 × 18.312 byte mỗi vòng. Phần truyền không phải tính toán, nên rơi vào phần server của vòng, như ở prototype RPC | bước 6 | trung bình 6,3–7,9 (2,5–16) |
| `transient_slow` | Như `slow_clients`, nhưng chỉ ở bước 6–12 | bước 6 | 1,48–1,59, rồi về 0,98–0,99 |

- **Cách tiêm:** wrapper gọi `ClientWorker.train` với đúng tham số, trả về đúng update, rồi **ngủ thật**. Thời gian vòng mà `CostMeter` đo vì vậy chứa độ trễ tiêm, còn trọng số không đổi (có test).
- **State cũ** là thuộc tính của twin, không phải của môi trường. Với **lag L = 3**:
  - twin nhận mọi quan sát (bản ghi chi phí, sai số) chậm 3 yêu cầu;
  - số mẫu từng client mà twin biết cũng cũ 3 yêu cầu (chưa trừ 3 lần xóa gần nhất).
  - Kịch bản "state cũ" trong bảng là `transient_slow` nhìn qua twin có lag 3; mọi tổ hợp (môi trường, lag) có trong `results/dt_extensions/stress/summary.csv`.

**Planner.**
- `precommit`: chọn `shard_full` nếu F̂ · m ≤ B; nếu không thì `shard_fixed` nếu X̂ · m ≤ B; nếu không thì từ chối. F̂, X̂ là dự báo 20 và 10 vòng.
- **Biên** m: phân vị 90 % của tỉ số thực / dự báo trên 10 thao tác gần nhất. Đây là cận trên kiểu conformal một phía, và có thể < 1 khi twin dự báo thừa. Lịch sử khởi đầu bằng sai số của lần fit.
- **Không biên:** m = 1.
- **Oracle** biết chi phí thật.
- **Biên chấm lại** (Adaptive): **thêm sau khi xem kết quả đầu**, vì biên thường bị kẹt sau khi stress bắt đầu (xem dưới). Thay vì tỉ số của dự báo lúc đó, mỗi thao tác cũ được dự báo lại bằng hồ sơ hiện tại. Sai số mà hồ sơ đã hấp thụ vì vậy thôi làm phồng biên. Với Static, cách này trùng với biên thường (có test).

**Thước đo** (mỗi trajectory 20 yêu cầu; gộp 4 trajectory = 80 yêu cầu mỗi ô):
- **Vi phạm:** yêu cầu được chấp nhận nhưng chi phí thật của phương án đã chạy > B. Tỉ lệ tính trên số yêu cầu chấp nhận.
- **Từ chối nhầm:** bị từ chối trong khi oracle chấp nhận (tức `shard_fixed` thật vừa B). Tỉ lệ tính trên số yêu cầu oracle chấp nhận.
- **Hạ cấp nhầm:** chọn `shard_fixed` trong khi `shard_full` thật vừa B.

### Kết quả

Mỗi ô: **số vi phạm / số chấp nhận · số từ chối nhầm**. Oracle không bao giờ vi phạm hay từ chối nhầm; cột của nó cho biết bao nhiêu yêu cầu thật sự vừa ngân sách. Bảng đầy đủ, cả hạ cấp nhầm và số giây vượt, ở `results/dt_extensions/stress/report.md`.

| Kịch bản | Ngân sách | Oracle chấp nhận | Static | Static + biên | Adaptive | Adaptive + biên | Adaptive + biên chấm lại |
|---|---|---|---|---|---|---|---|
| Không stress | chặt | 75/80 | 8/80 · 0 | 2/62 · 13 | 8/80 · 0 | 2/62 · 13 | 4/64 · 13 |
| Không stress | vừa | 80/80 | 0/80 · 0 | 0/80 · 0 | 0/80 · 0 | 0/80 · 0 | 0/80 · 0 |
| Không stress | rộng | 80/80 | 0/80 · 0 | 0/80 · 0 | 0/80 · 0 | 0/80 · 0 | 0/80 · 0 |
| Client chậm | chặt | 56/80 | 52/80 · 0 | 10/56 · 4 | 27/66 · 0 | 10/54 · 6 | 12/56 · 4 |
| Client chậm | vừa | 78/80 | 12/80 · 0 | 5/68 · 12 | 12/80 · 0 | 4/64 · 16 | 8/76 · 4 |
| Client chậm | rộng | 80/80 | 0/80 · 0 | 0/80 · 0 | 0/80 · 0 | 0/80 · 0 | 0/80 · 0 |
| Bandwidth giảm | chặt | 14/80 | 70/80 · 0 | 11/19 · 4 | 16/25 · 1 | 8/16 · 4 | 10/18 · 4 |
| Bandwidth giảm | vừa | 20/80 | 60/80 · 0 | 14/34 · 0 | 10/30 · 0 | 6/26 · 0 | 6/26 · 0 |
| Bandwidth giảm | rộng | 68/80 | 60/80 · 0 | 14/48 · 20 | 22/80 · 0 | 6/28 · 40 | 7/65 · 4 |
| State cũ (lag 3) | chặt | 64/80 | 30/80 · 0 | 11/59 · 10 | 20/72 · 4 | 10/58 · 10 | 11/59 · 10 |
| State cũ (lag 3) | vừa | 80/80 | 6/80 · 0 | 4/76 · 4 | 6/80 · 0 | 4/72 · 8 | 4/78 · 2 |
| State cũ (lag 3) | rộng | 80/80 | 0/80 · 0 | 0/80 · 0 | 0/80 · 0 | 0/80 · 0 | 0/80 · 0 |

**Sai số dự báo** của `shard_full`, khi mọi yêu cầu đều được chấp nhận (ngân sách rộng, không biên), tính bằng thực / dự báo − 1:

| Môi trường | Static: trung bình (lớn nhất) | Adaptive: trung bình (lớn nhất) |
|---|---|---|
| none | +2,6 % (+27 %) | +1,7 % (+28 %) |
| slow_clients | +46 % (+109 %) | +27 % (+109 %) |
| low_bandwidth | +478 % (+1.730 %) | +57 % (+1.260 %) |
| transient_slow | +20 % (+101 %) | +10 % (+101 %) |

**Đọc kết quả:**
- **Không stress,** ngân sách vừa và rộng không có vi phạm nào ở mọi planner. Ở mức chặt (cố ý sát ngưỡng), Static và Adaptive cùng vi phạm 8/80. Biên giảm xuống 2/62, nhưng đổi lấy 13/75 từ chối nhầm và 11/45 hạ cấp nhầm.
- **Static hỏng dưới stress:**
  - client chậm, mức chặt: 52/80 yêu cầu chấp nhận bị vượt;
  - bandwidth giảm: 60–70/80 ở cả ba mức, tổng vượt 520–744 giây.
  - Static không bao giờ học, nên vi phạm kéo dài tới hết chuỗi.
- **Adaptive giảm vi phạm còn khoảng một nửa** (client chậm, mức chặt: 27/66), nhưng vẫn vượt ngay sau khi stress bắt đầu: EWMA 0,3 cần vài quan sát mới theo kịp. Dưới bandwidth giảm, mức rộng: 22/80.
- **Biên từ sai số quá khứ cho ít vi phạm nhất,** với giá là từ chối nhầm:
  - client chậm, mức chặt: 10/54 vi phạm;
  - bandwidth giảm, mức vừa: 6/26.
  - Biên làm cả Static ít vi phạm hơn nhiều (bandwidth giảm, mức vừa: 60 → 14). Tuy vậy Static + biên vẫn kém Adaptive + biên, vì hồ sơ sai theo từng shard mà một hệ số chung không sửa được.
- **Biên bị kẹt sau khi stress bắt đầu.** Thao tác đầu tiên sau bước 6 có tỉ số thực / dự báo tới 13,6 (Adaptive, bandwidth giảm). Tỉ số đó nằm trong cửa sổ 10 thao tác và giữ phân vị 90 % ở mức rất cao, cả sau khi hồ sơ Adaptive đã theo kịp. Ví dụ bandwidth giảm, mức rộng: Adaptive + biên từ chối nhầm 40/68 yêu cầu vừa ngân sách.
  - **Biên chấm lại** sửa phần lớn: 4/68 từ chối nhầm, 7/65 vi phạm. Nó cũng giảm từ chối nhầm ở client chậm, mức vừa (16 → 4), nhưng đổi lại có thêm vi phạm (4/64 → 8/76).
  - Ở mức chặt, hai cách gần như bằng nhau.
- **State cũ:**
  - lag 3 làm Adaptive vi phạm nhiều hơn lúc stress bắt đầu: mức chặt, 20/72 so với 18/72 khi twin cập nhật kịp;
  - Adaptive cũng từ chối nhầm sau khi stress đã hết: 4 so với 2.
  - Với biên, vi phạm ở mức chặt tăng từ 6/50 lên 10/58.
  - Ảnh hưởng nhỏ hơn hai môi trường kia, vì đợt chậm ngắn và chỉ nhân chi phí khoảng 1,5 lần.

## Hướng 2: điều chỉnh giữa chừng

### Chính sách `shard_checkpoint`

- **Tương đương bit.** Trạng thái sau 10 vòng của `shard_full` trùng từng bit với `shard_fixed`, vì cùng `_Trainer` và cùng seed theo vòng (seed + 1000 · j + r · 10000). Có test cả hai chiều: dừng thì model bằng `shard_fixed`, đi tiếp thì bằng `shard_full`, và `verify_bit_exact` đạt.
- **Quy tắc:** sau 10 vòng, gọi e là tổng thời gian vòng đo được. Train tiếp khi e + e · (20 − 10)/10 · (1 + margin) ≤ budget_seconds.
  - Tham số `margin` để DT truyền biên của nó cho phép ngoại suy: phân vị 90 % của tỉ số giữa thời gian một vòng sau và trước checkpoint, trên các lần đã chạy đủ 20 vòng.
  - Chính sách không chấm model nào: `policy_eval_seconds` = 0 (có test).
- **Phạm vi chính xác: `unconditional`.** Đồng ý với nhận định ban đầu:
  - Kết quả luôn là một trong hai model, mỗi model có số vòng cố định trước (10 hoặc 20), và trùng từng bit với một lần train lại từ đầu trên dữ liệu còn giữ.
  - Lựa chọn giữa hai model dùng tín hiệu chi phí (thời gian của chính lần train lại trên dữ liệu đã xóa, ngân sách, biên), không dùng model hay điểm validation nào trước khi xóa. Đó là khác biệt với `utility_stop`, nơi ngưỡng F1₀ đến từ ensemble đã train trên cả dòng bị xóa.
  - Một điểm cần ghi: biên lấy từ thời gian của các thao tác trước, mà các thao tác đó train trên lượng dữ liệu có cả dòng sau này bị xóa. Thông tin duy nhất về yêu cầu mà thời gian có thể mang là **số dòng**, và history vốn đã lưu số dòng công khai. Không có nội dung bản ghi nào đi vào quyết định.
- **Không thuộc v3:** `POLICIES` không đổi. Muốn đưa vào gate của v3 thì phải quyết định trước khi chốt.

### Kết quả offline (cùng trajectory với hướng 1)

Chế độ `midcourse`:
- bắt đầu khi X̂ · m ≤ B (`shard_fixed` vừa theo dự báo);
- ở vòng 10, áp đúng quy tắc của `shard_checkpoint` lên thời gian đo thật của vòng 1–10;
- thời gian của vòng 11–20 cũng là số đo thật, nên kết quả "nếu đi tiếp" được biết chính xác.

Vi phạm, `precommit` → `midcourse` (số giây vượt trong ngoặc):

| Kịch bản | Ngân sách | Static | Adaptive | Adaptive + biên | Adaptive + biên chấm lại |
|---|---|---|---|---|---|
| Không stress | chặt | 8 → 5 (2 → 1) | 8 → 5 (2 → 1) | 2 → 0 (1 → 0) | 4 → 2 (1 → 0) |
| Client chậm | chặt | 52 → 24 (60 → 19) | 27 → 10 (24 → 8) | 10 → 2 (6 → 1) | 12 → 4 (7 → 2) |
| Client chậm | vừa | 12 → 2 (5 → 0) | 12 → 2 (5 → 0) | 4 → 2 (1 → 0) | 8 → 2 (2 → 0) |
| Bandwidth giảm | chặt | 70 → 68 (744 → 384) | 16 → 14 (51 → 34) | 8 → 8 (39 → 23) | 10 → 10 (50 → 34) |
| Bandwidth giảm | vừa | 60 → 60 (619 → 260) | 10 → 10 (51 → 35) | 6 → 6 (36 → 20) | 6 → 6 (36 → 20) |
| Bandwidth giảm | rộng | 60 → 12 (520 → 7) | 22 → 12 (77 → 7) | 6 → 0 (41 → 0) | 7 → 2 (41 → 0) |
| State cũ (lag 3) | chặt | 30 → 18 (28 → 8) | 20 → 14 (15 → 5) | 10 → 6 (6 → 2) | 11 → 6 (7 → 2) |
| State cũ (lag 3) | vừa | 6 → 0 (2 → 0) | 6 → 0 (2 → 0) | 4 → 0 (1 → 0) | 4 → 0 (1 → 0) |

- **Điều chỉnh giữa chừng xóa phần lớn vi phạm do dự báo thiếu cho 10 vòng sau.** Ví dụ:
  - bandwidth giảm, mức rộng: 60 → 12 với Static, 6 → 0 với Adaptive + biên;
  - client chậm, mức chặt: 27 → 10 với Adaptive.
  - Kết hợp tốt nhất là Adaptive + biên chấm lại + `midcourse`, ví dụ bandwidth giảm, mức rộng: 2/66 vi phạm, 4 từ chối nhầm.
- **Nó không cứu được khi chính 10 vòng đầu đã vượt** (bandwidth giảm, mức chặt và vừa): yêu cầu đã chạy quá ngân sách trước khi tới checkpoint. Khi đó chỉ quyết định bắt đầu hay không mới quyết định kết quả. Số giây vượt vẫn giảm khoảng một nửa, vì không còn chạy thêm 10 vòng sau.
- **Cái giá:** đôi khi dừng nhầm, vì vòng 11–20 nhanh hơn vòng 1–10. Không stress, mức chặt: 4/45 hạ cấp nhầm, so với 0/45 khi chọn trước. Ở những ô mà biên làm planner hạ cấp nhầm khi chọn trước, `midcourse` lại giảm hạ cấp nhầm, vì nó vẫn đi tiếp khi thời gian đo cho thấy `shard_full` vừa: client chậm, mức rộng: 16 → 2; state cũ, mức vừa: 8 → 0.
- Từ chối nhầm gần như không đổi giữa hai chế độ, vì cả hai đều từ chối khi `shard_fixed` không vừa theo dự báo.

### Chạy online

`run_midcourse_online.py`, seed 50042, `spread`, `slow_clients`, mức vừa (B = 7,02 giây), Adaptive + biên (biên thường, như lúc chạy), `results/dt_extensions/midcourse_online/`:
- 20 yêu cầu: 12 `shard_full`, 5 dừng ở checkpoint, 3 bị escalate. Không có vi phạm nào trong 17 yêu cầu được chấp nhận.
- Cả 3 lần escalate (shard 2, sau khi stress bắt đầu) là **từ chối nhầm**: 10 vòng đầu thật chỉ mất 5,9–6,3 giây, nhưng hệ số biên 1,56 đẩy dự báo `shard_fixed` lên 7,9–8,4 giây. Đây đúng là hiện tượng biên bị kẹt ở trên.
- Cuối chuỗi, `verify_bit_exact` đạt cả 4 shard. Mọi kết quả mang `exactness_scope = unconditional`, và các yêu cầu dừng có cảnh báo `STOPPED_AT_CHECKPOINT`.

**Lần chạy online thứ hai** (2026-11-16, `--margin-mode rescored`, có đo chi phí của DT; `results/dt_extensions/midcourse_online_rescored/`). Cùng seed, họ chuỗi và môi trường, nhưng dùng biên chấm lại:
- Ngân sách mức vừa của lần này là 6,33 giây, không phải 7,02 giây: mỗi lần chạy suy ngân sách từ lần fit của chính nó (8,44 so với 9,36 giây cho shard lớn nhất). Hai lần chạy vì vậy **không phải so sánh ghép cặp** giữa hai kiểu biên.
- 20 yêu cầu: 8 `shard_full`, 9 dừng ở checkpoint, 3 bị escalate.
- 1 vi phạm, vượt 0,04 giây: yêu cầu đầu tiên vào shard 2 sau khi stress bắt đầu, 10 vòng đầu đã mất 6,37 giây.
- Trong 3 lần escalate (đều ở shard 2), 2 lần là từ chối nhầm (10 vòng đầu thật mất 5,94 và 6,17 giây) và 1 lần là từ chối đúng (6,47 giây).
- Hệ số biên chấm lại lên tới 1,57 rồi giảm dần còn 1,29 ở cuối chuỗi. Nó giảm chậm vì Adaptive với α = 0,3 chỉ gặp lại mỗi shard sau 4 yêu cầu, nên hồ sơ chưa hấp thụ hết thay đổi: dự báo `shard_fixed` của shard 2 vẫn là 4,8–5,3 giây trong khi chi phí thật là 5,9–6,5 giây (94–102 % ngân sách). Sau khi nhân biên, dự báo thành 6,9–7,2 giây và vượt ngân sách. Ở shard này gần như không còn chỗ cho sai số theo hướng nào.
- Cuối chuỗi, `verify_bit_exact` đạt cả 4 shard.

## Đối chứng đơn giản cho hướng 1–2 (bổ sung 2026-11-16)

**Câu hỏi:** cost model theo client của DT có đáng với độ phức tạp của nó không? Đề cương (nội dung 4) yêu cầu chỉ ra khi nào DT có lợi và khi nào nó chỉ ngang một cách điều phối đơn giản.

**Cách làm.** Các đối chứng được thêm sau lần đánh giá đầu, nhưng **định nghĩa của chúng được cố định trước khi chạy** (`src/dt/planner.py`). Chúng được chấm offline trên đúng các trajectory đã đo, không train lại. Kết quả của các planner cũ không đổi một số nào sau khi thêm (đã so 1.008 dòng của `summary.csv`).

| Đối chứng | Cách dự báo |
|---|---|
| Luôn full, Luôn fixed | Không dự báo, không từ chối: mọi yêu cầu chạy một chính sách |
| Theo dòng (tĩnh) | "Chi phí tỉ lệ với số dòng": một hệ số duy nhất (giây mỗi dòng mỗi vòng) từ lần fit, không cập nhật |
| Theo dòng (EWMA) | Như trên, cập nhật EWMA 0,3 như Adaptive. So với Adaptive: cùng tốc độ học, không có cấu trúc theo client |
| Lần trước cùng shard | Thời gian một vòng của thao tác gần nhất trên cùng shard (lúc đầu: lần fit), điều chỉnh theo số dòng hiện tại. Không mô hình, không làm trơn |

Ba đối chứng sau chỉ thay phần dự báo; quy tắc quyết định, biên và lag giữ nguyên như Static và Adaptive.

**Độ nhạy theo trọng số EWMA (α).** Sau khi thấy bảng đối chứng, mình thử thêm Adaptive với α = 0,6 và α = 1. α = 0,3 là giá trị kế thừa từ `shard_dt.py`; α = 1 nghĩa là giữ giá trị quan sát gần nhất của từng client và của phần server. Đây là **thăm dò sau khi xem kết quả**, không phải cấu hình chọn trước.

### Kết quả (`precommit`, không biên)

Mỗi ô: **vi phạm / chấp nhận · từ chối nhầm · hạ cấp nhầm**. Bảng đầy đủ (cả `midcourse` và có biên) ở `results/dt_extensions/stress/report.md`.

| Kịch bản | Ngân sách | Luôn full | Luôn fixed | Theo dòng (tĩnh) | Theo dòng (EWMA) | Lần trước cùng shard | Static | Adaptive (α = 0,3) | Adaptive (α = 1, thăm dò) |
|---|---|---|---|---|---|---|---|---|---|
| Không stress | chặt | 35/80 · 0 · 0 | 5/80 · 0 · 45 | 2/74 · 3 · 7 | 0/66 · 9 · 13 | 8/78 · 2 · 2 | 8/80 · 0 · 0 | 8/80 · 0 · 2 | 8/78 · 2 · 2 |
| Không stress | vừa | 32/80 · 0 · 0 | 0/80 · 0 · 48 | 0/80 · 0 · 0 | 0/80 · 0 · 0 | 0/80 · 0 · 0 | 0/80 · 0 · 0 | 0/80 · 0 · 0 | 0/80 · 0 · 0 |
| Client chậm | chặt | 60/80 · 0 · 0 | 24/80 · 0 · 20 | 42/74 · 6 · 0 | 14/56 · 4 · 0 | 14/60 · 0 · 0 | 52/80 · 0 · 0 | 27/66 · 0 · 0 | 14/60 · 0 · 0 |
| Client chậm | vừa | 42/80 · 0 · 0 | 2/80 · 0 · 38 | 12/80 · 0 · 0 | 6/74 · 6 · 0 | 8/78 · 2 · 2 | 12/80 · 0 · 0 | 12/80 · 0 · 0 | 8/78 · 2 · 2 |
| Bandwidth giảm | chặt | 74/80 · 0 · 0 | 66/80 · 0 · 6 | 62/74 · 2 · 0 | 24/35 · 3 · 0 | 24/32 · 2 · 0 | 70/80 · 0 · 0 | 16/25 · 1 · 0 | 12/20 · 2 · 0 |
| Bandwidth giảm | vừa | 70/80 · 0 · 0 | 60/80 · 0 · 10 | 60/80 · 0 · 0 | 22/42 · 0 · 0 | 18/38 · 0 · 0 | 60/80 · 0 · 0 | 10/30 · 0 · 0 | 4/24 · 0 · 0 |
| Bandwidth giảm | rộng | 60/80 · 0 · 0 | 12/80 · 0 · 20 | 60/80 · 0 · 0 | 22/50 · 18 · 0 | 25/74 · 1 · 0 | 60/80 · 0 · 0 | 22/80 · 0 · 0 | 9/71 · 2 · 0 |
| State cũ (lag 3) | chặt | 46/80 · 0 · 0 | 16/80 · 0 · 34 | 20/72 · 6 · 2 | 8/56 · 12 · 8 | 16/66 · 6 · 8 | 30/80 · 0 · 0 | 20/72 · 4 · 6 | 16/66 · 6 · 8 |
| State cũ (lag 3) | vừa | 38/80 · 0 · 0 | 0/80 · 0 · 42 | 6/80 · 0 · 0 | 6/80 · 0 · 0 | 4/80 · 0 · 4 | 6/80 · 0 · 0 | 6/80 · 0 · 0 | 4/80 · 0 · 4 |

Sai số tuyệt đối trung bình của dự báo `shard_full` (giây; twin cập nhật kịp, mọi yêu cầu được chấp nhận):

| Môi trường | Static | Theo dòng (tĩnh) | Theo dòng (EWMA) | Lần trước cùng shard | Adaptive (α = 0,3) | Adaptive (α = 1) |
|---|---|---|---|---|---|---|
| none | 0,25 | 0,27 | 0,39 | 0,37 | 0,27 | 0,37 |
| slow_clients | 1,69 | 1,67 | 0,97 | 0,77 | 1,11 | 0,78 |
| low_bandwidth | 12,6 | 12,5 | 12,9 | 4,20 | 3,50 | 1,56 |
| transient_slow | 0,84 | 0,83 | 0,82 | 0,97 | 0,88 | 0,96 |

### Đọc kết quả

- **Phải có quyết định.**
  - Luôn full vi phạm 32–35/80 ngay cả khi không stress (mức chặt và vừa), và 38–74/80 dưới stress.
  - Luôn fixed ít vi phạm hơn, nhưng hạ cấp mọi yêu cầu lẽ ra chạy được full (48/80 ở mức vừa khi không stress). Nó vẫn vi phạm 60–66/80 khi bandwidth giảm ở mức chặt và vừa, vì không bao giờ từ chối.
- **Không học thì cấu trúc theo client không đem lại gì.** Static và "theo dòng (tĩnh)" gần như bằng nhau: 52/80 so với 42/74 (client chậm, mức chặt), 60/80 so với 60/80 (bandwidth giảm, mức vừa). Sai số dự báo cũng bằng nhau.
- **Cùng tốc độ học 0,3, cấu trúc theo client chỉ thắng rõ khi thay đổi không tỉ lệ với số dòng.**
  - Bandwidth giảm: Adaptive 10/30 so với 22/42 của "theo dòng (EWMA)" ở mức vừa; sai số 3,5 so với 12,9 giây. Thời gian truyền tính theo client-vòng, nên một hệ số theo dòng không biểu diễn được.
  - Client chậm: "theo dòng (EWMA)" ít vi phạm hơn (mức chặt: 14/56 so với 27/66), nhưng vì nó dè dặt hơn: chấp nhận ít hơn và từ chối nhầm 4–6 yêu cầu, trong khi Adaptive không từ chối nhầm yêu cầu nào.
- **"Lần trước cùng shard" ngang hoặc hơn Adaptive (α = 0,3) khi thay đổi nằm trong từng shard, và kém khi thay đổi trên toàn hệ thống.**
  - Client chậm, mức chặt: 14/60 so với 27/66, không từ chối nhầm.
  - Bandwidth giảm, mức vừa: 18/38 so với 10/30.
- **Điểm yếu của Adaptive là tốc độ học, không phải cấu trúc.**
  - Với α = 1, Adaptive cho **đúng cùng số** với "lần trước cùng shard" ở client chậm, state cũ và không stress.
  - Ở bandwidth giảm, α = 1 thắng rõ: 4/24 so với 18/38 (mức vừa), 9/71 so với 25/74 (mức rộng), sai số 1,56 so với 4,20 giây. Lý do: phần server dùng chung cho mọi shard, nên thay đổi toàn hệ thống học được ở shard này áp dụng ngay cho shard chưa quan sát. Bộ nhớ theo shard phải gặp lại từng shard một lần.
  - Cái giá của α = 1: khi không stress, dự báo nhiễu hơn (sai số 0,37 so với 0,27 giây) và có 2 từ chối nhầm ở mức chặt.
- **Với biên q90,** các bộ dự báo gần nhau hơn (ví dụ `precommit`, client chậm, mức chặt: "lần trước cùng shard" 8/44 · 14 từ chối nhầm; Adaptive 10/54 · 6). Biên chấm lại chỉ có tác dụng với mô hình học dần: với α = 1 nó gần như không đổi gì (14/60 · 0, giống không biên), vì hồ sơ đã khớp sẵn các thao tác gần nhất.
- **Điều chỉnh giữa chừng giúp mọi bộ dự báo như nhau.** Ví dụ client chậm, mức chặt: "lần trước cùng shard" 14 → 4, Adaptive (α = 0,3) 27 → 10, Adaptive (α = 1) 14 → 4.

**Kết luận của phần này.** Giá trị của DT trong các kịch bản này đến từ ba thứ, theo thứ tự: có phản hồi; phản hồi đủ nhanh; và một tham số dùng chung giữa các shard để khái quát thay đổi toàn hệ thống. Cấu trúc theo client tự nó không tạo ra lợi thế. Với α = 0,3 kế thừa, một luật không mô hình ("lần trước cùng shard") ngang hoặc hơn DT ở thay đổi cục bộ. Với α = 1, cost model của DT không thua "lần trước cùng shard" ở ô nào và thắng rõ ở thay đổi toàn hệ thống. Hai điều cần nhớ khi dùng kết luận này: α = 1 được thử sau khi xem kết quả, và mọi kịch bản stress ở đây đều là thay đổi dạng bậc thang, là dạng thuận cho học nhanh. "Theo dòng (EWMA)" có ít vi phạm hơn ở vài ô, nhưng ở các ô đó nó luôn từ chối nhầm nhiều hơn.

## Hướng 3: gom lô và hoãn theo deadline

### Thiết kế (`src/dt/scheduler.py`)

- **Thời gian mô phỏng:** yêu cầu đến theo thời điểm Poisson của v3. Việc xử lý tức thời theo thời gian mô phỏng, như `stream_v3.cost_plan`: giây tính toán trừ vào ngân sách, không trừ vào đồng hồ. Độ trễ xóa là khoảng từ lúc yêu cầu đến tới lúc model phục vụ không còn dữ liệu đó. **Deadline D = 5** (độ trễ lớn nhất của `batch_full_5` là 4,95).
- **Ngân sách gộp:** mỗi yêu cầu mang B giây (mức vừa hoặc chặt, như hướng 1). Một lần xử lý phục vụ n yêu cầu được tiêu n · B. Yêu cầu vượt ngân sách một mình **vào hàng đợi** của shard của nó, thay vì bị từ chối.
- **Mỗi lần xử lý** là một lệnh `unlearn_batch`.
- **Chế độ:**

| Chế độ | Khi nào xử lý | Chính sách |
|---|---|---|
| `immediate` | Ngay khi yêu cầu đến (v3, W = 0) | `shard_full` |
| `window5` | Cuối mỗi cửa sổ 5 đơn vị, mọi shard (v3 `batch_full_5`) | `shard_full` |
| `lazy` | Từng shard, khi yêu cầu cũ nhất của nó chạm deadline | `shard_full` |
| `dt_mid`, `dt_tight` | Sau mỗi yêu cầu đến: xử lý hàng đợi của shard đó **ngay** nếu dự báo · biên của một lần train lại vừa ngân sách gộp; nếu không thì chờ gom thêm. Khi yêu cầu cũ nhất chạm deadline: `shard_full` nếu vừa, nếu không thì `shard_fixed` (vẫn xử lý dù không vừa, vì deadline là ràng buộc cứng, và ghi `forced_over_budget`) | Adaptive + biên q90 |

- `immediate`, `window5` và `lazy` được chấm theo ngân sách mức vừa.
- **Kiểm chứng:** chế độ `window` và `immediate` tái lập **đúng** `stream_v3.cost_plan` (dòng × vòng, số lần train lại, độ trễ trung bình và lớn nhất) ở 4 họ và W ∈ {0, 2, 5}. Có test với executor đếm dòng.

### Kết quả (seed 50042; mỗi chế độ là một lần chạy thật)

| Họ | Chế độ | Lần train lại | Giây (so với `immediate`) | Độ trễ TB / max | Lần xử lý vượt ngân sách | Phục vụ bằng `shard_full` | F1 validation cuối |
|---|---|---|---|---|---|---|---|
| `proportional` | `immediate` | 20 | 124 (1) | 0 / 0 | 10/20 | 100 % | 0,898 |
| | `window5` | 10 | 42 (0,34) | 2,15 / 4,95 | 0/4 | 100 % | 0,898 |
| | `lazy` | 8 | 34 (0,28) | 3,83 / 5 | 0/8 | 100 % | 0,898 |
| | `dt_mid` | 15 | 74 (0,60) | 0,33 / 5 | 0/15 | 95 % | 0,898 |
| | `dt_tight` | 15 | 57 (0,46) | 0,83 / 5 | 0/15 | 85 % | 0,898 |
| `spread` | `immediate` | 20 | 91 (1) | 0 / 0 | 5/20 | 100 % | 0,896 |
| | `window5` | 11 | 40 (0,44) | 2,15 / 4,95 | 0/4 | 100 % | 0,896 |
| | `lazy` | 10 | 35 (0,39) | 3,78 / 5 | 0/10 | 100 % | 0,896 |
| | `dt_mid` | 18 | 62 (0,68) | 0,32 / 5 | 0/18 | 95 % | 0,895 |
| | `dt_tight` | 15 | 45 (0,49) | 1,10 / 5 | 0/15 | 85 % | 0,895 |
| `concentrated` | `immediate` | 20 | 165 (1) | 0 / 0 | 15/20 | 100 % | 0,898 |
| | `window5` | 4 | 28 (0,17) | 2,15 / 4,95 | 0/4 | 100 % | 0,898 |
| | `lazy` | 3 | 22 (0,13) | 2,62 / 5 | 0/3 | 100 % | 0,898 |
| | `dt_mid` | 10 | 80 (0,48) | 0,30 / 2,18 | 0/10 | 100 % | 0,898 |
| | `dt_tight` | 10 | 68 (0,41) | 0,30 / 2,18 | 0/10 | 100 % | 0,898 |
| `poisson_zipf` | `immediate` | 20 | 100 (1) | 0 / 0 | 5/20 | 100 % | 0,898 |
| | `window5` | 11 | 46 (0,46) | 2,15 / 4,95 | 0/4 | 100 % | 0,898 |
| | `lazy` | 9 | 42 (0,42) | 4,00 / 5 | 1/9 | 100 % | 0,898 |
| | `dt_mid` | 18 | 72 (0,72) | 0,50 / 5 | 0/18 | 95 % | 0,898 |
| | `dt_tight` | 17 | 56 (0,56) | 0,57 / 5 | 0/17 | 95 % | 0,898 |

- **Không chế độ nào trễ deadline.** Không lần xử lý nào phải `forced_over_budget`.
- **Tính chính xác:** `exact` và `structure_ok` đạt ở mọi lần xử lý. Cuối mỗi chế độ, `verify_bit_exact` đạt. Ba lịch chỉ dùng `shard_full` (`immediate`, `window5`, `lazy`) **kết thúc ở cùng một trạng thái, trùng từng bit**, ở cả 4 họ (`results/dt_extensions/batch/final_states.csv`). Đây là tính chất mà gate G1 của v3 kiểm tra cho `batch_full_W`.

**Đọc kết quả:**
- **Xử lý ngay vượt ngân sách vừa** ở 5–15/20 yêu cầu, và mọi lần vượt đều là yêu cầu vào shard lớn nhất:
  - `proportional`: 10/11 yêu cầu vào shard này vượt;
  - `concentrated`: 15/20. Shard này mất tới 28 % dữ liệu trong chuỗi, nên các yêu cầu cuối lại vừa ngân sách.
  - `concentrated` tệ nhất, như kế hoạch v3 đã tính.
- **Gom theo cửa sổ hoặc chờ tới deadline rẻ nhất:**
  - chỉ còn 13–46 % số giây;
  - tỉ lệ dòng × vòng của `window5` trên `proportional` là 0,379, đúng bằng 13,3 % / 35,1 % của kế hoạch v3;
  - đổi lại, độ trễ trung bình 2,15–4,0, và mọi yêu cầu chờ dù shard của nó rẻ.
- **DT chọn được điểm ở giữa:**
  - không lần xử lý nào vượt ngân sách gộp;
  - độ trễ trung bình chỉ 0,30–1,10, thấp hơn `window5` khoảng 2–7 lần;
  - chi phí 41–72 % của xử lý ngay.
  - DT xử lý ngay các yêu cầu vào shard nhỏ, và chỉ gom các yêu cầu vào shard lớn tới khi đủ ngân sách. Với `concentrated`, nó gom 2 yêu cầu mỗi lần (10 lần train lại) và không yêu cầu nào chờ quá 2,18.
  - Ngân sách chặt làm DT gom nhiều hơn, rẻ hơn, và hạ một phần nhỏ yêu cầu xuống `shard_fixed` ở deadline (85–95 % phục vụ bằng `shard_full`).
- **Utility trên validation** gần như như nhau ở mọi chế độ (macro-F1 cuối 0,895–0,898). Một seed không đủ để so sánh utility.
- **Cấu trúc chuỗi vẫn quyết định lợi ích của gom lô:**
  - `concentrated` gom được nhiều nhất (`lazy`: 3 lần train lại);
  - `spread` và `poisson_zipf` gom được ít nhất, vì hai yêu cầu liền nhau thường rơi vào hai shard khác nhau.
  - DT gom ít ở `spread` (18 lần train lại ở mức vừa), vì phần lớn yêu cầu đã vừa ngân sách một mình.
- **Thời gian của cùng một lịch dao động khoảng 15 %:** ở `concentrated`, `dt_mid` và `dt_tight` có cùng lịch và cùng dòng × vòng, nhưng mất 80 so với 68 giây. Lý do: 5 tiến trình chạy song song trên máy.

## Chi phí của chính DT (bổ sung 2026-11-16)

Đề cương và đặc tả (mục 7.2) yêu cầu tính chi phí thu thập trạng thái, lập kế hoạch và cập nhật DT. Công việc của twin quanh một yêu cầu gồm ba phần (`src/dt/overhead.py`):
- **state:** đọc số mẫu từng client của shard bị chạm từ hệ thống đang chạy;
- **plan:** dự báo hai phương án, tính biên, quyết định;
- **sync:** sau thao tác, đưa bản ghi chi phí vào hồ sơ và lịch sử sai số.

**Ba nguồn đo:**
- **Offline:** phát lại 320 yêu cầu đã đo (16 trajectory-môi trường), mỗi yêu cầu lấy trung vị của 5 lần (`scripts/dt/measure_dt_overhead.py`).
- **Trong vòng lặp:** lần chạy online thứ hai, 20 yêu cầu thật, twin Adaptive + biên chấm lại, mỗi phần đo một lần quanh từng yêu cầu (`results/dt_extensions/midcourse_online_rescored/`).
- **Live:** đọc state từ unlearner thật sau khi train ensemble, và so sánh bật / tắt phép đo payload trên 8 yêu cầu (`results/dt_extensions/overhead/live.json`).

### Thời gian của twin mỗi yêu cầu

| Phần | Offline: trung vị (lớn nhất) | Trong vòng lặp: trung vị (lớn nhất) |
|---|---|---|
| state: đọc số mẫu của một shard | 25–93 µs tùy shard (10.043–58.073 dòng; đo live) | 100 µs (184 µs) |
| plan, Adaptive không biên | 3,5 µs (4,4 µs) | — |
| plan, Adaptive + biên chấm lại | 173 µs (293 µs) | 342 µs (427 µs) |
| sync, Adaptive | 34–39 µs (66 µs) | 49 µs (69 µs) |
| **Tổng mỗi yêu cầu** | khoảng 0,3 ms (tối đa khoảng 0,45 ms) | **0,49 ms (0,61 ms)** |

- **So với thời gian FU:**
  - trong vòng lặp, twin dùng tổng cộng 9,9 ms cho 20 yêu cầu, trong khi các vòng FU mất 92,6 giây: **0,011 %**. Yêu cầu tệ nhất: 0,031 %;
  - offline, twin đắt nhất (Adaptive + biên chấm lại) chiếm 0,0026 % tổng thời gian FU. Yêu cầu tệ nhất, so với phương án rẻ nhất của nó (`shard_fixed`): 0,06 %.
- **Phần đắt nhất của twin là tính phân vị cho biên** (140–170 µs), không phải dự báo (3,5 µs).
- **Phần plan trong vòng lặp cao gấp khoảng 2 lần offline,** vì ở đó nó chỉ chạy một lần giữa các thao tác train, không được lặp lại để lấy trung vị.
- **Snapshot đầy đủ của execution-state twin** (`client_state()`, 20 client) mất 9,0 ms (lớn nhất 10,7 ms) và nặng 2.730 byte JSON. Hồ sơ chi phí của twin nặng 573 byte. Planner không cần snapshot này cho từng yêu cầu.

### Telemetry

Số byte một thao tác phải báo về để twin cập nhật, so với payload model của chính thao tác đó (5 client mỗi shard):

| | 20 vòng (`shard_full`) | 10 vòng (`shard_fixed`) |
|---|---|---|
| Payload model, tensor thô hai chiều | 3.662.400 byte | 1.831.200 byte |
| Bản ghi chi phí twin đang dùng (từng vòng, từng client-vòng; JSON) | 10,3 KB (0,28 %) | 5,1 KB (0,28 %) |
| Như trên, nén zlib | 1,7 KB (0,047 %) | 0,93 KB (0,051 %) |
| Thống kê đủ cho twin (tổng theo client, số vòng, thời gian vòng, phần server) | 315 byte (0,009 %) | 316 byte (0,017 %) |

- **Twin không cần bản ghi từng vòng.** `summary_rates()` dựng lại đúng hồ sơ mà `rates()` tính từ bản ghi đầy đủ (có test). Một thao tác chỉ cần báo khoảng 0,3 KB.

### Chi phí của việc đo trong mô phỏng

Phép đo kích thước payload của `CostMeter` (đóng gói từng model bằng `pack_state`) là chi phí của **việc đo**, không phải của twin. So sánh bật / tắt trên 8 yêu cầu từ cùng trạng thái, thứ tự xen kẽ:
- bật đo: thêm khoảng 1,1 ms mỗi lượt client-vòng, tức **2,7 % thời gian vòng**. Phần ngoài vòng của 8 yêu cầu là 1,11 giây khi bật so với 0,23 giây khi tắt; riêng đóng gói là 0,82 giây;
- trong lần chạy online: đóng gói mất 2,1 giây trên 92,6 giây, tức 2,3 %;
- trọng số giống hệt nhau ở hai cách (có kiểm tra);
- phần này nằm **ngoài** thời gian vòng, nên không đi vào ngân sách của bất kỳ kết quả nào ở trên. Trong triển khai thật, payload vốn đã được tuần tự hóa để gửi, nên chi phí này không phát sinh thêm.

### Đọc kết quả

- **Chi phí tính toán của DT không đáng kể:** dưới 1 ms mỗi yêu cầu, tức khoảng 0,01 % thời gian FU. Tính hay không tính nó vào ngân sách không đổi kết luận nào của hướng 1–3.
- **Telemetry cũng không đáng kể** so với lưu lượng model: 0,3 KB tới 10 KB so với 1,8–3,7 MB.
- **Giới hạn của phép đo:**
  - mọi thứ chạy trong một tiến trình. Thời gian *truyền* telemetry qua mạng không được đo; đây là kích thước, không phải độ trễ;
  - thời gian là của code Python trên máy này;
  - chi phí lưu twin xuống đĩa và chi phí khởi động của mỗi lệnh CLI (nạp dữ liệu, nạp checkpoint) không nằm trong các số trên. Ở đường CLI (`shard_dt.py`), phần khởi động đó lớn hơn nhiều so với phần của twin: riêng nạp dữ liệu đã mất khoảng 2,3 giây trên máy này.

## Không được tuyên bố

| Không được tuyên bố | Lý do |
|---|---|
| α = 1 là giá trị nên dùng cho Adaptive | Được thử sau khi xem bảng đối chứng; mọi kịch bản stress ở đây là thay đổi dạng bậc thang, thuận cho học nhanh; khi không stress nó nhiễu hơn |
| Cost model theo client luôn hơn luật đơn giản | Với α = 0,3 kế thừa, "lần trước cùng shard" ngang hoặc hơn ở thay đổi cục bộ theo shard |
| Chi phí của DT trong một triển khai phân tán | Chỉ đo trong một tiến trình; thời gian truyền telemetry và chi phí dịch vụ (API, lưu trữ) chưa đo |
| Tỉ lệ vi phạm / từ chối nhầm là của một hệ thống triển khai | Một máy Windows, mô phỏng tuần tự trong một tiến trình. Stress là tiêm nhân tạo; mức bandwidth nặng hơn dự định (mục "Trôi tốc độ của máy") |
| Planner A tốt hơn planner B một cách có ý nghĩa thống kê | 4 trajectory (2 seed × 2 họ), không có kiểm định. Các trajectory cùng họ ở hai seed gần như trùng thời gian, vì thời gian không phụ thuộc trọng số |
| Biên chấm lại là phương pháp được chọn trước | Nó được thêm sau khi thấy biên thường bị kẹt |
| Kết quả hướng 1–2 cho yêu cầu bị từ chối thật sự | Giả định escalate: mọi yêu cầu đều được thực hiện, chỉ khác có trong ngân sách hay không |
| `shard_checkpoint` là một phần của protocol v3 | Chưa được quyết định; v3 vẫn chưa chốt |
| Ưu thế về utility của chế độ nào trong hướng 3 | Một seed; macro-F1 cuối chênh không quá 0,003 |
| Kết quả trên test hay calibration, hoặc trên seed v3 | Không dùng |
| Thời gian tái lập trên máy khác | Thời gian phụ thuộc máy; chỉ số đếm (vòng, dòng × vòng, byte) là tất định |

## Tái lập

```bash
python -m unittest discover -s tests -p "test_dt_extensions.py" -v
```
```bash
for s in 50042 50043; do for f in spread proportional; do python scripts/dt/run_stress_trajectories.py runs/dt_stress/s4_${s}_$f --seed $s --family $f & done; done; wait
```
```bash
python scripts/dt/evaluate_stress.py runs/dt_stress results/dt_extensions/stress
```
```bash
python scripts/dt/run_midcourse_online.py runs/dt_midcourse/s4_50042_spread_slow_mid --seed 50042 --family spread --env slow_clients --budget-level mid
```
```bash
for f in proportional spread concentrated poisson_zipf; do python scripts/dt/run_batch_dt.py runs/dt_batch/s4_50042_$f --seed 50042 --family $f --verify & done; wait
```
```bash
python scripts/dt/analyze_batch.py runs/dt_batch results/dt_extensions/batch
```
```bash
python scripts/dt/run_midcourse_online.py runs/dt_midcourse/s4_50042_spread_slow_mid_rescored --seed 50042 --family spread --env slow_clients --budget-level mid --margin-mode rescored
```
```bash
python scripts/dt/measure_dt_overhead.py results/dt_extensions/trajectories results/dt_extensions/overhead --live --seed 50042
```

Bảng hướng 1–2, kể cả các đối chứng đơn giản và độ nhạy theo α, sinh lại từ dữ liệu đã lưu, không cần train: `python scripts/dt/evaluate_stress.py results/dt_extensions/trajectories OUT` (đã kiểm tra: trùng với `results/dt_extensions/stress/`). Phần offline của phép đo chi phí DT cũng vậy (bỏ `--live`). Thời gian trên máy này, sau khi sửa throttling: mỗi tiến trình trajectory 10–13 phút với 4 tiến trình song song; hướng 3 và lần chạy online xong trong khoảng 8 phút với 5 tiến trình song song.
