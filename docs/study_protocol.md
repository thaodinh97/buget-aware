# Protocol nghiên cứu FU nhiều seed — khóa trước khi chạy

Ngày 2026-10-30. Tài liệu này mô tả bốn cải tiến sau pilot (`validation_report.md`) và protocol chạy `scripts/run_study.py` với `configs/study.json`. Pilot cũ giữ nguyên: `run_pilot.py` và `configs/pilot.json` vẫn cho kết quả y hệt (có unit test so với bản cài đặt pilot nguyên văn). **Chưa chạy trên ToN-IoT thật**: môi trường triển khai không có dataset, nên mới chỉ kiểm tra bằng unit test và smoke trên dữ liệu tổng hợp (`scripts/make_synthetic_toniot.py`). Smoke luôn fail gate và không phải evidence.

## 1. Bước quên có tác dụng thật

Trong pilot, GA dùng SGD lr=5e-4, clip 1 và một cycle, nên dịch tham số tối đa khoảng 0,0075 (quên 10%). Con số này nhỏ hơn một bước Adam của recovery, và MAE(F) sau local còn xa retrain hơn mô hình gốc. `src/fu/conpfu_inspired.py` bổ sung:

- `forget_objective`:
  - `neg_ce`: như pilot, không bị chặn dưới.
  - `neggrad_plus`: (1−w)·(−CE(F)) + w·CE(R) trong cùng một bước, mỗi bước tính 2 batch.
  - `random_label`: CE tới nhãn sai ngẫu nhiên; với bài nhị phân đây là nhãn lật. Bị chặn dưới bởi 0.
  - `uniform`: CE tới phân phối đều, đẩy p về 0,5. Bị chặn dưới bởi log 2.
- `forget_optimizer` và `retain_optimizer` (`sgd`/`adam`). Optimizer được giữ suốt quá trình, nên Adam có moment liên tục. SGD không có state, nên vẫn khớp pilot.
- `cycles` (số cycle tối đa) và `retain_steps_per_cycle`.
- `stop_rule`:
  - `nonmember`: CE có trọng số lớp, theo tỷ lệ lớp của F, của mô hình gốc trên tập validation (nonmember).
  - `oracle_retrain`: CE trung bình của các retrain trên F. Chỉ dùng để chẩn đoán cận trên; gate luôn fail với lựa chọn này.
  - Cách dừng: kiểm tra *trước* mỗi bước quên, pha quên dừng khi CE(F) ≥ ngưỡng; cycle kết thúc khi CE(F) vẫn đạt ngưỡng sau pha sửa trên R. Nếu ngưỡng đã đạt từ đầu thì pha quên không làm gì.
  - Giả định: client hoặc server có một tập held-out để tính ngưỡng; trong mô phỏng đó là tập validation.
- Kiểm tra trước recovery: gate mới có check `local_effect_before_recovery`, yêu cầu D(original) − D(mô hình trước recovery) > margin nhiễu (định nghĩa ở mục 3).

## 2. KD có tín hiệu quên

`distill(..., kd_mode=...)`:

- `retain_only`: như pilot. Về cấu trúc không thể truyền sự quên.
- `full_local`: student khớp teacher đã unlearn trên cả R và F. Trên F chỉ dùng soft target, không bao giờ dùng hard label.
- `bad_teacher`: kiểu Chundawat et al. 2023. Trên R, teacher là mô hình gốc (competent); trên F, teacher "tệ" là phân phối đều (`kd_incompetent: uniform`) hoặc một mô hình khởi tạo ngẫu nhiên (`random_init`).

Tùy chọn thêm:
- `kd_student`: `original` (như ConPFU) hoặc `unlearned` (tiếp tục từ mô hình sau alternating).
- `kd_optimizer`.
- `kd_stop`: khi CE(F) đạt ngưỡng, các dòng F bị loại khỏi loss và KD tiếp tục chỉ trên R.

F chỉ được đọc trong thao tác local của client target; recovery không bao giờ thấy F.

## 3. Nhiễu nền và so sánh cùng compute

**Nhiễu nền (`src/eval/noise_floor.py`).** Mỗi case train N retrain từ đầu trên D\F. Retrain j dùng seed `seed + 104729·j`; j=0 trùng seed của mô hình gốc như pilot.
- D(M) = MAE(F) trung bình giữa M và từng retrain.
- Với chính các retrain, D là khoảng cách leave-one-out tới các retrain còn lại. Phân phối này cho biết một "unlearner hoàn hảo" (retrain thật) đạt được bao nhiêu.
- `margin = max(minimum_margin, margin_sd_factor × sd_LOO)`.
- `identifiable`: D(original) > max LOO và D(original) − mean LOO ≥ `identifiability_sd` × sd.

**Đối chứng cùng compute (`src/fu/baselines.py`).** `retain_matched` = fine-tune chỉ trên R, dùng optimizer/lr retain của method, với đúng số minibatch gradient evaluation của pha local FU (forget + retain + KD). Sau đó chạy cùng recovery và cùng seed. Baseline `retain_only` (chỉ recovery) vẫn được giữ.

**Gate case.** Một case chỉ *hợp lệ* khi:
- identifiable, và
- LiRA trên retained control của mô hình gốc có AUC ≥ 0,55.

Một method qua case khi đạt cả hai điều kiện trên và thêm tất cả các check sau:
- `local_effect_before_recovery`
- D(retain_matched) − D(final) > margin
- D(retain_only) − D(final) > margin
- macro-F1 ≥ 0,65, và không giảm quá 0,03 so với max(original, retrain trung bình)
- FPR tăng không quá 0,02 so với original
- LiRA AUC(F) gần AUC trung bình của các retrain hơn so với original
- không dùng oracle stop

**Gate nghiên cứu (theo method).**
- ≥ `minimum_valid_cases` case hợp lệ.
- Tỷ lệ qua ≥ `minimum_pass_rate`.
- Cận dưới CI 95% (phân phối t) của [D(retain_matched) − D(final) − margin] > 0.

Các case cùng seed dùng chung mô hình gốc nên không độc lập; CI chỉ là thống kê sàng lọc, không phải robustness claim. `dt_allowed` luôn là false.

## 4. Kịch bản nhận diện được và MIA mới

**Kịch bản (`src/data/scenarios.py`):**
- `random`: request phân tầng như pilot (10%/50%).
- `family`: ownership tổng hợp, trong đó *toàn bộ* dòng train của một họ attack thuộc client target; F = cả họ đó. Retrain vì vậy chưa từng thấy họ attack này.
  - Họ được khóa trước theo metadata: `ransomware` (phổ biến) và `mitm` (hiếm), không chọn theo kết quả.
- `atypical`: theo từng lớp, lấy các dòng có loss cao nhất dưới mô hình gốc và không có bản trùng encoded vector trong train (5%).
  - Chỉ dùng mô hình gốc và metadata, không dùng test hay kết quả FU.
  - Lưu ý: chọn F theo confidence thấp làm lệch MIA trên F (AUC có thể < 0,5). Ở kịch bản này chỉ nên đọc MIA ở dạng so sánh với retrain.

**LiRA offline (`src/eval/lira.py`, Carlini et al. 2022):**
- K shadow model được train bằng FedAvg (Dirichlet trên một nửa pool ngẫu nhiên). Pool = train trừ mọi bản ghi F, mọi retained control và mọi bản trùng encoded vector của chúng, nên mọi shadow đều OUT với mọi query.
- Partition của shadow là phía attacker: nếu Dirichlet tạo ra client suy biến, shadow thử lại một cách tất định với seed `s + 7919·attempt` (tối đa 10 lần, số lần thử được ghi trong `metrics.json`). Partition thí nghiệm của target vẫn giữ quy tắc không tự động repartition.
- Không loại cả connected group. Trên ToN-IoT thật, 14.705 dòng ransomware nằm trong 2 component có tổng 55.145 dòng, và với mức quên 50% thì các group chạm F chứa khoảng 99k/114k dòng train. Loại cả group sẽ chỉ còn 9–20k dòng cho shadow, khiến shadow yếu hơn hẳn target. Các bản gần trùng còn lại trong shadow làm attack thận trọng hơn (AUC thấp hơn), không làm thổi phồng membership.
- Mỗi bản ghi có thống kê φ = logit(p_y) dưới các shadow OUT (μ và một phương sai chung). Score = (φ_target − μ)/σ.
- Cách này hiệu chỉnh độ khó của từng bản ghi, nên tách được membership khỏi distribution shift, thứ làm attack loss cố định của pilot ra AUC 0,47 trên control.
- Ngưỡng operating point: phân vị 99 theo lớp trên calibration nonmembers của mô hình gốc.
- Nonmember đánh giá: test, match lớp, tối đa 2.000 mỗi lớp.
- Attack loss cũ vẫn được báo cáo để đối chiếu.
- Nếu control vẫn < 0,55 thì case là inconclusive, không đổi attack sau khi đã nhìn kết quả.

## Methods khóa trước

Danh sách nằm trong `configs/study.json`:
- `legacy_no_kd`, `legacy_retain_kd`: tái lập pilot.
- `neggrad_plus_adam`
- `uniform_adam_bad_teacher`
- `random_label_adam_full_kd`
- `uniform_adam_bad_teacher_oracle`: chỉ để chẩn đoán.

Hyperparameter là preset chưa tune. Nếu có tune thì chỉ trên validation và phải dùng run mới với config mới.

Mặc định: 3 seed (42/43/44); 2 target lớn nhất; 4 retrain; 8 shadow; recovery 2 round. Ước lượng thô từ timing pilot (chưa đo) là khoảng 1 giờ CPU 2 thread cho khoảng 30 case.

## Chạy

```powershell
python -m unittest discover -s tests -v
python scripts/run_study.py --data data/processed/connected_v3_seed42 --output runs/study_smoke --smoke
python scripts/run_study.py --data data/processed/connected_v3_seed42 --output runs/study_v1
```

Kết quả:
- `gate.json`: quyết định theo case và tổng hợp theo method.
- `cases.csv`: D(F), nhiễu, F1/FPR, LiRA theo từng model/stage.
- `cases/<case>/metrics.json`: đầy đủ metric, noise floor, ngưỡng stop, LiRA.
- Ngoài ra: request ID, evaluation ID, checkpoint, log local, `coverage.csv`, `timings.json`, `manifest.json`, `artifact_hashes.json`.
- Request không khả thi được ghi là `invalid_request`; mọi lỗi khác làm run fail.

Kiểm tra pipeline không cần dataset thật:

```powershell
python scripts/make_synthetic_toniot.py --output runs/synthetic/raw.csv
python src/data/prepare.py --input runs/synthetic/raw.csv --config configs/data_connected.json --output runs/synthetic/data
python scripts/run_study.py --data runs/synthetic/data --output runs/synthetic/study_smoke --smoke
```

## Giới hạn

- Đây vẫn là approximate unlearning: không có certified deletion, không có DP.
- Transformer cố định giữ thống kê của F.
- Ownership trong kịch bản `family` là tổng hợp.
- Ngưỡng `nonmember` là cận dưới thận trọng cho kịch bản family: một họ chưa từng thấy có thể có CE cao hơn nhiều so với nonmember trung bình. Chênh lệch với bản oracle sẽ đo khoảng hở này.
- Compute matching tính theo số minibatch gradient evaluation, không theo FLOP hay wall time.
- Retained control ở thang lớp/group, không match theo họ attack.

## v3 — khóa trước khi chạy (2026-10-30)

Các thay đổi sau được đề xuất sau khi đã xem `study_v2` (`docs/study_report_v2.md`), nên v3 là một run mới với config mới (`configs/study_v3.json`). Kết luận của v2 giữ nguyên.

### Chẩn đoán LiRA (`scripts/diagnose_lira.py`)

Lưới 2×2 trên checkpoint của `study_v2`:
- Shadow train trên nửa pool hoặc toàn pool (`subset_fraction` 0,5 / 1,0).
- Loại bản trùng chính xác, hoặc loại thêm cả bản gần trùng (`exclude` exact / near).

Định nghĩa bản gần trùng: một dòng train nằm trong bán kính r quanh một query member (F_eval hoặc control) trong không gian encoded. r là trung vị của khoảng cách từ mỗi dòng test tới dòng train gần nhất. Test là group-disjoint, nên trong bán kính này một dòng train gần query hơn mức một nonmember điển hình gần dữ liệu train.

Chẩn đoán chỉ train thêm shadow. Biến thể (0,5; exact) phải tái lập đúng v2.

**Quy tắc chọn (khóa trước, chỉ dựa trên tính hợp lệ của attack, không dùng kết quả FU):**
- Trên mọi case random, chọn biến thể có tỷ lệ case đạt LiRA AUC control (mô hình gốc) ≥ 0,55 cao nhất.
- Nếu bằng nhau, chọn biến thể có |trung vị score của test nonmember| gần 0 nhất, tức shadow mô phỏng đúng độ tự tin của target trên dữ liệu chưa thấy.
- Nếu tỷ lệ tốt nhất < 0,5, giữ biến thể của v2. Khi đó kịch bản random tiếp tục inconclusive.

Biến thể được chọn sẽ được ghi vào `configs/study_v3.json` trước khi chạy v3.

### Stop rule mới

**Ngưỡng proxy mà client tự tính được (`stop_rule: proxy_calibrated`).** Client target train một mô hình proxy từ đầu chỉ trên R của chính nó: cùng kiến trúc, Adam lr 1e-3, 15 epoch (bằng 15 round × 1 local epoch của FL). Proxy không bao giờ thấy F. Ngưỡng được tính là:

  ngưỡng = CE_gốc(validation, có trọng số lớp như F) × CE_proxy(F) / CE_proxy(validation, cùng trọng số)

Ý nghĩa: proxy cho biết F "lạ" đến đâu so với dữ liệu chưa thấy (validation, mà cả mô hình gốc lẫn proxy đều chưa thấy). Tỷ lệ đó được chuyển sang thang loss của mô hình gốc.
- Với xóa mẫu ngẫu nhiên, tỷ lệ gần 1, nên ngưỡng gần ngưỡng `nonmember`.
- Với một họ chưa từng thấy (mitm), ngưỡng cao hơn.
- `proxy_raw` = CE_proxy(F) chưa hiệu chỉnh, chỉ được log để đối chiếu.
- Chi phí proxy (số bước, thời gian) được ghi riêng và không cộng vào `retain_matched`, vì proxy không cập nhật mô hình được triển khai.

**Kiểm tra ngưỡng sau recovery ngắn (`stop_check: after_recovery`, `src/fu/recovery_check.py`).** Mỗi bước ngoài gồm:
1. Một cycle alternating, vẫn có chốt chặn local ở ngưỡng.
2. `check_rounds`=1 round FedAvg trên R.
3. Kiểm tra CE(F) của mô hình *đã recovery* so với ngưỡng.

Vòng ngoài chạy tối đa `max_outer`=5 bước. Mô hình cuối là mô hình sau vòng này, không chạy thêm recovery. Anchor L2 luôn là mô hình gốc. `retain_matched` nhận đúng số bước local và đúng số round FedAvg đã dùng, với cùng lịch seed. Chế độ này không kết hợp với KD.

Giới hạn đã biết: CE(F) không tăng đơn điệu theo bước quên. Ví dụ objective `uniform` kéo p về 0,5, nên CE của các điểm đang bị phân loại sai *giảm*, và CE trung bình có thể giảm theo.

### Ứng viên trước recovery

Với các method kiểm tra ngưỡng local, gate đánh giá hai ứng viên:
- `final`: sau recovery, so với `retain_matched`.
- `pre_recovery`: sau local/KD, chưa recovery, so với `retain_matched_local` (cùng số bước local, không có recovery). Ứng viên này phải qua cùng gate utility: macro-F1 ≥ 0,65; tụt ≤ 0,03 so với max(gốc, retrain); FPR tăng ≤ 0,02.

Gate tổng hợp tính riêng theo cặp `method@candidate`. Với `after_recovery`, chỉ có ứng viên `final`; `local_effect_before_recovery` dùng mô hình local đầu tiên, trước mọi recovery.

### Methods v3 (`configs/study_v3.json`)

- `legacy_retain_kd` và `neggrad_plus_adam`: giữ như v2 để đối chiếu.
- `neggrad_plus_proxy`: ngưỡng proxy, kiểm tra local.
- `neggrad_plus_proxy_recovery` và `random_label_proxy_recovery`: ngưỡng proxy, kiểm tra sau recovery.
- `uniform_bad_teacher_proxy`: KD bad-teacher với ngưỡng proxy.
- `neggrad_plus_oracle_recovery`: chỉ để chẩn đoán.

Kịch bản, seed, target, gate, số retrain và số shadow giữ nguyên như v2. Hyperparameter là preset, không tune.

### Kết quả chẩn đoán LiRA (`runs/lira_diag_v2`, tóm tắt ở `results/lira_diag_v2/`)

- Chạy 12 case random và 2 case family, mất 10,6 phút.
- Tổ hợp (0,5; exact) tái lập đúng AUC của v2, sai khác bằng 0.
- **Không tổ hợp nào đạt control ≥ 0,55 ở bất kỳ case random nào.** Tỷ lệ đạt bằng 0 ở cả 4 tổ hợp; AUC control trung bình 0,469–0,477.
- Theo quy tắc, v3 giữ tổ hợp của v2: `subset_fraction` 0,5, `exclude` exact. Kịch bản random vẫn inconclusive về MIA.

**Phát hiện về dữ liệu.** Bán kính "gần trùng" tính được là 1,46e-7. So với dòng train gần nhất:
- 33,9% số dòng test có khoảng cách float bằng 0; 91,8% nằm trong khoảng 1e-3; 96,7% nằm trong khoảng 1e-2.
- Chênh lệch tối đa giữa các cặp này khoảng 1e-8, chỉ ở feature `duration` đã chuẩn hóa.
- Nhãn của hàng xóm gần nhất trùng với nhãn test ở 98,4% số dòng.

Audit overlap encoded chỉ so hash chính xác nên không bắt được các bản gần trùng này. Hệ quả:
1. Nonmember trong test gần như luôn có bản gần trùng trong train, nên một MIA dựa trên độ tự tin không thể tách member khỏi nonmember ở thang mẫu.
2. Loại bản gần trùng khỏi shadow xóa mất 64–92k dòng train, vì các cụm gần trùng rất lớn.
3. Holdout hiện tại không phải dữ liệu "mới" trong không gian feature. Mọi kết luận về utility và MIA đều phải đọc với điều kiện này.

Sửa preprocessing (gộp bản gần trùng vào cùng group trước khi chia split, hoặc lượng tử hóa `duration`) là một thay đổi protocol cần quyết định riêng, chưa áp dụng cho v3.

## v5 — phương án C + log1p cho feature đếm (khóa trước khi chạy, 2026-10-30)

Quyết định: chọn phương án C trong `docs/preprocessing_near_duplicates.md`. Giữ cách group/split của v3, và **thêm log1p, fit chỉ trên train, cho 7 feature đếm** trước khi chuẩn hóa. Các feature đó là `src_bytes`, `dst_bytes`, `missed_bytes`, `src_pkts`, `src_ip_bytes`, `dst_pkts`, `dst_ip_bytes`.

- Config `configs/data_log1p.json`, data `data/processed/connected_log1p_v5_seed42`.
- `duration` không phải feature đếm nên giữ tuyến tính. Đổi `duration` sẽ là một thay đổi riêng.
- Gán split **trùng 100%** với v3 (190.474/190.474 row ID cùng split), với 2 lần refinement và component lớn nhất 38.869 như cũ. Vì vậy so sánh v3/v5 chỉ khác ở biến đổi feature.
- Độ lệch chuẩn sau biến đổi: `src_bytes` từ 1,7e7 xuống 2,29; `dst_ip_bytes` từ 2,5e4 xuống 2,59.

**Gate (`configs/study_v5.json`).** Giống v3, trừ `mia_required: false`.
- Ở mức mẫu, MIA không đánh giá được trên dataset này: 92% test nằm trong bán kính 1e-3 của train, và mọi biến thể LiRA đều có control < 0,55.
- Vì vậy LiRA vẫn được tính và báo cáo, nhưng `mia_positive_control` và `mia_closer_to_retrain` không còn quyết định tính hợp lệ hay việc qua case.
- Case hợp lệ khi và chỉ khi identifiable theo nhiễu nền retrain. Mọi check khoảng cách, utility và FPR giữ nguyên.
- Đây là nới lỏng có chủ đích. Kết quả v5 vì vậy không so trực tiếp được số case hợp lệ với v2/v3.

Methods, kịch bản, seed, target, số retrain/shadow và hyperparameter giữ nguyên như v3. Run: pilot `runs/pilot_v5` và study `runs/study_v5`.
