# Kết quả controlled FU diagnostics — 2026-10-19

Run: `runs/controlled_diagnostics_v2_seed42`. Source: `runs/pilot_v3_seed42`. Không đổi preprocessing/split/client assignment/forget requests. Không tune bằng test. DT chưa triển khai.

## Kết quả kiểm thử và checkpoint

15 tests đạt, gồm observation không đổi updates, full CE gradients, KD teacher freeze, null/inverted MIA và family matching. Runner kiểm tra final tensor hash của retain-only/FU-no-KD/FU-retain-KD bằng chính xác checkpoint cũ cho cả hai fractions. Mọi checkpoint diagnostic/final được reload và đối chiếu tensor hash. Source run được verify trước khi chạy.

18 method/request entries hoàn tất. Original/no-op/retrain reuse cached checkpoints có hash validation; runtime null và steps=0 trong CSV nghĩa không chạy lại, không phải retraining miễn phí. Historical reference costs lưu riêng.

## Vì sao local FU thay đổi ít

### Forget 10%

F=7,349, target R=66,155; 77.26% F có feature-equivalent record còn trong global R.

| Stage | Steps | CE(F) | CE(R) | Parameter L2 từ original | Prediction MAE(F) từ original |
|---|---:|---:|---:|---:|---:|
| cycle0_before_ga | 0 | 0.011890870 | 0.012515170 | 0.000000000 | 0.000000000 |
| cycle0_after_ga | 15 | 0.011902958 | 0.012526924 | 0.000297107 | 0.000008909 |
| cycle0_before_gd | 0 | 0.011902958 | 0.012526924 | 0.000297107 | 0.000008909 |
| cycle0_after_gd | 130 | 0.011800365 | 0.012425415 | 0.002272793 | 0.000068651 |

GA/GD steps: 15/130. Minibatches vượt clip bound 1: 0. Max pre-clip norm: 0.185872.

### Forget 50%

F=36,751, target R=36,753; 77.21% F có feature-equivalent record còn trong global R.

| Stage | Steps | CE(F) | CE(R) | Parameter L2 từ original | Prediction MAE(F) từ original |
|---|---:|---:|---:|---:|---:|
| cycle0_before_ga | 0 | 0.012335090 | 0.012570408 | 0.000000000 | 0.000000000 |
| cycle0_after_ga | 72 | 0.012402584 | 0.012627839 | 0.001559296 | 0.000047770 |
| cycle0_before_gd | 0 | 0.012402584 | 0.012627839 | 0.001559296 | 0.000047770 |
| cycle0_after_gd | 72 | 0.012345042 | 0.012578212 | 0.000277561 | 0.000007721 |

GA/GD steps: 72/72. Minibatches vượt clip bound 1: 0. Max pre-clip norm: 0.157920.

GA hoạt động, không có lỗi dấu hoặc checkpoint copy che update. Loss nhỏ, SGD LR nhỏ và retain-GD sau GA giải thích biến đổi nhỏ; KD reset student về original, teacher gần original. Không suy ra forgetting thành công chỉ từ CE tăng.

## So sánh controlled

| Fraction | Method | Macro-F1 | AP | FPR | MAE(F) tới retrain | Legacy MIA F/R | Family-matched MIA F/R | Steps mới |
|---|---|---:|---:|---:|---:|---|---|---:|
| 10% | original | 0.889700 | 0.989463 | 0.272284 | 0.001711640 | 0.6678/0.4720 | 0.5318/0.4461 | 0 |
| 10% | no_op | 0.889700 | 0.989463 | 0.272284 | 0.001711640 | 0.6678/0.4720 | 0.5318/0.4461 | 0 |
| 10% | retrain | 0.889038 | 0.989056 | 0.272702 | 0.000000000 | 0.6686/0.4717 | 0.5324/0.4440 | 0 |
| 10% | conpfu_local_only | 0.890101 | 0.989461 | 0.272284 | 0.001795246 | 0.6679/0.4721 | 0.5318/0.4462 | 289 |
| 10% | retain_only | 0.889299 | 0.989432 | 0.272284 | 0.000710837 | 0.6692/0.4740 | 0.5320/0.4453 | 422 |
| 10% | retain_gd | 0.889299 | 0.989437 | 0.272284 | 0.000710778 | 0.6692/0.4738 | 0.5320/0.4452 | 552 |
| 10% | fu_no_kd | 0.889299 | 0.989434 | 0.272284 | 0.000699525 | 0.6692/0.4739 | 0.5320/0.4454 | 567 |
| 10% | fu_retain_kd | 0.889299 | 0.989433 | 0.272284 | 0.000709338 | 0.6692/0.4740 | 0.5320/0.4453 | 697 |
| 10% | conpfu_inspired_baseline | 0.889299 | 0.989434 | 0.272284 | 0.000712215 | 0.6692/0.4740 | 0.5320/0.4454 | 711 |
| 50% | original | 0.889700 | 0.989463 | 0.272284 | 0.007400128 | 0.6530/0.4680 | 0.5374/0.4588 | 0 |
| 50% | no_op | 0.889700 | 0.989463 | 0.272284 | 0.007400128 | 0.6530/0.4680 | 0.5374/0.4588 | 0 |
| 50% | retrain | 0.886574 | 0.989475 | 0.268106 | 0.000000000 | 0.6640/0.4662 | 0.5540/0.4558 | 0 |
| 50% | conpfu_local_only | 0.890101 | 0.989461 | 0.272284 | 0.007479068 | 0.6530/0.4681 | 0.5373/0.4591 | 288 |
| 50% | retain_only | 0.892096 | 0.989209 | 0.259749 | 0.003285972 | 0.6600/0.4643 | 0.5389/0.4576 | 306 |
| 50% | retain_gd | 0.888851 | 0.989208 | 0.269359 | 0.003323098 | 0.6600/0.4643 | 0.5389/0.4575 | 378 |
| 50% | fu_no_kd | 0.892096 | 0.989206 | 0.259749 | 0.003297513 | 0.6600/0.4643 | 0.5389/0.4575 | 450 |
| 50% | fu_retain_kd | 0.888851 | 0.989207 | 0.269359 | 0.003323073 | 0.6600/0.4642 | 0.5389/0.4575 | 522 |
| 50% | conpfu_inspired_baseline | 0.892140 | 0.989204 | 0.259749 | 0.003309712 | 0.6600/0.4642 | 0.5389/0.4575 | 594 |

`retain_only` là recovery-only cũ. `retain_gd` thêm cùng retain-GD schedule rồi cùng recovery, giúp phân biệt ảnh hưởng GA. Không gọi compute-matched: số steps khác và đã ghi. FU-no-KD/FU-retain-KD giữ nguyên algorithm cũ. `conpfu_inspired_baseline` dùng teacher sau local, student original, KD trên F∪R rồi cùng recovery; `conpfu_local_only` không recovery. Cả hai chỉ inspired: MLP/5 clients/global-start/no incentives; không faithful ConPFU reproduction. F chỉ dùng trong local operation/KD variant được khai báo; recovery luôn D\F.

## MIA

Score log p(y|x) = negative CE, higher→member; không tự đảo khi AUC<0,5. Threshold calibration classwise percentile 99 của Original, strict >, cố định mọi model. Legacy dùng đúng evaluation IDs cũ. Family-matched là sensitivity cohort khác, match label+type và báo coverage mỗi stratum; không thay legacy âm thầm. Các cohort được cố định trước đánh giá model và lưu IDs. Không train attack classifier/shadow model (0 shadow models); đây là fixed-loss attack. Synthetic tests validate implementation, không chứng minh attack mạnh trong dữ liệu thật.

Retained controls loại group chạm F nên distribution khác; family matching không kiểm soát hết connection/difficulty shifts. Thiếu support ở một số họ attack làm giảm coverage. Không xem MIA AUC(F) giảm là xóa thành công. So với retrain và retained controls cùng protocol; khi controls không phân biệt được members, kết luận inconclusive. ROC TPR@1% là descriptive ROC point, khác realized FPR tại threshold calibration.

## Runtime và communication

CSV ghi optimizer_seconds = stage wall time trừ observer overhead, vẫn gồm logging/Python/optimizer initialization; không phải profiler kernel-only. diagnostic_seconds ghi riêng. Local branch dùng chung được tính vào từng phương pháp khi báo chi phí giả định chạy riêng; không cộng CSV thành tổng thời gian runner. First-use warmup và thứ tự cố định gây confound; không tuyên bố speedup. Historical retraining costs có nguồn và warning, không dùng như matched current timing.

Bytes là tensor payload theo dtype/numel: một download/upload local FU và các round recovery; không phải measured network bytes, không gồm serialization/protocol overhead, không phải DT cost. Checkpoints trung gian là offline diagnostic artifacts, không tính như network transfer.

## Giới hạn và hướng tiếp theo

Một seed/target, không có CI/statistical superiority claim. Full-local KD có nhiều steps/data hơn retain-KD. Preprocessing transformer giữ thống kê original train. Original test đã được xem trong pilot trước: đây là diagnostic trên benchmark cố định, không phải xác nhận độc lập mới. Không thay threshold hoặc tune LR từ test.

Tiếp theo: protocol shadow-model hoặc learned attack có auxiliary groups tách biệt; thiết kế validation-only GA/GD step-ratio ablation và compute-matched controls; giữ test khỏi tuning. Không tự mở DT, không tuyên bố unlearning thành công.

Lần đầu `runs/controlled_diagnostics_seed42` bị dừng sau fraction 10% do object-array feature-membership audit quá chậm. Giữ nguyên partial artifacts và execution_status.json. Run v2 dùng set membership tương đương để tăng tốc, không thay cohorts/model updates.
