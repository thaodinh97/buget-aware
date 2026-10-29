# Kế hoạch pilot controlled FU — 2026-10-19

Viết trước implementation/run. Không regenerate dataset; dùng data/processed/connected_v3_seed42 và source runs/pilot_v3_seed42, giữ transformer và 5-client partition. Không DT planner.

## Matrix và selection

Seed42, targets0 và4 (normal/attack khác rõ, chọn bằng counts đã biết), fractions10%/50%, random_stratified và low_overlap_groups: **8 requests**. Random dùng seeded class permutations/floor như code cũ; target4 random phải trùng requests cũ. Group request giữ nguyên toàn bộ phần thuộc target của mỗi connected component, ưu tiên component ít predictor-equivalent records ở các client khác, tie bằng seed. Greedy không vượt số dòng yêu cầu; báo actual fraction và group/label counts, không sửa request dựa trên model metrics. Không hứa low-overlap tuyệt đối khi ownership phân tán; invalid F/R hoặc không đủ lớp ghi failure, không resample.

6 arms: no_update(0), ga_only(24), gd_only(24), alt_3_1(18GA/6GD), alt_1_1(12/12), alt_1_3(6/18). Alternating có hai cycles. SGD lr/clipping/regularization giữ config cũ. Full batch512 mỗi step với deterministic cyclic permutations; không partial batch. GD24 là retained-training control khớp số optimizer steps, samples và architecture với GA/GD24. Không tuyên bố exact FLOP/time equality vì objective khác; đo CPU/runtime riêng.

Đánh giá validation trước/sau cùng 2 rounds FedAvg recovery all-clients trên D\\F. Chọn trong 3 alternating ratios theo validation prediction MAE tới retrain nhỏ nhất, với macro-F1 không giảm >0,03 và FPR không tăng >0,03 so với original validation. Tie theo tên arm. Ghi selection.json trước mọi test prediction. Nếu không khả thi, ghi NO_FEASIBLE_VALIDATION_ARM và dùng alt_1_1 chỉ làm fixed reference, không gọi đó là validated winner. Diagnostics F không dùng selection.

Thêm 3 arms: selected+retain_KD, selected+full_local_KD (ConPFU-inspired), gd24+retain_KD matched control. KD16 full batches512, frozen teacher sau local, student original. Matched KD control có cùng số student steps và teacher forward passes; không exact wall-time match. Full-local KD dùng F∪R chỉ trong local operation. Cả3 có recovery2rounds. Original/no-op/fresh-retrain là references. selected_without_KD alias selected alternating, retain-only recovery alias no_update/post, compute-matched retained alias gd_only/post; không đếm alias như experiment độc lập.

Tổng:9 local/recovery branches/request ×8=72 recovery runs (144 FL rounds);8 fresh retrains ×15rounds;3 shadow FL fits ×15rounds. Original cached, no-op copy.21 evaluated entries/request (3 references+9branches×2stages), dự kiến168 rows nếu mọi request hợp lệ. Không chạy full seed grid. Runtime dự kiến vài phút CPU, có thể tăng do diagnostics/I/O; không hứa time budget chưa đo.

## Learned MIA độc lập

Chỉ dùng frozen calibration split để tạo8 auxiliary group-disjoint buckets bằng seeded permutation; không thay split chính. Buckets0/1 là shadow1 member/nonmember,2/3 shadow2;4/5 held-out validation shadow;6 target attack-threshold calibration;7 reserve. Mỗi shadow dùng MLP/FedAvg config gốc nhưng auxiliary sample count nhỏ hơn và 5 balanced synthetic clients. Không dùng original train/F/test để fit attack hay shadow models. Lưu mọi row/group IDs, model hashes và source preprocessing limitation.

Attack logistic regression L2(C=1) với standardized [log true-label probability, entropy, confidence margin, label], fit member=1/nonmember=0 của2training shadows. Dùng label+type matching trong mỗi shadow pair, báo support và counts; không tune hyperparameters. Held-out shadow đánh giá generalization, không fit classifier. Original được score trên bucket6 để cố định threshold nominal FPR1% (strict >, ties conservative). Freeze attack và threshold cho Original/FU/Retain/Retrain.

Target evaluation: label+type matched F vs test nonmembers; retained controls gồm all-R và group-disjoint-R, mỗi cohort fixed một lần/request rồi shared mọi model. Báo AUC, TPR tại frozen nominal1%-FPR threshold, realized FPR, counts/coverage; không gọi realizedFPR đúng1% khi shift. Attack heldout AUC hoặc Original retained-group control AUC<0,55 => inconclusive heuristic, không chứng minh privacy ngay cả khi vượt. Không đảo attack score dựa trên test. Calibration auxiliaries không trùng group/encoded vectors với target train/validation/test; group disjoint giữa auxiliary buckets được assert. Nghiên cứu pilot chưa có group-bootstrap CI; mọi kết luận thận trọng.

## Profiling và artifacts

Reusable profiler dùng psutil: wall time, process user/system CPU delta, CPU-core-equivalent utilization, RSS baseline/max sampled20ms, timestamp, optimizer steps, local sample exposures, recovery rounds, tensor upload/download payload. RSS là sampled process memory, không exact allocation peak/per-client isolation. Sequential CPU simulation; wire bytes không đo. Diagnostics được profile riêng/exclude khỏi stage timings bằng pause/resume, không thay gradients. Checkpoint before/after GA/GD và KD giữ full F/R CE/gradient, displacement và prediction change.

Đầu ra: immutable manifest/config, auxiliary roles/shadow checkpoints/attack coefficients/threshold, requests/overlap/labels, validation selection trace, pre/post checkpoints, diagnostic logs, cost profiles, metrics CSV/JSON, MIA coverage, failures/coverage, hash inventory và báo cáo tiếng Việt. Source artifacts giữ nguyên. Test chỉ báo cáo sau selection; test cũ đã được xem nên không gọi independent external confirmation.

## Thay đổi code tối thiểu

Thêm modules: step-budget schedule tận dụng forget_step/kd_loss; group request builder tận dụng frozen metadata; learned shadow attack tận dụng FedAvg; psutil profiler; runner và tests. Giữ core model/aggregation/preprocessing cũ. Extend docs và dependency psutil. Không implement planner, DP, incentives hay certified-unlearning claim.
