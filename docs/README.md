# Tài liệu và lịch sử thực nghiệm

## Doc theo bước pipeline

| Bước | Protocol / kết quả |
|---|---|
| 0. Tiền xử lý | [preprocessing_validation.md](preprocessing_validation.md) |
| 2. Request (controlled phase) | [controlled_phase_plan.md](controlled_phase_plan.md) |
| 3. Seed floor | [seed_floor_results.md](seed_floor_results.md) |
| 4. Canary client 3 | [identifiable_fu_plan.md](identifiable_fu_plan.md), [identifiable_fu_results.md](identifiable_fu_results.md) |
| 5. Forget signal | [forget_signal_plan.md](forget_signal_plan.md), [forget_signal_results.md](forget_signal_results.md) |
| 6. Deployable forget (phương pháp ứng viên, chưa xác nhận) | [deployable_forget_plan.md](deployable_forget_plan.md), [deployable_forget_results.md](deployable_forget_results.md) |
| 7. MIA canary | [canary_mia.md](canary_mia.md) |
| 8. Utility chi tiết và xác nhận | [confirmation_plan.md](confirmation_plan.md); kết quả chế độ `existing`: [confirmation_existing_results.md](confirmation_existing_results.md); chế độ `confirm`: [confirmation_p43_results.md](confirmation_p43_results.md); chế độ `confirm2`: [confirmation_p44_results.md](confirmation_p44_results.md) |
| Baseline reset (forgetting position và test phân biệt) | [reset_distinguisher_plan.md](reset_distinguisher_plan.md), [reset_baseline_results.md](reset_baseline_results.md) |
| 9. LiRA canary và placebo test | [canary_lira_plan.md](canary_lira_plan.md), [canary_lira_results.md](canary_lira_results.md) |
| 10. Phân biệt unlearn với retrain | [indistinguishability_plan.md](indistinguishability_plan.md), [indistinguishability_results.md](indistinguishability_results.md) |
| Shadow MIA và LiRA | [membership_validation_results.md](membership_validation_results.md), [lira_plan.md](lira_plan.md), [lira_results.md](lira_results.md) |
| **Đóng băng FU trước phase DT** | [fu_freeze.md](fu_freeze.md) |
| Khám phá sau đóng băng: unlearn chính xác theo shard (kiểu SISA / DynamicFU) | [shard_exploration_results.md](shard_exploration_results.md) |
| Cấu hình train chuẩn cho hướng shard và xác nhận hai run 20 client (**không xác nhận**) | [shard_standard_config_plan.md](shard_standard_config_plan.md), [shard_standard_config_results.md](shard_standard_config_results.md) |
| Hướng shard, protocol v2: gate noninferiority, ≥ 4 client mỗi shard, sửa `utility_stop` (**không xác nhận**: G2, G5a, G5b fail; G3, G6 pass) | [shard_v2_plan.md](shard_v2_plan.md), [shard_v2_results.md](shard_v2_results.md) |
| Đặc tả ban đầu | [AGENT_IMPLEMENTATION_GUIDE_FU_BUDGET_AWARE_DT.md](AGENT_IMPLEMENTATION_GUIDE_FU_BUDGET_AWARE_DT.md) |

## Vai trò của các tập dữ liệu

Một số tập dữ liệu giữ nhiều vai trò cùng lúc. Bảng dưới liệt kê đầy đủ, để người đọc tự đánh giá mức độ độc lập của từng kết quả.

| Tập | Vai trò | Ở đâu |
|---|---|---|
| **train** (114.284 dòng, chia cho 5 client) | Train FedAvg; F của mọi request; nguồn của canary; nonmember của MIA canary và LiRA canary: các dòng attack khác của cùng client, được đóng dấu trigger nhưng không train | mọi bước; `run_canary_mia.py`, `run_canary_lira.py` |
| **validation** | **Chọn siêu tham số:** tỉ lệ GA/GD trong controlled phase | `run_controlled_phase.py` (`selection.json`) |
| | **Look-alike nonmember** cho mức dừng `nonmember`: attack có trigger cho canary, dòng cùng tỉ lệ lớp cho request tự nhiên | `run_forget_signal.py` |
| | **Dữ liệu phụ trợ:** pool của kNN (`knn_aux`), dữ liệu train reference (`ref_aux`) và control cùng pool | `run_deployable_forget.py`, `run_confirmation.py` |
| | Nonmember khi audit model | `run_membership_validation.py` |
| | **Fit tiền xử lý** (scaler, median, vocabulary) cho bản dữ liệu `connected_v3_auxfit_seed42`, và tín hiệu cho các chính sách shard (`utility_stop`, DynamicFU) | `configs/data_connected_auxfit.json`, `scripts/shard/` |
| **calibration** | Shadow model và ngưỡng của shadow MIA | `run_membership_validation.py`, `run_pilot.py` |
| **test** | Utility (macro-F1, FPR); **chỉ số chính của canary** (P(normal) trung bình trên attack test được đóng dấu); pool nonmember của LiRA trên request tự nhiên | mọi bước; `run_lira.py` |

**Hệ quả:**
- **Validation không phải dữ liệu "sạch".** Nó vừa là dữ liệu phụ trợ được phép dùng khi unlearn, vừa từng là nonmember và từng dùng để chọn siêu tham số.
- **Test cũng không hoàn toàn chưa được nhìn.** Các thay đổi phương pháp sau smoke của `deployable_forget` (thống kê `*_prob`, `joint_aux_prob`) được quyết định sau khi xem chỉ số chính của canary, vốn tính trên test. Chỉ chế độ `confirm` của bước 8 dùng seed và partition chưa được nhìn.

## Lịch sử: từ ConPFU-inspired tới phương pháp hiện tại

Mỗi mục ghi điều đã học được. Các mục đánh dấu † là thực nghiệm đã gỡ khỏi repo để gọn; code và doc của chúng vẫn còn trong lịch sử git, ở commit `1eea968` trên nhánh `feature/fu-experiment`.

1. **Pilot ConPFU-inspired † (2026-10-18), một seed.**
   - Pipeline gồm: GA/GD xen kẽ bằng SGD (lr 5e-4, clip 1) → KD chỉ trên R → 2 vòng recovery.
   - Ở mức quên 50%, MAE(F) tới retrain giảm từ 0,0074 (original) xuống 0,0033 (FU), nhưng retain-only cũng đạt 0,0033.
   - Gate thất bại: MIA trên nhóm control giữ lại chỉ đạt AUC 0,47, dưới ngưỡng 0,55. FPR của detector là 27%.
2. **Controlled diagnostics †.**
   - Bước GA gần như không làm model thay đổi: dự đoán trên F chỉ lệch khoảng 1e-5, chuẩn L2 của tham số khoảng 1e-3.
   - Nguyên nhân: loss trên F vốn đã nhỏ, lr của SGD nhỏ, bước GD sau đó gỡ lại phần đã quên, và KD đặt lại student về model gốc.
   - 77% dòng của F có bản trùng feature trong R.
3. **Controlled phase, 8 request.**
   - Mọi tỉ lệ GA/GD đều cho test MAE cách retain-only không quá khoảng 5e-4.
   - Request low-overlap 50% của client 4 chỉ còn 1 mẫu normal trong R.
   - Bước này được giữ vì nó tạo ra các request mà các bước sau dùng.
4. **Recovery trajectories †.**
   - Cập nhật của recovery có cosine âm so với hướng tới retrain trong không gian tham số, nhưng khoảng cách dự đoán tới retrain vẫn có thể giảm.
   - Client 4 chi phối hướng recovery: trọng số 0,47–0,65 và căn hướng âm.
5. **Shadow MIA, rồi LiRA 64 shadow.**
   - Attack không có sức mạnh ngay cả trên shadow: positive control chỉ đạt AUC 0,50–0,51.
   - Nguyên nhân: model hầu như không ghi nhớ từng record.
6. **Seed floor, 10 seed.**
   - Ở 6/8 request, việc xóa F làm model thay đổi ít hơn nhiễu do đổi seed.
   - Ở những request phân biệt được thì FU bằng retain-only.
   - Kết luận: phải so sánh với retrain cloud và với retain-only.
7. **Canary client 3.**
   - Đây là request duy nhất có hiệu ứng xóa vượt nhiễu.
   - GA-Adam và NegGrad+ xóa được backdoor trước recovery, nhưng recovery kéo nó quay lại (vị trí 0,68 → 0,49).
8. **Forget signal.**
   - Dừng theo mức retrain thật (oracle) thì có hiệu quả; dừng theo mức nonmember thì thất bại theo hai cách ngược nhau.
   - Các objective bị chặn (p→0,5, random-label), Bad-Teacher và SCRUB đi sai hướng trên canary.
   - KD chỉ trên R làm mất tín hiệu quên.
9. **Deployable forget.**
   - Reference kNN cộng một model reference mới train, kết hợp theo từng dòng (`joint_aux_prob`), bám mức retrain trong 0,001–0,005.
   - Projection giữ được phần đã quên qua recovery mà không làm giảm utility.
   - Đây là phương pháp hiện tại; giới hạn của nó được ghi trong doc kết quả.
10. **MIA canary khớp phân phối (kết luận đã rút lại ở mục 11).**
    - Ban đầu báo cáo: model gốc rò rỉ membership rất ít (AUC cao hơn retrain 0,007 trên c3 và 0,003 trên c0), retain-only vẫn rò rỉ, các arm có projection trở về mức retrain.
    - Xem [canary_mia.md](canary_mia.md).
11. **LiRA canary theo từng vector, 64 shadow mỗi request.**
    - Khoảng một nửa số dòng của cohort MIA canary có vector feature trùng với một dòng của cohort kia, nên membership không xác định. LiRA chấm theo vector duy nhất và bỏ các vector dùng chung.
    - Ngay trên shadow, nơi biết chắc membership, AUC chỉ 0,49–0,51, nên mọi verdict là `MIA_INCONCLUSIVE`.
    - Placebo test (post-hoc, hoán vị nhãn cohort): chênh AUC của mục 10 nằm trong phân phối null (p = 0,16 và 0,42). Power rule ghép seed trên một cohort cố định không phải là phép kiểm định membership.
    - Kết luận: model học quy tắc trigger chứ không ghi nhớ từng dòng. Rò rỉ đo được chỉ ở mức nhóm (backdoor), và phương pháp ứng viên đưa nó về mức retrain.
    - Xem [canary_lira_results.md](canary_lira_results.md).
12. **Phân biệt unlearn với retrain (kiểu U-LiRA), 40 seed.**
    - Attacker biết F và có 39 model tham chiếu mỗi loại; test được hiệu chỉnh đúng, vì retrain so với retrain không phân biệt được.
    - Mọi model unlearn đều phân biệt được với retrain trên cả 5 request.
    - Trên c3/c0, `ga_proj@joint_aux_prob` giảm AUC từ 1,00 xuống 0,67 / 0,71. Phần còn lại là độ lệch nhỏ có hệ thống ở thang logit, mà stop rule theo xác suất không nhìn thấy.
    - Trên c4 và hai request tự nhiên, AUC 0,98–1,00.
    - Xem [indistinguishability_results.md](indistinguishability_results.md).
13. **Baseline reset và chế độ `existing` của bước 8 (sau bản review).**
    - **Lối tắt cấu trúc.** Trigger của canary zero-dim nằm trên các chiều bằng 0 ở mọi dòng thật. Chỉ cần reset các cột W1 đó, không chạy bước quên nào, là đạt position 0,95–0,99. Model reset cũng khó phân biệt với retrain ngang `ga_proj`: AUC 0,65–0,80 so với 0,67–0,71. Kết quả canary zero-dim vì vậy bị loại khỏi bằng chứng cho phương pháp.
    - **Canary tổ hợp.** Trigger nằm trên các chiều đang hoạt động, nên reset không có tác dụng. Trên c3 và c4, `ga_proj` đạt position 0,95 / 0,98, còn GA không chiếu bị recovery kéo về 0,70 / 0,14; `ga_proj` thắng cả control cùng pool. Đây là bằng chứng canary đầu tiên cho projection, nhưng hẹp: 2 request, trên chính các seed dùng để phát triển. Canary tổ hợp c0 không qua gate.
    - **Request tự nhiên.** Control fine-tune cùng pool (R_target ∪ validation) đạt 0,56 trên low-overlap, và `ga_proj` (0,74) không vượt nó quá sàn nhiễu. Không còn bằng chứng nào cho bước quên trên dữ liệu tự nhiên.
    - **Model gốc yếu.** Macro-F1 0,89, nhưng FPR trên traffic normal là 27%.
    - Xem [confirmation_existing_results.md](confirmation_existing_results.md) và [reset_baseline_results.md](reset_baseline_results.md).
14. **Chế độ `confirm`: partition seed 43, seed 20042–20051 (2026-11-03).**
    - **Phạm vi.** Phương pháp, cấu hình và quy tắc đã được đóng băng trước khi chạy ([fu_freeze.md](fu_freeze.md)). Cả 6 canary qua gate; không request nào trong 9 request tự nhiên qua gate.
    - **Verdict: không xác nhận.** `ga_proj@joint_aux_prob` qua 4/6 request.
    - **Nguyên nhân trượt.** Cả hai lần trượt (canary tổ hợp c0, c1) đều do macro-F1 của client 4. Mọi model original và retrain cho cùng một giá trị ở đó, nên sàn nhiễu bằng 0. Trên c1, arm không kém retain-only ở seed nào mà vẫn trượt. Quy tắc không được sửa để chấm lại run này.
    - **Phần lặp lại được.** Projection giữ được việc quên qua recovery trên cả 3 canary tổ hợp và thắng control cùng pool, còn GA không chiếu thì không. Lối tắt zero-dim cũng lặp lại.
    - **Giới hạn mới.** Trên client 0 (R lớn, gần như toàn attack), bước chiếu không dịch chuyển được ở 3/10 seed.
    - Xem [confirmation_p43_results.md](confirmation_p43_results.md).
15. **Chế độ `confirm2`: partition seed 44, seed 30042–30051, lần xác nhận cuối (2026-11-04).**
    - **Phạm vi.** Protocol được ghi trước khi chạy: sàn utility không mịn hơn một dòng, và không có lần thứ ba. 7/12 request qua gate, gồm 2 request tự nhiên của client 2.
    - **Verdict: không xác nhận.** `ga_proj` qua 3/7 request. Sàn một dòng không làm đổi quyết định nào.
    - **Canary.** Hai canary trượt utility vì khoảng t bị nới rộng bởi các seed *tốt hơn* retain-only. Đây là điểm yếu thứ hai của quy tắc, và không được sửa.
    - **Request tự nhiên.** Trên random, bước chiếu không dịch chuyển. Trên low-overlap, mức dừng quá chặt và đẩy model ra xa retrain hơn cả original. Stop oracle cũng không giúp được.
    - **Phần lặp lại được.** Trên canary tổ hợp, projection giữ được việc quên và thắng control cùng pool, nay là 7/7 request trên 3 partition.
    - Xem [confirmation_p44_results.md](confirmation_p44_results.md).
