# Tài liệu và lịch sử thực nghiệm

## Doc theo bước pipeline

| Bước | Protocol / kết quả |
|---|---|
| 0. Tiền xử lý | [preprocessing_validation.md](preprocessing_validation.md) |
| 2. Request (controlled phase) | [controlled_phase_plan.md](controlled_phase_plan.md) |
| 3. Seed floor | [seed_floor_results.md](seed_floor_results.md) |
| 4. Canary client 3 | [identifiable_fu_plan.md](identifiable_fu_plan.md), [identifiable_fu_results.md](identifiable_fu_results.md) |
| 5. Forget signal | [forget_signal_plan.md](forget_signal_plan.md), [forget_signal_results.md](forget_signal_results.md) |
| 6. **Deployable forget (tốt nhất)** | [deployable_forget_plan.md](deployable_forget_plan.md), [deployable_forget_results.md](deployable_forget_results.md) |
| 7. MIA canary | [canary_mia.md](canary_mia.md) |
| Shadow MIA và LiRA | [membership_validation_results.md](membership_validation_results.md), [lira_plan.md](lira_plan.md), [lira_results.md](lira_results.md) |
| Đặc tả ban đầu | [AGENT_IMPLEMENTATION_GUIDE_FU_BUDGET_AWARE_DT.md](AGENT_IMPLEMENTATION_GUIDE_FU_BUDGET_AWARE_DT.md) |

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
10. **MIA canary khớp phân phối.**
    - Model gốc rò rỉ membership rất ít: AUC cao hơn retrain 0,007 trên c3 và 0,003 trên c0; trên c4 không có tín hiệu.
    - Retain-only vẫn rò rỉ ở mức đó.
    - Các arm có projection trở về mức retrain.
    - Xem [canary_mia.md](canary_mia.md).
