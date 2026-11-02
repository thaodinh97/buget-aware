# Báo cáo nghiên cứu xác nhận v7 — quên trọn họ attack, 2026-11-02

## Kết luận theo quy tắc đã khóa trước

**Giả thuyết chính (NegGrad+ proxy @final, kịch bản family): KHÔNG được xác nhận.**

Bốn trên năm điều kiện đạt, kèm bằng chứng thống kê mạnh. Điều kiện duy nhất trượt là (d), ý nghĩa thực tế: chỉ 34% số case (13/38) có mức cải thiện so với retain-only lớn hơn margin nhiễu của case, trong khi ngưỡng đã khóa là 50%.

| Điều kiện (protocol v7) | Kết quả | Đạt |
|---|---|:---:|
| (a) ≥ 15 case hợp lệ | 38 case | có |
| (b) Δ so với retain-only có recovery: cận dưới CI 95% > 0, Wilcoxon p < 0,05 | Δ = +0,0013 [+0,0006; +0,0021], p = 4,6e-7 | có |
| (c) Δ so với `retain_matched`: cận dưới CI 95% > 0, p < 0,05 | Δ = +0,0004 [+0,0001; +0,0008], p = 8,0e-5 | có |
| (d) ≥ 50% số case vượt retain-only quá margin nhiễu | 34% (13/38) | **không** |
| (e) Utility không tệ hơn (cận dưới ΔF1 > −0,03, cận trên ΔFPR < +0,02) | ΔF1 +0,0044 (cận dưới +0,0000); ΔFPR −0,0007 (cận trên +0,0005) | có |

**Giả thuyết phụ (bad-teacher proxy @final): không được xác nhận.** Δ so với retain-only là +0,0007 [−0,0001; +0,0016], p Holm 0,004. Δ so với `retain_matched` có p 0,125. Tỷ lệ vượt margin chỉ 13%.

Cách diễn đạt đúng: trên dữ liệu mới, NegGrad+@final **gần retrain hơn retain-only một cách nhất quán và có ý nghĩa thống kê**, đồng thời không làm giảm utility. Nhưng **hiệu ứng nhỏ**: ở đa số case, phần cải thiện nhỏ hơn dao động tự nhiên giữa các lần retrain. Vì vậy theo tiêu chí đã khóa, chưa chứng minh được ý nghĩa thực tế.

## Thông tin run

- Code và protocol ở commit `19d6aab`, `configs/study_v7.json`. Data `connected_log1p_v5_seed42`. Method giữ nguyên cấu hình đã khóa từ v6.
- Hai run:
  - `runs/study_v7_a`: seed 49–50, 38/38 case, 67,9 phút.
  - `runs/study_v7_b`: seed 51–52, 36 case hoàn tất và 2 `invalid_request` (xss ở seed 52), đúng như dự đoán trong protocol; 64,1 phút.
- Tóm tắt ở `results/study_v7/` (`report.md`, `confirmatory.json`, `summary.csv`, biểu đồ, gate/coverage/manifest của từng run).

## Tính hợp lệ

- Family: **38/70 case identifiable**, trung bình D gốc 0,0187, nhiễu nền 0,0108 (sd 0,0027).
- Theo họ attack (hợp lệ/tổng):

  | backdoor | ddos | dos | injection | mitm | password | ransomware | scanning | xss |
  |---:|---:|---:|---:|---:|---:|---:|---:|---:|
  | 7/8 | 7/8 | 2/8 | 2/8 | 0/8 | 3/8 | 4/8 | 8/8 | 5/6 |

  mitm không identifiable ở case nào, khác với v6.
- **Negative control: 1/4 case bị đánh dấu identifiable** (`random_control_s49_t2`, D gốc 0,0035). Gộp với v6 (0/10), tỷ lệ dương tính giả của tiêu chí identifiable là 1/14. Đây là một giới hạn cần nêu: tiêu chí dựa trên max của 10 retrain cộng 3 sd vẫn có thể dương tính giả ở xóa ngẫu nhiên.

## Chi tiết cho NegGrad+@final (38 case family hợp lệ)

- Gần retrain hơn retain-only ở **33/38 case**. **Không case nào xa retrain hơn retain-only quá margin** (0 thua). Trung bình khoảng cách giảm 21% so với retain-only (D 0,0114 so với 0,0127).
- Δ/margin theo phân vị: 10% −0,05; 25% 0,19; 50% 0,40; 75% 1,17; 90% 3,51. Ở trung vị, cải thiện chỉ bằng khoảng 40% margin nhiễu.
- **Tỷ lệ case nằm trong vùng nhiễu retrain:** NegGrad+@final **42%**, so với retain-only **8%** và mô hình gốc 0%.
- Theo họ attack:

  | Họ | n | Δ so với retain-only | Vượt margin | NegGrad+ trong vùng nhiễu | Retain-only trong vùng nhiễu |
  |---|---:|---:|---:|---:|---:|
  | backdoor | 7 | +0,0003 | 6/7 | 71% | 0% |
  | ransomware | 4 | +0,0060 | 3/4 | 75% | 0% |
  | scanning | 8 | +0,0003 | 3/8 | 50% | 12% |
  | xss | 5 | +0,0009 | 1/5 | 0% | 0% |
  | password | 3 | +0,0013 | 0/3 | 67% | 0% |
  | ddos | 7 | +0,0022 | 0/7 | 0% | 0% |
  | dos | 2 | +0,0000 | 0/2 | 50% | 50% |
  | injection | 2 | −0,0007 | 0/2 | 50% | 50% |

  Ransomware, họ đã sinh ra giả thuyết ở v6, lặp lại hiệu ứng (Δ +0,0060, 3/4 vượt margin). Ở ddos, Δ trung bình lớn (+0,0022) nhưng margin của họ này lớn hơn nhiều (0,0099), nên không case nào vượt.
- Utility (tại ngưỡng chọn trên validation): ΔF1 +0,0044 [cận dưới +0,0000], ΔFPR −0,0007 [cận trên +0,0005], `utility_preserved` 100%.
- Detection trên họ bị quên: mô hình gốc 0,865 → NegGrad+@final 0,849, so với retrain 0,843 và retain-only/legacy 0,854. Detection dịch về phía retrain.

## Các ứng viên khác

- `neggrad_plus_proxy@pre_recovery`: xa retrain hơn retain-only, Δ −0,0388 [−0,0788; +0,0013]; làm lệch R (D_R 0,0277 so với nhiễu 0,0033). Tiếp tục cho thấy mô hình trước recovery không dùng được.
- `uniform_bad_teacher_proxy@pre_recovery`: thắng `retain_matched_local` rất rõ (Δ +0,0421, 26/7), nhưng vẫn xa retrain hơn retain-only có recovery.
- `legacy_retain_kd`: trùng retain-only, Δ ≈ 0.

## Diễn giải và hệ quả

1. Kết quả v6 **lặp lại về chiều và ý nghĩa thống kê** trên seed mới và 7 họ attack mới. Đây là bằng chứng mạnh hơn hẳn v6 (p = 4,6e-7 với n = 38). Nhưng **độ lớn nhỏ hơn**: Δ trung bình +0,0013, so với +0,0106 ở v6 khi chỉ có ransomware/mitm.
2. Lợi ích rõ nhất của FU nằm ở chỉ số "đạt mức retrain": 42% so với 8% của retain-only. Phần lớn case retain-only nằm ngoài vùng nhiễu retrain, và phần cải thiện nhỏ của FU đủ đưa thêm khoảng một phần ba số case vào trong vùng đó.
3. Theo quy tắc đã khóa, không được nới ngưỡng 50% sau khi thấy kết quả. Phát biểu có thể bảo vệ được: *"NegGrad+ cộng recovery cải thiện nhỏ nhưng nhất quán so với retain-only khi xóa trọn một họ attack, không làm giảm utility; cải thiện vượt mức nhiễu giữa các lần retrain ở khoảng một phần ba số case."*
4. Đối với DT: FU + recovery là một lựa chọn **an toàn** (không thua retain-only quá margin ở case nào, utility giữ nguyên) và **hơi tốt hơn** khi mục tiêu là đạt mức retrain. Còn việc nó có đáng hay không phụ thuộc chi phí, vì FU tốn thêm bước local ở client target. Đây đúng là câu hỏi cost–quality mà DT phải trả lời.

## Giới hạn

- Negative control cho thấy 1/4 dương tính giả về identifiability.
- Các case cùng seed chia sẻ một phần partition, nên p-value coi các case là độc lập.
- MIA/privacy nằm ngoài v7.
- Chỉ một dataset và một kiến trúc.
- `forget_lr` nằm ở biên của lưới tune và không được tune lại.
