# Báo cáo FU — run `study_v6`: utility và khoảng cách tới retrain, 2026-11-02

## Kết luận

- **Gate nghiên cứu không đạt với method nào.** Tỷ lệ case qua tối đa 1/22, và cận dưới CI của (Δ − margin) âm ở mọi method. Không mở DT theo gate FU.
- **Điểm mới so với các vòng trước:** khi gộp mọi kịch bản, **NegGrad+ proxy (đã tune) @final** gần retrain hơn retain-only cùng compute một cách có ý nghĩa thống kê. Δ = +0,0023, CI 95% [+0,0002; +0,0045], Wilcoxon sau Holm p = 0,042; 6 thắng, 0 thua.
- Hiệu ứng tập trung ở kịch bản **quên trọn họ attack**:
  - Khoảng cách tới retrain: 0,0332, so với 0,0439 của retain-only có recovery.
  - Gần retrain hơn retain-only ở 7/8 case hợp lệ, trong đó 5 case vượt margin.
  - CI so với retain-only vẫn chứa 0 (n = 8).
- **Utility được bảo toàn** ở mọi ứng viên `final`: macro-F1 tại ngưỡng đã chọn tăng 0,001–0,015, FPR tăng ≤ 0,003. Detection trên họ bị quên dịch về phía retrain.

## Thông tin run

- Code `79dbcfb`, data `connected_log1p_v5_seed42`, `configs/study_v6.json` (protocol v6 kèm sửa đổi về seed).
- Seed 42, 43, 44, 47, 48. Seed 45 và 46 cho partition suy biến (`invalid_request`).
- Ba run:
  - `runs/study_v6_a`: 30 case, 50,0 phút.
  - `runs/study_v6_b`: 0 case, chỉ ghi lại 2 partition không hợp lệ.
  - `runs/study_v6_b2`: 20 case, 31,6 phút.
- Báo cáo gộp do `scripts/report_study.py` tạo ra trong `results/study_v6/` (`report.md`, `summary.csv`, biểu đồ, cùng gate/coverage/manifest của từng run).
- 10 retrain mỗi case. Mỗi method chạy lại 3 lần. Ngưỡng phát hiện được chọn trên validation với FPR ≤ 5%. LiRA tắt.
- Hyperparameter được tune trên seed 100–102 (`results/tune_v6/selection.json`).

## Tính hợp lệ

| Kịch bản | Case | Identifiable | D gốc trung bình | Nhiễu nền trung bình (sd) |
|---|---:|---:|---:|---:|
| random_control 10% (negative control) | 10 | **0** | 0,0036 | 0,0039 (0,0007) |
| random 50% | 10 | 6 | 0,0077 | 0,0046 (0,0007) |
| atypical 5% | 10 | 8 | 0,0414 | 0,0258 (0,0051) |
| family (ransomware, mitm) | 20 | 8 | 0,0383 | 0,0209 (0,0052) |

Negative control hoạt động đúng như kỳ vọng: không case 10% nào identifiable, và mô hình gốc nằm trong nhiễu của retrain.

Độ lệch chuẩn của D giữa 3 lần lặp (NegGrad+ 0,0024; retain-only 0,0025; bad-teacher 0,0033) cùng cỡ với nhiễu nền giữa các retrain (0,0034). Như vậy margin = 2 sd của nhiễu nền là phù hợp với độ dao động của chính các ứng viên.

## Khoảng cách tới retrain trên F (case hợp lệ; Δ > 0 nghĩa là ứng viên gần retrain hơn)

| Kịch bản | Ứng viên | D | Δ so với matched [CI 95%] | p Holm | Thắng/thua | Δ so với retain-only [CI 95%] | p Holm | Vượt retain-only |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| family (n=8) | neggrad_plus_proxy@final | 0,0332 | +0,0058 [+0,0001; +0,0116] | 0,234 | 3/0 | +0,0106 [−0,0022; +0,0234] | 0,547 | 5/8 |
| family | uniform_bad_teacher_proxy@final | 0,0371 | +0,0029 [−0,0102; +0,0161] | 1,000 | 2/2 | +0,0068 [−0,0051; +0,0187] | 0,547 | 4/8 |
| family | legacy_retain_kd@final | 0,0439 | −0,0001 | 1,000 | 0/0 | +0,0000 | 1,000 | 0/8 |
| atypical (n=8) | neggrad_plus_proxy@final | 0,0266 | +0,0004 [−0,0004; +0,0011] | 1,000 | 0/0 | +0,0002 [−0,0021; +0,0024] | 0,586 | 0/8 |
| atypical | uniform_bad_teacher_proxy@final | 0,0280 | +0,0024 [−0,0015; +0,0064] | 1,000 | 3/0 | −0,0012 | 0,766 | 0/8 |
| random 50% (n=6) | neggrad_plus_proxy@final | 0,0073 | +0,0003 [−0,0005; +0,0012] | 1,000 | 3/0 | −0,0001 | 1,000 | 0/6 |
| random 50% | uniform_bad_teacher_proxy@final | 0,0082 | +0,0003 | 1,000 | 2/0 | −0,0010 [−0,0018; −0,0002] | 0,188 | 0/6 |
| **Tất cả (n=22)** | **neggrad_plus_proxy@final** | 0,0238 | **+0,0023 [+0,0002; +0,0045]** | **0,042** | 6/0 | +0,0039 [−0,0007; +0,0085] | 0,052 | 5/22 |
| Tất cả | uniform_bad_teacher_proxy@final | 0,0259 | +0,0020 [−0,0022; +0,0063] | 0,776 | 7/2 | +0,0018 | 0,926 | 4/22 |

Ứng viên **trước recovery** luôn xa retrain hơn retain-only có recovery, kể cả khi gộp mọi kịch bản. Ví dụ NegGrad+: Δ = −0,0301 [−0,0483; −0,0119], p Holm < 0,001. Mô hình trước recovery còn làm lệch hành vi trên R: khi gộp mọi kịch bản, D_R = 0,0245, trong khi nhiễu nền trên R là 0,0034. Ngược lại, D_R của các ứng viên `final` là 0,0045–0,0050, gần nhiễu nền hơn nhiều. Riêng ở random, D_R của `final` là 0,007–0,008, so với nhiễu 0,0042.

Vì sao NegGrad+@final ở family vẫn không qua case dù gần retrain hơn retain-only ở 7/8 case:
- 5/8 case trượt `local_effect_before_recovery`: chính bước local không kéo mô hình về gần retrain; lợi ích chỉ xuất hiện sau recovery.
- 5/8 case trượt `beats_retain_only_matched` theo margin.
- Chỉ `family_s43_t1_ransomware` qua mọi check.

## Utility tại ngưỡng chọn trên validation (FPR ≤ 5%)

- Ứng viên `final`: macro-F1 thay đổi từ +0,001 (family) đến +0,015 (random) so với mô hình gốc, FPR tăng từ +0,0005 đến +0,003. `utility_preserved` đạt 88–100% theo kịch bản. Lần trượt duy nhất là `atypical_s44_t2`, nơi cả ba method, kể cả `legacy`, đều trượt.
- Ứng viên `pre_recovery`: `utility_preserved` từ 75% (atypical) đến 100% (family).
- **Detection trên họ bị quên (family, case hợp lệ):** tỷ lệ báo động trên test của họ đó ở mô hình gốc là 0,364, ở retrain là 0,301. Sau unlearning: NegGrad+ final 0,329, bad-teacher final 0,322, retain-only/legacy 0,337. Unlearning đưa detection về phía retrain. Đồng thời, ngay cả retrain chưa từng thấy họ đó vẫn phát hiện được khoảng 30% nhờ tổng quát hóa từ các họ attack khác.

## Diễn giải

1. Sau khi tune và đo kỹ, **FU có tín hiệu thật nhưng nhỏ**, và nó chỉ xuất hiện khi kết hợp với recovery. NegGrad+@final thắng retain-only cùng compute một cách nhất quán (6 thắng, 0 thua, p Holm 0,042). Riêng ở kịch bản family, nó gần retrain hơn retain-only ở 7/8 case.
2. Với xóa ngẫu nhiên và atypical, mọi ứng viên `final` đều tương đương retain-only trong phạm vi nhiễu. Ở đó không có lý do để dùng FU thay vì recovery.
3. Ứng viên trước recovery không dùng được: xa retrain hơn và làm lệch hành vi trên R.
4. Gate đã khóa trước vẫn không đạt, chủ yếu vì `local_effect_before_recovery` và margin trên từng case. Các case dùng chung seed có chung mô hình gốc, nên p-value gộp chỉ là thống kê sàng lọc.

## Bảng đầu vào cho DT (từ `results/study_v6/summary.csv`)

Theo tiêu chí xong pha FU đã đề ra, bảng chất lượng cho từng lựa chọn đã đầy đủ, mỗi kịch bản trên 5 seed có CI: no-op/original, retain-only (2 round), FU local (pre_recovery), FU + recovery (final), cùng nhiễu nền retrain. Phần chi phí (số round, số client tham gia, byte truyền, thời gian) nằm trong log và `timings.json`, nhưng chưa được tổng hợp thành bảng chi phí – chất lượng.

## Giới hạn

- MIA/privacy nằm ngoài v6.
- `forget_lr` được chọn nằm ở biên trên của lưới tune.
- Kịch bản family chỉ có 8 case hợp lệ.
- Holdout có bản gần trùng (phương án C).
- Chỉ dùng một dataset và một kiến trúc.
