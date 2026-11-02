# Báo cáo nghiên cứu FU — run `study_v5` (phương án C + log1p), 2026-10-30

## Kết luận

**Không method hay ứng viên nào qua gate nghiên cứu. Không mở DT.**

- log1p cải thiện utility của mọi mô hình, nhưng không thay đổi kết luận về FU.
- Tín hiệu tích cực của v3 (NegGrad+ proxy trước recovery: 6 thắng, 0 thua) **không lặp lại**. Ở v5, ứng viên đó chỉ thắng 3 và thua 3, và gate utility loại phần lớn các ứng viên trước recovery.
- Với xóa ngẫu nhiên, không method nào gần retrain hơn retain-only recovery đơn thuần với một margin lớn hơn nhiễu.

## Thông tin run

- Thư mục run: `runs/study_v5`, code commit `11654e9`.
- Data `connected_log1p_v5_seed42`: split trùng 100% với v3, khác ở log1p của 7 feature đếm.
- Config `configs/study_v5.json`: giống v3, trừ `mia_required: false`.
- Hoàn tất 30/30 case trong 29,5 phút. Tóm tắt ở `results/study_v5/`.
- Pilot trên cùng data (`results/pilot_v5/`): macro-F1 0,901 (v3 là 0,890), FPR 0,243 (v3 là 0,272).

## Tính hợp lệ

Có 13 case hợp lệ (identifiable theo nhiễu nền retrain): random 4 (cả 4 đều ở mức 50%), atypical 4, family 5. Do đã bỏ điều kiện MIA, con số này không so trực tiếp với 10 case của v3.

LiRA chỉ được báo cáo:
- AUC control ở random 0,446–0,497, ở atypical 0,497–0,549, ở family 0,550–0,644.

`family_s44_t1_mitm` có macro-F1 của chính mô hình gốc là 0,637, dưới sàn 0,65, nên không ứng viên nào qua được case này.

## Kết quả trên 13 case hợp lệ

Thắng/thua so với baseline đối chứng của chính ứng viên, với |Δ| > margin.

| method@candidate | Thắng | Thua | Qua case | Δ trung bình | utility_preserved | fpr_preserved | Cận dưới CI 95% của (Δ − margin) |
|---|---:|---:|---:|---:|---:|---:|---:|
| legacy_retain_kd@final | 0 | 0 | 0 | −0,0000 | 100% | 100% | −0,0086 |
| neggrad_plus_adam@final | 5 | 2 | 1 | 0,0012 | 100% | 100% | −0,0098 |
| neggrad_plus_adam@pre_recovery | 2 | 3 | 1 | 0,0013 | 23% | 23% | −0,0085 |
| neggrad_plus_proxy@final | 4 | 2 | 0 | 0,0010 | 100% | 100% | −0,0101 |
| neggrad_plus_proxy@pre_recovery | 3 | 3 | 0 | 0,0032 | 15% | 15% | −0,0091 |
| neggrad_plus_proxy_recovery@final | 1 | 3 | 0 | −0,0027 | 92% | 92% | −0,0150 |
| random_label_proxy_recovery@final | 0 | 2 | 0 | −0,0025 | 92% | 92% | −0,0145 |
| uniform_bad_teacher_proxy@final | 3 | 5 | 1 | −0,0017 | 100% | 100% | −0,0138 |
| uniform_bad_teacher_proxy@pre_recovery | 8 | 3 | 0 | 0,0119 | 46% | 38% | −0,0207 |
| neggrad_plus_oracle_recovery@final (chẩn đoán) | 2 | 3 | 0 | −0,0018 | 100% | 100% | −0,0142 |

Chỉ có 3 lần qua ở mức case:
- `family_s43_t1_ransomware`: NegGrad+ @final và @pre_recovery.
- `family_s43_t0_ransomware`: bad-teacher @final.

## Phân tích

1. **Xóa ngẫu nhiên.** Ở 3/4 case, `final` của FU thắng `retain_matched`. Nhưng đó là vì chính `retain_matched` bị hại: fine-tune Adam trên R đẩy mô hình *xa* retrain. Cả FU lẫn baseline đều thua retain-only recovery đơn thuần.
   - Ví dụ `random_s43_t0_f0.5`: D (khoảng cách tới retrain) của FU là 0,0018, của `retain_matched` là 0,0029, còn retain-only chỉ 0,0015.
   - Mô hình trước recovery luôn xa retrain hơn cả mô hình gốc: D_pre 0,0045–0,0121, so với D_gốc 0,0017–0,0076, ở cả 4 case và cả 3 method NegGrad+/bad-teacher.
   - Retain-only recovery có D nhỏ nhất ở 11/12 cặp (case, method). Ngoại lệ duy nhất là bad-teacher ở `random_s44_t2` (0,0021 so với 0,0023), nhưng chênh lệch nhỏ hơn margin 0,0004.
   - Với loại xóa này, phương án tốt nhất là không làm gì ở local và chạy FedAvg recovery trên R.
2. **Xóa trọn họ attack.** NegGrad+ thắng rõ ở hai case ransomware của seed 43. Ví dụ `family_s43_t1_ransomware`: D_final 0,0272, so với `retain_matched` 0,0371 và retain-only 0,0534. Nhưng nó thua ở `family_s44_t2_ransomware`, nên vẫn chưa ổn định theo seed và target.
3. **Ứng viên trước recovery.** Với feature log1p, các bước local làm hỏng utility nhiều hơn hẳn: `utility_preserved` chỉ 15–46%, so với 70–90% ở v3. Gate utility loại hầu hết những ứng viên đã thắng về khoảng cách. Vì vậy tín hiệu của v3 có thể phụ thuộc vào việc feature đếm bị nén (chuẩn hóa z-score làm các khác biệt nhỏ gần như biến mất).
4. **Kiểm tra ngưỡng sau recovery** tiếp tục không giúp gì: 1 thắng, 3 thua.

## Tổng kết qua v2, v3 và v5

Qua ba run với ba cấu hình khác nhau, không biến thể FU nào vượt ổn định retain-only có recovery với margin lớn hơn nhiễu nền retrain:
- Bước quên local có tác động đo được trước recovery (đã đạt mục tiêu sửa lỗi ban đầu).
- Lợi thế chỉ xuất hiện rải rác ở các case quên trọn họ attack, và đổi dấu theo seed hoặc target.
- Retain-only recovery (2 round FedAvg trên R) là baseline mạnh trên ToN-IoT với bộ feature này.
