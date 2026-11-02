# Báo cáo nghiên cứu FU nhiều seed — run `study_v2`, 2026-10-30

## Kết luận

**Không method nào qua gate nghiên cứu. Kết quả là inconclusive-to-negative. Không mở DT.**

Những gì run này cho thấy:
- Bước quên mới đã thực sự tác động: trên các case hợp lệ, check `local_effect_before_recovery` đạt 3–6/10 case, so với 0/10 của bản pilot.
- Tuy vậy, không method nào vượt retain-only cùng compute một cách ổn định với margin lớn hơn nhiễu nền.
- Kịch bản quên ngẫu nhiên vẫn không đánh giá được, vì LiRA không qua positive control.

## Thông tin run

- Thư mục run: `runs/study_v2`, code commit `df81048` (ToN-IoT `26ddc513…`, data `connected_v3_seed42`, `configs/study.json`).
- Hoàn tất 30/30 case trong 29,5 phút, CPU 2 thread, Linux, torch 2.6.0.
- Tóm tắt kết quả được commit tại `results/study_v2/`: `gate.json`, `cases.csv`, `coverage.csv`, `timings.json`, `manifest.json`.
- `runs/study_v1` là bản chẩn đoán: run dừng do partition shadow bị suy biến, đã sửa ở commit `df81048`. Không dùng làm evidence.
- Pilot được chạy lại trên cùng dữ liệu (`runs/pilot_repro`). Macro-F1 trùng chính xác; MAE(F) lệch ở chữ số thứ tư do khác nền tảng (ví dụ 0,001685 so với 0,001712). Kết luận pilot giữ nguyên.

## Tính hợp lệ theo kịch bản

| Kịch bản | Case | Identifiable | LiRA control ≥ 0,55 | Hợp lệ | LiRA AUC control (original) |
|---|---:|---:|---:|---:|---|
| random 10%/50% | 12 | 3 (chỉ ở 50%) | 0 | 0 | 0,448–0,493 |
| atypical 5% | 6 | 3 | 4 | 2 | 0,527–0,625 |
| family (ransomware, mitm) | 12 | 8 | 12 | 8 | 0,556–0,638 |

Ở kịch bản random, retrain không hề thấy F nhưng vẫn có LiRA AUC(F) trung bình 0,668, trong khi control lại dưới 0,5. Như vậy, kể cả với shadow OUT, score vẫn bị distribution shift giữa các group train và test chi phối.

Giả thuyết, chưa kiểm chứng: shadow chỉ train trên nửa pool nên kém tự tin hơn target trên nonmember. Đồng thời, các bản gần trùng của F và control vẫn nằm trong pool.

Theo protocol, các case random là inconclusive. Không đổi attack sau khi đã nhìn kết quả.

## Kết quả trên 10 case hợp lệ (8 family, 2 atypical)

Định nghĩa:
- D = MAE(F) tới ensemble 4 retrain.
- Δ = D(retain_matched) − D(final).
- "Thắng/thua" nghĩa là |Δ| > margin, trong đó margin = max(1e-4, 2 sd_LOO).

| Method | Local effect trước recovery | Thắng | Thua | Δ trung bình | Qua case | Cận dưới CI 95% của (Δ − margin) |
|---|---:|---:|---:|---:|---:|---:|
| legacy_no_kd | 0 | 0 | 0 | 0,0001 | 0 | −0,0085 |
| legacy_retain_kd | 0 | 0 | 0 | 0,0000 | 0 | −0,0086 |
| neggrad_plus_adam | 6 | 3 | 2 | 0,0005 | 2 | −0,0135 |
| random_label_adam_full_kd | 4 | 3 | 0 | 0,0033 | 2 | −0,0080 |
| uniform_adam_bad_teacher | 3 | 3 | 1 | 0,0025 | 1 | −0,0105 |
| uniform_adam_bad_teacher_oracle (chẩn đoán) | 5 | 4 | 2 | 0,0033 | 0 | −0,0103 |

Các check utility và FPR đều đạt; check LiRA gần retrain hơn đạt 70–100%.

Ví dụ thắng rõ:
- `family_s42_t4_ransomware`: NegGrad+ đạt D=0,0038, so với retain_matched 0,0209 (margin 0,0021).

Ví dụ thua rõ:
- `family_s43_t0_ransomware`: NegGrad+ đạt D=0,0317, so với 0,0099.
- `family_s44_t2_ransomware`: NegGrad+ đạt D=0,0301, so với 0,0132.

Dấu của hiệu ứng đổi theo seed và target, nên chưa có lợi thế ổn định.

## Vì sao chưa đạt

1. **Ảnh hưởng của F nhỏ về giá trị tuyệt đối.**
   - Retrain không thấy ransomware nhưng vẫn phân loại họ này là attack với CE(F) chỉ 0,026–0,037 (original là 0,001–0,009), nhờ tổng quát hóa từ các họ attack khác.
   - Nhiễu nền D của retrain là 0,004–0,012; khoảng cách của original chỉ là 0,022–0,032.
2. **Stop rule `nonmember` lệch theo cả hai chiều ở kịch bản family/atypical.**
   - Với ransomware, ngưỡng (0,030–0,106) cao hơn CE(F) của retrain ở 5/6 case. Pha quên chạy hết 10 cycle mà không dừng, đôi khi overshoot: D trước recovery lên tới 0,137.
   - Với mitm và atypical, ở cả 12/12 case ngưỡng (0,012–0,225) đều thấp hơn CE của original (0,04–0,86). Pha quên không chạy, method thoái hóa thành fine-tune trên R, tức gần với `retain_matched`.
   - Ví dụ `family_s43_t1_mitm` (277 dòng): chỉ 50 bước Adam trên R đã đẩy CE(F) từ 0,21 lên 0,86, tức quên thảm họa một họ hiếm và vượt quá retrain (0,39).
3. **Recovery chi phối kết quả cuối.**
   - 2 round FedAvg trên R kéo D của retain-only với ransomware xuống còn 0,009–0,031.
   - Recovery còn có thể phá kết quả local tốt. Ví dụ ở `family_s44_t2`, bad-teacher có D trước recovery 0,008 nhưng sau recovery lên 0,033.
   - Biến thể oracle có ngưỡng đúng bằng CE của retrain vẫn không ổn định, cho thấy vấn đề nằm ở tương tác với recovery chứ không chỉ ở ngưỡng.
4. **Kịch bản random không có MIA hợp lệ** (mục trên). Ở 10%, nó cũng không identifiable: D(original) 0,0007–0,0019 nằm trong nhiễu.

## Bước tiếp theo đề xuất (cần khóa trong protocol mới trước khi chạy)

1. **LiRA.** Chẩn đoán vì sao control < 0,5 ở kịch bản random: so phân phối score của nonmember, train shadow trên toàn pool (`subset_fraction` 1,0), và loại cả các bản gần trùng theo bidirectional five-tuple. Chỉ dùng lại MIA khi control ≥ 0,55.
2. **Ngưỡng dừng.** Ngưỡng nonmember theo lớp không ước lượng được CE của retrain cho một họ chưa từng thấy. Hai hướng cần cân nhắc:
   - Ngưỡng tính *sau* recovery: lặp quên → recovery ngắn → kiểm tra.
   - Ngưỡng từ một proxy mà client tính được, ví dụ một mô hình local train lại trên R.
   - Cả hai đều phải tính compute vào `retain_matched`.
3. **Recovery.** Đánh giá thêm mô hình trước recovery như một ứng viên (có kèm gate utility), và/hoặc thử số round recovery là 0/1/2, khóa trước.
4. **Báo cáo.** Giữ nguyên các kết luận ở đây. Mọi thay đổi trên phải dùng run và config mới.
