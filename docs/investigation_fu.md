# Điều tra: vì sao thuật toán FU chưa chứng minh được hiệu quả — 2026-11-03

## Kết luận

**Thuật toán FU (NegGrad+ cộng recovery) chưa được chứng minh, và với thiết kế hiện tại trên ToN-IoT, khó chứng minh được.** Lợi ích đo được trước hết đến từ recovery và từ việc train thêm trên R. Phần do *tín hiệu quên* đóng góp có thật nhưng rất nhỏ.

Có ba nguyên nhân chính, tất cả đều có số liệu bên dưới:
1. Mục tiêu quên trên dataset này rất tinh tế.
2. Bước quên đi sai cỡ: phá mô hình mạnh rồi để recovery sửa, và phần còn lại sau recovery chỉ là một hiệu chỉnh nhỏ.
3. Recovery chi phối kết quả cuối, kể cả khi bước quên được làm mạnh hơn.

Đây là thí nghiệm khám phá. Mọi giả thuyết mới đều phải xác nhận lại trên seed mới.

## Dữ liệu dùng

- **Phân tích trên 38 case family hợp lệ của v7** (seed đánh giá 49–52, không chạy lại mô hình nào): `scripts/investigate_fu.py`, kết quả ở `results/investigation_v7/`.
- **Thí nghiệm khám phá trên seed tune 100–102** (không đụng seed đánh giá): `configs/explore_v8.json`, 9 họ attack, 27 case trong đó 16 hợp lệ, 6 retrain mỗi case, 1 lần lặp, recovery 0/1/2/4 round. Tóm tắt bằng `scripts/summarize_sweep.py`, kết quả ở `results/explore_v8/`.

## Bằng chứng

### A. Lợi ích đến từ đâu (v7)

| Thành phần của D(retain-only) − D(FU) | Trung bình [CI 95%] |
|---|---|
| Tổng | +0,0013 [+0,0006; +0,0021] |
| Train thêm trên R, tức retain-only so với `retain_matched` | +0,0009 [+0,0003; +0,0014] |
| Tín hiệu quên, tức `retain_matched` so với FU | +0,00045 [+0,00006; +0,00084] |

**66% lợi ích của FU so với retain-only đến từ việc train thêm trên R.** Một baseline không hề dùng F cũng đạt được phần đó.

### B. Không phải do trần đo lường (v7)

Một bộ quên hoàn hảo (retrain thật) sẽ gần retrain hơn retain-only quá margin ở **92%** case. FU chỉ đạt 34%. Vậy điều kiện (d) của v7 *đạt được về nguyên tắc*, chứ không phải đặt quá cao.

Tỷ lệ khoảng cách (gốc → retrain) đã được rút ngắn, trung vị: retain-only 18%, `retain_matched` 41%, FU 59%, mô hình trước recovery **−51%**, tức bước local làm mô hình xa retrain hơn cả mô hình gốc.

### C. Đúng hướng nhưng quá yếu, xét từng mẫu (v7, trung vị qua các case)

- 82% mẫu của F có retain-only lệch so với trung bình retrain quá độ phân tán giữa các retrain.
- Trên các mẫu đó, FU dịch **đúng chiều 92%** và hầu như **không vượt quá đích** (0%).
- Nhưng độ dịch chỉ bằng **26%** độ dịch cần thiết. Cosine giữa thay đổi của FU và hướng cần đi là 0,48.
- Lợi ích tập trung: 10% mẫu đóng góp 61% phần cải thiện.
- Mục tiêu quên rất tinh tế: retrain vẫn phân loại 99,9% F là attack, giống mô hình gốc (100%). Khác biệt chỉ nằm ở độ tự tin. Nguyên nhân là các họ attack còn lại và R chứa nhiều mẫu tương tự, nên mô hình tổng quát hóa sang F.

### D. Stop rule

- Ngưỡng proxy lệch so với CE thật của retrain trên F trung vị khoảng 1,7 lần (|log| 0,55). Ngưỡng thấp hơn CE retrain ở 55% case.
- CE trên F trước recovery vượt CE của retrain ở 58% case: bước local quên quá tay, rồi recovery kéo lại.

### E. Hiệu ứng không lớn hơn khi ảnh hưởng của F mạnh hơn

Spearman giữa (D gốc − nhiễu)/sd và hiệu ứng/margin là 0,23, p = 0,17.

### P. Proxy không dùng được làm đích cho từng mẫu

Dự đoán của proxy (client train trên R của mình) trên F có D trung bình **0,074**, so với 0,013 của retain-only. Proxy gần retrain hơn retain-only ở chỉ 7/38 case. Proxy chỉ dùng được như ước lượng CE vô hướng cho stop rule.

### H1. Quên mạnh hơn không khắc phục được "quá yếu" (khám phá, seed tune, n = 16)

| Biến thể | D trước recovery | Δ so với retain-only tại k = 2 [cận dưới CI] | Vượt margin | Thua quá margin |
|---|---:|---:|---:|---:|
| NegGrad+ (cấu hình khóa, lr 0,003) | 0,0357 | +0,0012 [−0,0009] | 12% | 0% |
| lr 0,01 | 0,0520 | +0,0022 [−0,0004] | 19% | 0% |
| lr 0,01, 10 cycle | 0,0507 | +0,0013 [−0,0007] | 25% | 6% |
| lr 0,03 | 0,1145 | +0,0017 [−0,0008] | 19% | 12% |

Quên mạnh hơn làm mô hình trước recovery hỏng nặng hơn: D tăng tới 0,11 và F1 trước recovery giảm 0,019 ở lr 0,03. Nhưng sau recovery, kết quả chỉ nhích nhẹ, không có ý nghĩa thống kê với n = 16, và bắt đầu có case thua retain-only quá margin. **Recovery xóa phần lớn lượng quên thêm.**

### H2. Lợi thế của FU phụ thuộc ngân sách recovery (cấu hình khóa, n = 16)

| Recovery k | D retain-only | D FU | Δ [cận dưới CI] | Tín hiệu quên (so với `retain_matched`) [cận dưới CI] |
|---:|---:|---:|---:|---:|
| 0 | 0,0208 (= gốc) | 0,0357 | −0,0149 [−0,0285] | −0,0014 [−0,0081] |
| 1 | 0,0135 | 0,0125 | +0,0009 [−0,0017] | +0,0012 [+0,0002] |
| 2 | 0,0129 | 0,0117 | +0,0012 [−0,0009] | +0,0012 [+0,0000] |
| 4 | 0,0140 | 0,0135 | +0,0006 [−0,0005] | +0,0008 [+0,0000] |

- Không có recovery thì FU **tệ hơn không làm gì**.
- Lợi thế lớn nhất ở 1–2 round, rồi giảm dần khi recovery dài hơn.
- Retain-only không tiến thêm về phía retrain sau 2 round (D ở k = 4 còn tăng nhẹ), vì recovery từ mô hình gốc hội tụ về một trạng thái khác retrain.

## Diễn giải nguyên nhân gốc

1. **Mục tiêu nhỏ.** Trên ToN-IoT với bộ feature này, việc xóa một họ attack chỉ làm retrain bớt tự tin một chút trên F, không đổi nhãn. Mọi method đều chỉ có một "khoảng" rất hẹp để cải thiện, và margin nhiễu giữa các retrain chiếm phần lớn khoảng đó.
2. **Hướng tối ưu không khớp cỡ.** NegGrad+ đẩy F về phía nhãn sai (tăng loss), trong khi retrain chỉ giảm độ tự tin vừa phải. Kết quả là bước local phá nhiều, recovery sửa lại, và phần còn lại chỉ là một dịch chuyển nhỏ, đúng hướng.
3. **Recovery học lại từ R.** R chứa các họ attack khác và các mẫu tương tự, nên recovery khôi phục phần lớn hành vi trên F, bất kể bước quên mạnh hay yếu.

## Cần gì để chứng minh được một thuật toán FU

1. **Một mục tiêu quên đo được rõ ràng.** Chuẩn trong các bài FU là kiểm chứng bằng **backdoor/canary**: client cài một trigger vào F; mô hình gốc có tỷ lệ kích hoạt cao, retrain gần 0. Khi đó khác biệt nằm ở *nhãn*, không chỉ ở độ tự tin, nên hiệu ứng của FU (nếu có) sẽ vượt xa nhiễu. Đây cũng là cách giải quyết việc MIA không dùng được trên dataset này.
2. **Objective khớp với mục tiêu.** Ví dụ: giới hạn mức thay đổi trên F, hoặc kết hợp quên với recovery trong cùng một vòng tối ưu, thay vì phá rồi sửa.
3. **Dataset ít dư thừa hơn**, hoặc thêm một dataset IDS thứ hai, để kiểm tra phát hiện "mục tiêu nhỏ" có phải đặc thù của ToN-IoT không.
4. Mọi thay đổi trên phải được khóa protocol mới và xác nhận trên seed mới. Kết quả trên seed tune chỉ là giả thuyết.

## Hệ quả cho DT

- FU local không có recovery không dùng được.
- FU + 1–2 round recovery an toàn và tốt hơn retain-only một chút.
- Recovery dài hơn không mang lại thêm gì.

Với planner, các lựa chọn hợp lý gồm: retain-only 1–2 round, FU + 1–2 round, và retrain. Bảng chi phí (FU tốn thêm bước local ở client target) sẽ quyết định khi nào FU đáng dùng.
