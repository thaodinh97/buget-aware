# Báo cáo nghiên cứu FU — run `study_v3`, 2026-10-30

## Kết luận

**Không method hay ứng viên nào qua gate. Không mở DT.**

Ba thay đổi đã khóa trước (`docs/study_protocol.md`, mục v3) cho các kết quả sau:

1. **Chẩn đoán LiRA:** không biến thể nào cứu được MIA ở kịch bản random. Nguyên nhân nằm ở dữ liệu: test gần như trùng với train trong không gian feature.
2. **Stop rule `proxy_calibrated`:** ước lượng CE của retrain trên F tốt hơn hẳn `nonmember`. Ngược lại, kiểm tra ngưỡng sau recovery không giúp gì.
3. **Ứng viên trước recovery:** NegGrad+ với ngưỡng proxy là tín hiệu tích cực nhất trong cả chuỗi nghiên cứu. Nó thắng retain-only local cùng compute ở 6/10 case hợp lệ và không thua case nào. Nhưng nó vẫn kém retain-only *có* 2 round FedAvg, và utility đôi khi bị tụt.

## Thông tin run

- Thư mục run: `runs/study_v3`, code commit `2a0f1b6`, `configs/study_v3.json`, cùng dữ liệu `connected_v3_seed42`.
- Hoàn tất 30/30 case trong 32,9 phút.
- Tóm tắt kết quả ở `results/study_v3/`.
- Kiểm tra chéo: `neggrad_plus_adam@final` tái lập đúng v2 (excess trung bình −0,005244, cận dưới CI −0,013546).
- Có 10 case hợp lệ (8 family, 2 atypical), giống v2, vì tính hợp lệ không phụ thuộc method.

## Chẩn đoán LiRA (chi tiết trong protocol)

- Ở kịch bản random, cả 4 biến thể (nửa pool hoặc toàn pool; loại bản trùng chính xác hoặc gần trùng) đều có AUC control 0,469–0,477. Theo quy tắc, v3 giữ biến thể của v2.
- So với dòng train gần nhất, 33,9% dòng test có khoảng cách float bằng 0 và 91,8% nằm trong khoảng 1e-3. Nhãn của hàng xóm gần nhất trùng ở 98,4% số dòng.
- Hệ quả: nonmember không "mới" với mô hình, nên MIA dựa trên độ tự tin không đo được membership ở thang mẫu.

## Stop rule

Sai số |log(ngưỡng / CE trung bình của retrain trên F)|, trung bình theo kịch bản:

| Kịch bản | `nonmember` | `proxy_raw` | `proxy_calibrated` |
|---|---:|---:|---:|
| random | 1,030 | 0,444 | **0,259** |
| atypical | 1,530 | 0,438 | **0,274** |
| family | 1,412 | 1,432 | **0,639** |

Tỷ lệ case có ngưỡng cao hơn CE của mô hình gốc, tức pha quên thực sự hoạt động. Cột cuối là tỷ lệ case mà *nên* có pha quên, vì CE của retrain cao hơn CE gốc:

| Kịch bản | `nonmember` | `proxy_calibrated` | Retrain |
|---|---:|---:|---:|
| random | 100% | 75% | 100% |
| atypical | 0% | 67% | 100% |
| family | 50% | 92% | 92% |

Kiểm tra ngưỡng sau recovery (`*_proxy_recovery`) dừng như sau:
- 14/30 case dừng ngay sau 1 round, vì chính round FedAvg đó đã đẩy CE(F) vượt ngưỡng. Khi đó tín hiệu quên gần như không đóng góp.
- 13/30 case chạy hết 5 round mà không đạt ngưỡng. Khi đó 5 round FedAvg chi phối cả FU lẫn baseline đối chứng.
- Kết quả: `neggrad_plus_proxy_recovery@final` chỉ thắng 1 case và qua 0 case; `random_label_proxy_recovery@final` thắng 0 và thua 2.

## Kết quả trên 10 case hợp lệ

"Thắng/thua" so với baseline đối chứng của chính ứng viên đó, với |Δ| > margin nhiễu. `final` so với `retain_matched`; `pre_recovery` so với `retain_matched_local`.

| method@candidate | Thắng | Thua | Qua case | Δ trung bình | Cận dưới CI 95% của (Δ − margin) |
|---|---:|---:|---:|---:|---:|
| legacy_retain_kd@final | 0 | 0 | 0 | −0,0000 | −0,0086 |
| neggrad_plus_adam@final (như v2) | 3 | 2 | 2 | 0,0005 | −0,0135 |
| neggrad_plus_adam@pre_recovery | 5 | 0 | 1 | 0,0044 | −0,0065 |
| neggrad_plus_proxy@final | 2 | 2 | 1 | −0,0008 | −0,0129 |
| **neggrad_plus_proxy@pre_recovery** | **6** | **0** | 2 | 0,0076 | −0,0032 |
| neggrad_plus_proxy_recovery@final | 1 | 0 | 0 | 0,0002 | −0,0097 |
| random_label_proxy_recovery@final | 0 | 2 | 0 | −0,0024 | −0,0128 |
| uniform_bad_teacher_proxy@final | 3 | 2 | 1 | 0,0003 | −0,0131 |
| uniform_bad_teacher_proxy@pre_recovery | 6 | 3 | 1 | 0,0155 | −0,0339 |
| neggrad_plus_oracle_recovery@final (chẩn đoán) | 2 | 1 | 0 | 0,0002 | −0,0111 |

Với `neggrad_plus_proxy@pre_recovery`, ba lý do khiến ứng viên không qua case:
- Chỉ 2/10 case gần retrain hơn retain-only có 2 round FedAvg (`beats_retain_only_recovery`). Ví dụ `family_s43_t0_ransomware`: 0,0259 so với 0,0099.
- `utility_preserved` chỉ đạt 7/10. Ví dụ `family_s44_t2_ransomware`: macro-F1 tụt từ 0,890 xuống 0,834.
- `fpr_preserved` đạt 8/10. Ví dụ `atypical_s44_t2`: FPR tăng từ 0,272 lên 0,417.

Gate utility đã làm đúng vai trò: chặn các ứng viên quên tốt nhưng phát hiện tấn công kém.

## Diễn giải

- Bước quên local với ngưỡng proxy **có giá trị so với cùng lượng compute local chỉ trên R**: 6 thắng, 0 thua. Nhưng 2 round FedAvg trên mọi client (khoảng 394–450 minibatch) vẫn kéo mô hình về gần retrain tốt hơn ở đa số case.
- Lợi thế của FU local, nếu có, nằm ở chi phí: chỉ client target làm việc, không cần cả liên kết. Gate hiện tại chưa đo được điều này. Đây đúng là câu hỏi của phần budget-aware/DT, nhưng chưa được mở.
- Kiểm tra ngưỡng sau recovery không tách được tín hiệu quên khỏi tác động của recovery.
- Mọi kết luận về MIA và holdout bị giới hạn bởi phát hiện gần trùng giữa test và train.

## Bước tiếp theo đề xuất (cần quyết định)

1. **Sửa preprocessing trước.** Gộp các bản gần trùng (khoảng cách encoded ≤ một ngưỡng, hoặc `duration` được lượng tử hóa) vào cùng group trước khi chia split, rồi audit lại bằng khoảng cách hàng xóm gần nhất chứ không chỉ hash. Mọi run phải làm lại trên data mới.
2. Nếu muốn khẳng định lợi thế chi phí, cần một gate cost-aware khóa trước: so FU local với retain-only ở *cùng tổng chi phí*, tính cả số client tham gia và số round FedAvg.
