# FPR của detector: chọn ngưỡng trên validation (protocol thăm dò)

**Trạng thái:** viết ngày 2026-11-18, **trước khi chạy**. Thực nghiệm mô tả, không có gate. Seed dùng ở đây đã được xem ở v2, nên kết quả là dữ liệu phát triển, không phải xác nhận.

## Câu hỏi
Mọi kết quả shard và DT có FPR khoảng 24 % (validation) và 32 % (calibration) ở ngưỡng mặc định (tổng logit margin ≥ 0, tức P(attack) ≥ 0,5). Câu hỏi:

| # | Câu hỏi | Phép đo |
|---|---|---|
| F1 | FPR cao là do **ngưỡng** (model xếp hạng tốt nhưng ngưỡng 0,5 lệch, vì train có 80 % attack và loss không đặt trọng số lớp) hay do **bản thân model/feature** (xếp hạng kém)? | ROC-AUC, PR-AUC, FPR tại recall cố định trên calibration |
| F2 | Chọn ngưỡng trên validation theo quy tắc chốt trước thì FPR và macro-F1 trên calibration đổi ra sao? | Bảng theo quy tắc, S và seed |
| F3 | Phần lỗi nào không thể tránh với 11 feature hiện có? | Lỗi Bayes trên mẫu theo vector feature trùng khớp |
| F4 | Chọn lại ngưỡng sau mỗi lần unlearn có tốn kém, và có giữ tính chính xác không? | Thời gian chọn lại ngưỡng; lập luận ở mục "Tính chính xác" |

## Thiết kế
- **Dữ liệu:** `data/processed/connected_v3_auxfit_seed42`, hash nội dung `f130057e…` (tạo lại ngày 2026-11-18 bằng `src/data/prepare.py`, hash khớp). Chỉ nạp train, validation, calibration qua `shard_v3.load_protocol_data`. **Test không được mở.**
- **Model:** như v2/v3: 20 client, partition 42, cấu hình `standard`, gán shard seed 7. S ∈ {1, 4, 5}; S = 1 là model đơn (cách dựng như `full_retrain_single`). Chỉ trạng thái ban đầu sau `fit`, chưa có yêu cầu xóa.
- **Seed train:** 50042–50046 (seed v2, đã xem). Không dùng seed v3 60042–60051, seed DT 71042–71044, 72042–72044.
- **Điểm số:** tổng logit margin của ensemble (`ShardedUnlearner.margin`). Dự đoán attack khi margin ≥ t.

## Quy tắc chọn ngưỡng (chốt trước, chỉ đọc validation)
| Quy tắc | Định nghĩa |
|---|---|
| `default` | t = 0 (P = 0,5): mốc hiện tại |
| `max_f1` | t cực đại hóa macro-F1 trên validation |
| `youden` | t cực đại hóa TPR − FPR trên validation |
| `fpr10` | t nhỏ nhất có FPR validation ≤ 0,10 (recall lớn nhất dưới ràng buộc) |
| `fpr05` | như trên với 0,05 |

- Ứng viên t là các giá trị margin phân biệt trên validation. Hòa thì chọn t có |t| nhỏ nhất (gần mặc định nhất).
- `max_f1` phụ thuộc tỉ lệ lớp: validation có 31 % normal, calibration 23 %, train 20 %. `youden`, `fpr10`, `fpr05` không phụ thuộc tỉ lệ lớp. Điều này được ghi ra trước để không chọn quy tắc theo kết quả.

## Đo gì (trên calibration)
- Theo quy tắc: macro-F1, FPR, recall attack, F1 từng lớp.
- Không phụ thuộc ngưỡng: ROC-AUC và average precision (attack là lớp dương); FPR tại recall attack 0,95 và 0,99. Hai số sau tính trực tiếp trên calibration, chỉ để mô tả đường cong, **không phải quy tắc triển khai được**.
- **Lỗi không tránh được (F3):** trong từng split, gom các dòng có vector feature giống hệt nhau. Lỗi Bayes trên mẫu = Σ min(n_normal, n_attack) / N. "FPR sàn" = tỉ lệ dòng normal nằm trong vector có số attack lớn hơn số normal. Split gom nhóm theo vector feature nên không có vector trùng giữa các split; phép đo này chỉ thấy mâu thuẫn **trong** một split và bỏ qua các vector gần giống nhau, nên là cận dưới thô.
- **F4:** thời gian một lượt chọn lại ngưỡng (forward trên validation cộng tìm t), mỗi model.

## Tính chính xác khi dùng ngưỡng (F4)
Nếu sau mỗi lần unlearn, ngưỡng được chọn lại **từ model vừa train lại và validation**, thì cặp (model, ngưỡng) đang phục vụ là một hàm tất định của model đó và validation. Validation không chứa dòng train nào. Vì vậy cặp này trùng với cặp mà một lần train lại từ đầu trên dữ liệu còn giữ, cộng cùng quy tắc chọn ngưỡng, sẽ cho ra: phạm vi chính xác của từng chính sách không đổi. Nếu **giữ nguyên** ngưỡng của model cũ thì ngưỡng mang một số thực phụ thuộc dữ liệu đã xóa, và không còn chính xác vô điều kiện.

## Ngoài phạm vi lần này
- Trọng số lớp trong loss (cần train lại và sửa `src/fl/fedavg_core.py`). Chỉ xét nếu F1 cho thấy model xếp hạng kém.
- Thêm feature (port, DNS, HTTP, SSL): đổi tiền xử lý, phải chạy lại mọi thứ.
- Chạy lại v2/v3/DT với ngưỡng mới.

## Không được tuyên bố
| Không được tuyên bố | Lý do |
|---|---|
| FPR trên test hay khi triển khai | Test không mở; một partition; seed đã xem |
| Một quy tắc là tốt nhất | Không có gate; 5 seed |
| Kết quả v2/v3/DT với ngưỡng mới | Không chạy lại các protocol đó |

## Bổ sung sau khi xem kết quả (2026-11-18): ablation feature
**Thêm vào sau khi đọc `results/fpr_threshold/`**, viết trước khi chạy ablation. Phân tích post hoc (`scripts/shard/fpr_cluster_diagnosis.py`) cho thấy 75 % false positive ở ngưỡng mặc định là traffic normal `udp/dns/SF`, cùng vùng điểm số với DDoS/DoS `udp/dns/SF`; chọn ngưỡng chỉ đổi false positive lấy false negative. Câu hỏi bổ sung:

| # | Câu hỏi | Phép đo |
|---|---|---|
| F5 | Thêm các trường giao thức ít giá trị (DNS, HTTP, SSL) có tách được normal khỏi attack ở vùng đó không? | Như F1–F3, cùng quy tắc, seed và S, trên bản dữ liệu mới |

- **Bản dữ liệu riêng:** `configs/data_connected_auxfit_protocolfields.json` → `data/processed/connected_v3_auxfit_protocolfields_seed42`. Giữ nguyên 11 feature cũ, chính sách nhóm, tỉ lệ split, seed 42 và `transform_fit = validation_only`. **Bản dữ liệu chính không đổi.**
- **Feature thêm:** phân loại `dns_qclass`, `dns_qtype`, `dns_rcode`, `dns_AA`, `dns_RD`, `dns_RA`, `dns_rejected`, `http_method`, `http_status_code`, `http_trans_depth`, `ssl_version`, `ssl_resumed`, `ssl_established`, `weird_notice`; số `http_request_body_len`, `http_response_body_len`.
- **Cố ý loại:** trường chuỗi nhiều giá trị hoặc văn bản tự do (`dns_query`, `ssl_cipher`, `ssl_subject`, `ssl_issuer`, `http_uri`, `http_version`, `http_user_agent`, `http_orig_mime_types`, `http_resp_mime_types`, `weird_name`, `weird_addl`), vì dễ để model nhớ giá trị riêng của testbed; IP, port, `label`, `type` đã bị `prepare.py` cấm.
- Thêm feature đổi chữ ký feature, nên đổi cả cách gom nhóm và chia split; partition 20 client cũng đổi. So sánh với bản chính vì vậy **không ghép cặp theo dòng**, chỉ so mức FPR, AUC và cụm `udp/dns/SF`.
- Nếu audit tiền xử lý không đạt (`training_ready = false` hoặc có overlap mã hóa giữa các split) thì dừng và báo cáo, không đổi danh sách feature để cho qua.
- **Không được tuyên bố thêm:** bộ feature này không rò rỉ ngoài phạm vi audit; kết quả v2/v3/DT với bộ feature mới.
- **Độ lệch khi chạy (ghi trước lần chạy lại):** audit tiền xử lý đạt, nhưng `partition(y, 42, 0,5, 20)` trên train của bản mới dừng với "Degenerate client 17" (client có < 4 dòng hoặc chỉ một lớp; hàm không tự chia lại). Quy tắc thay thế: dùng **partition seed nhỏ nhất ≥ 43** không sinh client suy biến, tìm tuần tự chỉ dựa trên nhãn train, không train model nào. Seed train, S và quy tắc ngưỡng không đổi. Partition khác bản chính, nên điều này không làm thay đổi tính chất "không ghép cặp theo dòng" ở trên.

## Bổ sung 2 (2026-11-18): giới hạn nằm ở model hay ở feature
**Thêm vào sau khi đọc kết quả ablation**, viết trước khi chạy. Quan sát dẫn tới phần này: tiền xử lý chỉ chuẩn hóa z-score, không lấy log, nên trên calibration, với 51 % số dòng (`src_bytes` và `dst_bytes` < 500), `src_bytes` sau chuẩn hóa chỉ trải trong 2·10⁻⁵ dù giá trị gốc chênh tới 480 byte.

| # | Câu hỏi | Phép đo |
|---|---|---|
| D1 | Model đơn có fit tốt chính tập train không (thiếu khả năng biểu diễn hay thiếu khả năng tổng quát)? | FPR, recall, macro-F1 trên **train**, S = 1, ngưỡng `default` và `fpr05` |
| D2 | Bộ feature cho phép tới đâu khi không bị giới hạn bởi MLP và FL? | `HistGradientBoostingClassifier` của scikit-learn, **train tập trung** trên cả train, tham số mặc định, `random_state` = seed train; điểm số là `decision_function`; cùng 5 quy tắc ngưỡng trên validation |
| D3 | Lấy `log1p` cho feature số trước khi chuẩn hóa có giúp MLP không? | FL MLP như trước, S ∈ {1, 4, 5}, cùng seed và quy tắc |

- **Cách lấy `log1p` (D3):** khôi phục giá trị số từ mảng đã mã hóa bằng `mean`/`scale` trong `transformer.json`, làm tròn về số nguyên mọi cột trừ `duration`, lấy `log1p(max(·, 0))`, rồi chuẩn hóa lại bằng trung bình và độ lệch chuẩn (ddof = 0) **trên validation** như chính sách `validation_only`. Cột phân loại giữ nguyên. Phép biến đổi đơn điệu nên **split, nhóm và partition giữ nguyên**: kết quả D3 trên bản chính ghép cặp theo seed được với `results/fpr_threshold/`. Biến đổi chỉ fit trên validation nên việc xóa dữ liệu train không đổi cách mã hóa.
- Chạy cả hai bản dữ liệu: bản chính (partition seed 42) và bản có trường giao thức (partition seed 43). Seed train 50042–50046.
- **D2 không phải phương pháp của khóa luận:** model tập trung, không FL, không unlearn theo shard. Nó chỉ cho biết bộ feature cho phép tới đâu.
- **Không được tuyên bố thêm:** D2 là trần thật của bộ feature (chỉ là một model mạnh với tham số mặc định); `log1p` giúp ở dữ liệu hay partition khác; kết quả v2/v3/DT với biến đổi mới.
