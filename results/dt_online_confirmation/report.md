# Xác nhận DT online và deadline hoàn thành

Đã audit 54 chuỗi, 648 yêu cầu. Mỗi arm xử lý đủ yêu cầu đúng một lần; không dùng yêu cầu tương lai; latency = waiting + service; số liệu tổng hợp khớp event gốc; bit-exact đều đạt.

Thiết kế cố định trước: [protocol](../../docs/dt_online_confirmation_plan.md), `protocol_lock.json`. Ba lần lặp, hai họ chuỗi; thống kê mô tả theo block, không kiểm định ý nghĩa thống kê.

## Kết quả gộp

| environment | predictor | blocks | requests | joint_success | deadline_misses | latency_mean_seconds | waiting_mean_seconds | service_seconds | round_over_budget_events | events | full_requests | validation_f1_mean | validation_fpr_mean | bit_exact_all |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gradual_clients | rows_ewma | 6 | 72 | 50 | 21 | 8.719 | 4.635 | 202.802 | 5 | 52 | 35 | 0.893 | 0.241 | True |
| gradual_clients | adaptive | 6 | 72 | 49 | 23 | 8.835 | 4.715 | 205.405 | 5 | 52 | 39 | 0.894 | 0.240 | True |
| gradual_clients | last_shard | 6 | 72 | 48 | 24 | 9.181 | 5.029 | 209.529 | 5 | 53 | 39 | 0.894 | 0.240 | True |
| jitter_bandwidth | adaptive | 6 | 72 | 40 | 30 | 9.726 | 5.476 | 197.106 | 3 | 46 | 20 | 0.893 | 0.240 | True |
| jitter_bandwidth | last_shard | 6 | 72 | 41 | 28 | 9.418 | 5.216 | 194.678 | 5 | 47 | 23 | 0.893 | 0.240 | True |
| jitter_bandwidth | rows_ewma | 6 | 72 | 45 | 25 | 9.331 | 5.059 | 195.899 | 2 | 48 | 22 | 0.893 | 0.240 | True |
| none | last_shard | 6 | 72 | 62 | 10 | 6.546 | 3.464 | 157.853 | 0 | 55 | 48 | 0.894 | 0.240 | True |
| none | rows_ewma | 6 | 72 | 66 | 6 | 6.416 | 3.454 | 153.384 | 0 | 54 | 47 | 0.895 | 0.239 | True |
| none | adaptive | 6 | 72 | 62 | 10 | 6.774 | 3.659 | 158.186 | 0 | 53 | 47 | 0.895 | 0.239 | True |

Joint success: hoàn thành trong deadline và batch vừa ngân sách vòng FU. Deadline tính tới model-ready trong bộ nhớ, không gồm HTTP hay lưu checkpoint. Latency mean là trung bình các chuỗi có cùng số yêu cầu; p95 từng chuỗi có trong summary.csv.

## Đối chiếu ghép cặp

Delta = Adaptive − đối chứng; joint_success dương tốt hơn, service_seconds âm rẻ hơn. Win/tie/loss dựa trên số joint-success trong cùng block.

| environment | comparator | wins | ties | losses | joint_success_delta | service_seconds_delta |
|---|---|---|---|---|---|---|
| gradual_clients | last_shard | 1 | 5 | 0 | 1 | -4.124 |
| gradual_clients | rows_ewma | 1 | 3 | 2 | -1 | 2.602 |
| jitter_bandwidth | last_shard | 2 | 3 | 1 | -1 | 2.428 |
| jitter_bandwidth | rows_ewma | 1 | 2 | 3 | -5 | 1.206 |
| none | last_shard | 2 | 3 | 1 | 0 | 0.333 |
| none | rows_ewma | 0 | 3 | 3 | -4 | 4.802 |

## Tác động của phép đo deadline

Có **128/648 yêu cầu** mà thời gian chờ <= D nhưng completion latency > D: bỏ thời gian service sẽ đánh giá sai những yêu cầu này là đúng hạn. Đây là phân rã của lịch thực đã chạy, không phải đối chứng nhân quả với scheduler mô phỏng cũ.

## Giới hạn

- Một máy, stress được tiêm bằng sleep thật; điều kiện lấy mẫu đầu mỗi batch. Không đo mạng phân tán.
- Cả ba dùng luật gom lô full/fixed, không kiểm chứng trực tiếp lợi ích midcourse.
- Mọi yêu cầu cuối cùng được xử lý, không có loại bỏ hay phản hồi miễn phí. Forced jobs và deadline miss đều tính.
- Cùng budget tuyệt đối trong block, khác giữa block theo fit. Thứ tự chạy được xoay vòng; drift máy vẫn có thể còn.
- Utility chỉ trên validation; khác số vòng và lịch phục vụ. Không dùng test/calibration để chọn cấu hình.
- Ba repeat không đủ để suy rộng hoặc khẳng định ưu thế thống kê; không điều chỉnh alpha theo bảng này.
