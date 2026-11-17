# DT dự báo service và lập kế hoạch hàng đợi: kết quả

Cấu hình đã chốt: [protocol](../../docs/dt_predictive_plan.md), protocol_lock.json. Số liệu được audit từ timestamp và event, cùng phép chấm của bộ trước.

## Kết quả

| environment | predictor | requests | joint_success | deadline_misses | service_seconds | latency_mean | full_requests | validation_f1_mean |
|---|---|---|---|---|---|---|---|---|
| gradual_clients | dt_queue | 72 | 51 | 20 | 230.960 | 11.101 | 18 | 0.892 |
| gradual_clients | rows_legacy | 72 | 30 | 41 | 233.863 | 13.826 | 16 | 0.892 |
| gradual_clients | rows_queue | 72 | 55 | 16 | 219.238 | 9.835 | 15 | 0.892 |
| jitter_bandwidth | dt_queue | 72 | 35 | 32 | 217.750 | 12.953 | 11 | 0.892 |
| jitter_bandwidth | rows_legacy | 72 | 21 | 48 | 217.691 | 14.579 | 11 | 0.892 |
| jitter_bandwidth | rows_queue | 72 | 36 | 34 | 208.987 | 13.416 | 9 | 0.892 |
| none | dt_queue | 72 | 71 | 1 | 175.175 | 6.945 | 28 | 0.893 |
| none | rows_legacy | 72 | 62 | 10 | 176.270 | 8.935 | 40 | 0.894 |
| none | rows_queue | 72 | 72 | 0 | 163.199 | 6.264 | 25 | 0.892 |

`rows_legacy`: đối chứng mạnh nhất của bộ trước; `rows_queue`: cùng controller mới nhưng chỉ dự báo theo dòng; `dt_queue`: controller mới và twin kết hợp bốn bộ dự báo bằng sai số đã quan sát.

## So sánh ghép cặp DT − đối chứng

| comparator | wins | ties | losses | joint_delta | service_delta | f1_delta_mean |
|---|---|---|---|---|---|---|
| rows_legacy | 15 | 3 | 0 | 44 | -3.938 | -0.000 |
| rows_queue | 3 | 10 | 5 | -6 | 32.461 | 0.000 |

## Tiêu chí thực dụng chốt trước

```json
{
  "measured": {
    "joint_gain_pp": 20.370370370370374,
    "positive_environments": 3,
    "service_ratio": 0.9937270521846459,
    "mean_f1_delta": -9.634842195641345e-05,
    "worst_f1_delta": -0.002533767175264323
  },
  "targets": {
    "joint_success_gain_pp_vs_rows_legacy": 5.0,
    "environments_with_positive_gain": 2,
    "max_service_ratio_vs_rows_legacy": 1.1,
    "min_mean_paired_validation_f1_delta": -0.002,
    "min_block_validation_f1_delta": -0.005
  },
  "gates": {
    "joint_gain": true,
    "environments": true,
    "service_cost": true,
    "mean_utility": true,
    "worst_utility": true,
    "exactness": true
  },
  "all_practical_targets_met": true,
  "statistical_significance_tested": false
}
```

Đây là các ngưỡng thực dụng trên mẫu nhỏ, không phải kiểm định ý nghĩa thống kê hay chứng minh noninferiority. Nếu DT hơn rows_legacy nhưng không hơn rows_queue thì cải thiện thuộc chủ yếu về lập lịch/đo service; không được quy toàn bộ cho mô hình twin chi tiết.

Phạm vi: local CPU, một worker, 4 shard, full/fixed, chỉ yêu cầu đã đến; không đo HTTP, checkpoint bền vững hay distributed serving. Không dùng test/calibration hay seed v3. Không bảo đảm hard deadline.
