# Gom lô và hoãn theo deadline: bảng tổng hợp

| family | mode | budget_level | events | retrains | seconds_total | cost_vs_immediate | rows_x_rounds | latency_mean | latency_max | deadline_misses | over_budget_events | forced_over_budget_events | overrun_seconds | served_full_share | validation_f1_min | validation_f1_final | validation_fpr_final | exact_all_events | final_exact |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| concentrated | immediate | mid | 20 | 20 | 165 | 1 | 20301460 | 0 | 0 | 0 | 15 | 0 | 26.4 | 1 | 0.897 | 0.898 | 0.239 | True | True |
| concentrated | window5 | mid | 4 | 4 | 28.1 | 0.17 | 3732060 | 2.15 | 4.95 | 0 | 0 | 0 | 0 | 1 | 0.897 | 0.898 | 0.239 | True | True |
| concentrated | lazy | mid | 3 | 3 | 22.1 | 0.134 | 2891780 | 2.62 | 5 | 0 | 0 | 0 | 0 | 1 | 0.898 | 0.898 | 0.239 | True | True |
| concentrated | dt_mid | mid | 10 | 10 | 79.8 | 0.484 | 10072800 | 0.302 | 2.18 | 0 | 0 | 0 | 0 | 1 | 0.897 | 0.898 | 0.239 | True | True |
| concentrated | dt_tight | tight | 10 | 10 | 67.7 | 0.411 | 10072800 | 0.302 | 2.18 | 0 | 0 | 0 | 0 | 1 | 0.897 | 0.898 | 0.239 | True | True |
| poisson_zipf | immediate | mid | 20 | 20 | 100 | 1 | 11773140 | 0 | 0 | 0 | 5 | 0 | 12.3 | 1 | 0.898 | 0.898 | 0.242 | True | True |
| poisson_zipf | window5 | mid | 4 | 11 | 46.2 | 0.462 | 6064780 | 2.15 | 4.95 | 0 | 0 | 0 | 0 | 1 | 0.898 | 0.898 | 0.242 | True | True |
| poisson_zipf | lazy | mid | 9 | 9 | 42.1 | 0.421 | 5465120 | 4 | 5 | 0 | 1 | 0 | 1.6 | 1 | 0.898 | 0.898 | 0.242 | True | True |
| poisson_zipf | dt_mid | mid | 18 | 18 | 72 | 0.72 | 8882910 | 0.499 | 5 | 0 | 0 | 0 | 0 | 0.95 | 0.895 | 0.898 | 0.242 | True | True |
| poisson_zipf | dt_tight | tight | 17 | 17 | 56.3 | 0.563 | 8296570 | 0.568 | 5 | 0 | 0 | 0 | 0 | 0.95 | 0.895 | 0.898 | 0.242 | True | True |
| proportional | immediate | mid | 20 | 20 | 124 | 1 | 14707780 | 0 | 0 | 0 | 10 | 0 | 15.2 | 1 | 0.897 | 0.898 | 0.241 | True | True |
| proportional | window5 | mid | 4 | 10 | 42.4 | 0.342 | 5568400 | 2.15 | 4.95 | 0 | 0 | 0 | 0 | 1 | 0.898 | 0.898 | 0.241 | True | True |
| proportional | lazy | mid | 8 | 8 | 34.3 | 0.277 | 4408900 | 3.83 | 5 | 0 | 0 | 0 | 0 | 1 | 0.898 | 0.898 | 0.241 | True | True |
| proportional | dt_mid | mid | 15 | 15 | 73.7 | 0.596 | 9057230 | 0.331 | 5 | 0 | 0 | 0 | 0 | 0.95 | 0.897 | 0.898 | 0.241 | True | True |
| proportional | dt_tight | tight | 15 | 15 | 56.8 | 0.459 | 8470880 | 0.831 | 5 | 0 | 0 | 0 | 0 | 0.85 | 0.895 | 0.898 | 0.241 | True | True |
| spread | immediate | mid | 20 | 20 | 91 | 1 | 10694820 | 0 | 0 | 0 | 5 | 0 | 12.1 | 1 | 0.895 | 0.896 | 0.236 | True | True |
| spread | window5 | mid | 4 | 11 | 40.4 | 0.444 | 5171780 | 2.15 | 4.95 | 0 | 0 | 0 | 0 | 1 | 0.896 | 0.896 | 0.236 | True | True |
| spread | lazy | mid | 10 | 10 | 35.3 | 0.388 | 4632120 | 3.78 | 5 | 0 | 0 | 0 | 0 | 1 | 0.895 | 0.896 | 0.236 | True | True |
| spread | dt_mid | mid | 18 | 18 | 61.7 | 0.678 | 7875970 | 0.324 | 5 | 0 | 0 | 0 | 0 | 0.95 | 0.895 | 0.895 | 0.236 | True | True |
| spread | dt_tight | tight | 15 | 15 | 44.7 | 0.491 | 5328370 | 1.1 | 5 | 0 | 0 | 0 | 0 | 0.85 | 0.893 | 0.895 | 0.236 | True | True |

## Trạng thái cuối của các lịch chỉ dùng shard_full (immediate, window5, lazy)

| family | seed | shard_full_modes | final_states_equal | bit_exact_checked | bit_exact |
|---|---|---|---|---|---|
| concentrated | 50042 | immediate,window5,lazy | True | True | True |
| poisson_zipf | 50042 | immediate,window5,lazy | True | True | True |
| proportional | 50042 | immediate,window5,lazy | True | True | True |
| spread | 50042 | immediate,window5,lazy | True | True | True |
