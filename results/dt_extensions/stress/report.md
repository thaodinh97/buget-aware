# Stress, biên an toàn và điều chỉnh giữa chừng: bảng tổng hợp

Trajectory: s4_50042_proportional, s4_50042_spread, s4_50043_proportional, s4_50043_spread. Mỗi ô gộp mọi trajectory của kịch bản; mỗi trajectory có 20 yêu cầu.

## Chế độ `precommit`

### Vi phạm ngân sách (số yêu cầu vượt / số yêu cầu chấp nhận)

| scenario | budget_level | Oracle | Static | Static + biên | Adaptive | Adaptive + biên | Adaptive + biên chấm lại |
|---|---|---|---|---|---|---|---|
| none | tight | 0/75 | 8/80 | 2/62 | 8/80 | 2/62 | 4/64 |
| none | mid | 0/80 | 0/80 | 0/80 | 0/80 | 0/80 | 0/80 |
| none | wide | 0/80 | 0/80 | 0/80 | 0/80 | 0/80 | 0/80 |
| slow_clients | tight | 0/56 | 52/80 | 10/56 | 27/66 | 10/54 | 12/56 |
| slow_clients | mid | 0/78 | 12/80 | 5/68 | 12/80 | 4/64 | 8/76 |
| slow_clients | wide | 0/80 | 0/80 | 0/80 | 0/80 | 0/80 | 0/80 |
| low_bandwidth | tight | 0/14 | 70/80 | 11/19 | 16/25 | 8/16 | 10/18 |
| low_bandwidth | mid | 0/20 | 60/80 | 14/34 | 10/30 | 6/26 | 6/26 |
| low_bandwidth | wide | 0/68 | 60/80 | 14/48 | 22/80 | 6/28 | 7/65 |
| stale_state | tight | 0/64 | 30/80 | 11/59 | 20/72 | 10/58 | 11/59 |
| stale_state | mid | 0/80 | 6/80 | 4/76 | 6/80 | 4/72 | 4/78 |
| stale_state | wide | 0/80 | 0/80 | 0/80 | 0/80 | 0/80 | 0/80 |
| transient_fresh | tight | 0/64 | 30/80 | 6/50 | 18/72 | 6/50 | 8/50 |
| transient_fresh | mid | 0/80 | 6/80 | 4/74 | 6/80 | 4/68 | 4/79 |
| transient_fresh | wide | 0/80 | 0/80 | 0/80 | 0/80 | 0/80 | 0/80 |

### Từ chối nhầm (số yêu cầu bị từ chối dù oracle chấp nhận / số yêu cầu oracle chấp nhận)

| scenario | budget_level | Oracle | Static | Static + biên | Adaptive | Adaptive + biên | Adaptive + biên chấm lại |
|---|---|---|---|---|---|---|---|
| none | tight | 0/75 | 0/75 | 13/75 | 0/75 | 13/75 | 13/75 |
| none | mid | 0/80 | 0/80 | 0/80 | 0/80 | 0/80 | 0/80 |
| none | wide | 0/80 | 0/80 | 0/80 | 0/80 | 0/80 | 0/80 |
| slow_clients | tight | 0/56 | 0/56 | 4/56 | 0/56 | 6/56 | 4/56 |
| slow_clients | mid | 0/78 | 0/78 | 12/78 | 0/78 | 16/78 | 4/78 |
| slow_clients | wide | 0/80 | 0/80 | 0/80 | 0/80 | 0/80 | 0/80 |
| low_bandwidth | tight | 0/14 | 0/14 | 4/14 | 1/14 | 4/14 | 4/14 |
| low_bandwidth | mid | 0/20 | 0/20 | 0/20 | 0/20 | 0/20 | 0/20 |
| low_bandwidth | wide | 0/68 | 0/68 | 20/68 | 0/68 | 40/68 | 4/68 |
| stale_state | tight | 0/64 | 0/64 | 10/64 | 4/64 | 10/64 | 10/64 |
| stale_state | mid | 0/80 | 0/80 | 4/80 | 0/80 | 8/80 | 2/80 |
| stale_state | wide | 0/80 | 0/80 | 0/80 | 0/80 | 0/80 | 0/80 |
| transient_fresh | tight | 0/64 | 0/64 | 16/64 | 2/64 | 16/64 | 16/64 |
| transient_fresh | mid | 0/80 | 0/80 | 6/80 | 0/80 | 12/80 | 1/80 |
| transient_fresh | wide | 0/80 | 0/80 | 0/80 | 0/80 | 0/80 | 0/80 |

### Hạ cấp nhầm (chọn fixed dù full vừa ngân sách / số yêu cầu oracle chọn full)

| scenario | budget_level | Oracle | Static | Static + biên | Adaptive | Adaptive + biên | Adaptive + biên chấm lại |
|---|---|---|---|---|---|---|---|
| none | tight | 0/45 | 0/45 | 11/45 | 2/45 | 11/45 | 9/45 |
| none | mid | 0/48 | 0/48 | 0/48 | 0/48 | 0/48 | 0/48 |
| none | wide | 0/80 | 0/80 | 0/80 | 0/80 | 0/80 | 0/80 |
| slow_clients | tight | 0/20 | 0/20 | 0/20 | 0/20 | 0/20 | 0/20 |
| slow_clients | mid | 0/38 | 0/38 | 4/38 | 0/38 | 4/38 | 2/38 |
| slow_clients | wide | 0/80 | 0/80 | 12/80 | 0/80 | 16/80 | 0/80 |
| low_bandwidth | tight | 0/6 | 0/6 | 0/6 | 0/6 | 0/6 | 0/6 |
| low_bandwidth | mid | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| low_bandwidth | wide | 0/20 | 0/20 | 0/20 | 0/20 | 0/20 | 0/20 |
| stale_state | tight | 0/34 | 0/34 | 10/34 | 6/34 | 10/34 | 10/34 |
| stale_state | mid | 0/42 | 0/42 | 8/42 | 0/42 | 8/42 | 8/42 |
| stale_state | wide | 0/80 | 0/80 | 4/80 | 0/80 | 6/80 | 2/80 |
| transient_fresh | tight | 0/34 | 0/34 | 12/34 | 6/34 | 12/34 | 11/34 |
| transient_fresh | mid | 0/42 | 0/42 | 2/42 | 0/42 | 6/42 | 4/42 |
| transient_fresh | wide | 0/80 | 0/80 | 6/80 | 0/80 | 12/80 | 1/80 |

### Tổng số giây vượt ngân sách

| scenario | budget_level | Oracle | Static | Static + biên | Adaptive | Adaptive + biên | Adaptive + biên chấm lại |
|---|---|---|---|---|---|---|---|
| none | tight | 0 | 2 | 1 | 2 | 1 | 1 |
| none | mid | 0 | 0 | 0 | 0 | 0 | 0 |
| none | wide | 0 | 0 | 0 | 0 | 0 | 0 |
| slow_clients | tight | 0 | 60 | 5 | 24 | 6 | 7 |
| slow_clients | mid | 0 | 5 | 1 | 5 | 1 | 2 |
| slow_clients | wide | 0 | 0 | 0 | 0 | 0 | 0 |
| low_bandwidth | tight | 0 | 744 | 73 | 51 | 39 | 50 |
| low_bandwidth | mid | 0 | 619 | 73 | 51 | 36 | 36 |
| low_bandwidth | wide | 0 | 520 | 79 | 77 | 41 | 41 |
| stale_state | tight | 0 | 28 | 7 | 15 | 6 | 7 |
| stale_state | mid | 0 | 2 | 1 | 2 | 1 | 1 |
| stale_state | wide | 0 | 0 | 0 | 0 | 0 | 0 |
| transient_fresh | tight | 0 | 28 | 4 | 15 | 4 | 6 |
| transient_fresh | mid | 0 | 2 | 1 | 2 | 1 | 1 |
| transient_fresh | wide | 0 | 0 | 0 | 0 | 0 | 0 |

## Chế độ `midcourse`

### Vi phạm ngân sách (số yêu cầu vượt / số yêu cầu chấp nhận)

| scenario | budget_level | Oracle | Static | Static + biên | Adaptive | Adaptive + biên | Adaptive + biên chấm lại |
|---|---|---|---|---|---|---|---|
| none | tight | 0/75 | 5/80 | 0/62 | 5/80 | 0/62 | 2/64 |
| none | mid | 0/80 | 0/80 | 0/80 | 0/80 | 0/80 | 0/80 |
| none | wide | 0/80 | 0/80 | 0/80 | 0/80 | 0/80 | 0/80 |
| slow_clients | tight | 0/56 | 24/80 | 4/54 | 10/66 | 2/50 | 4/56 |
| slow_clients | mid | 0/78 | 2/80 | 2/68 | 2/80 | 2/64 | 2/76 |
| slow_clients | wide | 0/80 | 0/80 | 0/80 | 0/80 | 0/80 | 0/80 |
| low_bandwidth | tight | 0/14 | 68/80 | 11/19 | 14/25 | 8/16 | 10/18 |
| low_bandwidth | mid | 0/20 | 60/80 | 14/34 | 10/30 | 6/26 | 6/26 |
| low_bandwidth | wide | 0/68 | 12/80 | 0/49 | 12/80 | 0/28 | 2/66 |
| stale_state | tight | 0/64 | 18/80 | 6/60 | 14/72 | 6/60 | 6/60 |
| stale_state | mid | 0/80 | 0/80 | 0/76 | 0/80 | 0/73 | 0/78 |
| stale_state | wide | 0/80 | 0/80 | 0/80 | 0/80 | 0/80 | 0/80 |
| transient_fresh | tight | 0/64 | 18/80 | 2/52 | 12/72 | 2/50 | 2/53 |
| transient_fresh | mid | 0/80 | 0/80 | 0/74 | 0/80 | 0/68 | 0/79 |
| transient_fresh | wide | 0/80 | 0/80 | 0/80 | 0/80 | 0/80 | 0/80 |

### Từ chối nhầm (số yêu cầu bị từ chối dù oracle chấp nhận / số yêu cầu oracle chấp nhận)

| scenario | budget_level | Oracle | Static | Static + biên | Adaptive | Adaptive + biên | Adaptive + biên chấm lại |
|---|---|---|---|---|---|---|---|
| none | tight | 0/75 | 0/75 | 13/75 | 0/75 | 13/75 | 13/75 |
| none | mid | 0/80 | 0/80 | 0/80 | 0/80 | 0/80 | 0/80 |
| none | wide | 0/80 | 0/80 | 0/80 | 0/80 | 0/80 | 0/80 |
| slow_clients | tight | 0/56 | 0/56 | 6/56 | 0/56 | 8/56 | 4/56 |
| slow_clients | mid | 0/78 | 0/78 | 12/78 | 0/78 | 16/78 | 4/78 |
| slow_clients | wide | 0/80 | 0/80 | 0/80 | 0/80 | 0/80 | 0/80 |
| low_bandwidth | tight | 0/14 | 0/14 | 4/14 | 1/14 | 4/14 | 4/14 |
| low_bandwidth | mid | 0/20 | 0/20 | 0/20 | 0/20 | 0/20 | 0/20 |
| low_bandwidth | wide | 0/68 | 0/68 | 19/68 | 0/68 | 40/68 | 4/68 |
| stale_state | tight | 0/64 | 0/64 | 10/64 | 4/64 | 10/64 | 10/64 |
| stale_state | mid | 0/80 | 0/80 | 4/80 | 0/80 | 7/80 | 2/80 |
| stale_state | wide | 0/80 | 0/80 | 0/80 | 0/80 | 0/80 | 0/80 |
| transient_fresh | tight | 0/64 | 0/64 | 14/64 | 2/64 | 16/64 | 13/64 |
| transient_fresh | mid | 0/80 | 0/80 | 6/80 | 0/80 | 12/80 | 1/80 |
| transient_fresh | wide | 0/80 | 0/80 | 0/80 | 0/80 | 0/80 | 0/80 |

### Hạ cấp nhầm (chọn fixed dù full vừa ngân sách / số yêu cầu oracle chọn full)

| scenario | budget_level | Oracle | Static | Static + biên | Adaptive | Adaptive + biên | Adaptive + biên chấm lại |
|---|---|---|---|---|---|---|---|
| none | tight | 0/45 | 4/45 | 6/45 | 4/45 | 6/45 | 6/45 |
| none | mid | 0/48 | 0/48 | 0/48 | 0/48 | 0/48 | 0/48 |
| none | wide | 0/80 | 0/80 | 0/80 | 0/80 | 0/80 | 0/80 |
| slow_clients | tight | 0/20 | 0/20 | 0/20 | 0/20 | 0/20 | 0/20 |
| slow_clients | mid | 0/38 | 0/38 | 2/38 | 0/38 | 2/38 | 2/38 |
| slow_clients | wide | 0/80 | 2/80 | 2/80 | 2/80 | 2/80 | 2/80 |
| low_bandwidth | tight | 0/6 | 0/6 | 0/6 | 0/6 | 0/6 | 0/6 |
| low_bandwidth | mid | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| low_bandwidth | wide | 0/20 | 0/20 | 0/20 | 0/20 | 0/20 | 0/20 |
| stale_state | tight | 0/34 | 2/34 | 5/34 | 2/34 | 5/34 | 5/34 |
| stale_state | mid | 0/42 | 0/42 | 0/42 | 0/42 | 0/42 | 0/42 |
| stale_state | wide | 0/80 | 0/80 | 0/80 | 0/80 | 0/80 | 0/80 |
| transient_fresh | tight | 0/34 | 2/34 | 6/34 | 2/34 | 6/34 | 6/34 |
| transient_fresh | mid | 0/42 | 0/42 | 0/42 | 0/42 | 0/42 | 0/42 |
| transient_fresh | wide | 0/80 | 0/80 | 0/80 | 0/80 | 0/80 | 0/80 |

### Tổng số giây vượt ngân sách

| scenario | budget_level | Oracle | Static | Static + biên | Adaptive | Adaptive + biên | Adaptive + biên chấm lại |
|---|---|---|---|---|---|---|---|
| none | tight | 0 | 1 | 0 | 1 | 0 | 0 |
| none | mid | 0 | 0 | 0 | 0 | 0 | 0 |
| none | wide | 0 | 0 | 0 | 0 | 0 | 0 |
| slow_clients | tight | 0 | 19 | 2 | 8 | 1 | 2 |
| slow_clients | mid | 0 | 0 | 0 | 0 | 0 | 0 |
| slow_clients | wide | 0 | 0 | 0 | 0 | 0 | 0 |
| low_bandwidth | tight | 0 | 384 | 38 | 34 | 23 | 34 |
| low_bandwidth | mid | 0 | 260 | 39 | 35 | 20 | 20 |
| low_bandwidth | wide | 0 | 7 | 0 | 7 | 0 | 0 |
| stale_state | tight | 0 | 8 | 2 | 5 | 2 | 2 |
| stale_state | mid | 0 | 0 | 0 | 0 | 0 | 0 |
| stale_state | wide | 0 | 0 | 0 | 0 | 0 | 0 |
| transient_fresh | tight | 0 | 8 | 1 | 5 | 1 | 1 |
| transient_fresh | mid | 0 | 0 | 0 | 0 | 0 | 0 |
| transient_fresh | wide | 0 | 0 | 0 | 0 | 0 | 0 |

## Đối chứng đơn giản (thêm 2026-11-16)

Mỗi ô: vi phạm / chấp nhận · từ chối nhầm · hạ cấp nhầm. "Luôn full" và "Luôn fixed" không dự báo và không từ chối; ba cột "Theo dòng" và "Lần trước cùng shard" chỉ thay phần dự báo, giữ nguyên quy tắc quyết định.

### `precommit`, không biên

| scenario | budget_level | Luôn full | Luôn fixed | Theo dòng (tĩnh) | Theo dòng (EWMA) | Lần trước cùng shard | Static | Adaptive |
|---|---|---|---|---|---|---|---|---|
| none | tight | 35/80 · 0 · 0 | 5/80 · 0 · 45 | 2/74 · 3 · 7 | 0/66 · 9 · 13 | 8/78 · 2 · 2 | 8/80 · 0 · 0 | 8/80 · 0 · 2 |
| none | mid | 32/80 · 0 · 0 | 0/80 · 0 · 48 | 0/80 · 0 · 0 | 0/80 · 0 · 0 | 0/80 · 0 · 0 | 0/80 · 0 · 0 | 0/80 · 0 · 0 |
| none | wide | 0/80 · 0 · 0 | 0/80 · 0 · 80 | 0/80 · 0 · 0 | 0/80 · 0 · 0 | 0/80 · 0 · 0 | 0/80 · 0 · 0 | 0/80 · 0 · 0 |
| slow_clients | tight | 60/80 · 0 · 0 | 24/80 · 0 · 20 | 42/74 · 6 · 0 | 14/56 · 4 · 0 | 14/60 · 0 · 0 | 52/80 · 0 · 0 | 27/66 · 0 · 0 |
| slow_clients | mid | 42/80 · 0 · 0 | 2/80 · 0 · 38 | 12/80 · 0 · 0 | 6/74 · 6 · 0 | 8/78 · 2 · 2 | 12/80 · 0 · 0 | 12/80 · 0 · 0 |
| slow_clients | wide | 0/80 · 0 · 0 | 0/80 · 0 · 80 | 0/80 · 0 · 0 | 0/80 · 0 · 6 | 0/80 · 0 · 0 | 0/80 · 0 · 0 | 0/80 · 0 · 0 |
| low_bandwidth | tight | 74/80 · 0 · 0 | 66/80 · 0 · 6 | 62/74 · 2 · 0 | 24/35 · 3 · 0 | 24/32 · 2 · 0 | 70/80 · 0 · 0 | 16/25 · 1 · 0 |
| low_bandwidth | mid | 70/80 · 0 · 0 | 60/80 · 0 · 10 | 60/80 · 0 · 0 | 22/42 · 0 · 0 | 18/38 · 0 · 0 | 60/80 · 0 · 0 | 10/30 · 0 · 0 |
| low_bandwidth | wide | 60/80 · 0 · 0 | 12/80 · 0 · 20 | 60/80 · 0 · 0 | 22/50 · 18 · 0 | 25/74 · 1 · 0 | 60/80 · 0 · 0 | 22/80 · 0 · 0 |
| stale_state | tight | 46/80 · 0 · 0 | 16/80 · 0 · 34 | 20/72 · 6 · 2 | 8/56 · 12 · 8 | 16/66 · 6 · 8 | 30/80 · 0 · 0 | 20/72 · 4 · 6 |
| stale_state | mid | 38/80 · 0 · 0 | 0/80 · 0 · 42 | 6/80 · 0 · 0 | 6/80 · 0 · 0 | 4/80 · 0 · 4 | 6/80 · 0 · 0 | 6/80 · 0 · 0 |
| stale_state | wide | 0/80 · 0 · 0 | 0/80 · 0 · 80 | 0/80 · 0 · 0 | 0/80 · 0 · 0 | 0/80 · 0 · 0 | 0/80 · 0 · 0 | 0/80 · 0 · 0 |
| transient_fresh | tight | 46/80 · 0 · 0 | 16/80 · 0 · 34 | 20/74 · 4 · 2 | 6/60 · 6 · 8 | 14/66 · 4 · 6 | 30/80 · 0 · 0 | 18/72 · 2 · 6 |
| transient_fresh | mid | 38/80 · 0 · 0 | 0/80 · 0 · 42 | 6/80 · 0 · 0 | 6/78 · 2 · 0 | 4/80 · 0 · 4 | 6/80 · 0 · 0 | 6/80 · 0 · 0 |
| transient_fresh | wide | 0/80 · 0 · 0 | 0/80 · 0 · 80 | 0/80 · 0 · 0 | 0/80 · 0 · 2 | 0/80 · 0 · 0 | 0/80 · 0 · 0 | 0/80 · 0 · 0 |

### `precommit`, có biên q90

| scenario | budget_level | Theo dòng (EWMA) + biên | Lần trước cùng shard + biên | Adaptive + biên | Adaptive + biên chấm lại |
|---|---|---|---|---|---|
| none | tight | 0/60 · 15 · 15 | 2/62 · 13 · 12 | 2/62 · 13 · 11 | 4/64 · 13 · 9 |
| none | mid | 0/80 · 0 · 0 | 0/80 · 0 · 0 | 0/80 · 0 · 0 | 0/80 · 0 · 0 |
| none | wide | 0/80 · 0 · 0 | 0/80 · 0 · 0 | 0/80 · 0 · 0 | 0/80 · 0 · 0 |
| slow_clients | tight | 4/38 · 18 · 0 | 8/44 · 14 · 0 | 10/54 · 6 · 0 | 12/56 · 4 · 0 |
| slow_clients | mid | 0/60 · 18 · 4 | 4/64 · 16 · 10 | 4/64 · 16 · 4 | 8/76 · 4 · 2 |
| slow_clients | wide | 0/80 · 0 · 20 | 0/80 · 0 · 14 | 0/80 · 0 · 16 | 0/80 · 0 · 0 |
| low_bandwidth | tight | 6/16 · 4 · 0 | 12/20 · 4 · 0 | 8/16 · 4 · 0 | 10/18 · 4 · 0 |
| low_bandwidth | mid | 6/26 · 0 · 0 | 8/28 · 0 · 0 | 6/26 · 0 · 0 | 6/26 · 0 · 0 |
| low_bandwidth | wide | 6/36 · 32 · 0 | 8/36 · 32 · 0 | 6/28 · 40 · 0 | 7/65 · 4 · 0 |
| stale_state | tight | 6/46 · 20 · 12 | 10/50 · 18 · 6 | 10/58 · 10 · 10 | 11/59 · 10 · 10 |
| stale_state | mid | 4/64 · 16 · 10 | 4/72 · 8 · 10 | 4/72 · 8 · 8 | 4/78 · 2 · 8 |
| stale_state | wide | 0/80 · 0 · 16 | 0/80 · 0 · 8 | 0/80 · 0 · 6 | 0/80 · 0 · 2 |
| transient_fresh | tight | 4/50 · 16 · 12 | 6/44 · 22 · 8 | 6/50 · 16 · 12 | 8/50 · 16 · 11 |
| transient_fresh | mid | 0/68 · 12 · 2 | 4/70 · 10 · 8 | 4/68 · 12 · 6 | 4/79 · 1 · 4 |
| transient_fresh | wide | 0/80 · 0 · 12 | 0/80 · 0 · 10 | 0/80 · 0 · 12 | 0/80 · 0 · 1 |

### `midcourse`, không biên

| scenario | budget_level | Luôn full | Luôn fixed | Theo dòng (tĩnh) | Theo dòng (EWMA) | Lần trước cùng shard | Static | Adaptive |
|---|---|---|---|---|---|---|---|---|
| none | tight | 35/80 · 0 · 0 | 5/80 · 0 · 45 | 2/74 · 3 · 4 | 0/66 · 9 · 4 | 5/78 · 2 · 4 | 5/80 · 0 · 4 | 5/80 · 0 · 4 |
| none | mid | 32/80 · 0 · 0 | 0/80 · 0 · 48 | 0/80 · 0 · 0 | 0/80 · 0 · 0 | 0/80 · 0 · 0 | 0/80 · 0 · 0 | 0/80 · 0 · 0 |
| none | wide | 0/80 · 0 · 0 | 0/80 · 0 · 80 | 0/80 · 0 · 0 | 0/80 · 0 · 0 | 0/80 · 0 · 0 | 0/80 · 0 · 0 | 0/80 · 0 · 0 |
| slow_clients | tight | 60/80 · 0 · 0 | 24/80 · 0 · 20 | 24/74 · 6 · 0 | 4/56 · 4 · 0 | 4/60 · 0 · 0 | 24/80 · 0 · 0 | 10/66 · 0 · 0 |
| slow_clients | mid | 42/80 · 0 · 0 | 2/80 · 0 · 38 | 2/80 · 0 · 0 | 2/74 · 6 · 0 | 2/78 · 2 · 0 | 2/80 · 0 · 0 | 2/80 · 0 · 0 |
| slow_clients | wide | 0/80 · 0 · 0 | 0/80 · 0 · 80 | 0/80 · 0 · 2 | 0/80 · 0 · 2 | 0/80 · 0 · 2 | 0/80 · 0 · 2 | 0/80 · 0 · 2 |
| low_bandwidth | tight | 74/80 · 0 · 0 | 66/80 · 0 · 6 | 64/74 · 2 · 0 | 24/33 · 3 · 0 | 22/32 · 2 · 0 | 68/80 · 0 · 0 | 14/25 · 1 · 0 |
| low_bandwidth | mid | 70/80 · 0 · 0 | 60/80 · 0 · 10 | 60/80 · 0 · 0 | 22/42 · 0 · 0 | 18/38 · 0 · 0 | 60/80 · 0 · 0 | 10/30 · 0 · 0 |
| low_bandwidth | wide | 60/80 · 0 · 0 | 12/80 · 0 · 20 | 12/80 · 0 · 0 | 0/50 · 18 · 0 | 7/74 · 1 · 0 | 12/80 · 0 · 0 | 12/80 · 0 · 0 |
| stale_state | tight | 46/80 · 0 · 0 | 16/80 · 0 · 34 | 16/72 · 6 · 2 | 6/56 · 12 · 2 | 10/66 · 6 · 2 | 18/80 · 0 · 2 | 14/72 · 4 · 2 |
| stale_state | mid | 38/80 · 0 · 0 | 0/80 · 0 · 42 | 0/80 · 0 · 0 | 0/80 · 0 · 0 | 0/80 · 0 · 0 | 0/80 · 0 · 0 | 0/80 · 0 · 0 |
| stale_state | wide | 0/80 · 0 · 0 | 0/80 · 0 · 80 | 0/80 · 0 · 0 | 0/80 · 0 · 0 | 0/80 · 0 · 0 | 0/80 · 0 · 0 | 0/80 · 0 · 0 |
| transient_fresh | tight | 46/80 · 0 · 0 | 16/80 · 0 · 34 | 16/74 · 4 · 2 | 4/60 · 6 · 2 | 8/66 · 4 · 2 | 18/80 · 0 · 2 | 12/72 · 2 · 2 |
| transient_fresh | mid | 38/80 · 0 · 0 | 0/80 · 0 · 42 | 0/80 · 0 · 0 | 0/78 · 2 · 0 | 0/80 · 0 · 0 | 0/80 · 0 · 0 | 0/80 · 0 · 0 |
| transient_fresh | wide | 0/80 · 0 · 0 | 0/80 · 0 · 80 | 0/80 · 0 · 0 | 0/80 · 0 · 0 | 0/80 · 0 · 0 | 0/80 · 0 · 0 | 0/80 · 0 · 0 |

### `midcourse`, có biên q90

| scenario | budget_level | Theo dòng (EWMA) + biên | Lần trước cùng shard + biên | Adaptive + biên | Adaptive + biên chấm lại |
|---|---|---|---|---|---|
| none | tight | 0/60 · 15 · 6 | 0/62 · 13 · 6 | 0/62 · 13 · 6 | 2/64 · 13 · 6 |
| none | mid | 0/80 · 0 · 0 | 0/80 · 0 · 0 | 0/80 · 0 · 0 | 0/80 · 0 · 0 |
| none | wide | 0/80 · 0 · 0 | 0/80 · 0 · 0 | 0/80 · 0 · 0 | 0/80 · 0 · 0 |
| slow_clients | tight | 0/38 · 18 · 0 | 2/42 · 16 · 0 | 2/50 · 8 · 0 | 4/56 · 4 · 0 |
| slow_clients | mid | 0/60 · 18 · 2 | 2/64 · 16 · 2 | 2/64 · 16 · 2 | 2/76 · 4 · 2 |
| slow_clients | wide | 0/80 · 0 · 2 | 0/80 · 0 · 2 | 0/80 · 0 · 2 | 0/80 · 0 · 2 |
| low_bandwidth | tight | 8/16 · 4 · 0 | 12/20 · 4 · 0 | 8/16 · 4 · 0 | 10/18 · 4 · 0 |
| low_bandwidth | mid | 6/26 · 0 · 0 | 8/28 · 0 · 0 | 6/26 · 0 · 0 | 6/26 · 0 · 0 |
| low_bandwidth | wide | 0/36 · 32 · 0 | 0/36 · 32 · 0 | 0/28 · 40 · 0 | 2/66 · 4 · 0 |
| stale_state | tight | 2/46 · 20 · 5 | 4/49 · 19 · 3 | 6/60 · 10 · 5 | 6/60 · 10 · 5 |
| stale_state | mid | 0/64 · 16 · 0 | 0/72 · 8 · 0 | 0/73 · 7 · 0 | 0/78 · 2 · 0 |
| stale_state | wide | 0/80 · 0 · 0 | 0/80 · 0 · 0 | 0/80 · 0 · 0 | 0/80 · 0 · 0 |
| transient_fresh | tight | 2/50 · 16 · 6 | 2/45 · 21 · 6 | 2/50 · 16 · 6 | 2/53 · 13 · 6 |
| transient_fresh | mid | 0/68 · 12 · 0 | 0/70 · 10 · 0 | 0/68 · 12 · 0 | 0/79 · 1 · 0 |
| transient_fresh | wide | 0/80 · 0 · 0 | 0/80 · 0 · 0 | 0/80 · 0 · 0 | 0/80 · 0 · 0 |

## Độ nhạy theo trọng số EWMA của Adaptive (thăm dò, thêm sau khi xem bảng đối chứng)

Mỗi ô: vi phạm / chấp nhận · từ chối nhầm · hạ cấp nhầm. α = 0,3 là giá trị của `shard_dt.py`; α = 1 giữ giá trị quan sát gần nhất của từng client và của phần server.

### `precommit`

| scenario | budget_level | Lần trước cùng shard | Adaptive | Adaptive α=0.6 | Adaptive α=1 | Adaptive + biên chấm lại | Adaptive α=1 + biên chấm lại |
|---|---|---|---|---|---|---|---|
| none | tight | 8/78 · 2 · 2 | 8/80 · 0 · 2 | 8/80 · 0 · 2 | 8/78 · 2 · 2 | 4/64 · 13 · 9 | 8/73 · 7 · 6 |
| none | mid | 0/80 · 0 · 0 | 0/80 · 0 · 0 | 0/80 · 0 · 0 | 0/80 · 0 · 0 | 0/80 · 0 · 0 | 0/80 · 0 · 0 |
| none | wide | 0/80 · 0 · 0 | 0/80 · 0 · 0 | 0/80 · 0 · 0 | 0/80 · 0 · 0 | 0/80 · 0 · 0 | 0/80 · 0 · 0 |
| slow_clients | tight | 14/60 · 0 · 0 | 27/66 · 0 · 0 | 20/62 · 0 · 0 | 14/60 · 0 · 0 | 12/56 · 4 · 0 | 14/60 · 0 · 0 |
| slow_clients | mid | 8/78 · 2 · 2 | 12/80 · 0 · 0 | 12/80 · 0 · 0 | 8/78 · 2 · 2 | 8/76 · 4 · 2 | 8/78 · 2 · 2 |
| slow_clients | wide | 0/80 · 0 · 0 | 0/80 · 0 · 0 | 0/80 · 0 · 0 | 0/80 · 0 · 0 | 0/80 · 0 · 0 | 0/80 · 0 · 0 |
| low_bandwidth | tight | 24/32 · 2 · 0 | 16/25 · 1 · 0 | 12/20 · 2 · 0 | 12/20 · 2 · 0 | 10/18 · 4 · 0 | 12/20 · 2 · 0 |
| low_bandwidth | mid | 18/38 · 0 · 0 | 10/30 · 0 · 0 | 6/26 · 0 · 0 | 4/24 · 0 · 0 | 6/26 · 0 · 0 | 4/24 · 0 · 0 |
| low_bandwidth | wide | 25/74 · 1 · 0 | 22/80 · 0 · 0 | 12/74 · 0 · 0 | 9/71 · 2 · 0 | 7/65 · 4 · 0 | 8/71 · 1 · 0 |
| stale_state | tight | 16/66 · 6 · 8 | 20/72 · 4 · 6 | 18/68 · 6 · 10 | 16/66 · 6 · 8 | 11/59 · 10 · 10 | 14/60 · 10 · 10 |
| stale_state | mid | 4/80 · 0 · 4 | 6/80 · 0 · 0 | 6/80 · 0 · 0 | 4/80 · 0 · 4 | 4/78 · 2 · 8 | 4/80 · 0 · 8 |
| stale_state | wide | 0/80 · 0 · 0 | 0/80 · 0 · 0 | 0/80 · 0 · 0 | 0/80 · 0 · 0 | 0/80 · 0 · 2 | 0/80 · 0 · 0 |
| transient_fresh | tight | 14/66 · 4 · 6 | 18/72 · 2 · 6 | 16/68 · 4 · 10 | 14/66 · 4 · 6 | 8/50 · 16 · 11 | 12/58 · 10 · 8 |
| transient_fresh | mid | 4/80 · 0 · 4 | 6/80 · 0 · 0 | 6/80 · 0 · 0 | 4/80 · 0 · 2 | 4/79 · 1 · 4 | 4/76 · 4 · 4 |
| transient_fresh | wide | 0/80 · 0 · 0 | 0/80 · 0 · 0 | 0/80 · 0 · 0 | 0/80 · 0 · 0 | 0/80 · 0 · 1 | 0/80 · 0 · 4 |

### `midcourse`

| scenario | budget_level | Lần trước cùng shard | Adaptive | Adaptive α=0.6 | Adaptive α=1 | Adaptive + biên chấm lại | Adaptive α=1 + biên chấm lại |
|---|---|---|---|---|---|---|---|
| none | tight | 5/78 · 2 · 4 | 5/80 · 0 · 4 | 5/80 · 0 · 4 | 5/78 · 2 · 4 | 2/64 · 13 · 6 | 5/73 · 7 · 5 |
| none | mid | 0/80 · 0 · 0 | 0/80 · 0 · 0 | 0/80 · 0 · 0 | 0/80 · 0 · 0 | 0/80 · 0 · 0 | 0/80 · 0 · 0 |
| none | wide | 0/80 · 0 · 0 | 0/80 · 0 · 0 | 0/80 · 0 · 0 | 0/80 · 0 · 0 | 0/80 · 0 · 0 | 0/80 · 0 · 0 |
| slow_clients | tight | 4/60 · 0 · 0 | 10/66 · 0 · 0 | 6/62 · 0 · 0 | 4/60 · 0 · 0 | 4/56 · 4 · 0 | 4/60 · 0 · 0 |
| slow_clients | mid | 2/78 · 2 · 0 | 2/80 · 0 · 0 | 2/80 · 0 · 0 | 2/78 · 2 · 0 | 2/76 · 4 · 2 | 2/78 · 2 · 2 |
| slow_clients | wide | 0/80 · 0 · 2 | 0/80 · 0 · 2 | 0/80 · 0 · 2 | 0/80 · 0 · 2 | 0/80 · 0 · 2 | 0/80 · 0 · 2 |
| low_bandwidth | tight | 22/32 · 2 · 0 | 14/25 · 1 · 0 | 10/20 · 2 · 0 | 10/20 · 2 · 0 | 10/18 · 4 · 0 | 10/20 · 2 · 0 |
| low_bandwidth | mid | 18/38 · 0 · 0 | 10/30 · 0 · 0 | 6/26 · 0 · 0 | 4/24 · 0 · 0 | 6/26 · 0 · 0 | 4/24 · 0 · 0 |
| low_bandwidth | wide | 7/74 · 1 · 0 | 12/80 · 0 · 0 | 6/74 · 0 · 0 | 5/71 · 2 · 0 | 2/66 · 4 · 0 | 4/71 · 1 · 0 |
| stale_state | tight | 10/66 · 6 · 2 | 14/72 · 4 · 2 | 12/68 · 6 · 2 | 10/66 · 6 · 2 | 6/60 · 10 · 5 | 6/60 · 10 · 5 |
| stale_state | mid | 0/80 · 0 · 0 | 0/80 · 0 · 0 | 0/80 · 0 · 0 | 0/80 · 0 · 0 | 0/78 · 2 · 0 | 0/80 · 0 · 0 |
| stale_state | wide | 0/80 · 0 · 0 | 0/80 · 0 · 0 | 0/80 · 0 · 0 | 0/80 · 0 · 0 | 0/80 · 0 · 0 | 0/80 · 0 · 0 |
| transient_fresh | tight | 8/66 · 4 · 2 | 12/72 · 2 · 2 | 10/68 · 4 · 2 | 8/66 · 4 · 2 | 2/53 · 13 · 6 | 4/58 · 10 · 6 |
| transient_fresh | mid | 0/80 · 0 · 0 | 0/80 · 0 · 0 | 0/80 · 0 · 0 | 0/80 · 0 · 0 | 0/79 · 1 · 0 | 0/76 · 4 · 0 |
| transient_fresh | wide | 0/80 · 0 · 0 | 0/80 · 0 · 0 | 0/80 · 0 · 0 | 0/80 · 0 · 0 | 0/80 · 0 · 0 | 0/80 · 0 · 0 |

## Sai số dự báo của shard_full (precommit, không biên, ngân sách rộng)

| env | lag | planner | requests | mae_seconds | rel_err_mean | rel_err_max |
|---|---|---|---|---|---|---|
| low_bandwidth | 0 | adaptive | 80 | 3.5 | 0.573 | 12.6 |
| low_bandwidth | 0 | adaptive_a0.6 | 80 | 2.08 | 0.421 | 11.7 |
| low_bandwidth | 0 | adaptive_a1 | 80 | 1.56 | 0.349 | 10.7 |
| low_bandwidth | 0 | last_shard | 80 | 4.2 | 1.23 | 11.3 |
| low_bandwidth | 0 | rows_ewma | 80 | 12.9 | 0.674 | 14.7 |
| low_bandwidth | 0 | rows_static | 80 | 12.5 | 5.12 | 19.8 |
| low_bandwidth | 0 | static | 80 | 12.6 | 4.78 | 17.3 |
| low_bandwidth | 3 | adaptive | 80 | 5.77 | 1.21 | 12.6 |
| low_bandwidth | 3 | adaptive_a0.6 | 80 | 4.44 | 1.05 | 11.7 |
| low_bandwidth | 3 | adaptive_a1 | 80 | 3.92 | 0.969 | 10.7 |
| low_bandwidth | 3 | last_shard | 80 | 4.59 | 1.28 | 11.3 |
| low_bandwidth | 3 | rows_ewma | 80 | 9.42 | 1.34 | 14 |
| low_bandwidth | 3 | rows_static | 80 | 12.5 | 5.01 | 18.9 |
| low_bandwidth | 3 | static | 80 | 12.5 | 4.69 | 16.6 |
| none | 0 | adaptive | 80 | 0.271 | 0.0168 | 0.28 |
| none | 0 | adaptive_a0.6 | 80 | 0.303 | 0.0138 | 0.301 |
| none | 0 | adaptive_a1 | 80 | 0.373 | 0.0152 | 0.331 |
| none | 0 | last_shard | 80 | 0.369 | 0.0183 | 0.33 |
| none | 0 | rows_ewma | 80 | 0.386 | 0.0121 | 0.362 |
| none | 0 | rows_static | 80 | 0.267 | 0.0481 | 0.375 |
| none | 0 | static | 80 | 0.251 | 0.026 | 0.272 |
| none | 3 | adaptive | 80 | 0.282 | 0.00657 | 0.28 |
| none | 3 | adaptive_a0.6 | 80 | 0.306 | 0.00458 | 0.301 |
| none | 3 | adaptive_a1 | 80 | 0.368 | 0.00873 | 0.33 |
| none | 3 | last_shard | 80 | 0.365 | 0.0114 | 0.33 |
| none | 3 | rows_ewma | 80 | 0.357 | 0.0137 | 0.376 |
| none | 3 | rows_static | 80 | 0.304 | 0.0356 | 0.375 |
| none | 3 | static | 80 | 0.273 | 0.0148 | 0.272 |
| slow_clients | 0 | adaptive | 80 | 1.11 | 0.273 | 1.09 |
| slow_clients | 0 | adaptive_a0.6 | 80 | 0.835 | 0.193 | 1.09 |
| slow_clients | 0 | adaptive_a1 | 80 | 0.785 | 0.143 | 1.09 |
| slow_clients | 0 | last_shard | 80 | 0.767 | 0.145 | 1.11 |
| slow_clients | 0 | rows_ewma | 80 | 0.971 | 0.0967 | 1.14 |
| slow_clients | 0 | rows_static | 80 | 1.67 | 0.497 | 1.37 |
| slow_clients | 0 | static | 80 | 1.69 | 0.459 | 1.09 |
| slow_clients | 3 | adaptive | 80 | 1.15 | 0.285 | 1.09 |
| slow_clients | 3 | adaptive_a0.6 | 80 | 0.887 | 0.21 | 1.09 |
| slow_clients | 3 | adaptive_a1 | 80 | 0.802 | 0.158 | 1.09 |
| slow_clients | 3 | last_shard | 80 | 0.798 | 0.159 | 1.11 |
| slow_clients | 3 | rows_ewma | 80 | 0.931 | 0.18 | 1.34 |
| slow_clients | 3 | rows_static | 80 | 1.62 | 0.479 | 1.37 |
| slow_clients | 3 | static | 80 | 1.64 | 0.443 | 1.09 |
| transient_slow | 0 | adaptive | 80 | 0.879 | 0.0985 | 1.01 |
| transient_slow | 0 | adaptive_a0.6 | 80 | 0.902 | 0.0587 | 1.01 |
| transient_slow | 0 | adaptive_a1 | 80 | 0.956 | 0.0493 | 1.01 |
| transient_slow | 0 | last_shard | 80 | 0.965 | 0.0528 | 1.02 |
| transient_slow | 0 | rows_ewma | 80 | 0.824 | 0.0199 | 1.11 |
| transient_slow | 0 | rows_static | 80 | 0.826 | 0.227 | 1.27 |
| transient_slow | 0 | static | 80 | 0.845 | 0.199 | 1.01 |
| transient_slow | 3 | adaptive | 80 | 0.894 | 0.0921 | 1.01 |
| transient_slow | 3 | adaptive_a0.6 | 80 | 0.954 | 0.0477 | 1.01 |
| transient_slow | 3 | adaptive_a1 | 80 | 1.03 | 0.0251 | 1.01 |
| transient_slow | 3 | last_shard | 80 | 1.03 | 0.029 | 1.02 |
| transient_slow | 3 | rows_ewma | 80 | 1.09 | 0.0475 | 1.23 |
| transient_slow | 3 | rows_static | 80 | 0.794 | 0.213 | 1.27 |
| transient_slow | 3 | static | 80 | 0.818 | 0.187 | 1.01 |
