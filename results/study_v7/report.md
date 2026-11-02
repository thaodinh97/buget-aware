# Study report: runs/study_v7_a, runs/study_v7_b

## Validity

| scenario | role | cases | identifiable | distance_original | noise_mean_F | noise_sd_F |
|---|---|---|---|---|---|---|
| family | evaluation | 70 | 38 | 0.0187 | 0.0108 | 0.0027 |
| random_control | negative_control | 4 | 1 | 0.0032 | 0.0030 | 0.0003 |

## family (n valid = 38)

### Distance to retrain

| key | distance_candidate | delta_matched | delta_matched_ci | p_matched_holm | wins | losses | delta_retain_only | delta_retain_only_ci | p_retain_only_holm | beats_retain_only | passed | distance_R | noise_mean_R |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| legacy_retain_kd@final | 0.0127 | 0.0000 | [-0.0000, +0.0000] | 0.510 | 0 | 0 | 0.0000 | [-0.0000, +0.0000] | 0.378 | 0 | 0 | 0.0041 | 0.0033 |
| legacy_retain_kd@pre_recovery | 0.0161 | -0.0008 | [-0.0014, -0.0002] | 0.006 | 0 | 1 | -0.0034 | [-0.0056, -0.0012] | 0.000 | 0 | 0 | 0.0053 | 0.0033 |
| neggrad_plus_proxy@final | 0.0114 | 0.0004 | [+0.0001, +0.0008] | 0.000 | 7 | 0 | 0.0013 | [+0.0006, +0.0021] | 0.000 | 13 | 6 | 0.0048 | 0.0033 |
| neggrad_plus_proxy@pre_recovery | 0.0514 | -0.0061 | [-0.0142, +0.0021] | 0.561 | 9 | 15 | -0.0388 | [-0.0788, +0.0013] | 0.004 | 6 | 2 | 0.0277 | 0.0033 |
| uniform_bad_teacher_proxy@final | 0.0119 | 0.0004 | [-0.0002, +0.0010] | 0.376 | 5 | 4 | 0.0007 | [-0.0001, +0.0016] | 0.012 | 5 | 2 | 0.0051 | 0.0033 |
| uniform_bad_teacher_proxy@pre_recovery | 0.0167 | 0.0421 | [+0.0080, +0.0763] | 0.000 | 26 | 7 | -0.0040 | [-0.0139, +0.0059] | 0.677 | 6 | 5 | 0.0109 | 0.0033 |

### Utility at the validation operating point

| key | op_f1_delta | op_fpr_delta | utility_preserved | family_alert | family_alert_original | family_alert_retrain |
|---|---|---|---|---|---|---|
| legacy_retain_kd@final | 0.0011 | -0.0010 | 0.97 | 0.8535 | 0.8650 | 0.8427 |
| legacy_retain_kd@pre_recovery | 0.0004 | -0.0002 | 0.95 | 0.8633 | 0.8650 | 0.8427 |
| neggrad_plus_proxy@final | 0.0044 | -0.0007 | 1.00 | 0.8488 | 0.8650 | 0.8427 |
| neggrad_plus_proxy@pre_recovery | 0.0065 | -0.0002 | 1.00 | 0.8543 | 0.8650 | 0.8427 |
| uniform_bad_teacher_proxy@final | 0.0052 | -0.0004 | 1.00 | 0.8509 | 0.8650 | 0.8427 |
| uniform_bad_teacher_proxy@pre_recovery | 0.0048 | 0.0002 | 0.97 | 0.8558 | 0.8650 | 0.8427 |

### Utility non-inferiority (F1 drop < 0.03, FPR increase < 0.02, on 95% CI bounds)

| key | n | op_f1_delta | f1_delta_ci_low | f1_noninferior | op_fpr_delta | fpr_delta_ci_high | fpr_noninferior |
|---|---|---|---|---|---|---|---|
| legacy_retain_kd@final | 38 | 0.0011 | -0.0022 | yes | -0.0010 | -0.0003 | yes |
| legacy_retain_kd@pre_recovery | 38 | 0.0004 | -0.0004 | yes | -0.0002 | 0.0001 | yes |
| neggrad_plus_proxy@final | 38 | 0.0044 | 0.0000 | yes | -0.0007 | 0.0005 | yes |
| neggrad_plus_proxy@pre_recovery | 38 | 0.0065 | 0.0014 | yes | -0.0002 | 0.0011 | yes |
| uniform_bad_teacher_proxy@final | 38 | 0.0052 | 0.0004 | yes | -0.0004 | 0.0007 | yes |
| uniform_bad_teacher_proxy@pre_recovery | 38 | 0.0048 | -0.0003 | yes | 0.0002 | 0.0012 | yes |

### Share of cases within retrain noise (distance <= leave-one-out retrain maximum)

| key | n | within_noise | retain_only_within_noise | original_within_noise |
|---|---|---|---|---|
| legacy_retain_kd@final | 38 | 0.08 | 0.08 | 0.00 |
| legacy_retain_kd@pre_recovery | 38 | 0.00 | 0.08 | 0.00 |
| neggrad_plus_proxy@final | 38 | 0.42 | 0.08 | 0.00 |
| neggrad_plus_proxy@pre_recovery | 38 | 0.16 | 0.08 | 0.00 |
| uniform_bad_teacher_proxy@final | 38 | 0.24 | 0.08 | 0.00 |
| uniform_bad_teacher_proxy@pre_recovery | 38 | 0.21 | 0.08 | 0.00 |

## ALL (n valid = 38)

### Distance to retrain

| key | distance_candidate | delta_matched | delta_matched_ci | p_matched_holm | wins | losses | delta_retain_only | delta_retain_only_ci | p_retain_only_holm | beats_retain_only | passed | distance_R | noise_mean_R |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| legacy_retain_kd@final | 0.0127 | 0.0000 | [-0.0000, +0.0000] | 0.510 | 0 | 0 | 0.0000 | [-0.0000, +0.0000] | 0.378 | 0 | 0 | 0.0041 | 0.0033 |
| legacy_retain_kd@pre_recovery | 0.0161 | -0.0008 | [-0.0014, -0.0002] | 0.006 | 0 | 1 | -0.0034 | [-0.0056, -0.0012] | 0.000 | 0 | 0 | 0.0053 | 0.0033 |
| neggrad_plus_proxy@final | 0.0114 | 0.0004 | [+0.0001, +0.0008] | 0.000 | 7 | 0 | 0.0013 | [+0.0006, +0.0021] | 0.000 | 13 | 6 | 0.0048 | 0.0033 |
| neggrad_plus_proxy@pre_recovery | 0.0514 | -0.0061 | [-0.0142, +0.0021] | 0.561 | 9 | 15 | -0.0388 | [-0.0788, +0.0013] | 0.004 | 6 | 2 | 0.0277 | 0.0033 |
| uniform_bad_teacher_proxy@final | 0.0119 | 0.0004 | [-0.0002, +0.0010] | 0.376 | 5 | 4 | 0.0007 | [-0.0001, +0.0016] | 0.012 | 5 | 2 | 0.0051 | 0.0033 |
| uniform_bad_teacher_proxy@pre_recovery | 0.0167 | 0.0421 | [+0.0080, +0.0763] | 0.000 | 26 | 7 | -0.0040 | [-0.0139, +0.0059] | 0.677 | 6 | 5 | 0.0109 | 0.0033 |

### Utility at the validation operating point

| key | op_f1_delta | op_fpr_delta | utility_preserved | family_alert | family_alert_original | family_alert_retrain |
|---|---|---|---|---|---|---|
| legacy_retain_kd@final | 0.0011 | -0.0010 | 0.97 | 0.8535 | 0.8650 | 0.8427 |
| legacy_retain_kd@pre_recovery | 0.0004 | -0.0002 | 0.95 | 0.8633 | 0.8650 | 0.8427 |
| neggrad_plus_proxy@final | 0.0044 | -0.0007 | 1.00 | 0.8488 | 0.8650 | 0.8427 |
| neggrad_plus_proxy@pre_recovery | 0.0065 | -0.0002 | 1.00 | 0.8543 | 0.8650 | 0.8427 |
| uniform_bad_teacher_proxy@final | 0.0052 | -0.0004 | 1.00 | 0.8509 | 0.8650 | 0.8427 |
| uniform_bad_teacher_proxy@pre_recovery | 0.0048 | 0.0002 | 0.97 | 0.8558 | 0.8650 | 0.8427 |

### Utility non-inferiority (F1 drop < 0.03, FPR increase < 0.02, on 95% CI bounds)

| key | n | op_f1_delta | f1_delta_ci_low | f1_noninferior | op_fpr_delta | fpr_delta_ci_high | fpr_noninferior |
|---|---|---|---|---|---|---|---|
| legacy_retain_kd@final | 38 | 0.0011 | -0.0022 | yes | -0.0010 | -0.0003 | yes |
| legacy_retain_kd@pre_recovery | 38 | 0.0004 | -0.0004 | yes | -0.0002 | 0.0001 | yes |
| neggrad_plus_proxy@final | 38 | 0.0044 | 0.0000 | yes | -0.0007 | 0.0005 | yes |
| neggrad_plus_proxy@pre_recovery | 38 | 0.0065 | 0.0014 | yes | -0.0002 | 0.0011 | yes |
| uniform_bad_teacher_proxy@final | 38 | 0.0052 | 0.0004 | yes | -0.0004 | 0.0007 | yes |
| uniform_bad_teacher_proxy@pre_recovery | 38 | 0.0048 | -0.0003 | yes | 0.0002 | 0.0012 | yes |

### Share of cases within retrain noise (distance <= leave-one-out retrain maximum)

| key | n | within_noise | retain_only_within_noise | original_within_noise |
|---|---|---|---|---|
| legacy_retain_kd@final | 38 | 0.08 | 0.08 | 0.00 |
| legacy_retain_kd@pre_recovery | 38 | 0.00 | 0.08 | 0.00 |
| neggrad_plus_proxy@final | 38 | 0.42 | 0.08 | 0.00 |
| neggrad_plus_proxy@pre_recovery | 38 | 0.16 | 0.08 | 0.00 |
| uniform_bad_teacher_proxy@final | 38 | 0.24 | 0.08 | 0.00 |
| uniform_bad_teacher_proxy@pre_recovery | 38 | 0.21 | 0.08 | 0.00 |

## Pre-registered confirmatory analysis

| role | scenario | key | n | enough_cases | delta_retain_only | ci_low_retain_only | p_retain_only | delta_matched | ci_low_matched | p_matched | share_beyond_margin | practical | utility_noninferior | within_noise | confirmed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| primary | family | neggrad_plus_proxy@final | 38 | yes | 0.0013 | 0.0006 | 0.0000 | 0.0004 | 0.0001 | 0.0001 | 0.34 | no | yes | 0.42 | no |
| secondary | family | uniform_bad_teacher_proxy@final | 38 | yes | 0.0007 | -0.0001 | 0.0040 | 0.0004 | -0.0002 | 0.1252 | 0.13 | no | yes | 0.24 | no |
