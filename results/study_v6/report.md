# Study report: runs/study_v6_a, runs/study_v6_b, runs/study_v6_b2

## Validity

| scenario | role | cases | identifiable | distance_original | noise_mean_F | noise_sd_F |
|---|---|---|---|---|---|---|
| atypical | evaluation | 10 | 8 | 0.0414 | 0.0258 | 0.0051 |
| family | evaluation | 20 | 8 | 0.0383 | 0.0209 | 0.0052 |
| random | evaluation | 10 | 6 | 0.0077 | 0.0046 | 0.0007 |
| random_control | negative_control | 10 | 0 | 0.0036 | 0.0039 | 0.0007 |

## atypical (n valid = 8)

### Distance to retrain

| key | distance_candidate | delta_matched | delta_matched_ci | p_matched_holm | wins | losses | delta_retain_only | delta_retain_only_ci | p_retain_only_holm | beats_retain_only | passed | distance_R | noise_mean_R |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| legacy_retain_kd@final | 0.0268 | 0.0000 | [-0.0002, +0.0002] | 1.000 | 0 | 0 | -0.0001 | [-0.0001, +0.0000] | 0.766 | 0 | 0 | 0.0031 | 0.0026 |
| legacy_retain_kd@pre_recovery | 0.0412 | -0.0001 | [-0.0007, +0.0004] | 1.000 | 0 | 0 | -0.0145 | [-0.0244, -0.0046] | 0.062 | 0 | 0 | 0.0077 | 0.0026 |
| neggrad_plus_proxy@final | 0.0266 | 0.0004 | [-0.0004, +0.0011] | 1.000 | 0 | 0 | 0.0002 | [-0.0021, +0.0024] | 0.586 | 0 | 0 | 0.0030 | 0.0026 |
| neggrad_plus_proxy@pre_recovery | 0.0550 | 0.0011 | [-0.0038, +0.0059] | 1.000 | 2 | 0 | -0.0282 | [-0.0461, -0.0103] | 0.047 | 0 | 0 | 0.0119 | 0.0026 |
| uniform_bad_teacher_proxy@final | 0.0280 | 0.0024 | [-0.0015, +0.0064] | 1.000 | 3 | 0 | -0.0012 | [-0.0045, +0.0021] | 0.766 | 0 | 0 | 0.0032 | 0.0026 |
| uniform_bad_teacher_proxy@pre_recovery | 0.0404 | 0.0144 | [-0.0039, +0.0326] | 0.234 | 5 | 0 | -0.0137 | [-0.0187, -0.0086] | 0.047 | 0 | 0 | 0.0077 | 0.0026 |

### Utility at the validation operating point

| key | op_f1_delta | op_fpr_delta | utility_preserved | family_alert | family_alert_original | family_alert_retrain |
|---|---|---|---|---|---|---|
| legacy_retain_kd@final | 0.0070 | 0.0008 | 0.88 | — | — | — |
| legacy_retain_kd@pre_recovery | 0.0000 | -0.0000 | 0.75 | — | — | — |
| neggrad_plus_proxy@final | 0.0078 | 0.0008 | 0.88 | — | — | — |
| neggrad_plus_proxy@pre_recovery | -0.0039 | 0.0001 | 0.75 | — | — | — |
| uniform_bad_teacher_proxy@final | 0.0072 | 0.0007 | 0.88 | — | — | — |
| uniform_bad_teacher_proxy@pre_recovery | -0.0002 | 0.0003 | 0.75 | — | — | — |

## family (n valid = 8)

### Distance to retrain

| key | distance_candidate | delta_matched | delta_matched_ci | p_matched_holm | wins | losses | delta_retain_only | delta_retain_only_ci | p_retain_only_holm | beats_retain_only | passed | distance_R | noise_mean_R |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| legacy_retain_kd@final | 0.0439 | -0.0001 | [-0.0003, +0.0001] | 1.000 | 0 | 0 | 0.0000 | [-0.0001, +0.0001] | 1.000 | 0 | 0 | 0.0042 | 0.0037 |
| legacy_retain_kd@pre_recovery | 0.0617 | -0.0018 | [-0.0046, +0.0010] | 1.000 | 0 | 0 | -0.0178 | [-0.0284, -0.0073] | 0.047 | 0 | 0 | 0.0049 | 0.0037 |
| neggrad_plus_proxy@final | 0.0332 | 0.0058 | [+0.0001, +0.0116] | 0.234 | 3 | 0 | 0.0106 | [-0.0022, +0.0234] | 0.547 | 5 | 1 | 0.0045 | 0.0037 |
| neggrad_plus_proxy@pre_recovery | 0.0820 | 0.0002 | [-0.0145, +0.0149] | 1.000 | 2 | 1 | -0.0381 | [-0.0871, +0.0109] | 0.547 | 1 | 1 | 0.0184 | 0.0037 |
| uniform_bad_teacher_proxy@final | 0.0371 | 0.0029 | [-0.0102, +0.0161] | 1.000 | 2 | 2 | 0.0068 | [-0.0051, +0.0187] | 0.547 | 4 | 0 | 0.0047 | 0.0037 |
| uniform_bad_teacher_proxy@pre_recovery | 0.0483 | 0.0525 | [-0.0197, +0.1246] | 0.977 | 4 | 2 | -0.0044 | [-0.0258, +0.0170] | 1.000 | 2 | 1 | 0.0086 | 0.0037 |

### Utility at the validation operating point

| key | op_f1_delta | op_fpr_delta | utility_preserved | family_alert | family_alert_original | family_alert_retrain |
|---|---|---|---|---|---|---|
| legacy_retain_kd@final | 0.0006 | 0.0019 | 1.00 | 0.3368 | 0.3641 | 0.3014 |
| legacy_retain_kd@pre_recovery | 0.0001 | 0.0000 | 1.00 | 0.3651 | 0.3641 | 0.3014 |
| neggrad_plus_proxy@final | 0.0017 | 0.0021 | 1.00 | 0.3289 | 0.3641 | 0.3014 |
| neggrad_plus_proxy@pre_recovery | 0.0002 | 0.0009 | 1.00 | 0.3628 | 0.3641 | 0.3014 |
| uniform_bad_teacher_proxy@final | 0.0016 | 0.0025 | 1.00 | 0.3218 | 0.3641 | 0.3014 |
| uniform_bad_teacher_proxy@pre_recovery | -0.0007 | 0.0020 | 1.00 | 0.3177 | 0.3641 | 0.3014 |

## random (n valid = 6)

### Distance to retrain

| key | distance_candidate | delta_matched | delta_matched_ci | p_matched_holm | wins | losses | delta_retain_only | delta_retain_only_ci | p_retain_only_holm | beats_retain_only | passed | distance_R | noise_mean_R |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| legacy_retain_kd@final | 0.0072 | -0.0000 | [-0.0001, +0.0001] | 1.000 | 0 | 0 | 0.0000 | [-0.0000, +0.0001] | 1.000 | 0 | 0 | 0.0070 | 0.0042 |
| legacy_retain_kd@pre_recovery | 0.0118 | 0.0015 | [-0.0019, +0.0050] | 0.188 | 1 | 0 | -0.0046 | [-0.0106, +0.0014] | 0.188 | 0 | 0 | 0.0116 | 0.0042 |
| neggrad_plus_proxy@final | 0.0073 | 0.0003 | [-0.0005, +0.0012] | 1.000 | 3 | 0 | -0.0001 | [-0.0007, +0.0006] | 1.000 | 0 | 0 | 0.0077 | 0.0042 |
| neggrad_plus_proxy@pre_recovery | 0.0292 | 0.0006 | [-0.0004, +0.0016] | 1.000 | 1 | 0 | -0.0220 | [-0.0616, +0.0176] | 0.188 | 0 | 0 | 0.0496 | 0.0042 |
| uniform_bad_teacher_proxy@final | 0.0082 | 0.0003 | [-0.0014, +0.0021] | 1.000 | 2 | 0 | -0.0010 | [-0.0018, -0.0002] | 0.188 | 0 | 0 | 0.0079 | 0.0042 |
| uniform_bad_teacher_proxy@pre_recovery | 0.0220 | 0.0078 | [-0.0050, +0.0207] | 0.188 | 6 | 0 | -0.0148 | [-0.0419, +0.0124] | 0.188 | 0 | 0 | 0.0364 | 0.0042 |

### Utility at the validation operating point

| key | op_f1_delta | op_fpr_delta | utility_preserved | family_alert | family_alert_original | family_alert_retrain |
|---|---|---|---|---|---|---|
| legacy_retain_kd@final | 0.0095 | 0.0005 | 1.00 | — | — | — |
| legacy_retain_kd@pre_recovery | 0.0000 | -0.0000 | 1.00 | — | — | — |
| neggrad_plus_proxy@final | 0.0117 | 0.0018 | 1.00 | — | — | — |
| neggrad_plus_proxy@pre_recovery | -0.0028 | 0.0013 | 0.83 | — | — | — |
| uniform_bad_teacher_proxy@final | 0.0148 | 0.0028 | 1.00 | — | — | — |
| uniform_bad_teacher_proxy@pre_recovery | 0.0085 | 0.0042 | 1.00 | — | — | — |

## ALL (n valid = 22)

### Distance to retrain

| key | distance_candidate | delta_matched | delta_matched_ci | p_matched_holm | wins | losses | delta_retain_only | delta_retain_only_ci | p_retain_only_holm | beats_retain_only | passed | distance_R | noise_mean_R |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| legacy_retain_kd@final | 0.0277 | -0.0000 | [-0.0001, +0.0000] | 0.624 | 0 | 0 | -0.0000 | [-0.0001, +0.0000] | 0.949 | 0 | 0 | 0.0046 | 0.0034 |
| legacy_retain_kd@pre_recovery | 0.0407 | -0.0003 | [-0.0016, +0.0010] | 0.849 | 1 | 0 | -0.0130 | [-0.0182, -0.0078] | 0.000 | 0 | 0 | 0.0078 | 0.0034 |
| neggrad_plus_proxy@final | 0.0238 | 0.0023 | [+0.0002, +0.0045] | 0.042 | 6 | 0 | 0.0039 | [-0.0007, +0.0085] | 0.052 | 5 | 1 | 0.0048 | 0.0034 |
| neggrad_plus_proxy@pre_recovery | 0.0578 | 0.0006 | [-0.0041, +0.0054] | 0.630 | 5 | 1 | -0.0301 | [-0.0483, -0.0119] | 0.000 | 1 | 1 | 0.0245 | 0.0034 |
| uniform_bad_teacher_proxy@final | 0.0259 | 0.0020 | [-0.0022, +0.0063] | 0.776 | 7 | 2 | 0.0018 | [-0.0024, +0.0059] | 0.926 | 4 | 0 | 0.0050 | 0.0034 |
| uniform_bad_teacher_proxy@pre_recovery | 0.0383 | 0.0264 | [+0.0018, +0.0511] | 0.015 | 15 | 2 | -0.0106 | [-0.0196, -0.0016] | 0.030 | 2 | 1 | 0.0159 | 0.0034 |

### Utility at the validation operating point

| key | op_f1_delta | op_fpr_delta | utility_preserved | family_alert | family_alert_original | family_alert_retrain |
|---|---|---|---|---|---|---|
| legacy_retain_kd@final | 0.0054 | 0.0011 | 0.95 | 0.3368 | 0.3641 | 0.3014 |
| legacy_retain_kd@pre_recovery | 0.0000 | -0.0000 | 0.91 | 0.3651 | 0.3641 | 0.3014 |
| neggrad_plus_proxy@final | 0.0067 | 0.0016 | 0.95 | 0.3289 | 0.3641 | 0.3014 |
| neggrad_plus_proxy@pre_recovery | -0.0021 | 0.0007 | 0.86 | 0.3628 | 0.3641 | 0.3014 |
| uniform_bad_teacher_proxy@final | 0.0072 | 0.0019 | 0.95 | 0.3218 | 0.3641 | 0.3014 |
| uniform_bad_teacher_proxy@pre_recovery | 0.0020 | 0.0020 | 0.91 | 0.3177 | 0.3641 | 0.3014 |
