# FPR của detector: kết quả chọn ngưỡng và ablation feature

**Protocol:** [fpr_threshold_plan.md](fpr_threshold_plan.md), viết trước khi chạy; phần ablation feature được thêm sau khi xem kết quả đầu, và ghi trước khi chạy ablation. **Kết quả:** `results/fpr_threshold/` (dữ liệu chính) và `results/fpr_threshold_protocolfields/` (ablation), mỗi thư mục có `run.json`, `thresholds.csv`, `curves.csv`, `bayes.json` và `posthoc_*.csv`; chẩn đoán D1–D3 ở `results/fpr_ceiling/` và `results/fpr_ceiling_protocolfields/`.

**Thực hiện:** 2026-11-18, CPU, 2 thread. 5 seed train 50042–50046 (seed v2, đã xem) × S ∈ {1, 4, 5}, trạng thái ban đầu sau `fit`. Chỉ đọc train, validation, calibration; **test không được mở**. Mọi số là trung bình 5 seed trên calibration, trừ khi ghi khác. Không có gate.

## Kết luận
- **FPR cao chủ yếu do cách chuẩn hóa feature số, không phải do ngưỡng hay do thiếu feature** (chẩn đoán D1–D3, mục cuối). Tiền xử lý chỉ chuẩn hóa z-score, nên các flow nhỏ (một nửa số dòng) có số byte sau chuẩn hóa gần như bằng nhau. Chỉ cần lấy `log1p` trước khi chuẩn hóa, cùng 11 feature, cùng split, partition và seed, FPR ở ngưỡng mặc định đã giảm từ 0,320 xuống 0,078 (S = 1), macro-F1 tăng từ 0,881 lên 0,960.
- **Bộ 11 feature đủ để tách gần như hoàn toàn trên split này:** gradient boosting train tập trung (tham số mặc định, không FL) đạt FPR 0,018 với recall 0,997, ROC-AUC 0,999. Khoảng cách còn lại giữa MLP FL (`log1p`, FPR 0,05–0,08 tùy quy tắc) và mức này nằm ở model/cách train, không ở feature.
- **Với z-score hiện tại, chọn ngưỡng chỉ đổi lỗi này lấy lỗi khác:** muốn recall attack 0,95 thì FPR phải khoảng 0,29–0,30. Chọn ngưỡng trên validation hạ FPR xuống khoảng 7 %, nhưng recall giảm từ 0,999 xuống khoảng 0,87 (DDoS 0,70, DoS 0,46) và macro-F1 giảm khoảng 0,03.
- **75 % false positive là traffic normal `udp/dns/SF`,** cùng vùng điểm số với DDoS/DoS `udp/dns/SF` (phân tích post hoc). Đây là triệu chứng của việc số byte bị ép phẳng, không phải bằng chứng là thông tin phân biệt không có trong 11 feature.
- **Thêm 16 trường giao thức (DNS, HTTP, SSL)** cũng hạ FPR mặc định xuống 0,089, nhưng sau khi đã lấy `log1p` thì gần như không giúp thêm. Các trường DNS dạng phân loại chủ yếu bù cho số byte bị ép phẳng. Thêm chúng đổi split và mang rủi ro đặc điểm riêng của testbed, nên **không cần thiết**.
- **Chọn lại ngưỡng sau mỗi lần unlearn rất rẻ** (10–60 ms) và giữ được phạm vi chính xác của chính sách, nếu ngưỡng được tính lại từ model vừa train lại (lập luận trong protocol). Nó cũng chặn được các seed rơi vào chế độ FPR cao ở ngưỡng mặc định.

> **Sửa kết luận (2026-11-18):** bản đầu của tài liệu này kết luận "FPR nằm ở bộ feature", dựa trên ablation thêm trường giao thức. Chẩn đoán D1–D3, chạy sau đó, cho thấy kết luận đó sai: cùng bộ feature, chỉ đổi cách chuẩn hóa, đã cho mức cải thiện tương đương.

## Dữ liệu chính: theo quy tắc chọn ngưỡng
| S | Quy tắc | FPR | Recall attack | Macro-F1 |
|---|---|---|---|---|
| 1 | `default` | 0,320 | 0,999 | 0,881 |
| | `max_f1` | 0,302 | 0,992 | 0,882 |
| | `youden` | 0,067 | 0,871 | 0,854 |
| | `fpr10` | 0,124 | 0,882 | 0,844 |
| | `fpr05` | 0,072 | 0,872 | 0,853 |
| 4 | `default` | 0,318 | 0,999 | 0,882 |
| | `max_f1` | 0,316 | 0,995 | 0,879 |
| | `youden` | 0,067 | 0,859 | 0,844 |
| | `fpr10` | 0,128 | 0,868 | 0,831 |
| | `fpr05` | 0,078 | 0,864 | 0,844 |
| 5 | `default` | 0,323 | 0,999 | 0,880 |
| | `max_f1` | 0,315 | 0,994 | 0,879 |
| | `youden` | 0,108 (0,062–0,269) | 0,880 | 0,848 |
| | `fpr10` | 0,112 | 0,867 | 0,836 |
| | `fpr05` | 0,075 | 0,859 | 0,841 |

- `max_f1` gần như không đổi gì: macro-F1 ưa điểm recall cao.
- **Có lệch giữa validation và calibration:** mục tiêu FPR 0,05 trên validation cho 0,072–0,078 trên calibration; mục tiêu 0,10 cho 0,112–0,128.
- `fpr10` có FPR cao hơn `fpr05` mà macro-F1 lại thấp hơn. Ở S = 1, ngưỡng của hai quy tắc chỉ cách nhau khoảng 3·10⁻⁴ trên thang margin, nhưng FPR cách nhau 5 điểm phần trăm: một khối lớn dòng normal có gần như cùng một điểm số (xem phân tích dưới).
- `youden` ở S = 5 không ổn định: một seed chọn ngưỡng ở phía FPR cao (0,269).

**Không phụ thuộc ngưỡng (calibration):**

| S | ROC-AUC | Average precision | FPR tại recall 0,95 | FPR tại recall 0,99 |
|---|---|---|---|---|
| 1 | 0,953 | 0,983 | 0,293 | 0,298 |
| 4 | 0,952 | 0,982 | 0,300 | 0,318 |
| 5 | 0,944 | 0,977 | 0,297 | 0,322 |

**Lỗi không tránh được theo vector trùng khớp (F3):** lỗi Bayes trên mẫu là 0,09 % (train, 12 vector có cả hai nhãn, 5.671 dòng), 0,31 % (validation) và 0 (calibration); "FPR sàn" theo đa số là 0,18 % / 0,47 % / 0. Vậy lỗi không đến từ các dòng giống hệt nhau mà từ các dòng **gần** giống nhau, mà phép đo này không thấy.

**Chi phí chọn lại ngưỡng (F4):** forward trên validation cộng tìm ngưỡng mất trung bình 11 ms (S = 1), 26 ms (S = 4), 40 ms (S = 5), so với khoảng 17 giây cho một lần `fit`.

## Phân tích post hoc: false positive đến từ đâu
`scripts/shard/fpr_cluster_diagnosis.py`, **thêm sau khi xem kết quả**. Train lại đúng model S = 1, seed 50042 (ngưỡng khớp `thresholds.csv`), nối dòng calibration về CSV gốc qua `row_id`. Một model, một seed: mô tả, không phải kiểm định.

| Nhóm normal (`proto`/`service`/`conn_state`) | Số dòng | FPR `default` | FPR `fpr05` | Phần trong FP `default` |
|---|---|---|---|---|
| `udp/dns/SF` | 1.042 | 0,999 | 0,034 | 75,3 % |
| `tcp/http/SF` | 167 | 1,000 | 0,988 | 12,1 % |
| `tcp/ssl/SF` | 43 | 1,000 | 0,605 | 3,1 % |
| `tcp/-/S0` | 1.865 | 0,014 | 0,014 | 2,0 % |

- Ở ngưỡng mặc định có 1.382 trên 4.313 dòng normal bị gắn attack.
- Trong dải margin giữa ngưỡng `fpr10` và `fpr05` có 588 dòng normal, 523 DDoS và 187 DoS, **tất cả đều `udp/dns/SF`**. Đó là khối dòng có cùng điểm số nói ở trên.
- Hạ FPR bằng ngưỡng là đổi các dòng DNS normal lấy các dòng DNS tấn công. Recall theo loại ở `fpr05`: DDoS 0,70, DoS 0,46, MITM 0,36, XSS 0,85, scanning 0,88; injection, password, backdoor ≥ 0,92.
- Trong CSV gốc, các trường DNS tách được một phần các nhóm này. Ví dụ, ở `udp/dns/SF`, normal có nhiều `dns_qtype` (1, 28, 12, 2, 43), còn attack gần như chỉ có 1 và 28; DDoS chủ yếu có `dns_rcode` = 3.

## Ablation: thêm trường giao thức
Bản dữ liệu `connected_v3_auxfit_protocolfields_seed42` (`configs/data_connected_auxfit_protocolfields.json`): 11 feature cũ cộng 14 trường phân loại DNS/HTTP/SSL/weird và 2 trường số độ dài body HTTP, 118 chiều sau mã hóa. Các trường chuỗi nhiều giá trị (`dns_query`, `http_uri`, `http_user_agent`, …) bị loại có chủ ý. Audit tiền xử lý đạt: `training_ready`, không có overlap nhóm, chữ ký, chiều ngược hay mã hóa giữa các split. Thêm feature đổi cách gom nhóm, nên split khác bản chính; partition seed 42 sinh client suy biến, nên theo quy tắc ghi trước, dùng **partition seed 43**. Vì vậy so sánh **không ghép cặp theo dòng**.

| S | Quy tắc | FPR | Recall attack | Macro-F1 |
|---|---|---|---|---|
| 1 | `default` | 0,089 | 0,994 | 0,964 |
| | `max_f1` | 0,090 | 0,997 | 0,967 |
| | `youden` | 0,075 | 0,989 | 0,963 |
| | `fpr05` | 0,082 | 0,985 | 0,957 |
| 4 | `default` | 0,093 | 0,987 | 0,955 |
| | `max_f1` | 0,098 | 0,998 | 0,965 |
| | `youden` | 0,071 | 0,983 | 0,958 |
| | `fpr05` | 0,082 | 0,985 | 0,956 |
| 5 | `default` | 0,182 (0,072–0,561) | 0,994 | 0,924 |
| | `max_f1` | 0,078 | 0,995 | 0,968 |
| | `youden` | 0,078 | 0,995 | 0,968 |
| | `fpr05` | 0,076 | 0,990 | 0,964 |

`fpr10` cho FPR 0,105–0,116; bảng đầy đủ ở `results/fpr_threshold_protocolfields/thresholds.csv`.

| S | ROC-AUC | Average precision | FPR tại recall 0,95 | FPR tại recall 0,99 |
|---|---|---|---|---|
| 1 | 0,982 | 0,991 | 0,065 | 0,087 |
| 4 | 0,979 | 0,987 | 0,068 | 0,095 |
| 5 | 0,980 | 0,988 | 0,067 | 0,076 |

- **FPR tại recall 0,95 giảm từ khoảng 0,30 xuống khoảng 0,07:** vách trên đường ROC gần như biến mất.
- **Phân tích post hoc cùng cách** (S = 1, seed 50042): FP ở ngưỡng mặc định còn 366 trên 4.389 dòng normal. `udp/dns/SF` normal chỉ còn FPR 0,147 (trước là 0,999); recall DDoS vẫn 0,998.
- **Phần còn lại:** `tcp/http/SF` normal vẫn có FPR 0,711 (166 dòng; 32 % số FP), và recall DoS chỉ còn 0,67 ở ngưỡng mặc định.
- **Ngưỡng mặc định ở S = 5 không ổn định:** seed 50046 có FPR 0,561, tức chế độ FPR cao (> 0,45). Mọi quy tắc chọn ngưỡng trên validation đều đưa seed này về khoảng 0,08. Đây là một lý do để luôn chọn lại ngưỡng thay vì giữ 0,5.
- Một lần `fit` mất khoảng 19,5 giây so với 17 giây ở bản chính.

## Chẩn đoán D1–D3: giới hạn nằm ở model hay ở feature
Bổ sung 2 của protocol, viết sau khi đọc ablation và trước khi chạy. `scripts/shard/fpr_ceiling_check.py`; kết quả `results/fpr_ceiling/` (bản chính, partition seed 42) và `results/fpr_ceiling_protocolfields/` (partition seed 43). 5 seed, metric trên calibration.

**Bản chính (11 feature).** `fl_mlp_zscore` là model của các mục trên (calibration trùng `results/fpr_threshold/`); `fl_mlp_log1p` chỉ khác ở phép `log1p` cho 8 feature số, cùng split, partition và seed, nên **ghép cặp được**.

| Model | S | Quy tắc | FPR | Recall | Macro-F1 | FPR trên train |
|---|---|---|---|---|---|---|
| FL MLP, z-score | 1 | `default` | 0,320 | 0,999 | 0,881 | 0,082 |
| | | `youden` | 0,067 | 0,871 | 0,854 | 0,015 |
| FL MLP, `log1p` | 1 | `default` | 0,078 | 0,987 | 0,960 | 0,019 |
| | | `max_f1` | 0,084 | 0,994 | 0,965 | 0,020 |
| | | `youden` | 0,056 | 0,975 | 0,955 | 0,013 |
| | 4 | `default` | 0,090 (0,076–0,123) | 0,982 | 0,950 | |
| | | `youden` | 0,070 | 0,979 | 0,955 | |
| | 5 | `default` | 0,096 | 0,982 | 0,949 | |
| | | `youden` | 0,075 | 0,974 | 0,947 | |
| Gradient boosting tập trung | – | `default` | 0,018 | 0,997 | 0,991 | 0,003 |
| | | `youden` | 0,008 | 0,984 | 0,980 | 0,001 |

| Model | S | ROC-AUC | FPR tại recall 0,95 | FPR tại recall 0,99 |
|---|---|---|---|---|
| FL MLP, z-score | 1 | 0,953 | 0,293 | 0,298 |
| FL MLP, `log1p` | 1 | 0,985 | 0,046 | 0,081 |
| | 4 | 0,982 | 0,059 | 0,109 |
| | 5 | 0,973 | 0,060 | 0,147 |
| Gradient boosting tập trung | – | 0,999 | 0,003 | 0,010 |

**Bản có trường giao thức.** FL MLP z-score: FPR mặc định 0,089; `log1p`: 0,090 (S = 1), 0,074 (S = 4), 0,098 (S = 5); với `youden`: 0,049 / 0,048 / 0,054. Gradient boosting tập trung: 0,015. ROC-AUC của MLP `log1p` 0,981–0,982, gần bằng bản chính có `log1p` (0,973–0,985).

**Đọc kết quả:**
- **D1:** MLP z-score có FPR 0,082 trên chính tập train ở ngưỡng mặc định, so với 0,003 của gradient boosting. Model không fit được tập train, nên đây không chỉ là chuyện tổng quát hóa.
- **D3:** `log1p` một mình đưa MLP lên mức của bản có trường giao thức. Cộng thêm trường giao thức thì không cải thiện thêm rõ rệt.
- **D2:** cùng 11 feature, một model mạnh không bị giới hạn bởi MLP/FL đạt FPR khoảng 0,02. Khoảng cách còn lại (MLP `log1p` 0,05–0,08) là của model và cách train: kích thước MLP, số vòng, non-IID của FedAvg, hoặc chính cách chuẩn hóa. Các yếu tố này chưa được tách riêng.
- Ở S ∈ {4, 5}, FPR mặc định của MLP `log1p` dao động theo seed nhiều hơn S = 1; các quy tắc chọn ngưỡng làm hẹp khoảng này.
- **Ghi chú về D3:** phép `log1p` *thay* cột z-score, làm trong bộ nhớ, tạo ra 16 vector mã hóa trùng nhau giữa các split (train–validation 4, train–calibration 8, validation–calibration 4). Lý do: ở float32, cột log gộp các flow rất lớn chỉ chênh nhau vài chục byte. Con số này quá nhỏ để đổi kết quả D3, nhưng nó vi phạm cổng chống rò rỉ. Vì vậy bản cài vào tiền xử lý (mục dưới) *thêm* cột log thay vì thay thế.

## Bản cài vào tiền xử lý (`append_log1p`)
`src/data/prepare_log1p.py` mã hóa lại bản chính: mỗi feature số giữ cột z-score và có thêm một cột `log1p` chuẩn hóa (fit trên validation), cột phân loại giữ nguyên, tổng 45 chiều. **Split, nhóm, thứ tự dòng, nhãn và `row_id` được giữ nguyên, không chia lại.** Đã kiểm tra: dòng, nhãn và nhóm trùng bản chính; khối z-score và khối one-hot trùng từng bit; partition 42 khớp số dòng đã chốt cho từng client; cổng chống rò rỉ đạt (0 vector trùng giữa các split); khối log lệch D3 không quá 10⁻⁶. `src/data/prepare.py` không đổi, hash bản chính vẫn là `f130057e…`.

Chia lại split bằng `prepare.py` với bộ mã hóa mới **không dùng được**. Nếu chỉ thay cột log, cổng chống rò rỉ không hội tụ sau 10 vòng tinh chỉnh nhóm, nên không ghi ra artifact nào. Nếu thêm cột log, nó hội tụ sau 3 vòng (bản chính cần 8) nhưng ra split khác, và partition 42 khi đó sinh client suy biến. Do đó module chỉ mã hóa lại và chỉ kiểm tra cổng, không tinh chỉnh.

Kết quả (`results/fpr_threshold_log1p/`, cùng 5 seed, S và quy tắc; **ghép cặp theo seed** với `results/fpr_threshold/`):

| S | Quy tắc | FPR | Recall | Macro-F1 | ΔFPR so với z-score (xấu nhất) | ΔF1 (xấu nhất) |
|---|---|---|---|---|---|---|
| 1 | `default` | 0,080 | 0,989 | 0,962 | −0,240 (−0,236) | +0,081 (+0,077) |
| | `youden` | 0,063 | 0,978 | 0,956 | −0,004 (+0,007) | +0,102 (+0,100) |
| | `fpr05` | 0,071 | 0,982 | 0,957 | −0,001 (+0,008) | +0,104 (+0,101) |
| 4 | `default` | 0,084 | 0,982 | 0,952 | −0,235 (−0,232) | +0,071 (+0,070) |
| | `youden` | 0,071 | 0,979 | 0,954 | +0,005 (+0,013) | +0,110 (+0,108) |
| | `fpr05` | 0,077 | 0,980 | 0,954 | −0,001 (+0,002) | +0,109 (+0,107) |
| 5 | `default` | 0,094 | 0,984 | 0,952 | −0,230 (−0,224) | +0,072 (+0,069) |
| | `youden` | 0,085 (0,065–0,101) | 0,979 | 0,950 | −0,024 (+0,024) | +0,101 (+0,082) |
| | `fpr05` | 0,072 | 0,956 | 0,930 | −0,003 (−0,001) | +0,089 (+0,080) |

`max_f1` và `fpr10` cho FPR 0,070–0,102; bảng đầy đủ ở `thresholds.csv`.

| S | ROC-AUC | FPR tại recall 0,95 | FPR tại recall 0,99 | Chọn lại ngưỡng |
|---|---|---|---|---|
| 1 | 0,984 (z-score: 0,953) | 0,048 (0,293) | 0,085 (0,298) | 9 ms |
| 4 | 0,982 (0,952) | 0,059 (0,300) | 0,122 (0,318) | 33 ms |
| 5 | 0,975 (0,944) | 0,063 (0,297) | 0,099 (0,322) | 33 ms |

- **Ở ngưỡng mặc định, cả 15 cặp (seed, S) đều cải thiện:** FPR giảm 0,22–0,24 và macro-F1 tăng 0,069–0,081.
- **Ở các quy tắc nhắm FPR (`youden`, `fpr05`), FPR gần như không đổi so với z-score (khoảng 0,06–0,08), nhưng recall tăng từ khoảng 0,86 lên khoảng 0,98.** Đây là chỗ khác biệt thật: trước đây muốn FPR thấp thì phải mất recall, nay thì không.
- Mức đạt được ngang D3 (thay cột): FPR mặc định 0,080 / 0,084 / 0,094 so với 0,078 / 0,090 / 0,096.
- `fpr05` cho FPR ổn định nhất giữa các S (0,071–0,077), nhưng ở S = 5 recall chỉ còn 0,956. `youden` ở S = 5 dao động theo seed (0,065–0,101).

## Dùng kết quả này thế nào
- **Kết quả v2, v3 và DT giữ nguyên.** Chúng dùng bản dữ liệu chính, z-score và ngưỡng mặc định; tài liệu này giải thích vì sao FPR của chúng cao, nhưng không chấm lại chúng.
- **Cho thí nghiệm mới** (ví dụ protocol DT tiếp theo): nên dùng `data/processed/connected_v3_auxfit_log1p_seed42` (11 feature cũ cộng cột `log1p`, cùng split và partition 42), kèm một quy tắc ngưỡng chọn lại sau mỗi lần unlearn. Cả hai phải được **chốt trước khi chạy**. Không cần thêm trường giao thức. Bản dữ liệu này có hash nội dung khác bản chính, nên protocol mới phải ghi hash của nó. Kết quả ở đây là dữ liệu phát triển cho quyết định đó.
- **Hệ quả có thể có với phần FU xấp xỉ (giả thuyết, chưa kiểm tra):** giai đoạn đó cũng dùng z-score không log. Một phần kết luận "model không ghi nhớ từng bản ghi" (MIA không có sức mạnh, hiệu ứng xóa nhỏ hơn nhiễu seed) có thể đến từ việc số byte bị ép phẳng, chứ không chỉ từ dữ liệu dư thừa.
- **Vai trò mới của validation:** chọn ngưỡng. Nó vốn đã dùng để fit tiền xử lý và làm tín hiệu cho `utility_stop`/DynamicFU; xem bảng vai trò dữ liệu trong [README.md](README.md).

## Không được tuyên bố
| Không được tuyên bố | Lý do |
|---|---|
| FPR trên test hay khi triển khai | Test không mở; một partition mỗi bản dữ liệu; seed đã xem |
| Bộ trường giao thức không rò rỉ | Chỉ qua audit tiền xử lý. Các trường như `dns_rcode` có thể mang đặc điểm riêng của testbed ToN-IoT |
| Thêm trường giao thức là nguyên nhân của cải thiện trong ablation | Split và partition cũng đổi; D3 cho thấy `log1p` một mình đạt mức tương đương |
| Gradient boosting là trần thật của bộ feature, hay là phương pháp nên dùng | Một model với tham số mặc định; train tập trung, không FL, không unlearn theo shard |
| `log1p` giúp ở partition hay dữ liệu khác | Một partition mỗi bản dữ liệu; 5 seed đã xem |
| Một quy tắc ngưỡng là tốt nhất | Không có gate; 5 seed; quy tắc `max_f1` phụ thuộc tỉ lệ lớp |
| Kết quả v2/v3/DT với feature hoặc ngưỡng mới | Chưa chạy lại |
| Phân tích nhóm normal và loại attack là tổng quát | Post hoc, một model, một seed |

## Tái lập
```bash
python src/data/prepare.py --config configs/data_connected_auxfit.json --output data/processed/connected_v3_auxfit_seed42
python src/data/prepare.py --config configs/data_connected_auxfit_protocolfields.json --output data/processed/connected_v3_auxfit_protocolfields_seed42
export FL_DEVICE=cpu
python scripts/shard/fpr_threshold_check.py results/fpr_threshold
python scripts/shard/fpr_cluster_diagnosis.py results/fpr_threshold
python scripts/shard/fpr_threshold_check.py results/fpr_threshold_protocolfields --data-dir data/processed/connected_v3_auxfit_protocolfields_seed42 --partition-seed 43
python scripts/shard/fpr_cluster_diagnosis.py results/fpr_threshold_protocolfields data/processed/connected_v3_auxfit_protocolfields_seed42 43
python scripts/shard/fpr_ceiling_check.py results/fpr_ceiling
python scripts/shard/fpr_ceiling_check.py results/fpr_ceiling_protocolfields --data-dir data/processed/connected_v3_auxfit_protocolfields_seed42 --partition-seed 43
python src/data/prepare_log1p.py --source data/processed/connected_v3_auxfit_seed42 --config configs/data_connected_auxfit_log1p.json --output data/processed/connected_v3_auxfit_log1p_seed42
python scripts/shard/fpr_threshold_check.py results/fpr_threshold_log1p --data-dir data/processed/connected_v3_auxfit_log1p_seed42
```
Thư mục kết quả không được tồn tại trước khi chạy. Dữ liệu chính tạo lại phải có hash nội dung `f130057e…` (script kiểm tra).
