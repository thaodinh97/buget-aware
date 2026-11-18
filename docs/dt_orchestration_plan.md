# Điều phối unlearn chính xác theo shard: controller đề xuất so với chiến lược đã có (C1) và ablation (C2)

**Trạng thái: BẢN NHÁP, chưa chốt; smoke đã xong, lần chạy chính hoãn lại (2026-11-19).** Viết ngày 2026-11-18, sau khi xem mọi kết quả DT trước đó ([dt_extensions.md](dt_extensions.md), [dt_online_confirmation_results.md](dt_online_confirmation_results.md), [dt_predictive_results.md](dt_predictive_results.md)) và phần FPR ([fpr_threshold_results.md](fpr_threshold_results.md)). Các kết quả đó là dữ liệu phát triển của protocol này.
- Trước khi chốt: một lần **smoke** (`configs/dt_orchestration_smoke.json`, seed 7, không phải bằng chứng) để đo thời gian và kiểm tra đường ống. Sau smoke chỉ được đổi **số seed** trong `configs/dt_orchestration.json` (để vừa thời gian máy); mọi thứ khác giữ nguyên. Thay đổi sau smoke ghi vào mục "Lịch sử bản nháp".
- Chốt = commit tài liệu này cùng cấu hình và code; runner ghi hash (sau khi đổi CRLF thành LF) của chúng vào `protocol_lock.json` trước lần fit đầu tiên.

## Luận điểm
| # | Luận điểm | Phép kiểm |
|---|---|---|
| **C1** (chính) | Với cùng twin, cùng yêu cầu, cùng ngân sách và deadline, controller đề xuất (`dt_queue`) phục vụ được **nhiều yêu cầu vừa đúng hạn vừa trong ngân sách hơn** mỗi chiến lược đối chứng, và mọi chuỗi đều unlearn chính xác. Về utility: so với đối chứng cũng chọn full/fixed (`edf`, `rows_legacy`) thì **không giảm** quá 0,005 macro-F1; so với đối chứng luôn chạy full (`immediate`, `window`, `lazy`) thì là **một đánh đổi có cận**, giảm không quá 0,02 | Wilcoxon một phía theo block, Holm trên 5 đối chứng; biên utility theo từng đối chứng; tính chính xác |
| **C2** (ablation) | Đóng góp riêng của từng thành phần: twin kết hợp, lập kế hoạch thứ tự, mô hình overhead, chờ để gộp ngân sách | Wilcoxon hai phía theo block, Holm trên 4 ablation; khoảng tin cậy bootstrap |
| Mệnh đề (lý thuyết) | Mọi quyết định chỉ dựa trên tín hiệu chi phí (thời gian, số dòng, số vòng, ngân sách, deadline) và chọn giữa các chính sách có số vòng cố định trước thì giữ unlearn chính xác vô điều kiện | Chứng minh trong khóa luận; thí nghiệm chỉ kiểm lại bằng so trùng từng bit |

**Không phải luận điểm:** twin kết hợp dự báo tốt hơn dự báo đơn giản. Ở [dt_predictive_results.md](dt_predictive_results.md), chênh lệch giữa `dt_queue` và `rows_queue` là −0,33 yêu cầu/block với độ lệch chuẩn 1,41: cần khoảng 142 block mới phát hiện được. `rows_queue` vẫn được chạy như một ablation của C2 và báo cáo nguyên trạng.

## Arm
Mọi arm dùng vòng lặp thời gian thực chung (`src/dt/orchestration.py`, `src/dt/realtime.py`): yêu cầu đến theo đồng hồ, một worker, chỉ thấy yêu cầu đã đến, mọi yêu cầu cuối cùng đều được xử lý và tính chi phí, phản hồi chỉ từ thao tác đã chạy.

| Arm | Nhóm | Định nghĩa |
|---|---|---|
| `dt_queue` | đề xuất | Twin kết hợp 4 bộ dự báo (`predictive.ServiceTwin`); lập kế hoạch thứ tự và full/fixed trên mọi shard đang chờ; mô hình overhead; chờ để gộp ngân sách khi cả fixed cũng vượt. Trùng từng quyết định với `predictive.run` (có test) |
| `immediate` | C1 | Worker rảnh thì xử lý hàng đợi có yêu cầu cũ nhất, luôn `shard_full`; không dùng dự báo để quyết định |
| `window` | C1, kiểu SISA | Tại các mốc k·W (W = 0,5 F), xử lý các yêu cầu đã đến trước mốc, luôn `shard_full`; không dùng dự báo |
| `lazy` | C1, kiểu ERASER | Hoãn mỗi hàng đợi tới thời điểm bắt đầu muộn nhất của `shard_full` trước deadline của yêu cầu cũ nhất, rồi xử lý cả hàng đợi, luôn `shard_full`; deadline sớm nhất trước |
| `edf` | C1 | Worker rảnh thì xử lý hàng đợi có deadline sớm nhất; `shard_full` nếu dự báo vừa ngân sách gộp và deadline, nếu không thì `shard_fixed`; không bao giờ chờ |
| `rows_legacy` | C1, controller cũ của repo | `realtime.run` với dự báo theo dòng EWMA 0,3 (như ở [dt_predictive_results.md](dt_predictive_results.md)) |
| `rows_queue` | C2 | `dt_queue` nhưng twin chỉ dùng dự báo theo dòng EWMA 0,3 |
| `queue_edf_order` | C2 | `dt_queue` nhưng thứ tự cố định theo EDF (sau cùng quy tắc không để worker rảnh); chỉ lập kế hoạch full/fixed |
| `queue_no_overhead` | C2 | `dt_queue` nhưng lập kế hoạch bằng thời gian vòng, bỏ overhead đã học |
| `queue_no_pooling` | C2 | `dt_queue` nhưng không bao giờ chờ để gộp ngân sách |

- `lazy` và `edf` cần dự báo để quyết định; chúng nhận **cùng twin kết hợp** như `dt_queue` (twin cũng học từ thao tác của chúng). Khác biệt giữa các arm vì vậy chỉ nằm ở logic điều phối.
- Chọn W = 0,5 F trước khi chạy, để một yêu cầu chờ mốc không quá 0,5 F trong khi deadline là 1,5 F.

## Thiết kế
- **Dữ liệu:** `data/processed/connected_v3_auxfit_log1p_seed42` (11 feature, thêm cột `log1p`, cùng split và partition với bản chính; [fpr_threshold_results.md](fpr_threshold_results.md)). Runner kiểm tra `numeric_transform = append_log1p`, ghi hash nội dung vào lock, chỉ nạp train, validation, calibration. **Test không được mở.**
- **Model:** 20 client, partition 42, S = 4 (`assign_shards`), cấu hình `standard` (R_base = 20, `shard_fixed` = 10 vòng).
- **Ngưỡng:** sau mỗi thao tác unlearn, ngưỡng được chọn lại trên validation theo quy tắc `youden` từ model vừa train lại. Thời gian này nằm trong thời gian phục vụ của thao tác. Utility báo trên calibration với ngưỡng đó (kèm ngưỡng mặc định để tham khảo).
- **Block** = seed × họ chuỗi × mức tải × môi trường:
  - seed train **73042, 73043** (chưa từng chạy; số seed có thể tăng sau smoke);
  - họ `spread`, `proportional`;
  - mức tải: khoảng cách trung bình giữa hai yêu cầu = 0,2 F, 0,5 F, 1,0 F (0,2 F là mức của các thí nghiệm trước, quá tải);
  - môi trường `none`, `gradual_clients`, `jitter_bandwidth` (như [dt_online_confirmation_plan.md](dt_online_confirmation_plan.md), seed stress 93000).
  - Với 2 seed: 2 × 2 × 3 × 3 = **36 block, 360 chuỗi**.
- Mỗi block: một lần fit; F = thời gian 20 vòng của shard lớn nhất trong lần fit đó; ngân sách B = 0,75 F mỗi yêu cầu, deadline D = 1,5 F; 12 yêu cầu sample-level (`stream_requests`, seed chuỗi 83000 + 100·lần_lặp + offset). Cùng một chuỗi yêu cầu cho mọi mức tải và môi trường của cùng seed và họ; chỉ thời điểm đến được nhân với mức tải.
- **Thứ tự chạy:** môi trường xoay vòng theo (lần lặp, họ, mức tải); thứ tự arm xoay vòng theo chỉ số block.
- **Tính chính xác:** mọi thao tác phải `exact` và `structure_ok`; cuối mỗi chuỗi `verify_bit_exact` đạt; mọi arm của một block phải còn cùng số dòng. Sai một điều kiện thì runner dừng.
- Lần fit có vòng > 60 giây thì block không hợp lệ và runner dừng (như các protocol trước).

## Thước đo và phép kiểm (cố định khi chốt)
- **Thước đo chính:** số yêu cầu **thành công đồng thời** mỗi block: hoàn thành trước deadline (đo tới model-ready, gồm chờ, xử lý và chọn ngưỡng) **và** batch chứa nó không vượt ngân sách thời gian vòng gộp.
- **C1:** với mỗi đối chứng b, d = `dt_queue` − b theo block; Wilcoxon signed-rank một phía (d > 0), bỏ các cặp bằng 0; Holm trên 5 đối chứng; α = 0,05. b **bị vượt** khi p Holm < 0,05 và trung bình d > 0. Biên utility: trung bình chênh macro-F1 calibration ghép cặp (`dt_queue` − b) ≥ **−0,005** với `edf` và `rows_legacy`, ≥ **−0,02** với `immediate`, `window` và `lazy` (`analysis.calibration_f1_floor`). **C1 đạt** khi đủ 36 block, mọi chuỗi chính xác, mọi đối chứng bị vượt và mọi biên utility đạt.
- **Vì sao hai biên:** đối chứng luôn chạy full có utility cao hơn theo cấu tạo, vì `shard_fixed` (10 vòng) kém `shard_full` (20 vòng). v3 đo được cận dưới của chênh `shard_fixed` − `shard_full` từ −0,0108 đến −0,0137 ([shard_v3_results.md](shard_v3_results.md), G4). Giữ ngân sách chặt mà không mất utility so với chúng là điều không thể; biên −0,02 cho phép đúng cái giá đó cộng một khoảng dư, và vẫn chặn một controller hạ cấp quá mức. Với đối chứng cùng không gian full/fixed, chênh utility chỉ đến từ khác biệt quyết định, nên giữ biên chặt −0,005.
- **C2:** với mỗi ablation a, d = `dt_queue` − a; Wilcoxon hai phía, Holm trên 4 ablation. Thành phần **có đóng góp** khi p Holm < 0,05 và trung bình d > 0; **có hại** khi p Holm < 0,05 và trung bình d < 0; còn lại **không phát hiện được**. Mọi trung bình kèm khoảng tin cậy bootstrap 95 % theo block (10.000 lần, seed 1).
- **Mô tả (không kiểm định):** chênh theo mức tải, deadline miss, tổng thời gian phục vụ, latency p95, số yêu cầu chạy full, FPR calibration, thời gian lập kế hoạch và chọn ngưỡng.
- **Cỡ mẫu:** độ lệch chuẩn của chênh lệch theo block ở thí nghiệm trước là 1,65 yêu cầu/block. Với 36 block, α hiệu chỉnh khoảng 0,01 và power 0,8, chênh lệch nhỏ nhất phát hiện được khoảng 3,42 · 1,65 / √36 ≈ 0,94 yêu cầu/block. Chênh của controller mới so với controller cũ trước đây là +2,44. Các ablation có hiệu ứng nhỏ hơn mức này có thể ra "không phát hiện được"; điều đó được ghi trước ở đây.
- Bộ phân tích: `scripts/dt/analyze_orchestration.py`. Nó audit lại từ log sự kiện (mỗi yêu cầu đúng một lần, không có yêu cầu tương lai trong batch, một worker, latency = chờ + xử lý, thành công đồng thời khớp tổng hợp) trước khi tính.

## Không được tuyên bố, kể cả khi đạt
| Không được tuyên bố | Lý do |
|---|---|
| Kết quả cho hệ thống phân tán thật, nhiều worker, hay SLA triển khai | Một máy, một worker, stress tiêm nhân tạo |
| Ưu thế so với bản cài đặt gốc của ERASER hay SISA | `lazy` và `window` là chiến lược *theo tinh thần* các công trình đó, cài trên cùng hệ thống; không phải tái lập |
| Twin kết hợp dự báo tốt hơn dự báo đơn giản | Không đủ cỡ mẫu (mục "Luận điểm") |
| Kết quả ở partition, S, kích thước chuỗi hay mức ngân sách/deadline khác | Một partition, S = 4, 12 yêu cầu, B = 0,75 F, D = 1,5 F |
| Kết quả trên test | Test không mở |
| Kết quả ngoài phạm vi yêu cầu sample-level | Chỉ dùng chuỗi sample-level |

## Lệnh chạy
```powershell
python scripts/dt/run_orchestration.py runs/dt_orchestration_smoke --config configs/dt_orchestration_smoke.json
python scripts/dt/run_orchestration.py results/dt_orchestration --config configs/dt_orchestration.json
python scripts/dt/analyze_orchestration.py results/dt_orchestration
```
Thêm `--resume` để tiếp tục sau gián đoạn với cùng cấu hình và code; block dở dang phải được chuyển đi trước, không ghi đè.

## Lịch sử bản nháp
- 2026-11-18: bản nháp đầu tiên; code, test (15 test trong `tests/test_dt_orchestration.py`) và cấu hình smoke.
- 2026-11-18, **sau khi xem block đầu của smoke** (seed 7, `spread`, tải 0,2 F, `jitter_bandwidth`; không phải bằng chứng): biên utility một mức (−0,005 với mọi đối chứng) được thay bằng hai mức (mục "Thước đo và phép kiểm"). Lý do: trong block đó, các đối chứng luôn chạy full có macro-F1 calibration 0,963, còn `dt_queue` có 0,949 (chênh −0,014), vì nó chạy fixed cho 10/12 yêu cầu để giữ ngân sách. Với biên cũ, C1 thất bại theo cấu tạo. Biên −0,02 được chọn từ chênh `shard_fixed` − `shard_full` đo ở v3 cộng khoảng dư; giá trị −0,014 của smoke cũng đã được nhìn thấy khi chọn. Quyết định do người dùng chọn giữa ba phương án (giữ −0,005; tách hai mức; bỏ biên). Cấu hình smoke giữ nguyên biên cũ, vì smoke đã chạy với nó; bộ phân tích đọc được cả hai dạng.
- 2026-11-19, **smoke hoàn tất** (`runs/dt_orchestration_smoke/`, 2 block: seed 7, `spread`, `jitter_bandwidth`, tải 0,2 F và 1,0 F; không phải bằng chứng):
  - Đường ống: 20/20 chuỗi chính xác từng bit, mọi arm của một block còn cùng số dòng; bộ phân tích chạy được đầu–cuối (`--smoke`); 15 test đạt.
  - Thời gian một block ở môi trường chậm nhất: 8,6 phút (tải 0,2 F), 17,7 phút (tải 1,0 F); tải 0,5 F chưa đo, nội suy khoảng 12 phút. Ước lượng cận trên: 2 seed (36 block) khoảng 7,7 giờ; 3 seed (54 block) khoảng 11,5 giờ.
  - Kết quả quan sát được (thành công đồng thời trên 24): `dt_queue` 12, `edf` 12, `rows_legacy` 7, `immediate` 6, `window` 3, `lazy` 0; ablation `queue_no_overhead` 18, `queue_edf_order` 15, `rows_queue` 14, `queue_no_pooling` 13.
  - **Quyết định của người dùng: giữ `dt_queue` làm controller đề xuất**, không chỉnh thiết kế theo smoke. Nếu thành phần nào có hại, C2 sẽ báo cáo nguyên trạng.
  - Số seed giữ **2** (73042, 73043) trong cấu hình; người dùng chọn hoãn lần chạy chính. Nếu tăng số seed thì phải làm trước khi chốt và ghi vào mục này.
  - **Chưa chốt.** Khi chạy: commit tài liệu, cấu hình và code ở trạng thái này, rồi chạy runner; runner ghi lock trước lần fit đầu tiên.
