# Đóng băng thành phần FU trước phase Digital Twin

> † **Code của giai đoạn này đã được gỡ khỏi nhánh** (gồm `run_deployable_forget.py`, `run_confirmation.py`, các verifier và `src/fu/` trừ `conpfu_inspired.py`). Chúng vẫn còn nguyên ở tag `pre-cleanup`: `git checkout pre-cleanup`. Các doc kết quả được nhắc tới ở đây nằm trong [archive/](archive/). Hướng hiện tại là unlearn chính xác theo shard: [shard_v2_results.md](shard_v2_results.md).

**Ngày đóng băng:** 2026-11-03, trên nhánh `feature/fu-experiment`. Hai lần xác nhận (partition 43 và 44) đã chạy xong. Verdict cuối cùng ở mục 4: **ứng viên, không được xác nhận**. Commit và tag `fu-freeze` chưa được tạo.

**Mục đích.** Theo [đặc tả](AGENT_IMPLEMENTATION_GUIDE_FU_BUDGET_AWARE_DT.md), Digital Twin (DT) là đóng góp chính của khóa luận, còn federated unlearning (FU) là thành phần hỗ trợ. Planner của DT giữ cố định bước FU và chỉ chọn tập client tham gia recovery. Tài liệu này cố định bước FU đó, ghi những kết luận được phép tuyên bố, và đặt ranh giới dữ liệu cho phase DT.

**Từ thời điểm này:**
- Không thay đổi phương pháp, siêu tham số hay quy tắc quyết định của FU.
- Mọi thực nghiệm FU mới phải ghi là "sau đóng băng" và không được làm thay đổi đầu vào của DT.

## 1. Cấu hình đóng băng

**Nền** (không đổi kể từ `controlled_phase_v2_seed42`):

| Thành phần | Giá trị |
|---|---|
| Dữ liệu | `data/processed/connected_v3_seed42`; split train / validation / calibration / test = 60 / 15 / 10 / 15, chia theo connected component |
| Partition | 5 client, Dirichlet α = 0,5, seed 42 (`runs/pilot_v3_seed42/partitions.npz`) |
| Model | MLP 38→64→32→2, khởi tạo bằng `model(38, [64, 32], seed)` |
| FedAvg | 15 vòng, 1 epoch local, Adam lr 1e-3, batch 512, 2 luồng torch, `FL_DEVICE=cpu`, thuật toán tất định |

**Phương pháp FU: `ga_proj@joint_aux_prob`.** Hàm `run_deployable_forget.standard_arm`; cấu hình nằm trong `run_deployable_forget.CONFIG`.

| Bước | Ai chạy | Cấu hình |
|---|---|---|
| 1. Reference | Client đích, local | kNN với k = 10 trên R_target ∪ aux, dùng margin trung bình của model original. Model reference mới: train 5 epoch trên R_target ∪ aux (Adam lr 1e-3, batch 512, seed s + 700000) |
| 2. Mức dừng `joint_aux_prob` | Client đích | Ở từng dòng của F, lấy reference cho xác suất nhãn đúng thấp hơn trong hai reference; mức dừng là trung bình của chúng |
| 3. Bước quên | Client đích | Gradient ascent trên CE(F) bằng Adam lr 1e-3, batch 128, clip 5. Gradient và update được chiếu ra khỏi basis SVD của input từng layer trên ≤ 4.000 dòng của R_target (năng lượng 0,999999). Dừng khi xác suất nhãn đúng trung bình trên F ≤ mức dừng; tối đa 300 bước; guard dừng khi CE tăng quá 20 nats |
| 4. Recovery | Server và các client | 2 vòng FedAvg trên D \ F, seed s + 900000. **Đây là chỗ planner của DT can thiệp:** chọn tập client tham gia recovery |

- **Dữ liệu phụ trợ (aux)** là tập validation.
- **Đầu ra của bước 1–3** là model trung gian θ_pre: trạng thái FU cố định mà planner nhận vào.

**Baseline giữ lại để so sánh trong phase DT:**
- **retrain:** train lại 15 vòng trên D \ F, là chuẩn vàng;
- **retain-only:** chỉ recovery, không có bước quên;
- **`reset_zero@global`:** phòng vệ gần như miễn phí cho trigger nằm trên các chiều không được dùng;
- **original.**

## 2. Kết luận được phép và không được phép

| Được tuyên bố (có điều kiện) | Bằng chứng |
|---|---|
| Trên partition 42, mức dừng `joint_aux_prob` bám mức retrain trong khoảng 0,001–0,005, không cần retrain | [deployable_forget_results.md](archive/deployable_forget_results.md), 5 request. **Không** đúng trên mọi partition (xem bảng dưới) |
| Trên canary tổ hợp, có trigger nằm trên các chiều đang dùng, projection giữ được việc quên qua recovery và thắng cả retain-only lẫn control cùng pool. GA không chiếu bị recovery kéo về và chỉ thắng control cùng pool ở 1/7 request | 7/7 request trên 3 partition: 2 trên seed phát triển ([confirmation_existing_results.md](archive/confirmation_existing_results.md)), 3 trên partition 43 ([confirmation_p43_results.md](archive/confirmation_p43_results.md)), 2 trên partition 44 ([confirmation_p44_results.md](archive/confirmation_p44_results.md)). Lặp lại trên dữ liệu chưa nhìn, nhưng **không phải "đã xác nhận"** (mục 4) |
| Trung bình qua seed, utility trên test không kém retain-only: macro-F1 và FPR | Mọi request của cả 3 run. Trong `confirm2`, check noninferiority trượt trên 2 canary vì khoảng tin cậy bị nới rộng bởi các seed **tốt hơn**; không seed nào có FPR kém hơn quá 0,0004 |
| Macro-F1 trên dữ liệu giữ lại của từng client không giảm | 7/7 request của `existing`. Trong `confirm`, chênh lệch xấu nhất là −0,00016, và check trượt duy nhất là của client 4, nơi sàn nhiễu bằng 0 (mục 4) |
| Rò rỉ ở mức nhóm (backdoor) về mức retrain | Các canary có `forget_ok` ở trên |
| Bước quên rẻ về giao tiếp: chạy local trong 2–5 s, sau đó 2 vòng FedAvg, so với 15 vòng khi retrain | [deployable_forget_results.md](archive/deployable_forget_results.md) |

| **Không** được tuyên bố | Lý do |
|---|---|
| "Phương pháp tốt nhất" hoặc "đã xác nhận" | Phương pháp được chọn sau khi xem dữ liệu, và cả hai lần xác nhận đều không qua quy tắc đặt trước: 4/6 và 3/7 request. Theo protocol, không có lần thứ ba (mục 4) |
| Projection luôn quên được trigger nằm trên các chiều đang dùng | Trên canary tổ hợp c0 của partition 43, position chỉ 0,64; ở 3/10 seed bước chiếu không dịch chuyển được trong 300 bước |
| Mức dừng `joint_aux_prob` luôn bám mức retrain | Trên low-overlap của client 2 (partition 44), mức dừng quá chặt: CE trên F gấp đôi retrain, và model bị đẩy ra xa retrain hơn cả original (position −0,27 sau recovery) |
| Canary zero-dim chứng minh giá trị của projection | Đó là lối tắt cấu trúc, và lối tắt này lặp lại trên cả 3 partition ([reset_baseline_results.md](archive/reset_baseline_results.md)) |
| Bước quên có ích, hay ít nhất là vô hại, trên request tự nhiên | Có 4 request tự nhiên đo được trên 3 partition, và ở request nào bước quên cũng không vượt được control công bằng. Trên low-overlap của client 2 (partition 44), bước quên còn gây hại |
| Unlearn làm tăng privacy theo bản ghi, hay "MIA giống retrain" | MIA không có sức mạnh, ngay cả trên model original |
| Model unlearn không phân biệt được với retrain | AUC của attacker 0,67–1,00 ([indistinguishability_results.md](archive/indistinguishability_results.md)) |
| Đảm bảo xóa dữ liệu, DP, hay certified unlearning | Không có cơ chế hình thức nào |
| FU rẻ hơn retrain về compute | Chưa có bảng chi phí đầy đủ: một lần retrain chỉ mất 10–13 s, trong khi kNN và reference cũng tốn tương đương |

## 3. Bàn giao cho phase DT

**Giao diện:**
- **Đầu vào:** model original, request F tại client đích, R_target và dữ liệu aux.
- **Đầu ra của FU:** θ_pre, cùng các đại lượng chi phí:
  - `local_seconds`, `steps`, `backward_samples`, `check_samples`;
  - `reference_backward_samples`;
  - `knn_seconds` (trong `request_state.csv`).
- **Recovery:** planner chọn tập P gồm các client đủ điều kiện, với **số vòng cố định là 2**, đúng như đặc tả.

**Chưa từng được đo, và là phép đo đầu tiên của phase DT:**
1. **Recovery với tập con client.** Mọi thực nghiệm FU đều recovery với cả 5 client. Đảm bảo của projection giả định dữ liệu recovery nằm trong span của client đích, và chất lượng quên trên request tự nhiên chủ yếu đến từ recovery (retain-only đạt 0,72 trên request random). Lựa chọn client của planner vì vậy ảnh hưởng trực tiếp tới chất lượng unlearn, không chỉ tới chi phí.
2. **Chi phí theo client và theo vòng.** `fedavg` có ghi `compute_seconds` và `payload_bytes` cho từng client, nhưng các runner FU hiện không lưu log recovery. Phase DT phải lưu chúng.

**Request nào đo được chất lượng unlearn.**
- **Request tự nhiên xóa 50% hầu như không đo được.** Trên cả 3 partition, phần lớn request này không vượt nhiễu seed: 6/8 request trên partition 42, 9/9 trên partition 43 và 4/6 trên partition 44.
  - Hai request đo được trên partition 44 đều thuộc client 2, client lớn nhất (53 nghìn dòng).
- **Chất lượng unlearn của planner chỉ đo được một cách đáng tin trên canary tổ hợp.** Trên request tự nhiên, planner nên được đánh giá theo chi phí, utility và khoảng cách tới retrain, luôn kèm retain-only để so sánh.
- **Trên request tự nhiên, θ_pre của `ga_proj` có thể tệ hơn không làm gì.** Trên low-overlap của client 2 (partition 44), model sau recovery xa retrain hơn cả original.
  - Phase DT cần ghi rõ nó nhận θ_pre của phương pháp đóng băng hay θ_pre = original (retain-only) cho request tự nhiên.
  - **Đây là quyết định của người làm khóa luận, chưa được chốt.**
- **Lựa chọn client có thể quyết định projection có tác dụng hay không,** vì hiệu quả của nó phụ thuộc vào client đích. Trên client 0 của partition 43, R lớn và gần như toàn attack, và bước chiếu bị nghẽn ở một số seed.

**Ngưỡng utility cho planner.** Model original là một detector yếu: macro-F1 0,89, FPR 27%, macro-F1 của client 1 chỉ 0,56. Ràng buộc utility nên gồm macro-F1, FPR và macro-F1 của từng client, mỗi đại lượng so với original của cùng seed, với biên đặt trước. Không chỉ dùng macro-F1 toàn cục.

**Ranh giới dữ liệu.** Đặc tả cấm dùng test để chọn siêu tham số hoặc để học planner.
- **train:** chỉ dùng cho FL và FU, như hiện nay.
- **validation:** dữ liệu aux của FU, như cấu hình đã đóng băng. **Không** dùng để tinh chỉnh DT.
- **calibration:** hiệu chỉnh cost model, ngưỡng utility proxy và tinh chỉnh planner của DT. Trước đây tập này chỉ dùng cho shadow MIA, nên không còn ràng buộc nào với FU.
- **test:** chỉ dùng cho báo cáo cuối cùng.

## 4. Trạng thái xác nhận

**Verdict cuối cùng: ỨNG VIÊN, KHÔNG ĐƯỢC XÁC NHẬN.**
- Cả hai lần xác nhận đều trượt quy tắc đặt trước.
- Protocol đã chốt từ trước là không có lần thứ ba, dù lý do trượt là gì.
- Cấu hình ở mục 1 vẫn là bước FU cố định cho phase DT.

| Lần | Partition / seed | Request được admit | `ga_proj@joint_aux_prob` | Lý do trượt |
|---|---|---|---|---|
| 1 (`confirm`) | 43 / 20042–20051 | 6 canary, 0 tự nhiên | 4/6 | sàn utility bằng 0 ở client 4 (2 canary) |
| 2 (`confirm2`) | 44 / 30042–30051 | 5 canary, 2 tự nhiên | 3/7 | khoảng tin cậy utility bị nới bởi các seed tốt hơn (2 canary); bước quên không có ích (random) hoặc gây hại (low-overlap) trên client 2 |

### 4.1 Lần 1: `run_confirmation.py --mode confirm`
- **Thiết lập.** Partition mới (Dirichlet seed 43, α = 0,5), seed mới 20042–20051, 15 request hợp lệ:
  - 6 canary: zero-dim và tổ hợp, trên client 0, 1, 2;
  - 9 request tự nhiên trên client 0–4.
- **Request bị loại.** `client0_low_overlap_groups_0.5` không hợp lệ: F hoặc R chỉ còn một lớp.
- **Run:** `runs/confirmation_p43`, 4.189 s. Verifier pass: 172 artifact khớp hash, 84 dòng quyết định được tính lại, 32 stage chạy lại khớp từng bit. Chi tiết ở [confirmation_p43_results.md](archive/confirmation_p43_results.md).

**Verdict: KHÔNG XÁC NHẬN theo quy tắc đặt trước.**
- **Không arm nào pass.** `ga_proj@joint_aux_prob` qua 4/6 request được admit, và trượt canary tổ hợp c0 và c1.
- **Không request tự nhiên nào qua gate.** Effect size chỉ 0,09–3,67, nên `confirm` chỉ kiểm tra được trên 6 canary (client 0, 1, 2).
- **Cả hai lần trượt đều do một check utility suy biến, không phải do bước quên:**
  - `ga_proj` đạt `forget_ok` trên 6/6 request.
  - Check trượt là macro-F1 trên dữ liệu giữ lại của client 4. Mọi model original và retrain cho cùng một giá trị 0,961545, nên sàn nhiễu bằng 0.
  - Trên canary tổ hợp c1, `ga_proj` không kém retain-only ở seed nào: 9 seed bằng nhau, 1 seed đúng thêm một dòng. Check vẫn trượt.
- **Phân tích độ nhạy, post hoc nên không thay được verdict.** Nếu thay sàn bằng 0 bằng độ mịn của một dòng (0,0012), `ga_proj` qua 6/6 request trên 3 client đích.

**Những gì lặp lại trên partition và seed mới:**
- Projection giữ được việc quên trên canary tổ hợp: position 0,64 / 1,03 / 0,98, so với 0,06 / 0,29 / 0,39 khi không chiếu. Nó thắng control cùng pool trên cả 3 request.
- Lối tắt của canary zero-dim: `reset_zero@global` cách `ga_proj` không quá 0,01.
- Utility trên test không giảm.
- Model gốc yếu: FPR 27%.

**Giới hạn mới.** Trên canary tổ hợp c0, bước chiếu chạm trần 300 bước ở 7/10 seed và không dịch chuyển được ở 3/10 seed. Stop oracle cũng không cứu được (0,59).

**Xử lý.** Quy tắc không được sửa để chấm lại run này. Bản sửa (sàn không mịn hơn một dòng) được ghi vào protocol rồi mới áp dụng cho lần 2.

### 4.2 Lần 2: `run_confirmation.py --mode confirm2`
- **Thiết lập.** Protocol được ghi trước khi chạy, trong mục "Confirm lần 2" của [confirmation_plan.md](archive/confirmation_plan.md).
  - Partition Dirichlet seed 44, seed 30042–30051. Sàn utility không mịn hơn một dòng.
  - Không đổi phương pháp. Không chạy smoke trên dữ liệu thật.
- **Partition rất lệch.** Normal / attack của từng client: 658 / 2, 51 / 25.665, 449 / 52.921, 2.777 / 13.161 và 18.599 / 1.
- **Request.** 12 request hợp lệ, 7 qua gate: canary zero-dim c2, c1, c3; canary tổ hợp c2, c1; random và low-overlap của client 2. Các request này trải trên client đích 1, 2, 3.
- **Run:** `runs/confirmation_p44`, 2.665 s. Verifier pass: 142 artifact, 98 dòng quyết định, 32 stage. Chi tiết ở [confirmation_p44_results.md](archive/confirmation_p44_results.md).

**Verdict: KHÔNG XÁC NHẬN.**
- **Không arm nào pass.** `ga_proj` qua 3/7 request, `nkd_proj` 4/7.
- **Sàn một dòng không làm đổi quyết định nào** của run này.

**Bốn lần trượt của `ga_proj`:**
- **Hai canary (zero-dim c2, tổ hợp c1): phần quên đạt (position 1,00 / 1,04), nhưng check utility trượt.**
  - Trung bình qua seed, arm *tốt hơn* retain-only: FPR thấp hơn 0,001 / 0,020.
  - Nhưng một số seed tốt hơn rất nhiều. Ví dụ, ở một seed retain-only có FPR 0,416, còn `ga_proj` chỉ 0,272.
  - Các seed đó nới rộng khoảng t đối xứng, đẩy cận phía xấu vượt sàn 0,0008.
  - Không seed nào có FPR kém hơn quá 0,0004.
  - Đây là một điểm yếu thứ hai của quy tắc. Theo protocol, nó không được sửa.
- **Random 50% của client 2: bước quên không dịch chuyển.** Nó chạy hết 300 bước; sau recovery đạt 0,37, trong khi retain-only đạt 0,40. F trùng với R, nên phép chiếu không để lại hướng nào để đi, như ở partition 42.
- **Low-overlap 50% của client 2: bước quên gây hại.**
  - F chứa 446/449 dòng normal của client 2.
  - Mức dừng `joint_aux_prob` quá chặt: CE trên F gấp đôi retrain.
  - Position là −4,4 trước recovery và −0,27 sau recovery, tức xa retrain hơn cả original và hơn retain-only (0,06).
  - Stop oracle cũng không giúp được (0,24): khớp loss trung bình trên F không kéo dự đoán theo từng dòng về phía retrain.

**Những gì lặp lại lần thứ ba:**
- Trên canary tổ hợp, projection giữ được việc quên (1,02 / 1,04, so với 0,09 / 0,25 khi không chiếu) và thắng control cùng pool.
- Lối tắt zero-dim: `reset_zero@global` cách `ga_proj` không quá 0,02.
- Model gốc yếu: FPR 27%.

### 4.3 Điều được phép viết trong khóa luận
- **Bước FU là một ứng viên, không được xác nhận.** Báo cáo cả hai lần xác nhận cùng với lý do trượt.
- **Bằng chứng dương tính duy nhất lặp lại qua các partition:** trên backdoor có trigger nằm trên các chiều đang dùng, phép chiếu trực giao với dữ liệu giữ lại giữ được việc quên qua recovery (7/7 request trên 3 partition).
- **Trên dữ liệu tự nhiên, FU không có tác dụng.**
  - Hầu hết request không đo được.
  - Ở 4 request đo được, không request nào vượt control công bằng, và có một request bị hại.
  - Chất lượng unlearn trên request tự nhiên chủ yếu đến từ recovery, tức đúng phần mà planner DT điều khiển.

## 5. Hướng FU để lại cho sau (không chặn phase DT)
- **Kiểm định noninferiority bền với ngoại lệ thuận lợi,** ví dụ một phía, hoặc dùng seed xấu nhất. Đây chỉ là hướng cho công việc sau, không dùng để chấm lại hai run trên.
- **Mức dừng khi R_target gần như mất một lớp,** như ở low-overlap của client 2 trên partition 44 và client 4 trên partition 42: reference đánh giá quá thấp xác suất mà retrain giữ lại.
- **Bước quên trên request tự nhiên.** Khớp loss trung bình không kéo dự đoán từng dòng về phía retrain.
- Bước chiếu bị nghẽn khi R của client đích phủ các chiều của trigger (canary tổ hợp c0 trên partition 43).
- Stop rule theo thang logit, nhắm vào phần dư khiến model vẫn phân biệt được với retrain.
- Probe control: chạy test phân biệt trên dữ liệu ngoài F.
- Partition α = 0,1, và canary có nhiều trigger.
- Áp các chỉ số của bài ConPFU (unlearning accuracy, UASR) kèm giá trị của retrain.
- Certified unlearning, ví dụ noisy recovery.
