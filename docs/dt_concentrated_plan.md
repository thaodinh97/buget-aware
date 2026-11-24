# Mở rộng C1/C2 sang họ chuỗi `concentrated`

**Trạng thái: CHỐT ngày 2026-11-24, trước lần fit đầu tiên.** Lock nằm trong `results/dt_concentrated/protocol_lock.json`. Runner hash tài liệu này, `configs/dt_concentrated.json`, `src/**`, `scripts/dt/*.py` và các file `scripts/shard` liên quan, sau khi đổi CRLF thành LF.

## Vì sao làm
- C1/C2 ([dt_orchestration_plan.md](dt_orchestration_plan.md)) và đợt quét độ nhạy chỉ dùng hai họ `spread` và `proportional`. Hai họ này được kế thừa từ hướng 1–2 của [dt_extensions.md](dt_extensions.md) mà không ghi lý do.
- `concentrated` (mọi yêu cầu vào shard lớn nhất lúc thiết kế, [shard_v3_plan.md](shard_v3_plan.md)) là cực trị chưa được kiểm tra. Ở họ này:
  - chỉ có một hàng đợi;
  - gom lô hiệu quả nhất;
  - mỗi yêu cầu chạm thao tác đắt nhất.

Đây là **phần mở rộng mô tả, không phải xác nhận**. Kết luận C1/C2 giữ nguyên.

## Thiết kế
- **Giống hệt C1/C2** (`configs/dt_orchestration.json`, runner `scripts/dt/run_orchestration.py` không sửa), trừ ba khóa:
  - `families` = [`concentrated`];
  - `arms` = `dt_queue`, `edf`, `rows_legacy`, `immediate`, `window`, `lazy`;
  - `c2_ablations` = [].
- **Giữ nguyên:**
  - seed 73042–73044, seed chuỗi 83000, seed stress 93000;
  - tải 0,2 / 0,5 / 1,0 F; môi trường `none`, `gradual_clients`, `jitter_bandwidth`;
  - 12 yêu cầu mỗi chuỗi; B = 0,75 F; D = 1,5 F; W = 0,5 F.
- **27 block** (3 seed × 3 tải × 3 môi trường), bằng số block của mỗi họ ở C1/C2.
- **Ghép cặp được với C1/C2.** Vì cùng seed, mỗi block có:
  - cùng model ban đầu (trùng từng bit);
  - cùng thời điểm đến và cùng kích thước yêu cầu như block tương ứng của `spread` và `proportional`;
  - cùng đường stress.

  Chỉ khác: mọi yêu cầu rơi vào shard 2. Đã kiểm tra cả ba lần lặp: mọi yêu cầu ở shard 2, không yêu cầu nào bị cắt theo trần 50 %, thời điểm đến trùng `spread`.
- **Thời gian máy ước tính:** khoảng 3–4,5 giờ.

## Báo cáo (chốt trước)
- **Theo arm:** số yêu cầu thành công đồng thời (trên 324), tỉ lệ đúng hạn, số yêu cầu trong ngân sách, số lần train lại, số yêu cầu chạy full, latency p95, macro-F1 và FPR trên calibration, tính chính xác.
- **`dt_queue` − mỗi arm khác:**
  - trung bình theo block, khoảng tin cậy bootstrap percentile 95 % (10.000 lần, seed 1), thắng/hòa/thua;
  - toàn bộ và tách theo tải, theo môi trường.
- **Đặt cạnh C1/C2 theo từng họ:** cùng các chênh lệch cho `spread` và `proportional`, lấy từ `results/dt_orchestration`.
- **Không kiểm định giả thuyết:** không có p-value, không có Holm.

**Dự đoán ghi trước khi chạy.** Mục đích là để đối chiếu, không phải để kiểm định:
1. `dt_queue` ≈ `edf`. Chỉ có một hàng đợi nên lợi thế lập kế hoạch thứ tự biến mất. `shard_fixed` của shard lớn nhất (≈ 0,5 F) luôn vừa ngân sách một yêu cầu (0,75 F), nên `dt_queue` không bao giờ chờ để gộp ngân sách.
2. Khoảng cách giữa `dt_queue` và `window` / `lazy` **hẹp hơn** ở C1/C2, vì gom lô có lợi nhất ở họ này.
3. `immediate` vẫn kém, vì mỗi lần chạy full trên shard lớn nhất (≈ F) vượt ngân sách một yêu cầu.
4. Mọi arm unlearn chính xác, trùng từng bit.

## Không phải luận điểm
- Không thay đổi hay sửa kết luận C1/C2.
- `poisson_zipf` chưa được chạy.
