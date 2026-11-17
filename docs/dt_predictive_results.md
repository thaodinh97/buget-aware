# DT cải tiến: kết quả xác nhận online

Ngày 2026-11-17. **DT mới cải thiện so với đối chứng dùng controller cũ, đạt toàn bộ ngưỡng thực dụng đã chốt. Tuy nhiên, bộ dự báo đơn giản dùng cùng controller mới có kết quả đúng hạn/trong ngân sách cao hơn DT. Lợi thế riêng của bộ dự báo DT kết hợp chưa được xác nhận.**

## Thay đổi đã thực hiện

Đã thêm [controller và ServiceTwin](../src/dt/predictive.py), [runner](../scripts/dt/run_predictive_confirmation.py), [bộ phân tích](../scripts/dt/analyze_predictive_confirmation.py) và 9 test mới. Controller dự báo cả thời gian xử lý ngoài các vòng FU, xét thứ tự cùng lựa chọn full/fixed của các shard đang chờ, và chạy fixed ngay khi phù hợp thay vì chờ sát deadline. DT kết hợp bốn bộ dự báo bằng sai số của các thao tác đã thực thi.

[Protocol](dt_predictive_plan.md) và [cấu hình](../configs/dt_predictive_confirmation.json) được chốt trước fit; mã/cấu hình giữ nguyên suốt lần chạy. Kết quả online trước là dữ liệu phát triển cho bản này. Không chỉnh tham số sau khi xem kết quả mới.

## Phép so sánh

18 block ghép cặp × 3 arm = **54 chuỗi, 648 yêu cầu**. Mỗi arm có 216 yêu cầu; ba seed train mới 72042–72044, hai họ yêu cầu và ba môi trường. Trong mỗi block, các arm dùng cùng model ban đầu, yêu cầu, ngân sách, deadline và quy luật stress; thứ tự chạy xoay vòng.

- `rows_legacy`: dự báo theo số dòng, controller cũ; là đối chứng có điểm tổng cao nhất ở bộ trước, được chạy lại trong từng block mới.
- `rows_queue`: dự báo theo số dòng, controller mới.
- `dt_queue`: DT kết hợp, **cùng controller mới** với `rows_queue`.

“Thành công đồng thời” nghĩa là hoàn thành trước deadline và batch không vượt ngân sách thời gian vòng FU. Deadline bao gồm **chờ + xử lý tới khi model sẵn sàng trong bộ nhớ**. Service time là thời gian thực xử lý batch, rộng hơn round time dùng để chấm ngân sách.

## Kết quả tổng

| Chỉ số | `rows_legacy` | `dt_queue` | `rows_queue` |
|---|---:|---:|---:|
| Thành công đồng thời | 113/216 | 157/216 | **163/216** |
| Tỷ lệ thành công | 52,31% | 72,69% | **75,46%** |
| Yêu cầu trễ deadline | 99 | 53 | **50** |
| Yêu cầu trong ngân sách vòng FU | 205 | 206 | **210** |
| Completion latency trung bình (giây) | 12,447 | 10,333 | **9,838** |
| Tổng service time (giây) | 627,824 | 623,886 | **591,424** |
| Yêu cầu xử lý full | 67 | 57 | 49 |
| Macro-F1 validation trung bình | 0,892638 | 0,892542 | 0,892060 |
| FPR validation trung bình | 0,242361 | 0,242154 | 0,242204 |

So với `rows_legacy`, DT tăng **44 yêu cầu thành công, tương đương 20,37 điểm phần trăm**; giảm 46 yêu cầu trễ, service time giảm khoảng 0,63%. Ghép cặp theo block: **15 thắng, 3 hòa, 0 thua** về số yêu cầu thành công đồng thời.

So với `rows_queue`, DT ít hơn **6 yêu cầu thành công (−2,78 điểm phần trăm)** và dùng thêm 32,461 giây service (khoảng 5,49%). Ghép cặp: **3 thắng, 10 hòa, 5 thua**. DT có thêm 8 yêu cầu full và F1 trung bình cao hơn 0,000482; đây là khác biệt quan sát được, chưa chứng minh ưu thế chất lượng có ý nghĩa thống kê.

### Theo môi trường

Mỗi ô là số yêu cầu thành công đồng thời trên 72 yêu cầu:

| Môi trường | `rows_legacy` | `dt_queue` | `rows_queue` |
|---|---:|---:|---:|
| Không stress | 62/72 | 71/72 | 72/72 |
| Client chậm tăng dần | 30/72 | 51/72 | 55/72 |
| Bandwidth dao động | 21/72 | 35/72 | 36/72 |

Controller mới giúp cả hai bộ dự báo tăng kết quả ở cả ba môi trường. Các thành phần của controller được thay cùng nhau, nên chưa tách riêng đóng góp của mô hình overhead, bỏ chờ hay chọn thứ tự. Bảng này không hỗ trợ quy toàn bộ lợi ích cho DT kết hợp.

## Đối chiếu tiêu chí đã chốt

Các tiêu chí sau so `dt_queue` với `rows_legacy`:

| Tiêu chí | Ngưỡng chốt trước | Quan sát | Kết quả |
|---|---:|---:|---|
| Tăng tỷ lệ thành công đồng thời | ≥ 5 điểm phần trăm | +20,370 | Đạt |
| Số môi trường có cải thiện | ≥ 2/3 | 3/3 | Đạt |
| Tỷ số tổng service time | ≤ 1,10 | 0,993727 | Đạt |
| Chênh F1 ghép cặp trung bình | ≥ −0,002 | −0,000096 | Đạt |
| Chênh F1 của block xấu nhất | ≥ −0,005 | −0,002534 | Đạt |
| Exactness và bit-exact | Mọi chuỗi | 54/54 | Đạt |

Đạt ngưỡng utility không có nghĩa F1 hoàn toàn không giảm. Các gate là mục tiêu thực dụng trên mẫu nhỏ; **không phải kiểm định noninferiority hoặc bằng chứng ưu thế thống kê**. Full ở đây dùng 20 vòng, fixed dùng 10 vòng; bit-exact được so với retrain có **cùng số vòng đã chọn**, không khẳng định fixed bằng retrain 20 vòng.

## Kiểm tra tính đúng đắn và chi phí điều khiển

- **108 test đạt**, gồm 9 test mới cho controller, học từ dự báo trước cập nhật, xử lý ngân sách không khả thi và tích hợp cả ba arm.
- Audit lại từ event: 18 block đủ ba arm, 648 yêu cầu được xử lý đúng một lần, không bỏ yêu cầu; không thực thi yêu cầu tương lai; một worker không chạy chồng job; latency, cờ vượt ngân sách và số tổng hợp khớp timestamp.
- Cả 54 chuỗi đạt kiểm tra cấu trúc/dữ liệu còn lại và bit-exact. Số dòng còn lại giống nhau giữa các arm trong mỗi block; số vòng cuối khớp policy đã thực thi.
- Tính lại trọng số từ các sai số **đã quan sát trước quyết định**: khớp cả 152 batch DT, 148 batch đối chứng mới và trọng số cuối. Việc này kiểm tra chuỗi cập nhật trọng số; không tái dựng toàn bộ trạng thái dự báo theo client từ raw telemetry.
- 33 file mã/cấu hình trong manifest mới và 30 file của lần online trước còn đúng hash. Vòng fit dài nhất 0,718 giây, dưới giới hạn gián đoạn 60 giây đã chốt.
- Tổng thời gian lập kế hoạch và cập nhật DT là **0,288 giây**, trung bình **1,893 ms/batch**, bằng khoảng **0,046%** tổng service time. `rows_queue` tương ứng 0,362 giây, 2,443 ms/batch; khác lịch phục vụ và số lần lập lại kế hoạch nên đây không phải phép benchmark tốc độ predictor riêng. Telemetry/log sau model-ready vẫn trì hoãn job kế tiếp. Log legacy không cộng các lượt lập kế hoạch chỉ dẫn tới chờ, nên không dùng tổng đó để tuyên bố so sánh overhead đầy đủ.

## Kết luận sử dụng và giới hạn

Kết quả ủng hộ giữ **controller mới**. Trong ba cấu hình đã chạy, `rows_queue` có tỷ lệ thành công cao nhất và service time thấp nhất; nó là mốc cần vượt ở vòng nghiên cứu tiếp theo. DT kết hợp vẫn là cấu hình thử nghiệm, chưa có bằng chứng cần chọn nó thay cho đối chứng đơn giản cùng controller. Chưa thay thuật toán mặc định của runner cũ.

Phạm vi chỉ gồm một máy CPU, một worker, 4 shard, full/fixed, stress nhân tạo lấy mẫu đầu mỗi batch và ba lần lặp. Dữ liệu/partition/validation vẫn như các giai đoạn trước; seed/chuỗi mới không biến validation thành tập dữ liệu độc lập. Không dùng test/calibration trong lượt này. FPR vẫn khoảng 24,2%; cải thiện deadline không giải quyết chất lượng nền của bộ phát hiện. Chưa kiểm chứng HTTP ingress, truyền thông phân tán thật, checkpoint bền vững, điều chỉnh giữa chừng, hard deadline hay ưu thế so với mọi phương pháp ngoài repo.

## Tệp kết quả và tái kiểm tra

- [Bảng do analyzer xuất](../results/dt_predictive_confirmation/report.md), [so sánh ghép cặp](../results/dt_predictive_confirmation/paired.csv), [tiêu chí](../results/dt_predictive_confirmation/targets.json).
- [Tổng hợp chính xác](../results/dt_predictive_confirmation/aggregate_metrics.json), [audit bổ sung](../results/dt_predictive_confirmation/supplemental_audit.json), [xác nhận kiểm thử](../results/dt_predictive_confirmation/validation.json).
- [Protocol lock](../results/dt_predictive_confirmation/protocol_lock.json); mỗi thư mục block có `run.json`, chi phí fit, `events.jsonl` và `final.json` cho từng arm.

```powershell
python scripts/dt/analyze_predictive_confirmation.py results/dt_predictive_confirmation
python results/dt_predictive_confirmation/audit_replay.py
```

Hai lệnh trên chỉ phân tích log, không huấn luyện lại. Audit bổ sung được viết sau khi chạy để kiểm tra tính nhất quán; nó không thay đổi thuật toán, ngưỡng hay dữ liệu thực nghiệm đã chốt.
