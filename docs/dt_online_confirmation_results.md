# Kết quả xác nhận DT online và deadline hoàn thành

Ngày 2026-11-17. **Đã thực hiện đủ hai việc:** chạy online với đối chứng trên cấu hình chốt trước; đo completion latency bằng thời gian chờ cộng xử lý thật tới khi model trong bộ nhớ sẵn sàng phục vụ.

**Kết luận:** bộ mới chưa xác nhận ưu thế của DT Adaptive (alpha=0,3) so với luật đơn giản. DT và `last_shard` cùng đạt 151/216 yêu cầu vừa ngân sách vừa đúng hạn; `rows_ewma` đạt 161/216. Việc tính cả service time là cần thiết: có 128 yêu cầu được đánh giá nhầm là đúng hạn nếu chỉ xét thời gian chờ. Đây là bằng chứng trên controller và máy đã thử, không phải kết luận thống kê cho mọi triển khai.

## Thiết kế đã thực hiện

[Protocol](dt_online_confirmation_plan.md) và [cấu hình](../configs/dt_online_confirmation.json) được chốt trước các lần đo. [Manifest](../results/dt_online_confirmation/protocol_lock.json) ghi hash mã Python và cấu hình; đối chiếu sau chạy không có file nào lệch.

- 3 seed train mới 71042–71044, 2 họ `spread`/`proportional`, 3 môi trường, 3 predictor; 12 yêu cầu mỗi chuỗi: **18 block ghép cặp, 54 chuỗi, 648 yêu cầu**.
- Ba phương án fork cùng model ban đầu, cùng lịch và nội dung yêu cầu, cùng giá trị budget/deadline trong block. Cùng luật gom lô/chọn full hoặc fixed; chỉ khác bộ dự báo. Thứ tự arm và môi trường xoay vòng.
- DT và `rows_ewma` giữ alpha=0,3; cả ba dùng biên q90, cửa sổ 10 quan sát. Không thử lại alpha hoặc kiểu biên sau khi thấy bảng này.
- F là thời gian fit shard lớn nhất của block; F hợp lệ nằm trong **6,985–8,025 giây**. B=0,75F mỗi yêu cầu, deadline D=1,5F, interarrival trung bình 0,2F. Các block khác nhau không có cùng budget tuyệt đối.
- Stress mới: client chậm tăng dần và bandwidth dao động, cộng điều kiện không stress. Sleep thật làm tăng chi phí thực thi; update model không đổi vì stress.
- Mọi yêu cầu cuối cùng đều được xử lý, kể cả khi không thể đáp ứng dự báo ngân sách/deadline; mọi chi phí và vi phạm đều được ghi. Không có phản hồi miễn phí từ yêu cầu bị từ chối hay phương án chưa chạy.
- Chỉ đánh giá utility trên validation. Không dùng seed v3 60042–60051; không chấm calibration/test.

## So sánh chính

Một yêu cầu thành công khi model mới sẵn sàng trước deadline của yêu cầu **và** batch chứa nó dùng thời gian vòng FU không quá ngân sách gộp. Mỗi ô có 72 yêu cầu (3 repeat × 2 họ × 12).

| Môi trường | DT Adaptive | `last_shard` | `rows_ewma` |
|---|---:|---:|---:|
| Không stress | 62/72 | 62/72 | 66/72 |
| Client chậm tăng dần | 49/72 | 48/72 | 50/72 |
| Bandwidth dao động | 40/72 | 41/72 | 45/72 |
| **Tổng** | **151/216 (69,91%)** | **151/216 (69,91%)** | **161/216 (74,54%)** |

Đối chiếu từng block theo chỉ số trên:

| DT so với | DT thắng | Hòa | DT thua | Chênh tổng số yêu cầu thành công |
|---|---:|---:|---:|---:|
| `last_shard` | 5 | 11 | 2 | 0 |
| `rows_ewma` | 2 | 8 | 8 | −10 |

DT thắng nhiều block hơn thua khi so với `last_shard`, nhưng tổng số yêu cầu thành công vẫn bằng nhau: độ lớn chênh lệch cũng quan trọng. `rows_ewma` có số điểm tốt hơn trong bộ này; không suy ra rằng nó luôn tốt hơn DT.

## Chi phí và chất lượng đi kèm

| Chỉ số gộp 18 chuỗi mỗi phương án | DT Adaptive | `last_shard` | `rows_ewma` |
|---|---:|---:|---:|
| Deadline miss, trên 216 yêu cầu | 63 | 62 | 52 |
| Completion latency trung bình (giây) | 8,445 | 8,381 | 8,155 |
| Thời gian chờ trung bình (giây) | 4,617 | 4,570 | 4,383 |
| Tổng service time (giây) | 560,696 | 562,059 | 552,086 |
| Số batch | 151 | 155 | 154 |
| Batch vượt ngân sách vòng FU | 8 | 10 | 7 |
| Yêu cầu được phục vụ bằng full 20 vòng | 106 | 110 | 104 |
| Macro-F1 validation cuối, trung bình theo chuỗi | 0,893837 | 0,893817 | 0,893739 |
| FPR validation cuối, trung bình theo chuỗi | 0,239811 | 0,239937 | 0,239956 |

Service time đo từ lúc gọi thực thi đến lúc model sẵn sàng; không cộng thời gian chờ. Budget vẫn có đơn vị tổng thời gian vòng FU, để giữ nghĩa từ thí nghiệm trước. `summary.csv` còn ghi số lần service time vượt budget để phân biệt hai đại lượng này.

F1 trung bình gần nhau, nhưng **không có kiểm định tương đương/noninferiority**. Chênh F1 ghép cặp của DT so với `last_shard` nằm trong [−0,003279; +0,003808], so với `rows_ewma` trong [−0,003088; +0,003780]. Full/fixed mang số vòng khác nhau; bit-exact không chứng minh fixed giữ nguyên utility của full.

`forced_over_budget` nghĩa là dự báo fixed vượt budget nhưng vẫn phải xử lý, không phải số vi phạm thực đo. Ba phương án lần lượt có 12, 11, 20 lần forced; số vi phạm thực đo trong bảng nhỏ hơn. Không dùng hai cột này thay cho nhau.

## Deadline khi có xử lý thật

Đồng hồ `perf_counter` chạy liên tục trong lúc train. Yêu cầu tới khi worker bận vẫn tích lũy thời gian chờ. Model-ready được ghi khi thao tác FU trả về và model mới dùng được trong bộ nhớ. Sync/log sau đó làm chậm yêu cầu kế tiếp; validation và retrain kiểm chứng chỉ chạy sau khi kết thúc cả chuỗi.

- **177/648 yêu cầu trễ deadline hoàn thành.**
- Trong đó **128 yêu cầu có thời gian chờ <= D nhưng completion latency > D**. Nếu chỉ chấm thời gian chờ thì chỉ thấy 49 yêu cầu trễ, bỏ sót 128 trường hợp.
- Riêng DT: 63 yêu cầu trễ, 46 trường hợp trễ trên 1 giây; trung vị phần trễ của các yêu cầu vi phạm là **3,321 giây**, lớn nhất **12,724 giây**. Không thể giải thích toàn bộ bằng nhiễu vài mili giây ở ngưỡng.

Đây là phân rã thời gian của **chính lịch thực đã chạy**. Nó không phải phép so sánh nhân quả với scheduler cũ, vốn dùng workload và luật thời gian khác. Dự báo hiện vẫn dựa vào thời gian vòng FU cộng 0,05 giây dự phòng service; không có bảo đảm deadline cứng khi sai số dự báo hoặc tải cao. Luật cũng chưa tối ưu ảnh hưởng của một batch đang chạy lên deadline của các shard khác.

## Kiểm tra và sự cố vận hành

- **99/99 unit test pass**, gồm 7 test mới: yêu cầu đến khi đang train, tách wait/service, gom lô, dự báo thiếu gây deadline miss, xử lý bắt buộc, thời gian log làm chậm job kế tiếp, ba arm xóa cùng dữ liệu và bit-exact.
- **54/54 chuỗi đạt bit-exact** đối chiếu với retrain tại số vòng tương ứng; exactness và cấu trúc đạt ở mọi event.
- Audit xác nhận mỗi chuỗi xử lý đủ 12 yêu cầu, đúng một lần; không đưa yêu cầu tương lai vào batch; không chồng thời gian thực thi của một worker; latency = wait + service; số deadline miss và bảng tổng hợp khớp event gốc.
- Hash mã Python/cấu hình khớp manifest trước chạy. Lệnh Windows `-Resume` đã được thử sau khi hoàn tất: nhận đủ block đã chạy, không train lại.
- Một lần fit bị host gián đoạn hơn bốn giờ trước khi có yêu cầu online trong block đó. Đã giữ nguyên dữ liệu ở [thư mục loại khỏi thống kê](../results/dt_online_confirmation_interrupted/), ghi [incident](../results/dt_online_confirmation/incident.json), chạy lại riêng block bị ảnh hưởng với cùng cấu hình. Năm block hoàn chỉnh trước đó được giữ. Không xác minh độc lập được nguyên nhân là sleep hay host pause. Phần tiếp tục dùng cơ chế giữ máy thức theo tiến trình; không đổi power plan.

## Phạm vi kết luận và file bàn giao

Đây là **xác nhận triển vọng cho một cấu hình local**, với 3 repeat, một mức tải/budget/deadline và stress nhân tạo. Các yêu cầu trong chuỗi có phụ thuộc; không xem 648 yêu cầu là 648 mẫu độc lập. Kết quả không chứng minh ưu thế thống kê hay SLA triển khai phân tán.

Controller này chọn trước full/fixed và gom lô; **không dùng checkpoint giữa chừng**. Do đó không dùng bảng này để xác nhận hay bác bỏ trực tiếp lợi ích midcourse của bộ cũ. Điểm hoàn thành chưa gồm HTTP ingress, mạng telemetry, checkpoint bền vững hay chuyển model sang dịch vụ khác. Scheduler mô phỏng và kết quả cũ được giữ nguyên để đối chiếu lịch sử.

- [Bảng chi tiết và báo cáo sinh tự động](../results/dt_online_confirmation/report.md); [mỗi chuỗi](../results/dt_online_confirmation/summary.csv); [ghép cặp từng block](../results/dt_online_confirmation/paired.csv).
- [Scheduler thời gian thực](../src/dt/realtime.py), [runner](../scripts/dt/run_online_confirmation.py), [lệnh chạy Windows](../scripts/dt/run_online_confirmation.ps1), [script audit/tổng hợp](../scripts/dt/analyze_online_confirmation.py), [test mới](../tests/test_dt_realtime.py).

Nếu đổi alpha, biên hoặc controller sau bảng này, cần ghi thành cấu hình thăm dò mới và có một tập chạy xác nhận khác. Bộ hiện tại nên được giữ nguyên làm mốc.
