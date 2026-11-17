# DT online có đối chứng và deadline hoàn thành

Protocol ngày 2026-11-17, viết trước khi chạy dữ liệu xác nhận. Cấu hình máy đọc: `configs/dt_online_confirmation.json`; runner ghi `protocol_lock.json` (cấu hình, hash mã, thời điểm UTC) **trước lần fit đầu tiên**. Đây là xác nhận triển vọng trên máy local cho giả thuyết đã có, không phải chứng minh triển khai phân tán hay gate FU v3.

## Câu hỏi và chỉ số chốt trước

1. Khi cùng luật lập lịch và cùng điều kiện, DT Adaptive có cải thiện số yêu cầu vừa ngân sách vừa hoàn thành đúng deadline so với `last_shard` và `rows_ewma` không?
2. Kết quả deadline thay đổi thế nào khi tính cả chờ, lập kế hoạch và thực thi thật đến model-ready, thay vì coi train tức thời?

Chỉ số chính: số yêu cầu có completion latency <= D **và** lần xử lý của nó có tổng thời gian vòng FU <= ngân sách gộp. Chỉ số phụ: deadline miss, latency mean/p95/max, thời gian chờ, số lần và số giây vượt ngân sách, tổng service time, dòng × vòng, tỉ lệ full, macro-F1/FPR validation cuối, bit-exact. Tất cả yêu cầu đều ở mẫu số. Không chấm false rejection hoặc regret khi không đo phản thực tế.

## Thiết kế

- CPU local, 2 torch thread, standard 20/10 vòng, 20 client, S=4, cùng dữ liệu và partition như DT trước. Không sửa thành phần FU hay protocol v3.
- 3 lần lặp: seed train 71042–71044; 2 họ `spread`, `proportional`; 3 môi trường; 3 predictor = **18 block ghép cặp, 54 chuỗi, 648 yêu cầu**. Mỗi chuỗi 12 yêu cầu xóa mẫu, seed chuỗi mới 81000 + 100·lần_lặp + offset. Không dùng seed v3 60042–60051. Chỉ train và chấm validation, không đánh giá calibration/test.
- Mỗi block fit một lần. Ba arm fork cùng model, cùng dòng cần xóa, thời điểm đến, ngân sách, deadline và hàm stress. Hiệu chuẩn không stress và không cập nhật bằng kết quả của arm khác. Thứ tự arm xoay vòng theo repeat/family/environment; ba repeat đặt mỗi predictor ở mỗi vị trí một lần. Chạy tuần tự trên cùng máy, không chạy benchmark khác đồng thời.
- F = thời gian 20 vòng của shard lớn nhất trong fit của block; B = 0,75F mỗi yêu cầu, D = 1,5F; interarrival Poisson trung bình 0,2F. Các hệ số cố định trước; cả ba arm dùng **đúng cùng giá trị số** trong block. Không so trực tiếp các block như có cùng budget tuyệt đối.
- Cả ba predictor dùng alpha=0,3 nếu có EWMA, biên q90 của 10 sai số gần nhất, cùng cách khởi tạo từ fit; không chọn alpha=1 hoặc biên rescored sau khi xem số mới.
- Môi trường: `none`; `gradual_clients` (một client/shard, hệ số chậm tăng tuyến tính 1→4 từ 20% đến 80% horizon arrival); `jitter_bandwidth` (sau 20% horizon, bandwidth Uniform[5,20] Mbit/s theo bucket 1 giây, seed 91000+repeat). Stress theo đồng hồ elapsed của từng arm; lấy mẫu khi bắt đầu mỗi batch và giữ trong batch. Độ trễ được sleep thật, không đổi update model. Đây là stress nhân tạo, không phải đo mạng thực.

## Luật chung và ranh giới thời gian

Một worker, hàng đợi theo shard. Chỉ yêu cầu đã tới mới được predictor nhìn thấy. Mỗi yêu cầu mang B giây vòng FU; batch n yêu cầu được nB. Chạy full ngay nếu dự báo × biên vừa ngân sách gộp và vừa thời gian còn lại tới deadline sớm nhất (cộng 0,05 giây dự phòng service). Nếu chưa vừa, chờ thêm yêu cầu, tối đa tới deadline sớm nhất trừ dự báo fixed × biên và dự phòng. Tới lúc đó chạy fixed kể cả không khả thi và ghi rõ forced/overrun/miss. Chọn batch đủ điều kiện có deadline sớm nhất. Không giả định có thể bảo đảm deadline khi dự báo sai hoặc hệ thống quá tải.

Luật này dùng lựa chọn trước khi chạy full/fixed và gom lô, **không dùng checkpoint giữa chừng**: thành phần `unlearn_batch` hiện hỗ trợ hai chính sách đó. Vì vậy kết quả xác nhận so sánh predictor trong controller gom lô này, không xác nhận trực tiếp bảng midcourse cũ.

- Arrival: timestamp đã định trước trên `perf_counter`, độc lập với lúc worker rảnh. Harness phát traffic local; không có HTTP ingress. Yêu cầu tới trong lúc train vẫn bắt đầu tính thời gian chờ, được nhận vào hàng đợi ở lần worker kiểm tra tiếp theo.
- Model-ready: lúc `unlearn_batch` trả về, model trong bộ nhớ đã thay thế và có thể dự đoán. Completion latency = model-ready − arrival; waiting = bắt đầu execute − arrival; service = model-ready − bắt đầu execute. Hạch toán này gồm lập kế hoạch/chờ trước execute và toàn bộ thao tác FU, kể cả payload packing nằm ngoài vòng train. Ghi thêm service-over-budget để không nhầm round-budget với tổng thời gian.
- Sync telemetry và ghi log sau model-ready không tăng latency yêu cầu vừa xong, nhưng làm chậm worker và tăng chờ của yêu cầu kế tiếp. Chấm validation và verify bit-exact **sau toàn chuỗi**, không chen vào lịch.
- Không từ chối rồi chạy ngoài ngân sách để lấy feedback miễn phí. Mọi yêu cầu cuối cùng được xử lý và hạch toán; không có quan sát từ phương án chưa chạy.
- Không đo HTTP/network ingress, lưu checkpoint bền vững, distributed serving hoặc truyền telemetry. Model-ready là ranh giới in-process, chưa phải SLA dịch vụ triển khai.

## Cách kết luận và tái lập

So DT với từng đối chứng **trong block**: chênh số joint-success, deadline miss, service cost, full share và utility. Báo cả từng block và gộp môi trường; đếm win/tie/loss. Ba lần lặp chưa đủ để khẳng định ý nghĩa thống kê; không xem 648 yêu cầu là 648 mẫu độc lập. Không chọn lại cấu hình sau khi thấy kết quả. Nếu lỗi triển khai làm hỏng phép đo, giữ lại dữ liệu cũ, ghi nguyên nhân và tạo lần chạy mới.

```powershell
$env:TMP = 'E:\budget-aware-code\tmp'
$env:TEMP = $env:TMP
python -m unittest discover -s tests -p 'test_dt_realtime.py' -v
python scripts/dt/run_online_confirmation.py results/dt_online_confirmation
python scripts/dt/analyze_online_confirmation.py results/dt_online_confirmation
```

Đây là thay đổi có chủ đích so với scheduler mô phỏng cũ. Không ghi đè kết quả ngày 15–16/11; tên trường mới luôn ghi rõ seconds, deadline hoàn thành và phạm vi budget.

## Nhật ký vận hành sau khi khóa cấu hình

Trong lần chạy 2026-11-17, host bị gián đoạn hơn bốn giờ giữa lúc fit của block `r0_proportional_none`, làm F thành 14.861,62 giây. Chưa yêu cầu online nào trong block đó được thực thi. Đã dừng, giữ nguyên dữ liệu gián đoạn tại `results/dt_online_confirmation_interrupted/`, và chạy lại riêng block đó với mã/cấu hình đã khóa; năm block hoàn chỉnh trước đó được giữ. Chi tiết tại `results/dt_online_confirmation/incident.json`. Không xác minh độc lập được nguyên nhân là sleep hay host pause.

Phần tiếp tục dùng `ES_SYSTEM_REQUIRED` để ngăn Windows tự sleep trong phạm vi tiến trình, tự trả lại trạng thái khi thoát; không đổi power plan hay tham số thí nghiệm. Wrapper tương đương được lưu để tái lập trên Windows:

```powershell
./scripts/dt/run_online_confirmation.ps1 -Output results/dt_online_confirmation_new
# Chỉ tiếp tục khi các block đã tạo đều hoàn chỉnh, cùng mã/cấu hình:
./scripts/dt/run_online_confirmation.ps1 -Output results/dt_online_confirmation_new -Resume
```
