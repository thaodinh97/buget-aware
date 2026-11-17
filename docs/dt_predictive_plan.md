# DT dự báo service và lập kế hoạch hàng đợi — protocol phát triển và xác nhận

Ngày 2026-11-17. Mục tiêu: cải thiện DT dựa trên nguyên nhân thất bại trong bộ online trước; kiểm tra trên lần chạy mới, không sửa số liệu cũ hay chỉnh tham số cho đến khi thắng.

## Bằng chứng phát triển đã xem

Nguồn: `results/dt_online_confirmation/` và thực nghiệm mở rộng trước đó. Những dữ liệu này từ nay là dữ liệu phát triển cho bản mới, không phải holdout của bản mới.

- DT cũ có 59 batch trễ; 57 dùng fixed, chỉ 2 dùng full.
- 41/59 batch trễ có thời gian vòng FU thực <= dự báo sau biên. Dự báo round cost tốt hơn một mình không sửa được các lỗi này.
- Service time trừ round time có trung vị 0,120 giây (max 0,265), trong khi scheduler cũ chỉ dự phòng 0,05 giây.
- Cơ chế cũ chờ tới latest-start của fixed khi full không vừa ngân sách; nó không kiểm tra tác động của một job lên các deadline còn lại.

## Thay đổi được chốt trước lần chạy mới

Mã mới ở `src/dt/predictive.py`; giữ nguyên `src/dt/realtime.py`, predictor cũ, FU và mọi kết quả cũ.

1. **Dự báo service:** round cost × biên q90 (không nhỏ hơn 1) cộng overhead học từ các service thực đã hoàn tất, chuẩn hóa theo client-vòng, cộng 0,05 giây dự phòng. Khởi tạo phần đóng gói bằng số đo fit; không đưa thời gian chấm validation vào overhead mỗi job.
2. **DT kết hợp:** bốn expert: mô hình theo client alpha=0,3; mô hình theo client alpha=1; `last_shard`; `rows_ewma` alpha=0,3. Cập nhật EWMA 0,3 của sai số log tuyệt đối giữa số đo và dự báo **trước cập nhật**; trọng số softmax(−3·loss), trộn 10% uniform. Mọi expert chỉ học từ thao tác đã thực thi. Đây là một cấu hình phát triển, không có khẳng định lý thuyết rằng ensemble luôn tốt hơn từng expert.
3. **Dự tính hàng đợi:** chỉ nhìn các yêu cầu đã tới, gom theo shard; thử mọi thứ tự và full/fixed của tối đa 4 shard (384 kế hoạch). Tối đa hóa lần lượt số yêu cầu dự báo vừa budget/đúng hạn, giảm số yêu cầu vượt budget, tăng số yêu cầu full đúng hạn, giảm tổng phần trễ rồi tổng completion latency. Chỉ thực thi hành động đầu và lập lại kế hoạch sau đó.
4. **Bỏ chờ không cần thiết:** fixed vừa ngân sách thì có thể chạy ngay; không chờ sát hạn chỉ để mong gom đủ full. Nếu ngay cả fixed chưa vừa ngân sách thì chỉ chờ gom thêm khi còn slack và không có shard khác đủ ngân sách để chạy; tới hạn bắt đầu phải xử lý và ghi đầy đủ overrun/miss. Không bỏ yêu cầu.

Bản mới có thể hạ số vòng nhiều hơn; vì vậy không chỉ chấm deadline. Phải báo full share, utility và service cost. Không dùng thông tin về future arrivals, chi phí thật của phương án chưa chạy hoặc utility validation để chọn hành động.

## Đối chứng và lần chạy xác nhận mới

`configs/dt_predictive_confirmation.json` được hash và lưu trước fit. Không đổi cấu hình khi xem kết quả.

| Arm | Bộ dự báo | Controller |
|---|---|---|
| `dt_queue` | Twin kết hợp bốn expert | Service-aware và dự tính hàng đợi mới |
| `rows_queue` | Chỉ `rows_ewma` | **Cùng controller, overhead và biên mới** |
| `rows_legacy` | `rows_ewma` | Controller cũ nguyên bản |

`rows_legacy` là đối chứng có điểm tổng cao nhất ở bộ trước. `rows_queue` tách lợi ích của controller khỏi lợi ích riêng của predictor DT. Không được quy mọi cải thiện so với legacy cho mô hình theo client, và không được tuyên bố hơn mọi phương pháp hiện có ngoài phạm vi các đối chứng này.

- Seed train mới 72042–72044; seed chuỗi 82000+100·repeat+offset; stress seed 92000. Không dùng lại seed xác nhận trước 71042–71044 hay seed v3 60042–60051.
- 3 repeat × 2 họ × 3 môi trường × 3 arm × 12 yêu cầu = **18 block, 54 chuỗi, 648 yêu cầu**. Họ `spread`, `proportional`; môi trường không stress, client chậm tăng dần, bandwidth dao động. Dữ liệu/partition/S=4/standard 20–10 vòng giữ như cũ.
- Mỗi block fit một lần và fork cùng model; cùng giá trị B=0,75F, D=1,5F, interarrival trung bình 0,2F và cùng đường stress cho ba arm. Thứ tự xoay vòng. Không so trực tiếp số tổng này với bộ 71042; đối chứng legacy được chạy mới trong cùng block.
- Chạy tuần tự, giữ máy thức theo tiến trình trên Windows. Nếu một vòng fit >60 giây, dừng trước mọi request, giữ bản ghi để kiểm tra gián đoạn máy; không tự điều chỉnh ngân sách theo kết quả online.
- Deadline và budget giữ nguyên nghĩa ở protocol trước: completion latency = chờ + service tới model trong bộ nhớ; budget = tổng giây các vòng FU của batch. Validation/bit-exact chấm sau toàn chuỗi. Không chấm calibration/test.

## Tiêu chí thực dụng chốt trước

So `dt_queue` với `rows_legacy`, trên toàn bộ dữ liệu mới:

1. Tỉ lệ yêu cầu vừa budget vừa đúng hạn tăng ít nhất **5 điểm phần trăm**.
2. Chênh chỉ số đó dương ở ít nhất **2/3 môi trường**.
3. Tổng service time không quá **1,10 lần** legacy.
4. Chênh Macro-F1 validation ghép cặp: trung bình không dưới **−0,002**, block xấu nhất không dưới **−0,005**.
5. Exactness/cấu trúc và bit-exact đạt ở mọi chuỗi.

Các ngưỡng này là mục tiêu thực dụng trên mẫu nhỏ, không phải yêu cầu nghiệp vụ, kiểm định noninferiority hoặc bảo đảm thống kê. Báo cả tiêu chí fail. So với `rows_queue` bằng chênh ghép cặp, win/tie/loss, utility và chi phí, không chọn lại tham số sau khi xem bảng.

Nếu hơn legacy nhưng không hơn `rows_queue`, kết luận là **cải thiện chủ yếu từ service/queue-aware control; lợi thế riêng của twin kết hợp chưa được xác nhận**. Nếu gate utility fail thì không tuyên bố cải thiện không đánh đổi chất lượng. Một cấu hình không thể được sửa trên bộ mới rồi tiếp tục gọi bộ đó là xác nhận.

## Kiểm tra và tái lập

Test phát triển gồm trường hợp fixed vừa budget nhưng chờ sát hạn gây miss, chọn order/policy để giữ nhiều yêu cầu full đúng hạn, không nhìn future arrival, học sai số trước cập nhật, hạch toán job không khả thi, và đối chiếu model cuối của ba arm trên dữ liệu synthetic. Các test này không dùng seed/dữ liệu xác nhận mới.

```powershell
$env:TMP = 'E:\budget-aware-code\tmp'
$env:TEMP = $env:TMP
python -m unittest discover -s tests -p 'test_dt_predictive.py' -v
python scripts/dt/run_predictive_confirmation.py results/dt_predictive_confirmation
python scripts/dt/analyze_predictive_confirmation.py results/dt_predictive_confirmation
```

Phạm vi: một worker CPU local, traffic phát theo đồng hồ thực, stress nhân tạo và lấy mẫu khi bắt đầu batch. Không đo HTTP ingress, network telemetry, checkpoint bền vững hay distributed serving. Controller không có bảo đảm hard deadline và không biết các yêu cầu chưa tới.
