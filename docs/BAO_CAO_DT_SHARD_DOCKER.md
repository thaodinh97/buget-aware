# Báo cáo Digital Twin cho federated unlearning theo shard trên Docker GPU

## Vai trò của DT trong khóa luận

Khi một client yêu cầu xóa dữ liệu, hệ thống cần trả lời hai câu hỏi. FU xác định phần mô hình cần huấn luyện lại để đáp ứng yêu cầu xóa. DT ước lượng việc đó sẽ tiêu tốn bao nhiêu thời gian và dữ liệu trao đổi, rồi chọn một phương án FU có thể chạy trong ngân sách. Sau khi chạy, coordinator đối chiếu dự báo với số đo. Mục đích là có thể giải thích cả quyết định chấp nhận hoặc từ chối lẫn chi phí và chất lượng của mô hình thu được.

| Phase | Đầu vào và quyết định | Đầu ra bàn giao |
|---|---|---|
| Tiền xử lý | CSV TON-IoT, cấu hình connected_v3_auxfit_seed42; fit phép biến đổi trên validation để việc xóa train không đổi mã hóa | Các split có row_id, manifest, audit và hash nội dung |
| FL | 20 client, Dirichlet α=0,5, partition seed 42, assignment seed 7; 4 shard; FedAvg riêng ở mỗi shard, 20 vòng, 2 local epoch, Adam lr 0,01, batch 512 | Ensemble, trọng số từng shard, cấu hình, danh sách client và trạng thái dữ liệu |
| FU | Checkpoint FL và yêu cầu xóa toàn client hoặc một tập row_id; xác định dữ liệu giữ lại, shard bị ảnh hưởng và thay đổi cấu trúc | Có thể huấn luyện lại shard từ đầu trên dữ liệu giữ lại theo chính sách được chọn; xuất checkpoint và kiểm tra exactness |
| DT và coordinator | Trước khi FU thực thi, đọc phạm vi công việc mà FU xác định, chính sách ứng viên, dự báo chi phí và ngân sách; sau khi chạy, thu số đo | Kế hoạch, quyết định, chi phí dự báo/thực tế, sai số, trạng thái ngân sách và checkpoint FU nếu được chấp nhận |

Trong báo cáo, “DT” là mô hình số về chi phí thực thi của hệ thống FL/FU. Có hai cách duy trì mô hình này. **DT Static** giữ thông số hiệu chuẩn ban đầu, cung cấp mốc để đánh giá giá trị của việc cập nhật số đo. **DT Adaptive** cập nhật dự báo bằng EWMA (trọng số quan sát mới 0,3) để theo dõi thay đổi của thời gian thực thi. Trong chuỗi một tiến trình, predictor cập nhật theo client và phần server. Trong pilot RPC lặp một yêu cầu, predictor giữ một ước lượng thời gian riêng cho full và fixed. Cả hai dùng dự báo được ghi trước khi có kết quả của lượt đang chấm. Full/fixed là hai chính sách FU (20/10 vòng); wide/mid/low là các mức ngân sách; Docker là môi trường thực thi. Planner dùng quy tắc ưu tiên full rồi fixed. Pilot admission HTTP dùng predictor hiệu chuẩn tĩnh trong PowerShell.

## Đối chiếu với nhánh feature/shard-fu ngày 11/11/2026

**Đồng bộ về thiết kế và đường bàn giao FL → FU → DT; phiên bản mã và quy trình đo có khác biệt.** DT local được phát triển trên nền commit 22cec5a. Đầu nhánh GitHub kiểm tra ngày 11/11 là 3e4617dd0e4cdcd29b1110398a69e029a48f506d. Commit mới bổ sung warm-up backend trước lần train đầu trong tiến trình, trường backend_warmup_seconds, replay trên dữ liệu thật và đối chứng trên cùng máy. Số thời gian và bit-exact của một môi trường cần được đối chiếu với reference cùng môi trường. [Thay đổi giữa hai phiên bản](https://github.com/thaodinh97/buget-aware/compare/22cec5a...3e4617dd0e4cdcd29b1110398a69e029a48f506d)

* Cùng giao diện: FL lưu state, FU nạp state, áp dụng ForgetRequest và ghi checkpoint kế tiếp. DT sử dụng chính ShardedUnlearner, cấu trúc shard, seed và FedAvg của luồng này. Exactness được kiểm tra bằng retrain tại số vòng tương ứng. [Giao diện upstream](https://github.com/thaodinh97/buget-aware/blob/3e4617dd0e4cdcd29b1110398a69e029a48f506d/docs/shard_dt_interface.md)
* Khác phạm vi: upstream có sáu chính sách FU. Planner local nhận hai chính sách số vòng cố định để dự toán trước khi train. Local bổ sung lựa chọn CUDA, hook worker RPC, dự báo, ngân sách và Compose. RPC thử yêu cầu xóa client 10, còn chuỗi xóa mẫu và xóa client chạy trong một tiến trình. CPU replay upstream báo kết quả đối chiếu trên máy của tác giả; kết quả Docker GPU dưới đây là thực nghiệm của đồ án. [Kết quả upstream](https://github.com/thaodinh97/buget-aware/blob/3e4617dd0e4cdcd29b1110398a69e029a48f506d/results/shard_dt/analysis.txt)
* Kiểm chứng tại chỗ: kiểm tra hash dữ liệu v2, cùng checkpoint cha cho FU local và RPC, kiểm tra client tham gia, số vòng và trọng số từng shard. Mô hình fixed khớp reference 10 vòng; tác động chất lượng so với full 20 vòng phải đọc từ Macro-F1 và FPR.

## Mục tiêu và thiết kế

Thí nghiệm kiểm tra liệu coordinator có thể chọn số vòng huấn luyện lại theo giới hạn thời gian và byte, đồng thời định lượng sự đánh đổi chất lượng phát hiện. Mỗi lần xóa tác động lên một ensemble gồm bốn shard. FU xác định shard và dữ liệu còn lại, còn DT chọn phương án huấn luyện. Hệ thống hỗ trợ xóa toàn bộ client và xóa một phần mẫu của client.

Chuỗi đánh giá gồm 20 client logic và 10 yêu cầu tích lũy (5 xóa client, 5 xóa mẫu). Ba seed huấn luyện 50042, 50043 và 50044 dùng cùng phân vùng và chuỗi yêu cầu. Trong mỗi seed, ba cấu hình ngân sách xuất phát từ cùng checkpoint. Baseline wide chạy đủ 20 vòng. Macro-F1 và tỷ lệ dương tính giả (FPR) được đối chiếu trên validation sau từng bước. Chênh lệch tại một bước thể hiện cả lịch sử xóa trước đó.

## Thuật toán DT và cách chọn budget

Ở mỗi yêu cầu, planner lấy các client còn hoạt động trong shard cần train sau khi áp dụng quy tắc cấu trúc của FU. Với R vòng, C client và mô hình có B byte trọng số, trao đổi tensor hai chiều là 2 × R × C × B. Trong cấu hình này một client trao đổi 36.624 byte mỗi vòng. Dự báo thời gian vòng của bản tuần tự cộng thời gian tính theo số mẫu từng client với phần server. Static dùng hiệu chuẩn từ lần fit. Adaptive cập nhật ước lượng chi phí của các client quan sát được và phần server sau tác vụ, với trọng số 0,3 cho quan sát mới. Hai predictor ghi dự báo cho cùng phương án thực thi để tính MAE theo cặp.

Planner ưu tiên shard_full (20 vòng), sau đó xét shard_fixed (10 vòng) nếu được bật. Phương án cần đáp ứng cả giới hạn byte và thời gian dự báo. Khi không phương án nào đáp ứng, planner trả NO_FEASIBLE_PLAN trước khi train. Replay dừng tại yêu cầu bị từ chối vì các bước sau giả định yêu cầu đó đã hoàn thành.

| Budget tensor/yêu cầu | Lý do chọn | Kết quả |
|---|---|---|
| 1.464.960 | 4 client × 10 vòng | Seed 50042 hoàn thành bước 1, từ chối bước 2 có 5 client |
| 1.831.200 | 5 client × 10 vòng | 10 lượt fixed ở mỗi seed |
| 2.929.920 | 4 client × 20 vòng | 8 full, 2 fixed ở mỗi seed |
| 10.000.000 | Đủ rộng để lấy mốc full | 10 full ở mỗi seed |

Giới hạn thời gian 300 giây cho mỗi yêu cầu cho phép khảo sát riêng tác động của ngân sách byte. Số giây trong bảng là tổng thời gian các vòng huấn luyện. Thời gian dựng container, nạp dữ liệu và xác minh độc lập nằm ngoài chỉ số này.

## Kết quả chuỗi trong một tiến trình Docker CUDA

| Seed | Giây wide / mid / low | Chênh lệch Macro-F1 xấu nhất của mid | FPR tăng lớn nhất của mid (điểm %) |
|---|---|---:|---:|
| 50042 | 29,116 / 25,713 / 15,282 | −0,006616 | 0,8817 |
| 50043 | 33,855 / 25,307 / 14,594 | −0,002850 | 0,0226 |
| 50044 | 28,815 / 23,927 / 14,686 | −0,007170 | 1,1529 |

Mỗi chuỗi hoàn thành dùng lần lượt 29.299.200, 25.636.800 và 14.649.600 byte. Mid tiết kiệm 12,5% byte và 11,7–25,2% thời gian so với wide. Macro-F1 và FPR cuối chuỗi của mid khớp wide ở độ chính xác báo cáo trên cả ba seed. Đánh đổi trung gian lớn nhất về FPR xuất hiện ở bước 4 của seed 50044. Low tiết kiệm 50% byte nhưng làm Macro-F1 trung gian giảm tới 0,009846 ở seed 50044. Trên seed 50044, MAE thời gian Static/Adaptive lần lượt là 0,648/0,550 giây (wide), 0,611/0,528 (mid), 0,341/0,298 (low). Các seed khảo sát biến thiên huấn luyện trên cùng workload.

## Pilot RPC nhiều container trên cùng RTX 3050 Ti

Seed 50042, xóa client 10 khỏi shard 2. Bốn worker 0, 5, 9, 12 đọc bốn phân vùng riêng, lần lượt 4.768, 4.508, 4.445 và 15.035 mẫu. Coordinator gửi trọng số qua HTTP theo thứ tự client, nhận cập nhật và tổng hợp theo số mẫu. Worker chỉ mount file dữ liệu của chính mình. Coordinator pilot còn truy cập dữ liệu tham chiếu để nạp ensemble và kiểm tra tính tương đương.

| Phương án | Số RPC train | Byte tensor | Byte HTTP body hai chiều | Tổng giây các vòng | Macro-F1 | FPR |
|---|---:|---:|---:|---:|---:|---:|
| full | 80 | 2.929.920 | 2.974.272 | 33,667 | 0,895577 | 0,243811 |
| fixed | 40 | 1.464.960 | 1.485.037 | 6,753 | 0,894788 | 0,247542 |

Cả hai lượt khớp từng bit khi đối chiếu phép tổng hợp vòng đầu. Retrain xác minh cuối tác vụ khớp trọng số của cả bốn shard, đúng tập shard đang phục vụ. HTTP body là độ dài dữ liệu NPZ trong request và response; header và chi phí TCP nằm ngoài phép đo. Với full, body vượt mốc tensor 44.352 byte. Do đó cần chọn rõ đơn vị ngân sách khi chuyển sang mạng thật.

Pilot dùng một lần chạy mỗi phương án, full trước fixed, các worker dùng chung GPU. Thời gian trên phản ánh lần đo đó và có thể chịu ảnh hưởng khởi động. Hai lượt này cung cấp số đo hiệu chuẩn cho các thí nghiệm tiếp theo. Các số MAE và quyết định ngân sách của chuỗi 10 bước ở trên thuộc bản một tiến trình. Hai lượt RPC chọn chính sách rõ ràng để kiểm tra tính tương đương.

Log gốc của pilot: runs/shard_dt/rpc_pair_20261110_211922/{shard_full,shard_fixed}/rpc_report.json và rpc_calls.json.

## Chọn phương án trước khi train theo ngân sách HTTP body

Trên cùng checkpoint và yêu cầu, planner dùng số đo pilot để dự báo trước khi gọi coordinator. Với ngân sách 3.100.000 byte, planner chọn full và dùng 2.974.272 byte qua 80 RPC. Tại ngưỡng 2.929.920 byte, planner chọn fixed và dùng 1.485.037 byte qua 40 RPC. Hai checkpoint đều khớp từng bit. Ở mức 1.400.000 byte, planner trả NO_FEASIBLE_PLAN và phát 0 RPC train. Ngưỡng giữa cho thấy tác động của việc tính ngân sách theo dữ liệu thực gửi trong HTTP body.

## So sánh dự báo thời gian RPC

Sáu lượt đo mới xuất phát từ cùng checkpoint seed 50042, cùng yêu cầu xóa client 10 và cùng bốn worker. Thứ tự là full, fixed, fixed, full, full, fixed. Cả hai predictor ghi dự báo trước mỗi lượt. Static giữ thời gian pilot. Adaptive cập nhật theo EWMA với trọng số 0,3 cho số đo vừa nhận và duy trì một giá trị riêng cho mỗi chính sách. MAE tính trên tổng thời gian các vòng của mỗi tác vụ, đơn vị giây.

| Chính sách | Số lượt | MAE Static | MAE Adaptive | MAE giảm | Khớp byte dự báo | Checkpoint bit-exact |
|---|---:|---:|---:|---:|---:|---:|
| shard_full | 3 | 26,0288 | 19,5708 | 24,81% | 3/3 | 3/3 |
| shard_fixed | 3 | 3,9395 | 2,9316 | 25,59% | 3/3 | 3/3 |

Dự báo ban đầu cao hơn cả sáu số đo. EWMA điều chỉnh theo các quan sát mới nhưng còn trễ. Ở lượt full thứ ba, Adaptive dự báo 20,731 giây và thực tế là 5,139 giây. Ba lần lặp cho mỗi chính sách mô tả cùng một yêu cầu trên một GPU dùng chung. Đánh giá khả năng đáp ứng deadline chặt hoặc tải thay đổi có chủ đích cần một thí nghiệm riêng.

Dữ liệu đối chiếu nằm trong `runs/shard_dt/rpc_adaptation_20261111_200131/runs.csv` và `summary.csv` cùng thư mục. Chuỗi 10 yêu cầu được đánh giá trong Docker một tiến trình. Thí nghiệm nhiều container kiểm chứng việc thực thi và dự báo cho một yêu cầu xóa client cố định.
## Kịch bản trình diễn từ dữ liệu đến DT Docker

Trên máy của đồ án, mở Docker Desktop ở chế độ Linux containers, vào D:\budget-aware. Script dùng image shard-fu-dt-shard, môi trường .venv-gpu, các phân vùng worker đã xuất và hai log RPC hiệu chuẩn trong runs/shard_dt. Các lượt demo ghi vào thư mục demo có timestamp. Worker chạy trong project Compose riêng và được dừng sau demo.

~~~powershell
Set-Location D:\budget-aware
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\demo_dt_docker.ps1
~~~

Để trình diễn cả bước xử lý từ CSV thô, thêm -PrepareRaw vào cuối lệnh. Script tạo split ở thư mục mới, đối chiếu hash nội dung với split đã chuẩn bị và dùng bản đã kiểm chứng được mount vào Docker. Khi thời gian trên lớp giới hạn, dùng lệnh mặc định; giải thích rõ đây là dữ liệu đã tiền xử lý. Để dựng lại image từ checkout hiện tại khi thiếu image, chạy docker build -t shard-fu-dt-shard . trước buổi demo.

| Điểm dừng | Trình bày với hội đồng | Dấu hiệu kiểm tra |
|---|---|---|
| Dữ liệu | Với cùng dữ liệu và phân vùng, chúng ta mới so sánh hai cách thực thi được | Hash v2 bắt đầu f130057e và kết thúc a0ef3a |
| FL | Fit mới ensemble 4 shard tại seed 50042 | 80 vòng tổng, checkpoint base ở step 0 |
| FU đối chứng | Từ checkpoint này, xóa client 10; tạo kết quả 10 vòng tại cùng máy để so trọng số | Shard 2 train lại; các client còn lại 0,5,9,12 |
| DT trước khi train qua RPC | Cho xem hai file plan.json; dự báo lấy từ phép đo trước, không lấy từ lượt đang demo | 1.400.000 byte từ chối, 2.929.920 byte chọn fixed vì dự báo full 2.974.272 byte |
| Thực thi và đối chứng | Mở rpc_calls.json để xem body gửi/nhận từng client; mở rpc_report.json để xem kết quả | 40 RPC, kỳ vọng khoảng 1.485.037 byte; kiểm tra vòng đầu, checkpoint cuối và 4/4 trọng số đối chiếu với FU local |
| Kết quả đã đo | Mở summary.csv của sáu lượt để giải thích Static/Adaptive; mở kết quả ba seed để giải thích ảnh hưởng chất lượng | Phân biệt số liệu live với các benchmark đã lưu; thời gian phụ thuộc tải máy |

Nếu CUDA hoặc worker chưa sẵn sàng, kiểm tra Docker Desktop, image và các file mount của compose.shard-rpc.yaml. Nếu hash hoặc checkpoint lệch, ghi nhận lỗi và kiểm tra seed, phiên bản image và dataset trước khi so sánh. Có thể trình chiếu lại log đã lưu, ghi rõ đó là lần chạy trước.