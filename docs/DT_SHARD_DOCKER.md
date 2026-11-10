# Digital Twin cho shard FU v2

Chạy PowerShell từ checkout này. Compose chạy 20 client logic tuần tự trong một tiến trình CUDA. Ngân sách thời gian tính bằng tổng wall time các vòng FedAvg; ngân sách byte tính hai chiều trên tensor thô. Chi phí đóng gói, kiểm tra validation, nạp và ghi checkpoint được loại khỏi ngân sách vòng. Xem result.json để đọc wall_seconds_total của thao tác FU và kích thước NPZ. Thời gian từ lúc gọi CLI còn bao gồm nạp dữ liệu và checkpoint.

## Dữ liệu và GPU

Dataset chuẩn là connected_v3_auxfit_seed42. Nếu đã có thư mục này, có thể đặt biến SHARD_DATA_DIR trỏ đến đường dẫn tuyệt đối để bind mount. Nếu cần tạo lại:

~~~powershell
$PY = 'D:\budget-aware\.venv-gpu\Scripts\python.exe'
& $PY src/data/prepare.py --input D:/budget-aware/dataset/train_test_network.csv --config configs/data_connected_auxfit.json --output data/processed/connected_v3_auxfit_seed42
~~~

~~~powershell
docker build -t shard-fu-dt-shard .
$dc = @('compose','-p','shard-fu-dt','-f','compose.shard-dt.yaml')
docker @dc run --rm shard -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name())"
~~~

## Tạo ensemble và so sánh trên cùng yêu cầu

Dùng tên mới cho mỗi lần chạy. Ví dụ này dùng state tạo bên trong Docker để hiệu chuẩn predictor đúng môi trường.

~~~powershell
$tag = Get-Date -Format yyyyMMdd_HHmmss
$root = 'runs/shard_dt'
$base = "$root/base_$tag"
New-Item -ItemType Directory -Force $root | Out-Null
'{"client":10}' | Set-Content "$root/request_$tag.json" -Encoding ascii
docker @dc run --rm shard -m scripts.shard.shard_service init --device cuda --shards 4 --seed 50042 --output $base
if ($LASTEXITCODE) { throw 'Khoi tao shard that bai' }
$common = @('-m','scripts.shard.shard_dt','--original-run',$base,'--forget-request',"$root/request_$tag.json",'--budget-seconds','300')
docker @dc run --rm shard @common --mode static --budget-raw-bytes 0 --allow-fixed --output "$root/probe_$tag"
if ($LASTEXITCODE) { throw 'Lap ke hoach that bai' }
$p = Get-Content "$root/probe_$tag/dt_report.json" -Raw | ConvertFrom-Json
$p.candidates | Format-Table
$full = ($p.candidates | Where-Object policy -eq shard_full).raw_bytes
$fixed = ($p.candidates | Where-Object policy -eq shard_fixed).raw_bytes
docker @dc run --rm shard @common --mode static --budget-raw-bytes $full --output "$root/full_$tag"
docker @dc run --rm shard @common --mode adaptive --budget-raw-bytes $fixed --allow-fixed --output "$root/fixed_$tag"
docker @dc run --rm shard -m scripts.shard.shard_service verify --original-run "$root/fixed_$tag"
Get-Content "$root/fixed_$tag/dt_report.json" -Raw
~~~

## Đọc thí nghiệm

* Probe với 0 byte chỉ tạo báo cáo từ chối nếu cần train. Tiếp tục từ checkpoint cha khi yêu cầu bị từ chối.
* Hai lần full/fixed minh họa ảnh hưởng của ngân sách và số vòng. Để so sánh sai số predictor, dùng predicted_seconds_same_policy trong mỗi lần hoàn thành: cả hai dự báo được lập trước khi thực hiện cùng tác vụ. Ở yêu cầu đầu tiên chúng trùng nhau.
* Với yêu cầu tiếp theo, truyền thư mục output đã COMPLETED cho --original-run. Tạo request cho client đang hoạt động hoặc dùng forget_row_ids thuộc tập train còn lại của client. Twin Static giữ bảng hiệu chuẩn, Adaptive cập nhật giá trị của các client đã quan sát và phần thời gian server bằng EWMA 0,3 sau mỗi tác vụ.
* Khi replay nhiều yêu cầu, so sánh MAE của từng predictor trên cùng các yêu cầu; tính trị tuyệt đối của chênh lệch predicted_seconds_same_policy và actual_seconds. Đánh giá chất lượng bằng validation_macro_f1 và validation_fpr trên các cặp full/fixed cùng state cha. So sánh trên nhiều seed trước khi kết luận về hiệu quả.
* exact và structure_ok là các kiểm tra của FU; verify chạy retrain độc lập tại cùng số vòng. Việc planner chọn số vòng dựa vào chi phí đã quan sát cần được ghi rõ khi diễn giải exactness của cả quy trình.
* Phạm vi planner là shard_full và shard_fixed. Chính sách utility_stop, dynamicfu và drop_backfill có chi phí phụ, fallback hoặc backlog riêng; có thể chạy độc lập qua CLI gốc. Ở bản Docker này byte là lưu lượng logic; benchmark nhiều máy cần triển khai worker RPC và đo thời gian mạng thực tế.
