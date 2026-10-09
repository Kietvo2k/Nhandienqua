# Chạy model nhận diện quả trên Jetson Orin Nano 8GB

> Trạng thái: các bước dưới đây theo tài liệu Ultralytics/NVIDIA và **chưa được kiểm chứng trên Jetson thật** của dự án.
> Số FPS chỉ là tham khảo, cần tự đo.

## 1. Chuẩn bị Jetson

1. Cài **JetPack 6.x** (Ubuntu 22.04) bằng ảnh thẻ nhớ/NVMe hoặc SDK Manager.
2. Cài Docker + NVIDIA Container Runtime (JetPack 6 thường có sẵn): `sudo apt install -y docker.io nvidia-container`.
3. Chép thư mục `vision/` sang Jetson. Tối thiểu cần `nhandienqua/`, `pyproject.toml`, `configs/`, `jetson/`, và `weights/fruit_det.pt` + `weights/fruit_quality.pt`.

## 2. Chạy nhanh (Docker)

```bash
bash vision/jetson/run.sh 0
```

Script sẽ làm các việc sau:
- Bật chế độ công suất tối đa (`nvpmodel -m 0`, `jetson_clocks`).
- Chạy image `ultralytics/ultralytics:latest-jetson-jetpack6`.
- Build engine TensorRT FP16 **ngay trên Jetson** ở lần chạy đầu. Engine phụ thuộc phiên bản GPU/TensorRT, nên không copy được từ PC sang.
- Chạy `nhandienqua predict` với camera `/dev/video0`.

## 3. Nguồn camera

| Loại | Tham số `source` |
|---|---|
| USB (UVC) | `0`, `1`… |
| CSI (IMX219/IMX477…) | `"gst:nvarguscamerasrc ! video/x-raw(memory:NVMM),width=1280,height=720,framerate=60/1 ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink drop=1"` |
| Camera IP | `rtsp://user:pass@ip/stream` |
| Video thử | `duong/dan/video.mp4` |

Khuyến nghị cho băng chuyền đang chạy:
- Dùng camera **global shutter** (tránh méo hình khi vật di chuyển).
- Thời gian phơi sáng ngắn, khoảng ≤ 1/1000 s.
- Đèn LED tán xạ đều phía trên.
- Đặt camera nhìn thẳng từ trên xuống, giống ảnh tổng hợp đã dùng để train.

## 4. Đầu ra để nối cơ cấu gạt

Với video/camera, pipeline theo dõi từng quả (ByteTrack). Khi tâm quả vượt vạch `--line` (tỉ lệ chiều ngang khung hình, mặc định 0.6), chương trình in **một dòng JSON** cho mỗi quả:

```json
{"track_id": 12, "fruit": "mango", "action": "reject", "p_defect": 0.83, "frames": 7, "low_evidence": false, "frame": 1520}
```

Phần nối phần cứng (GPIO, Arduino, PLC) nằm ngoài phạm vi đồ án. Muốn nối thì đọc stdout này, rồi kích van khí hoặc cần gạt sau một khoảng trễ:

`trễ = khoảng cách từ vạch tới cơ cấu gạt (m) / tốc độ băng (m/s)`

## 5. Tham khảo tốc độ

Ultralytics công bố cho Orin Nano Super, YOLO26n TensorRT FP16: khoảng 4,6 ms/ảnh (chỉ suy luận). Pipeline thật còn có đọc camera, tiền xử lý và phân loại từng quả, nên FPS sẽ thấp hơn.

Tự đo bằng dòng tổng kết mà `nhandienqua predict` in ra cuối mỗi lần chạy (FPS trung bình).

## 6. Cài không dùng Docker (tuỳ chọn)

Làm theo hướng dẫn Ultralytics "NVIDIA Jetson" cho JetPack 6:
1. Cài `torch`/`torchvision` bản dựng sẵn cho Jetson.
2. `pip install ultralytics`.
3. Trong `vision/`: `pip install -e .`

Sau đó export engine như trong `run.sh`.
