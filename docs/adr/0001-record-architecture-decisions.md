# ADR 0001 — Kiến trúc hai phần, pipeline hai giai đoạn

- Trạng thái: chấp nhận — 2026-10-08

## Bối cảnh
Cần (1) mô phỏng 3D dây chuyền sấy và (2) model nhận diện quả chạy thật trên Jetson Orin Nano 8GB. Dataset bệnh/hư public đều ở dạng phân loại (không có bbox); dataset bbox thì không có nhãn bệnh.

## Quyết định
1. Tách `vision/` (Python, chạy thật) và `web/` (Three.js, kết quả AI giả lập để nhẹ tài nguyên). Hai phần không phụ thuộc runtime.
2. Pipeline hai giai đoạn: YOLO26n detect 5 loại quả → cắt ảnh → một model YOLO26n-cls chung, lúc suy luận chỉ xét các lớp thuộc loại quả đã phát hiện. Jetson chạy 2 engine.
3. Dùng Ultralytics (AGPL-3.0) vì là đồ án/nghiên cứu; nếu thương mại hoá cần Enterprise license hoặc đổi sang model Apache-2.0.
4. Engine TensorRT build trên chính Jetson (JetPack 6); máy dev export ONNX.

## Hệ quả
- Dùng lại được toàn bộ dataset phân loại bệnh/hư.
- Hai model phải train/đánh giá riêng; độ chính xác thực tế phụ thuộc domain gap.
