# Nhandienqua

Đồ án nhận diện và xử lý quả trên dây chuyền sấy, gồm hai phần chạy độc lập:

| Phần | Thư mục | Nội dung | Trạng thái |
|---|---|---|---|
| **Vision** | [`vision/`](vision/) | YOLO26 phát hiện quả + YOLO26-cls phân loại tốt / hư. Export ONNX/TensorRT cho Jetson Orin Nano 8GB. | Chạy thật trên ảnh, video, camera |
| **Web mô phỏng** | [`web/`](web/) | Mô phỏng 3D (Three.js) toàn dây chuyền: xe tải → dỡ hàng → băng chuyền → camera AI → thùng rác / lò sấy → QC → đóng gói → QR truy xuất. | Giả lập. Kết quả AI trong web không phải model thật |

Quả mục tiêu: xoài (`mango`), chuối (`banana`), thanh long (`dragon_fruit`), mít (`jackfruit`), khoai lang (`sweet_potato`).

> **Quy tắc môi trường:** mọi cache, file tạm và trọng số đều nằm trong thư mục dự án, không ghi ra ổ C. Chạy `source env.sh` (Git Bash) hoặc `. .\env.ps1` (PowerShell) trước mọi lệnh.

## Yêu cầu

- Windows 10/11, Git Bash hoặc PowerShell.
- Node.js 20+ cho `web/`.
- Python 3.11 cho `vision/` (dùng `py -3.11`, không dùng Python hệ thống 3.14).
- GPU NVIDIA (khuyến nghị RTX 3060 trở lên) để train và chạy nhanh. Jetson Orin Nano 8GB để chạy thực tế tại xưởng.

## Chạy nhanh: web mô phỏng

```bash
source env.sh
cd web && npm install && npm run dev -- --host
```

- Mở địa chỉ **Network** mà Vite in ra. Mã QR trên màn hình chứa IP LAN để điện thoại cùng Wi-Fi quét được.
- Kéo chuột để xoay, cuộn để thu phóng. Bấm ô camera để phóng to, `Esc` để về toàn cảnh.
- Nút **Lô hàng & QR** hiển thị danh sách lô, mã QR từng lô và link trang sản phẩm. Video lô được ghi tự động khi lô vào máy đóng gói.
- Nút **Thử model** (`#/lab`) chạy hai model ONNX ngay trong trình duyệt (WebGPU / WASM) trên ảnh, video hoặc webcam. Có bản đồ nhiệt (Phát hiện / Hư hỏng) và làm mượt theo thời gian cho video.
  - Trước đó chạy `vision/.venv/Scripts/python vision/scripts/export_web.py` để chép model sang `web/public/models/`.
- Test: `npm test`. Build kiểm tra kiểu và đóng gói: `npm run build`.

## Chạy nhanh: vision (model thật)

Cài môi trường một lần:

```bash
source env.sh
py -3.11 -m venv vision/.venv
vision/.venv/Scripts/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
vision/.venv/Scripts/python -m pip install -e "vision[dev,onnx]" pillow-heif
```

Chạy pipeline trên ảnh, thư mục, video hoặc camera (`0`):

```bash
vision/.venv/Scripts/nhandienqua predict <ảnh|thư mục|video|0>
vision/.venv/Scripts/nhandienqua heatmap <ảnh|thư mục>
```

- `predict`: phát hiện quả, bỏ phiếu theo track và phân loại tốt / hư.
- `heatmap`: bản đồ nhiệt cho phát hiện và CAM "hư hỏng" của từng quả.

### Pipeline dữ liệu và train

Các bước này tốn thời gian và GPU. Chỉ chạy khi đã quyết định train lại.

```bash
cd vision
.venv/Scripts/python -m pytest -q                    # test
.venv/Scripts/python scripts/fetch_datasets.py       # tải dataset (~13 GB, không cần token)
.venv/Scripts/python scripts/build_cutouts.py        # tách quả bằng YOLOE (~40 phút, GPU)
.venv/Scripts/python scripts/build_quality_ds.py     # dataset tốt / hư
.venv/Scripts/python scripts/build_detect_ds.py      # dataset phát hiện (ảnh tổng hợp + ảnh thật)
.venv/Scripts/python scripts/train.py quality        # train phân loại tốt / hư
.venv/Scripts/python scripts/train.py detect         # train phát hiện
.venv/Scripts/python scripts/evaluate.py             # sinh docs/ket-qua-model.md
.venv/Scripts/python scripts/export.py               # ONNX + kiểm tra khớp PyTorch
.venv/Scripts/python scripts/consistency.py <video>  # đo độ nhất quán giữa các khung
```

Số liệu trong tài liệu chỉ lấy từ `scripts/evaluate.py`. Đừng sửa tay.

Trên Jetson, xem [docs/huong-dan-jetson.md](docs/huong-dan-jetson.md).

## Tài liệu

- [CONTEXT.md](CONTEXT.md): thuật ngữ miền.
- [docs/WORKFLOW.md](docs/WORKFLOW.md): quy trình phát triển.
- [docs/ket-qua-model.md](docs/ket-qua-model.md): kết quả model (sinh tự động).
- [docs/nguon-du-lieu.md](docs/nguon-du-lieu.md): nguồn dữ liệu và license từng dataset.
- [docs/huong-dan-jetson.md](docs/huong-dan-jetson.md): chạy trên Jetson.
- [docs/adr/](docs/adr/): quyết định kiến trúc.

## Giấy phép

Ultralytics YOLO là AGPL-3.0, phù hợp đồ án và nghiên cứu. Dùng thương mại cần license Enterprise của Ultralytics. License từng dataset xem [docs/nguon-du-lieu.md](docs/nguon-du-lieu.md).
