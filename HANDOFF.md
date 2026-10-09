# Handoff — Nhandienqua

Cập nhật: 2026-10-08. Chế độ: Max. Kế hoạch gốc: `C:\Users\Admin\.claude\plans\gi-p-t-i-xem-v-jolly-blum.md`.

## Đã xong

- **Web 3D (`web/`)**
  - Mô phỏng toàn dây chuyền, 8 camera CCTV, dashboard, QR + trang sản phẩm theo TT 02/2024, video lô.
  - `npm test` 37/37, `npm run build` qua.
- **Model đã train** (Google Colab L4, qua Colab CLI, phiên đã dừng):
  - `vision/weights/fruit_det.pt` + `.onnx`
  - `vision/weights/fruit_quality.pt` + `.onnx`
  - ONNX khớp PyTorch (lệch 0,02% / 0,00%).
  - Số liệu: `docs/ket-qua-model.md`, sinh bởi `scripts/evaluate.py`:
    - Phát hiện trên ảnh thật: mAP50 0,879.
    - Tốt/hư trên tập test: top-1 0,946.
    - Video thử: 24/24 quả đúng loại, 22/24 đúng tốt/hư, 39,8 FPS trên RTX 3060.
- **Pipeline thật:** `nhandienqua predict` cho ảnh / video / camera / GStreamer, bỏ phiếu theo track, đã chống gạt 2 lần khi tracker đổi ID.
  - `pytest` 55/55.
- **Train trên Colab bằng CLI:** `vision/scripts/colab_job.py all --gpu L4`.
  - Có sẵn `colab.sh`. Shim `termios` cho Windows nằm ở `.cache/colab-shim`.
  - Token ở `.cache/colab-home` (gitignore, không commit). Lưu ý `colab.log` ở mức DEBUG có ghi token phiên.

- **Trang thử model `#/lab`:** onnxruntime-web chạy trên WebGPU, nhận ảnh / video / webcam.
  - So khớp với Python (cùng ONNX): 15/15 quyết định giống nhau, độ tin cậy trùng 3 chữ số.
  - `vision/scripts/export_web.py` chép model và mẫu sang web.
  - Đã sửa lỗi: model ONNX/engine có batch cố định = 1 trước đây sập khi khung hình có ≥ 2 quả (`predict_in_batches`).

- **Bản đồ nhiệt + độ nhất quán** (`vision/nhandienqua/explain.py`, `voting.py`, `web/src/lab/heatmap.ts`, `track.ts`):
  - CAM chính xác: đầu Classify là conv → GAP → Linear nên không cần gradient. ONNX `fruit_quality_cam.onnx` có 2 đầu ra, lệch xác suất 3,3e-7.
  - Bản đồ phát hiện lấy từ 8400 anchor. Lệnh `nhandienqua heatmap`, nút trên trang `#/lab`.
  - Làm mượt hiển thị: EMA + vùng trễ ±0,1, bỏ khung quả chạm mép. Đổi tốt/hư giữa 2 khung: 3,60% → 0,73% (`docs/ket-qua-model.md`).
  - TTA lật ảnh đã thử nhưng không cải thiện, nên không bật.
- **3D + UI xưởng (subagent, đã kiểm lại):**
  - Sửa 12 lỗi kết xuất/chuyển động, ví dụ: khối bị thiếu vì `box()` không gắn vào cảnh; băng 2/3 chạy ngược; quả hiện ra từ không khí; túi biến mất…
  - Mô hình quả mới ở `web/src/scene/fruitModels.ts`; quả hư có texture đốm.
  - Gộp mesh ở `batch.ts`: còn khoảng 1 ms/khung.
  - Hết bôi xanh khi kéo chuột. Giao diện xưởng làm lại.
  - Tôi sửa thêm: dải trạm ở màn hẹp (<1100px) thành 1 hàng trượt (che 17% thay vì khoảng 50%).
  - Còn lại: pallet đầy chưa có xe nâng; phần chờ sấy lại chưa có hoạt cảnh vào lò; inox hơi tối (không có env map).

## Việc còn lại

1. **Mức sấy:** gán nhãn 1.629 ảnh trong `vision/data/drying_unlabeled/` vào `data/drying/<split>/<under|ok|over>/`, rồi train (local `train.py drying`, hoặc thêm task vào `colab/train_remote.py`).
2. **Chạy trên Jetson thật** (`vision/jetson/run.sh`). Engine phát hiện build với `imgsz=384,640` (camera 16:9).
3. **Fine-tune bằng ảnh chụp trên băng chuyền thật** (khoai lang và mít đặc biệt cần).
4. Tuỳ chọn: trạm AI trong web phát lại kết quả model thật.
5. Review cuối: code-review, ponytail-review, hallmark UI, security.

## Lưu ý

- Chạy `source env.sh` trước mọi lệnh. Dùng Python 3.11 trong `vision/.venv`. Riêng Colab CLI dùng `.cache/colab-venv` (Python 3.14).
- Không tự train khi người dùng chưa đồng ý.
- **Luôn `colab stop`** khi xong để không tốn compute unit.
- Gọi Colab CLI từ Python thì dùng thẳng `colab.exe`, không gọi `bash`, vì trên Windows lệnh `bash` có thể là WSL.
- Lệnh tay: dùng `bash colab.sh …`. Script đã đặt `MSYS_NO_PATHCONV=1` để Git Bash không đổi đường dẫn `/content/…`.
