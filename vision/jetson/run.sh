#!/usr/bin/env bash
# Chạy pipeline nhận diện quả trên Jetson Orin Nano 8GB (JetPack 6) bằng Docker Ultralytics.
# Dùng:  bash vision/jetson/run.sh [nguồn]      (mặc định nguồn = 0, camera USB /dev/video0)
# Yêu cầu: thư mục vision/weights/ có fruit_det.pt và fruit_quality.pt (copy từ máy train).
set -euo pipefail
cd "$(dirname "$0")/.."            # → vision/
SRC="${1:-0}"
IMAGE="ultralytics/ultralytics:latest-jetson-jetpack6"

sudo nvpmodel -m 0 || true         # chế độ công suất tối đa
sudo jetson_clocks || true

DEV=()
for d in /dev/video*; do [ -e "$d" ] && DEV+=(--device "$d"); done

sudo docker run --rm -it --ipc=host --runtime=nvidia "${DEV[@]}" \
  -v "$PWD":/work/vision -w /work/vision \
  -e YOLO_CONFIG_DIR=/work/vision/.cache/ultralytics -e YOLO_AUTOINSTALL=false \
  "$IMAGE" bash -lc "
    set -e
    pip install --no-deps -e . >/dev/null
    python -c "import lap" 2>/dev/null || pip install "lap>=0.5.12" >/dev/null   # ByteTrack cần lap
    cd weights
    # Engine TensorRT phải build NGAY trên Jetson (lần đầu mất vài phút, sau đó dùng lại)
    [ -f fruit_det.engine ]     || yolo export model=fruit_det.pt     format=engine imgsz=384,640 half=True device=0
    [ -f fruit_quality.engine ] || yolo export model=fruit_quality.pt format=engine imgsz=224 half=True device=0
    cd ..
    nhandienqua predict '$SRC' --det weights/fruit_det.engine --cls weights/fruit_quality.engine --device 0 --line 0.6
  "
