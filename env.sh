# Dùng: source env.sh  (Git Bash)
# Giữ MỌI cache / file tạm / cấu hình công cụ bên trong thư mục dự án — không ghi ra ổ C.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
C="$ROOT/.cache"
export PIP_CACHE_DIR="$C/pip"
export PIP_DISABLE_PIP_VERSION_CHECK=1
export UV_CACHE_DIR="$C/uv"
export YOLO_AUTOINSTALL=false  # không để Ultralytics tự cài gói
export npm_config_cache="$C/npm"
export TMP="$C/tmp" TEMP="$C/tmp" TMPDIR="$C/tmp"
export YOLO_CONFIG_DIR="$C/ultralytics"
export TORCH_HOME="$C/torch"
export HF_HOME="$C/hf"
export MPLCONFIGDIR="$C/matplotlib"
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
mkdir -p "$UV_CACHE_DIR" "$PIP_CACHE_DIR" "$npm_config_cache" "$TMP" "$YOLO_CONFIG_DIR" "$TORCH_HOME" "$HF_HOME" "$MPLCONFIGDIR"
