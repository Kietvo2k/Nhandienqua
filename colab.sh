#!/usr/bin/env bash
# Chạy Google Colab CLI mà KHÔNG ghi gì ra ổ C:
# token đăng nhập, phiên, lịch sử, cài đặt của CLI nằm trong .cache/colab-home (đã gitignore — KHÔNG commit, chứa token).
# Dùng:  bash colab.sh <lệnh colab…>      vd. bash colab.sh status -s ndq
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/env.sh"
W="$(cygpath -w "$ROOT")"  # đường dẫn dạng Windows cho chương trình gốc
export PYTHONPATH="$W\\.cache\\colab-shim"  # shim termios (CLI chưa hỗ trợ Windows chính thức)
export HOME="$W\\.cache\\colab-home" USERPROFILE="$W\\.cache\\colab-home"
export TMP="$W\\.cache\\tmp" TEMP="$W\\.cache\\tmp"
export MSYS_NO_PATHCONV=1  # giữ nguyên đường dẫn /content/... của Colab (Git Bash hay tự đổi)
mkdir -p "$ROOT/.cache/colab-home"
exec "$ROOT/.cache/colab-venv/Scripts/colab.exe" "$@"
