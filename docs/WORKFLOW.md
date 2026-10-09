# Workflow phát triển

1. Làm rõ yêu cầu (tối đa 6 câu), lập plan khi đổi hành vi.
2. Test đỏ trước (pytest cho `vision/`, vitest cho `web/src/sim/`), code tối thiểu cho xanh, refactor.
3. Tự review diff; chạy `pytest`, `npm test`, `npm run build`.
4. Web: chạy `npm run dev`, kiểm tra bằng trình duyệt, console không lỗi.
5. Model: số liệu chỉ lấy từ `scripts/evaluate.py`; ghi vào `docs/ket-qua-model.md`.
6. Cập nhật `HANDOFF.md` khi dừng.

Windows: UTF-8 cho file tiếng Việt; dùng `vision/.venv/Scripts/python` (Python 3.11), không dùng Python 3.14 hệ thống.
