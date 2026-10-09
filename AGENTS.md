# Hướng dẫn agent — Nhandienqua

Tuân thủ policy toàn cục tại `C:\Users\Admin\.codex\AGENTS.md`. Chế độ hiện tại: **Max**.

## Dự án gồm hai phần độc lập

| Thư mục | Nội dung | Chạy thật? |
|---|---|---|
| `vision/` | Python 3.11, package `nhandienqua`: YOLO26 phát hiện quả + YOLO26-cls phân loại tốt/hư + model mức sấy. Export ONNX/TensorRT cho Jetson Orin Nano 8GB. | **Có** — đưa lên máy thật |
| `web/` | Vite + TypeScript + Three.js: mô phỏng 3D dây chuyền sấy (xe tải → phân loại → sấy → QC → đóng gói → QR). | Không — kết quả AI là giả lập |

Quả mục tiêu: xoài (`mango`), chuối (`banana`), thanh long (`dragon_fruit`), mít (`jackfruit`), khoai lang (`sweet_potato`).

## Lệnh dự án

| Lệnh (từ thư mục gốc) | Mục đích |
|------|----------|
| `vision/.venv/Scripts/python -m pytest vision/tests -q` | Test Python |
| `vision/.venv/Scripts/python vision/scripts/fetch_datasets.py --list` | Xem danh mục dataset |
| `vision/.venv/Scripts/nhandienqua predict <ảnh/thư mục/video>` | Chạy pipeline thật |
| `cd web && npm test` | Test engine mô phỏng (vitest) |
| `cd web && npm run build` | Typecheck + build web |
| `cd web && npm run dev -- --host` | Chạy web (mở được từ điện thoại cùng LAN) |

## Quy tắc

- **KHÔNG ghi ra ổ khác (đặc biệt ổ C).** Mọi cache, file tạm, trọng số, cấu hình công cụ nằm trong dự án:
  chạy `source env.sh` (Git Bash) hoặc `. .\env.ps1` (PowerShell) trước mọi lệnh; pip của venv đã trỏ `.cache/pip`
  (`vision/.venv/pip.ini`), npm trỏ `.cache/npm` (`web/.npmrc`). Trọng số model để trong `vision/weights/`.
- **KHÔNG tự chạy train** (`scripts/train.py`, `model.train`). Chuẩn bị dữ liệu xong thì báo người dùng lệnh +
  thời gian ước tính và chờ họ đồng ý.
- Đổi hành vi → test đỏ trước (pytest / vitest), rồi code.
- `web/src/sim/` là TypeScript thuần, KHÔNG import `three` — để test được bằng vitest.
- Dữ liệu và trọng số (`vision/data/`, `vision/runs/`, `*.pt`, `*.onnx`, `*.engine`) không commit.
- Mỗi dataset mới phải ghi URL + license vào `vision/configs/datasets.yaml` và `docs/nguon-du-lieu.md`.
- Không ghi token Kaggle/Roboflow vào repo; đọc từ biến môi trường `ROBOFLOW_API_KEY` / `~/.kaggle/kaggle.json`.
- Số liệu model trong tài liệu phải đến từ lệnh đã chạy (`scripts/evaluate.py`).
- File tiếng Việt: UTF-8 có dấu.

## Unit test skills

<available_skills>
  <skill><name>generate-tests</name><description>Generate unit tests: analyze code, review test cases, then write tests.</description></skill>
  <skill><name>generate-test-cases</name><description>List test cases in Given-When-Then without generating code.</description></skill>
</available_skills>

## An toàn

- Không commit secret, token, PII. Không commit/push nếu người dùng chưa yêu cầu.

## Tài liệu

- `CONTEXT.md` — thuật ngữ miền; `docs/adr/` — quyết định kiến trúc; `docs/WORKFLOW.md` — quy trình.
- `docs/nguon-du-lieu.md`, `docs/ket-qua-model.md`, `docs/huong-dan-jetson.md`.
- `HANDOFF.md` — trạng thái phiên.
