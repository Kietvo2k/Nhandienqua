# Kết quả model

Sinh tự động bởi `vision/scripts/evaluate.py` lúc 2026-10-08 22:00 trên NVIDIA GeForce RTX 3060. Không sửa tay số liệu — chạy lại script.

> **Cảnh báo domain gap:** dữ liệu là ảnh public (điện thoại, vườn, phòng lab) và ảnh tổng hợp. Trên băng chuyền thật (ánh sáng, camera, giống quả Việt Nam khác) độ chính xác sẽ THẤP HƠN số dưới đây. Muốn dùng thật cần chụp vài trăm ảnh/loại trên máy và fine-tune.

## 1. Phát hiện quả (YOLO26n, 640 px)

### Ảnh THẬT (test_real) — 463 ảnh

mAP50 = **0.879**, mAP50-95 = **0.782**, precision = 0.898, recall = 0.838
Tốc độ (RTX 3060, PyTorch): 3.4 ms/ảnh suy luận

| Lớp | Số bbox | mAP50-95 |
|---|---:|---:|
| Xoài | 52 | 0.954 |
| Chuối | 601 | 0.584 |
| Thanh long | 20 | 0.937 |
| Mít | 4 | 0.459 |
| Khoai lang | 30 | 0.974 |

Biểu đồ: `vision/runs/eval/det_test_real/`

### Ảnh tổng hợp băng chuyền (test_synth) — 600 ảnh

mAP50 = **0.994**, mAP50-95 = **0.956**, precision = 0.983, recall = 0.988
Tốc độ (RTX 3060, PyTorch): 2.7 ms/ảnh suy luận

| Lớp | Số bbox | mAP50-95 |
|---|---:|---:|
| Xoài | 620 | 0.965 |
| Chuối | 614 | 0.966 |
| Thanh long | 616 | 0.936 |
| Mít | 624 | 0.950 |
| Khoai lang | 652 | 0.965 |

Biểu đồ: `vision/runs/eval/det_test_synth/`

## 2. Phân loại tốt / hư (YOLO26n-cls, 224 px)

Tập test: 1177 ảnh (ảnh cắt dán lên nền băng chuyền, nhóm ảnh gốc KHÔNG trùng train).
Top-1 trên 10 lớp = **0.946** · 12.6 ms/ảnh (gồm đọc file, RTX 3060).

Theo cách pipeline thật ra quyết định (chỉ xét lớp của loại quả mà detector báo, ngưỡng 0.5):

| Quả | Ảnh test (tốt/hư) | Độ chính xác | Precision (hư) | Recall (hư) | Loại nhầm quả tốt |
|---|---|---:|---:|---:|---:|
| Chuối | 161/60 | 0.968 | 1.000 | 0.883 | 0.000 |
| Thanh long | 130/104 | 0.974 | 0.980 | 0.962 | 0.015 |
| Mít | 76/22 | 0.827 | 0.609 | 0.636 | 0.118 |
| Xoài | 198/371 | 0.956 | 0.965 | 0.968 | 0.066 |
| Khoai lang | 25/30 | 1.000 | 1.000 | 1.000 | 0.000 |

**Đọc số liệu cho đúng:**

- **Khoai lang:** ảnh *hư* chỉ từ một bộ chụp phòng lab (GitHub aipal-nchu), ảnh *tốt* từ Fruits-360 + ảnh web. Hai lớp khác NGUỒN → model có thể học khác biệt nguồn (ánh sáng, độ nét) thay vì vết hư. Số liệu cao ở đây **không đáng tin**; cần ảnh khoai tốt và hư chụp cùng điều kiện.
- **Mít:** *hư* = quả méo/dị dạng (không phải bệnh), ít ảnh → yếu nhất.
- **Xoài, chuối, thanh long:** tốt và hư cùng nguồn trong mỗi bộ dữ liệu → đáng tin hơn, nhưng vẫn là ảnh public, không phải camera băng chuyền.

## 3. Chế độ video (theo dõi ByteTrack + bỏ phiếu khi qua vạch)

Video băng chuyền mô phỏng từ ảnh cắt tập TEST (`scripts/make_demo_video.py`), chạy `nhandienqua predict runs/demo/bang_chuyen_test.mp4 --line 0.6`.

- Số quả: 24 · số quyết định: 24 (mỗi quả đúng 1 lần gạt/nhận)
- Đúng loại quả: 24/24 · đúng tốt/hư: 22/24

**Độ nhất quán giữa các khung** (`scripts/consistency.py`; cùng một quả qua nhiều khung):

| Chế độ | Đổi loại quả | Đổi tốt/hư giữa 2 khung liền | Khung lệch đa số | Track có lúc đổi |
|---|---:|---:|---:|---:|
| thô (từng khung) | 0.00% | 3.60% | 7.24% | 17/24 |
| EMA + vùng trễ, bỏ khung chạm mép (hiển thị) | 0.00% | 0.73% | 5.84% | 11/24 |

Quyết định gạt dùng trung bình cả track (không phụ thuộc làm mượt hiển thị).

