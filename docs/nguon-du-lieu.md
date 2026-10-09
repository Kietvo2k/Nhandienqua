# Nguồn dữ liệu

Tra cứu ngày 2026-10-08. Danh mục máy đọc được: `vision/configs/datasets.yaml`. Tải về `vision/data/raw/` bằng:

```bash
source env.sh
vision/.venv/Scripts/python vision/scripts/fetch_datasets.py
```

Kaggle cho tải dataset công khai **không cần token**. Hugging Face cũng vậy. Mendeley Data lúc tra bị lỗi phía máy chủ (500/502), nên các bộ gốc Mendeley được lấy qua bản mirror trên Kaggle hoặc Hugging Face.

## Dùng cho model

| Tên trong dự án | Nguồn | License | Dùng cho | Ghi chú |
|---|---|---|---|---|
| `kg_mango_disease` | Kaggle `anilsandhii/mango-fruit-disease` | CC0 | Chất lượng xoài | Dẫn xuất MangoFruitDDS (Senegal). Ảnh đã augment, chia tập theo ảnh gốc (`aug_<k>_<tên gốc>`) |
| `hf_banana_guava_quality` | HF `Project-AgML/banana_guava_quality_classification` | CC BY 4.0 | Chất lượng chuối | Class_A/B → tốt, Defect → hư |
| `hf_dragonfruit_quality` | HF `Project-AgML/dragonfruit_quality_classification` (Mendeley 2jpzbx8tm6) | CC BY 4.0 | Chất lượng thanh long | Ảnh hư đã tách nền trắng, ảnh tươi chụp ngoài vườn. Tất cả được tách quả rồi dán lên nền băng chuyền để tránh model học nền |
| `kg_dragonfruit_vn` | Kaggle `nguynphancminh/dragonfruitdataset-qma` | Apache-2.0 | Chất lượng thanh long | Chỉ lấy thư mục `Fruit`, bỏ bản xoay sẵn |
| `kg_jackfruit_shapes` | Kaggle `sridhargutam/jackfruit-fruit-shapes` | CC0 | Chất lượng mít | perfect → tốt; deformed/malformed → hư (**lỗi hình dạng**, không phải bệnh) |
| `gh_sweet_potato_defect` | GitHub `aipal-nchu/sweet_potato_defect_dataset` | Apache-2.0 | Chất lượng khoai lang | 4 lỗi; ảnh nhãn màu cho mask cả củ |
| `hf_fruits360` | HF `PedroSampaio/fruits-360` | CC BY-SA 4.0 (Fruits-360 gốc) | Ảnh cắt (nền trắng) | Khoai lang tốt, xoài, chuối, thanh long. Mỗi lớp chỉ có vài quả chụp xoay vòng |
| `kg_kritik_sweetpotato` | Kaggle `kritikseth/fruit-and-vegetable-image-recognition` | CC0 | Khoai lang tốt | Chỉ giải nén lớp `sweetpotato` (89 ảnh web) |
| `hf_fruitseg30` | HF `Project-AgML/fruitseg30_segmentation` | CC BY 4.0 | Ảnh cắt có mask | Xoài, chuối, thanh long |
| `kg_conveyor_blueberry` | Kaggle `josembm/industrial-fruit-defect-detection-conveyor-belt` | CC0 | Nền băng chuyền | Ảnh băng trống + quả mọng (làm vật gây nhiễu) |
| `kg_fruits10_yolo` | Kaggle `afsananadia/fruits-images-dataset-object-detection` | CC0 | Phát hiện (ảnh thật) | Xoài/chuối/mít, lớp lấy theo tên file |
| `kg_lakshay_fruits` | Kaggle `lakshaytyagi01/fruit-detection` | CC0 | Phát hiện (ảnh thật) | Chỉ lớp Banana; ảnh quả khác làm mẫu âm |
| `kg_lvis_fruits_veg` | Kaggle `henningheyen/lvis-fruits-and-vegetables-dataset` | MIT (ảnh COCO: CC BY theo từng ảnh) | Phát hiện (ảnh thật) | banana, sweet potato (chỉ 2 ảnh) |
| `kg_mango_dried` | Kaggle `luftmensch2409/mango-dried-dataset` | CC BY 4.0 (Roboflow) | Mức sấy (tham khảo) | Bbox "xoai-say", **không có nhãn mức sấy** |
| `kg_fruits_by_yolo` | Kaggle `kapturovalexander/fruits-by-yolo-fruits-detection` | CC BY 4.0 | Không dùng | Bản Kaggle không có bbox |

## Model dùng để tách quả

YOLOE-26s-seg (Ultralytics, AGPL-3.0) với câu lệnh văn bản kiểu `["jackfruit", "durian", "fruit"]`. Ảnh nào YOLOE không tìm thấy quả, hoặc mask gần như hình chữ nhật, thì **bị bỏ**, không đoán bừa.

## Khoảng trống dữ liệu (đã biết)

- **Không có dataset Việt Nam** cho xoài, chuối, mít, khoai lang. Thanh long có `kg_dragonfruit_vn`.
- **Thanh long và mít gần như không có bbox thật**, nên phát hiện dựa chủ yếu vào ảnh tổng hợp.
- **Mức độ sấy:** không có dataset ảnh xoài/chuối sấy có nhãn mức sấy (chưa đạt / đạt / quá khô).
- **Mít "hư"** chỉ là lỗi hình dạng; dữ liệu bệnh trên quả mít còn rất ít.
- Ảnh public chụp bằng điện thoại, ngoài vườn hoặc trong phòng lab, không phải camera băng chuyền. Phải có ảnh chụp trên máy thật để fine-tune trước khi dùng thật.
