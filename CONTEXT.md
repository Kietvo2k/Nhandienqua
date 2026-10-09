# Thuật ngữ miền

| Thuật ngữ | Nghĩa trong dự án |
|---|---|
| **Lô (batch)** | Toàn bộ quả từ một xe tải, cùng loại quả; đơn vị truy xuất nguồn gốc. |
| **Trạm (station)** | Một công đoạn trên dây chuyền: nhận hàng, phân loại AI, lò sấy, đánh giá mức sấy, lấy mẫu QC, đóng gói, dán nhãn. |
| **Phát hiện (detect)** | YOLO tìm bbox + loại quả (5 lớp). |
| **Chất lượng (quality)** | Phân loại ảnh cắt của một quả: `<quả>__good` hoặc `<quả>__<lỗi>`. |
| **Quyết định (Decision)** | `accept` (vào lò sấy) hoặc `reject` (ra thùng rác) cho một quả, kèm lý do + độ tin cậy. |
| **Track** | Một quả được theo dõi qua nhiều khung hình (ByteTrack); quyết định cuối được bỏ phiếu theo track. |
| **Mức sấy (drying level)** | `under` (chưa đạt), `ok` (đạt), `over` (quá khô). |
| **Mẫu QC** | Một số túi/miếng lấy ra theo kế hoạch lấy mẫu để kiểm nghiệm độ ẩm, aw, vi sinh. |
| **Mã truy xuất (trace code)** | Mã định danh lô in trong QR; trang SP hiển thị 7 thông tin tối thiểu theo Thông tư 02/2024/TT-BKHCN. |
| **Domain gap** | Khác biệt giữa ảnh dataset public (chụp điện thoại, ngoài vườn) và ảnh camera băng chuyền thật. |
