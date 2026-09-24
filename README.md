# 🚦 Hệ Thống Giám Sát & Ưu Tiên Phương Tiện Khẩn Cấp (ITS)

Dự án ứng dụng Deep Learning (YOLOv8) để giám sát lưu lượng giao thông tại nút giao, phát hiện xe vi phạm vượt đèn đỏ và tự động cảnh báo khi có xe khẩn cấp (Cứu thương, cứu hỏa, cảnh sát).

> **⚠️ LƯU Ý QUAN TRỌNG:** Do giới hạn dung lượng của GitHub, các tài nguyên có kích thước lớn (bao gồm Video Demo, Báo cáo chi tiết và Source code đầy đủ có chứa các file trọng số) đã được lưu trữ trên Google Drive.

## 🔗 Tài Nguyên Dự Án (Google Drive)
- 📁 **[Toàn bộ thư mục dự án (Source code & Weights)](https://drive.google.com/drive/folders/1g4AEXWpgEFe_V4z4riNVd22e_ilkIa-0?usp=sharing)**
- 📄 **[Báo cáo chi tiết dự án (PDF/Word)](https://drive.google.com/file/d/1X9ktSQnh2tQG398tBDglmRIfMQvmI_ZD/view?usp=sharing)**
- 🎥 **[Video Demo kết quả hệ thống](https://drive.google.com/file/d/1LnOr1bjX31DKjZo79e5K7UeUfcLDuB87/view?usp=drive_link)**

## 🌟 Tính Năng Nổi Bật
- **Nhận diện & Tracking:** Phân loại và theo dõi 5 loại phương tiện.
- **Xử lý Logic Giao Thông:** Đếm số lượng xe dừng chờ đèn đỏ tại ngã tư.
- **Phát hiện Vi phạm:** Ghi nhận và thống kê phương tiện vượt đèn đỏ.
- **Hệ thống Ưu tiên (Emergency Override):** Phát ra cảnh báo thị giác khẩn cấp (chớp đỏ màn hình) khi phát hiện xe ưu tiên lọt vào khu vực ngã tư.

## 📂 Cấu Trúc Mã Nguồn

- `config/`: Chứa các tệp cấu hình cho hệ thống.
- `models/`: Chứa trọng số mô hình YOLOv8 đã tối ưu (`best.pt`).
- `src/`: Chứa mã nguồn cốt lõi (Detector, Tracker, Logic Controller, Counter, Visualizer).
- `tool/` & `training_pipeline/`: Các script hỗ trợ tiền xử lý dữ liệu và huấn luyện mô hình.
- `benchmark.py` & `benchmark_results.csv`: Script đánh giá hiệu năng (mAP, FPS, sai số đếm) và file lưu kết quả.
- `gui.py` & `main.py`: Các script để khởi chạy giao diện người dùng và luồng chương trình chính.
- `yolov8n.pt` & `yolov8s.pt`: Các file trọng số cơ sở của mô hình YOLOv8.
- `project.label_config.xml`: Cấu hình nhãn dữ liệu của dự án.
- `demo/`: (Lưu trên Drive) Chứa Video thử nghiệm đầu vào và video kết quả.

## 🚀 Hướng Dẫn Sử Dụng

Hệ thống đã được thiết kế linh hoạt để dễ dàng trình diễn thông qua GUI hoặc chạy trực tiếp bằng mã nguồn.

**Cách 1: Chạy bằng file thực thi (Không cần cài đặt Python)**
1. Mở thư mục dự án đã tải về, chạy file `HeThongGiamSatITS.exe`.
2. Trên giao diện, bấm **"Chọn Video"** và trỏ đến file demo (tải từ link Google Drive).
3. Bấm **"Khởi động hệ thống"**. Có thể dùng phím `Space` để Tạm dừng/Phát hoặc kéo thanh Trackbar để tua nhanh đến đoạn có xe ưu tiên.

**Cách 2: Chạy bằng Mã nguồn (Môi trường Python)**
1. Cài đặt các thư viện phụ thuộc:
   ```bash
   pip install -r requirements.txt