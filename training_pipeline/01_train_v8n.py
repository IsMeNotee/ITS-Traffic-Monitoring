from ultralytics import YOLO
import torch
import multiprocessing

def train_baseline_model():
    print("="*50)
    print("🚀 BẮT ĐẦU HUẤN LUYỆN MÔ HÌNH CƠ SỞ (YOLOv8 NANO)")
    print("="*50)
    
    # Kiểm tra trạng thái GPU
    device = '0' if torch.cuda.is_available() else 'cpu'
    if device == '0':
        print(f"✅ Đã nhận diện GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ CẢNH BÁO: Đang chạy bằng CPU, sẽ rất chậm!")

    # Khởi tạo mô hình Nano
    model = YOLO('yolov8n.pt')
    
    # Cấu hình tham số huấn luyện an toàn cho VRAM 4GB
    model.train(
        data='config/data.yaml',
        epochs=100,             # Số vòng lặp
        imgsz=640,              # Kích thước ảnh chuẩn
        batch=8,                # Giữ mức 8 để chống tràn VRAM
        device=device,
        workers=2,              # Tránh quá tải CPU trên Windows
        patience=15,            # Tự động dừng nếu sau 15 epoch không cải thiện
        project='models',       # Thư mục gốc lưu kết quả
        name='train_v8_nano',   # Thư mục con chứa file best.pt và biểu đồ
        exist_ok=True           # Ghi đè nếu bạn lỡ chạy lại
    )
    
    print("\n🎉 HOÀN TẤT HUẤN LUYỆN BẢN NANO!")
    print("👉 Toàn bộ trọng số và biểu đồ đã được lưu tại: models/train_v8_nano/")

if __name__ == '__main__':
    # Bắt buộc phải có dòng này trên Windows để không bị lỗi đa luồng
    multiprocessing.freeze_support()
    train_baseline_model()