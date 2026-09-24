from ultralytics import YOLO
import torch
import multiprocessing
import os

# Ép hệ thống lấy đường dẫn tuyệt đối của thư mục 'models' hiện tại
PROJECT_DIR = os.path.abspath("models")

def train_improved_model():
    print("="*50)
    print("🚀 BẮT ĐẦU HUẤN LUYỆN MÔ HÌNH CẢI TIẾN (YOLOv8 SMALL)")
    print("="*50)
    
    # Kiểm tra trạng thái GPU
    device = '0' if torch.cuda.is_available() else 'cpu'
    if device == '0':
        print(f"✅ Đã nhận diện GPU: {torch.cuda.get_device_name(0)}")
    
    # Khởi tạo mô hình Small (Nặng hơn, thông minh hơn bản Nano)
    model = YOLO('yolov8s.pt')
    
    # Cấu hình tham số huấn luyện (ĐÃ ĐIỀU CHỈNH CHO VRAM 4GB)
    model.train(
        data='config/data.yaml',
        epochs=100,             
        imgsz=640,              
        batch=4,                # ⚠️ BẮT BUỘC: Ép xuống 4 (vì bản s nặng gấp 3 lần bản n, để 8 sẽ tràn VRAM ngay)
        device=device,
        workers=2,              
        patience=15,            
        project=PROJECT_DIR,    # Ép lưu thẳng vào thư mục tuyệt đối D:\Code\Final_CV\models
        name='train_v8_small',  # Tên thư mục con chứa kết quả
        exist_ok=True           
    )
    
    print("\n🎉 HOÀN TẤT HUẤN LUYỆN BẢN SMALL!")
    print(f"👉 Toàn bộ trọng số đã được lưu tại: {os.path.join(PROJECT_DIR, 'train_v8_small')}")

if __name__ == '__main__':
    # Bắt buộc phải có dòng này trên Windows
    multiprocessing.freeze_support()
    train_improved_model()