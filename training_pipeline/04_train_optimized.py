from ultralytics import YOLO
import torch
import multiprocessing
import os

PROJECT_DIR = os.path.abspath("models")

def train_ultimate_model():
    print("="*50)
    print("🚀 HUẤN LUYỆN BẢN MÔ HÌNH CẢI TIẾN + TỐI ƯU HÓA TOÀN DIỆN")
    print("="*50)
    
    device = '0' if torch.cuda.is_available() else 'cpu'
    # Đã đổi sang Nano để khớp với kịch bản thực nghiệm
    model = YOLO('yolov8n.pt') 
    
    model.train(
        data='config/data.yaml',
        epochs=100,             
        imgsz=640,              
        batch=4,                
        device=device,
        workers=2,              
        patience=20,            
        project=PROJECT_DIR,    
        name='train_v8_nano_optimized',  # Đổi tên thư mục thành nano
        exist_ok=True,
        
        # 1. TỐI ƯU TOÁN HỌC (ALGORITHMIC OPTIMIZATION)
        optimizer='AdamW',      
        cos_lr=True,            
        lr0=0.001,              
        weight_decay=0.0005,    
        
        # 2. ĐÓNG BĂNG TRỌNG SỐ (TRANSFER LEARNING)
        freeze=10,              
        
        # 3. CHỐNG CHỊU MÔI TRƯỜNG (ROBUSTNESS AUGMENTATION)
        erasing=0.4,            # Giả lập dính bụi, che khuất
        degrees=15.0,           # Giả lập gió rung camera
        shear=2.0,              # Lệch góc camera
        perspective=0.0001,     # Góc nhìn chúi từ trên cột đèn xuống
        hsv_h=0.015,            # Chói nắng
        hsv_s=0.7,              # Sương mù nhạt màu
        hsv_v=0.4,              # Chạng vạng / tối trời
        translate=0.1,          
        scale=0.5,              
        fliplr=0.5,             
        mosaic=1.0,             
        flipud=0.0              # Giữ xe đi đúng hướng, không lộn ngược gầm lên trời
    )
    
    print("\n🎉 HOÀN TẤT HUẤN LUYỆN BẢN NANO OPTIMIZED!")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    train_ultimate_model()