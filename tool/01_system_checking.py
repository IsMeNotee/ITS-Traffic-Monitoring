import torch
import psutil
import sys

def run_diagnostics():
    print("="*60)
    print("🖥️ KIỂM TRA SỨC MẠNH HỆ THỐNG (SYSTEM HEALTH CHECK)")
    print("="*60)
    
    # 1. CPU & RAM
    print("[1] THÔNG SỐ HỆ THỐNG:")
    print(f"    - Python Version: {sys.version.split(' ')[0]}")
    print(f"    - RAM System: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    print(f"    - CPU Cores: {psutil.cpu_count(logical=True)} cores")
    
    # 2. GPU (CUDA)
    print("\n[2] THÔNG SỐ CARD ĐỒ HỌA (AI ACCELERATOR):")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        # Mẹo: Dùng getattr để qua mặt bộ quét lỗi tĩnh của Pylance
        torch_version = getattr(torch, "version", None)
        cuda_version = torch_version.cuda if torch_version else "Unknown"
        
        print(f"    ✅ Trạng thái: Kích hoạt thành công")
        print(f"    - Thiết bị: {gpu_name}")
        print(f"    - VRAM Tổng: {vram_total:.1f} GB")
        print(f"    - CUDA Version: {cuda_version}")
    else:
        print("    ❌ KHÔNG TÌM THẤY GPU! Hệ thống sẽ chạy cực kỳ chậm trên CPU.")
        
    print("="*60)
    print("✅ Hệ thống đã sẵn sàng để huấn luyện và chạy YOLOv8!")

if __name__ == "__main__":
    run_diagnostics()