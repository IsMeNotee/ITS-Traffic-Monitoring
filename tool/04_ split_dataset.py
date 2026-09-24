import os
import shutil
import glob
import random

# Đường dẫn thư mục tổng hợp gốc
SRC_IMG = "data/images/train"
SRC_LBL = "data/labels/train"

# Thư mục đích chuẩn chỉnh cuối cùng
OUT_DIR = "dataset_final"

# Tất cả các nhóm đều giới hạn ở mức 1000 ảnh. Thiếu thì lấy hết, đủ thì ngưng.
LIMITS = {
    0: 1000, # Fire_Truck
    1: 1000, # Ambulance
    2: 1000, # Police_Car
    3: 1000, # Warning_Light
    "civil": 1000 # Nhóm dân sự (Class 4 & 5)
}

def setup_dirs():
    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR) # Xóa sạch thư mục cũ, đảm bảo không bao giờ lỗi khi chạy lại
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(OUT_DIR, 'images', split))
        os.makedirs(os.path.join(OUT_DIR, 'labels', split))

def select_images():
    print("=== BẮT ĐẦU CHỌN LỌC DỮ LIỆU (TẤT CẢ TỐI ĐA 1000) ===")
    
    txt_files = glob.glob(os.path.join(SRC_LBL, "*.txt"))
    random.shuffle(txt_files) # Xáo trộn để lấy ngẫu nhiên
    
    selected_pairs = set() 
    counts = {0: 0, 1: 0, 2: 0, 3: 0, "civil": 0}
    
    for txt_path in txt_files:
        # Tối ưu hóa: Nếu đã đủ TẤT CẢ chỉ tiêu thì dừng vòng lặp luôn cho nhanh
        if all(counts[k] >= LIMITS[k] for k in LIMITS):
            break
            
        with open(txt_path, 'r') as f:
            classes = set([int(line.split()[0]) for line in f.readlines() if line.strip()])
            
        img_name = os.path.basename(txt_path).replace('.txt', '.jpg')
        img_path = os.path.join(SRC_IMG, img_name)
        if not os.path.exists(img_path):
            img_path = img_path.replace('.jpg', '.png')
            if not os.path.exists(img_path): continue
            
        needs_this_image = False
        for c in classes:
            # Kiểm tra xem class c có đang bị thiếu so với chỉ tiêu không
            if c in LIMITS and counts[c] < LIMITS[c]:
                needs_this_image = True
                break
            elif c in [4, 5] and counts["civil"] < LIMITS["civil"]:
                needs_this_image = True
                break
                
        if needs_this_image:
            selected_pairs.add((txt_path, img_path))
            # Cộng điểm cho các class có mặt trong bức ảnh vừa nhặt
            for c in classes:
                if c in counts: counts[c] += 1
                elif c in [4, 5]: counts["civil"] += 1

    print("\n📊 TỔNG KẾT SỐ LƯỢNG ẢNH ĐÃ THU THẬP ĐƯỢC:")
    print(f"   + Fire_Truck (0): {counts[0]} ảnh")
    print(f"   + Ambulance (1):  {counts[1]} ảnh")
    print(f"   + Police_Car (2): {counts[2]} ảnh")
    print(f"   + Warning_Light (3): {counts[3]} ảnh")
    print(f"   + Nhóm dân sự (4&5): {counts['civil']} ảnh")
    
    return list(selected_pairs)

def split_and_copy(selected_pairs):
    random.shuffle(selected_pairs)
    total = len(selected_pairs)
    
    train_idx = int(total * 0.7)
    val_idx = train_idx + int(total * 0.2)
    
    splits = {
        'train': selected_pairs[:train_idx],
        'val': selected_pairs[train_idx:val_idx],
        'test': selected_pairs[val_idx:]
    }
    
    print("\n⏳ Đang copy vào thư mục Train(70%) / Val(20%) / Test(10%)...")
    for split_name, pairs in splits.items():
        for txt_path, img_path in pairs:
            shutil.copy(img_path, os.path.join(OUT_DIR, 'images', split_name, os.path.basename(img_path)))
            shutil.copy(txt_path, os.path.join(OUT_DIR, 'labels', split_name, os.path.basename(txt_path)))
            
        print(f" ✅ {split_name.upper()}: {len(pairs)} ảnh")

    print(f"\n🎉 HOÀN TẤT! Tổng cộng {total} ảnh độc lập đã được lưu vào '{OUT_DIR}'.")

if __name__ == "__main__":
    setup_dirs()
    pairs = select_images()
    split_and_copy(pairs)