import os

def scan_yolo_dataset(dataset_path):
    """Quét và thống kê số lượng file cùng chi tiết các class trong chuẩn dataset YOLO"""
    print(f"\n🔍 ĐANG QUÉT THƯ MỤC: {dataset_path}")
    
    if not os.path.exists(dataset_path):
        print("❌ Thư mục không tồn tại! Kiểm tra lại đường dẫn.")
        return

    # Từ điển ánh xạ ID class sang tên hiển thị (Khớp với hệ thống ITS của bạn)
    DISPLAY_NAMES = {
        0: "4_Wheel_Vehicle (Car)",
        1: "2_Wheel_Vehicle (Motorbike)",
        2: "Police_Car",
        3: "Ambulance",
        4: "Fire_Truck",
        5: "Warning_Light"
    }

    subsets = ['train', 'val', 'test']
    total_imgs = 0
    total_labels = 0
    total_size = 0
    
    # Tính tổng dung lượng toàn bộ thư mục
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            total_size += os.path.getsize(os.path.join(root, file))
    
    print("-" * 65)
    print(f"{'Tập con':<10} | {'Số lượng Ảnh':<15} | {'Số lượng Nhãn (.txt)'}")
    print("-" * 65)

    for subset in subsets:
        img_dir = os.path.join(dataset_path, 'images', subset)
        lbl_dir = os.path.join(dataset_path, 'labels', subset)
        
        # Đếm file ảnh (jpg, png, jpeg)
        img_count = 0
        if os.path.exists(img_dir):
            img_count = len([f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            
        # Đếm file nhãn (txt) và thống kê class
        lbl_count = 0
        class_counts = {} # Dùng để đếm riêng cho từng tập con
        
        if os.path.exists(lbl_dir):
            txt_files = [f for f in os.listdir(lbl_dir) if f.lower().endswith('.txt')]
            lbl_count = len(txt_files)
            
            # Đọc từng file txt để đếm class
            for txt_file in txt_files:
                file_path = os.path.join(lbl_dir, txt_file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            parts = line.strip().split()
                            if parts: # Nếu dòng không trống
                                class_id = int(parts[0]) # ID class luôn ở đầu dòng
                                class_counts[class_id] = class_counts.get(class_id, 0) + 1
                except Exception:
                    pass # Bỏ qua nếu file lỗi không đọc được
            
        if img_count > 0 or lbl_count > 0:
            print(f"{subset:<10} | {img_count:<15} | {lbl_count}")
            
            # Cảnh báo nếu số ảnh và số nhãn không khớp nhau
            if img_count != lbl_count:
                print(f"   ⚠️ CẢNH BÁO LỆCH DATA: Tập {subset} có {img_count} ảnh nhưng lại có {lbl_count} nhãn!")
            
            # Hiển thị chi tiết số lượng từng class của tập con này
            if class_counts:
                print(f"   📊 Chi tiết các Class trong tập [{subset}]:")
                for cid in sorted(class_counts.keys()):
                    c_name = DISPLAY_NAMES.get(cid, f"Unknown (ID: {cid})")
                    print(f"      + {c_name:<30}: {class_counts[cid]} đối tượng")
                print("   " + "-" * 55)
                
        total_imgs += img_count
        total_labels += lbl_count

    print("-" * 65)
    print(f"Tổng cộng  | {total_imgs:<15} | {total_labels}")
    print(f"Dung lượng | {total_size / (1024*1024):.2f} MB")
    print("=" * 65)

if __name__ == "__main__":
    # ĐƯỜNG DẪN TUYỆT ĐỐI ĐƯỢC ĐIỀN THẲNG
    path_data = r"D:\code\Final_cv\dataset\data_raw" 
    path_final = r"D:\code\Final_cv\dataset\dataset_final"
    
    # Chạy quét báo cáo
    scan_yolo_dataset(path_data)
    scan_yolo_dataset(path_final)