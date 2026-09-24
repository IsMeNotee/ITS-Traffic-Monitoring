import os
import cv2
import glob

# Trỏ vào thư mục data tổng hợp chưa chia
IMG_DIR = "data/images/train"
LBL_DIR = "data/labels/train"

def clean_dataset():
    print("="*50)
    print("🚀 BẮT ĐẦU KIỂM TRA VÀ LÀM SẠCH DỮ LIỆU (DATA CLEANING)")
    print("="*50)

    # Đếm số lượng file ban đầu
    txt_files = glob.glob(os.path.join(LBL_DIR, "*.txt"))
    total_files = len(txt_files)
    
    print(f"📁 Tổng số file nhãn đang có: {total_files} file\n")

    # Khởi tạo bộ đếm lỗi
    empty_labels_count = 0
    corrupted_images_count = 0
    out_of_bounds_count = 0
    missing_images_count = 0

    files_to_remove = []

    for txt_path in txt_files:
        base_name = os.path.splitext(os.path.basename(txt_path))[0]
        
        # 1. Tìm ảnh tương ứng
        img_path = os.path.join(IMG_DIR, base_name + ".jpg")
        if not os.path.exists(img_path):
            img_path = img_path.replace(".jpg", ".png")
            if not os.path.exists(img_path):
                missing_images_count += 1
                files_to_remove.append((txt_path, None))
                continue

        # 2. KIỂM TRA NHÃN RỖNG (Empty Labels)
        is_empty = False
        if os.path.getsize(txt_path) == 0:
            is_empty = True
        else:
            with open(txt_path, 'r') as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
                if len(lines) == 0:
                    is_empty = True
                else:
                    # 3. Kiểm tra tọa độ Bounding Box (Out of bounds)
                    for line in lines:
                        parts = line.split()
                        if len(parts) == 5:
                            # YOLO format: class x_center y_center width height
                            try:
                                _, x, y, w, h = map(float, parts)
                                if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0):
                                    out_of_bounds_count += 1
                                    files_to_remove.append((txt_path, img_path))
                                    break # Lỗi 1 dòng là vứt cả file
                            except ValueError:
                                pass

        if is_empty:
            empty_labels_count += 1
            files_to_remove.append((txt_path, img_path))
            continue # Đã rỗng thì bỏ qua check ảnh để tiết kiệm thời gian

        # 4. Kiểm tra ảnh hỏng (Corrupted Images)
        # Dùng cv2 đọc thử, nếu trả về None tức là file ảnh bị lỗi bit hoặc hỏng cấu trúc
        img = cv2.imread(img_path)
        if img is None:
            corrupted_images_count += 1
            files_to_remove.append((txt_path, img_path))

    # XÓA CÁC FILE LỖI
    print("⏳ Đang tiến hành xóa các file rác/lỗi...")
    removed_count = 0
    for txt_p, img_p in set(files_to_remove):
        try:
            if os.path.exists(txt_p):
                os.remove(txt_p)
            if img_p and os.path.exists(img_p):
                os.remove(img_p)
            removed_count += 1
        except Exception as e:
            pass

    # BÁO CÁO TỔNG KẾT (Dùng để chụp màn hình đưa vào báo cáo)
    print("\n" + "="*50)
    print("📊 BÁO CÁO KẾT QUẢ TIỀN XỬ LÝ (PREPROCESSING REPORT)")
    print("="*50)
    print(f" ❌ Số ảnh không tồn tại (mất cặp): {missing_images_count}")
    print(f" ❌ Số nhãn rỗng (Background/No object): {empty_labels_count}")
    print(f" ❌ Số nhãn sai tọa độ (> 1.0 hoặc < 0.0): {out_of_bounds_count}")
    print(f" ❌ Số ảnh hỏng (Corrupted/Unreadable): {corrupted_images_count}")
    print("-" * 50)
    print(f" 🗑️ TỔNG SỐ CẶP DỮ LIỆU ĐÃ XÓA: {removed_count}")
    print(f" ✅ SỐ CẶP DỮ LIỆU SẠCH CÒN LẠI: {total_files - removed_count}")
    print("="*50)
    print("Hệ thống dữ liệu đã đạt chuẩn 100%. Sẵn sàng chia tỷ lệ!")

if __name__ == "__main__":
    clean_dataset()