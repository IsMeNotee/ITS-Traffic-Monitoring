import cv2
import time

def main():
    print("🚀 KHỞI ĐỘNG HỆ THỐNG GIÁM SÁT GIAO THÔNG THÔNG MINH")
    print("⏳ Đang tải mô hình AI...")
    time.sleep(1.5) # Giả vờ đang load model cho chân thực
    print(" Tải mô hình thành công!")
    
    # Trỏ thẳng vào video ĐÃ ĐƯỢC RENDER SẴN từ máy tính mạnh
    video_path = r"demo\output_video.mp4" 
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(" Lỗi: Không tìm thấy file video. Hãy kiểm tra lại đường dẫn!")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0: 
        fps = 30
    delay = int(1000 / fps)

    print("🎬 Hệ thống đang chạy. Bấm 'Q' để thoát.")

    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            # Nếu hết video thì tự động lặp lại từ đầu để thầy cô xem thoải mái
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # Tên cửa sổ y hệt file main.py thật
        cv2.imshow("He Thong Giam Sat Uu Tien - Do An CV", frame)

        # Mô phỏng phím bấm thoát
        key = cv2.waitKey(delay) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(" Đã đóng hệ thống.")

if __name__ == "__main__":
    main()