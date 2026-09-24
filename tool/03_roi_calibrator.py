import cv2
import numpy as np

# Cấu hình video
VIDEO_PATH = 'demo/test.mp4'

# Lưu trữ tọa độ cho 2 vùng
areas = {
    'active': [],      # Vùng nhận diện toàn cục (Detection Area)
    'violation': []    # Vùng vi phạm đèn đỏ (Violation ROI)
}
current_step = 'active' # Bắt đầu với bước vẽ vùng active trước

def mouse_click(event, x, y, flags, param):
    global current_step
    if event == cv2.EVENT_LBUTTONDOWN:
        areas[current_step].append([x, y])
        print(f"📍 Đã chấm điểm cho vùng {current_step}: [{x}, {y}]")

def draw_polygon(img, points, color, thickness=2, fill_alpha=0.2):
    if len(points) == 0: return
    
    # Vẽ các điểm và viền
    for i in range(len(points)):
        cv2.circle(img, tuple(points[i]), 5, color, -1)
        if i > 0:
            cv2.line(img, tuple(points[i-1]), tuple(points[i]), color, thickness)
    if len(points) > 2:
        cv2.line(img, tuple(points[-1]), tuple(points[0]), color, thickness)
        
        # Đổ màu mờ
        pts = np.array(points, np.int32)
        overlay = img.copy()
        cv2.fillPoly(overlay, [pts], color)
        cv2.addWeighted(overlay, fill_alpha, img, 1 - fill_alpha, 0, img)

def main():
    global current_step
    print("🎨 TOOL VẼ ROI KÉP (ACTIVE AREA & VIOLATION ROI)")
    print("--------------------------------------------------")
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("❌ Không đọc được video!")
        return

    window_name = "Calibrator (Doc huong dan tren man hinh)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_click)

    while True:
        img_copy = frame.copy()
        
        # In hướng dẫn lên màn hình cho dễ dùng
        if current_step == 'active':
            instruction = "B1: Ve VUNG HOAT DONG (Xanh). Xong nhan 'N' de tiep tuc."
            color = (0, 255, 0)
        else:
            instruction = "B2: Ve VUNG VI PHAM (Vang). Xong nhan 'S' de Luu."
            color = (0, 255, 255)
            
        cv2.putText(img_copy, instruction, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(img_copy, "Phim 'C': Xoa ve lai | Phim 'Q': Thoat", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Vẽ các vùng đã/đang chấm
        draw_polygon(img_copy, areas['active'], (0, 255, 0))        # Xanh lá
        draw_polygon(img_copy, areas['violation'], (0, 255, 255))   # Vàng

        cv2.imshow(window_name, img_copy)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c') or key == ord('C'):
            areas[current_step].clear()
            print(f"🧹 Đã xóa vùng {current_step}. Hãy vẽ lại!")
            
        elif key == ord('n') or key == ord('N'):
            if current_step == 'active':
                if len(areas['active']) < 3:
                    print("⚠️ Vùng hoạt động cần ít nhất 3 điểm!")
                else:
                    current_step = 'violation'
                    print("✅ Đã chốt Vùng Hoạt Động. Chuyển sang vẽ Vùng Vi Phạm.")
                    
        elif key == ord('s') or key == ord('S'):
            if len(areas['violation']) < 3:
                print("⚠️ Vùng vi phạm cần ít nhất 3 điểm!")
            else:
                print("\n" + "="*50)
                print("🎉 ĐÃ LƯU THÀNH CÔNG! Copy cấu hình YAML dưới đây:")
                print("="*50)
                print("roi:")
                print(f"  active_area: {areas['active']}")
                print(f"  intersection_area: {areas['violation']}")
                print("="*50 + "\n")
                break
                
        elif key == ord('q') or key == ord('Q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()