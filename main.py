import cv2
import time
import csv
import os
from ultralytics import YOLO

# Import các Module xử lý của bạn
from src.logic import TrafficLightController
from src.visualizer import SystemVisualizer
from src.counter import TrafficCounter # Đã thêm Module Counter vào hệ thống
from src.datastruct import FrameState

class Vehicle:
    def __init__(self, bbox, track_id, cls_id, cls_name):
        self.bbox = bbox
        self.track_id = track_id
        self.cls_id = cls_id
        self.cls_name = cls_name

# --- HÀM GHI LOG BENCHMARK ---
def save_benchmark(model_name, total_count, avg_latency, fps):
    try:
        with open('benchmark_results.csv', 'r') as f:
            pass
    except FileNotFoundError:
        with open('benchmark_results.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Model Name", "Total Vehicles", "Avg Latency", "Actual FPS"])

    with open('benchmark_results.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([model_name, total_count, f"{avg_latency:.2f}ms", f"{fps:.2f}"])
    print("📊 Đã lưu kết quả benchmark vào file 'benchmark_results.csv'!")

def main():
    print("🚀 KHỞI ĐỘNG HỆ THỐNG GIÁM SÁT GIAO THÔNG THÔNG MINH")
    
    # Sử dụng đường dẫn tương đối để sau này đổi máy không bị lỗi
    MODEL_PATH = r"models\train_v8_nano_combined\weights\best.pt"
    VIDEO_PATH = r"demo\test_2mins.mp4"
    OUTPUT_PATH = r"demo\output_video.mp4" 
    
    print("⏳ Đang tải mô hình AI...")
    model = YOLO(MODEL_PATH)
    print("✅ Tải mô hình thành công!")
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))
    print(f"📁 Video kết quả sẽ được lưu tại: {OUTPUT_PATH}")
    
    # Khởi tạo các Module điều khiển
    logic_controller = TrafficLightController(video_fps=fps)
    traffic_counter = TrafficCounter() # Khởi tạo bộ đếm xe
    visualizer = SystemVisualizer()
    
    frame_id = 0
    total_latency = 0.0
    system_start_time = time.time()
    
    print("🎬 Bắt đầu phân tích & render video. Vui lòng để hệ thống chạy hết video...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_id += 1
        state = FrameState(frame_id, frame)
        
        frame_start_time = time.time()
        
        # Phân tích AI & Tracking liên tục
        results = model.track(frame, persist=True, verbose=False, conf=0.4)
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            cls_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            names = model.names
            
            for box, track_id, cls_id in zip(boxes, track_ids, cls_ids):
                bbox = [int(box[0]), int(box[1]), int(box[2]), int(box[3])]
                cls_name = names[cls_id]
                state.vehicles.append(Vehicle(bbox, track_id, cls_id, cls_name))
        
        # Xử lý luồng dữ liệu qua các bộ não điều khiển
        state = traffic_counter.process(state) # 1. Đếm xe và phát hiện xe ưu tiên trước
        state = logic_controller.process(state)  # 2. Xử lý logic đèn tín hiệu dựa trên trạng thái
        
        # Vẽ giao diện lên frame (Đã xử lý chớp tắt cảnh báo trong này, không lo đè chữ)
        output_frame = visualizer.draw(state)
        
        total_latency += (time.time() - frame_start_time) * 1000
        
        # Ghi frame vào video output
        out.write(output_frame)
        
        # Hiển thị xem trực tiếp trên PC để theo dõi
        cv2.imshow("He Thong Giam Sat Uu Tien - Do An CV (PC RUN)", output_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("🛑 Đang dừng tiến trình theo yêu cầu...")
            break

    if frame_id > 0:
        total_duration = time.time() - system_start_time
        actual_fps = frame_id / total_duration
        avg_latency = total_latency / frame_id
        
        # Lấy tổng số xe đếm được thực tế từ bộ counter
        total_count = sum(traffic_counter.counts.values())
        
        save_benchmark("Proposed_Model", total_count, avg_latency, actual_fps)

    out.release() 
    cap.release()
    cv2.destroyAllWindows()
    print("💾 Đã tạo xong file video kết quả 'output_video.mp4'!")

if __name__ == "__main__":
    main()