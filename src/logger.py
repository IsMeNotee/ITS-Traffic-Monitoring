import csv
import threading
from pathlib import Path
from datetime import datetime
import cv2
import numpy as np
from queue import Queue, Empty

class TrafficLogger:
    """
    Module ghi log và chụp ảnh bất đồng bộ (Không làm giật lag video chính)
    """
    def __init__(self, output_dir="output"):
        self.output_dir = Path(output_dir)
        self.evidence_dir = self.output_dir / "evidence_images"
        self.log_file = self.output_dir / "traffic_log.csv"
        
        # Hàng đợi để giao việc cho Thread ngầm
        self.evidence_queue = Queue()
        self.log_queue = Queue()
        self._running = True 
        
        # Khởi động công nhân chạy ngầm
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
        
        self._init_files()
        print(f"📁 Khởi tạo Logger thành công (Dữ liệu lưu tại: {self.output_dir}/)")
        
    def _init_files(self):
        """Tạo sẵn thư mục và file CSV nếu chưa có"""
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        if not self.log_file.exists():
            with open(self.log_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Frame_ID', 'Timestamp', 'Vehicle_ID', 'Class_Name', 'Event_Type', 'Confidence', 'BBox'])

    def log_event(self, frame_id: int, vehicle_id: int, class_name: str, 
                  event_type: str, confidence: float, bbox: tuple):
        """Đẩy dữ liệu vào hàng đợi để ghi file CSV"""
        log_entry = {
            'frame_id': frame_id, 
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
            'vehicle_id': vehicle_id,
            'class_name': class_name, 
            'event_type': event_type,
            'confidence': f"{confidence:.4f}", 
            'bbox': str(bbox)
        }
        self.log_queue.put(log_entry)

    def save_image(self, frame: np.ndarray, vehicle_id: int, 
                   bbox: tuple, event_type: str, frame_id: int):
        """Đẩy ảnh vào hàng đợi để lưu ổ cứng (Chụp ảnh bằng chứng)"""
        evidence_item = {
            'frame': frame.copy(), # Bắt buộc copy để tránh xung đột RAM
            'vehicle_id': vehicle_id,
            'bbox': bbox,
            'event_type': event_type,
            'frame_id': frame_id,
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
        }
        self.evidence_queue.put(evidence_item)

    def _worker(self):
        """Hàm chạy ngầm liên tục để dọn dẹp hàng đợi"""
        while self._running or not self.log_queue.empty() or not self.evidence_queue.empty():
            # Xử lý ghi CSV
            while not self.log_queue.empty():
                try:
                    entry = self.log_queue.get_nowait()
                    with open(self.log_file, 'a', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            entry['frame_id'], entry['timestamp'], entry['vehicle_id'], 
                            entry['class_name'], entry['event_type'], entry['confidence'], entry['bbox']
                        ])
                    self.log_queue.task_done()
                except Empty: break
                except Exception as e: print(f"Lỗi ghi log: {e}")

            # Xử lý chụp ảnh
            while not self.evidence_queue.empty():
                try:
                    item = self.evidence_queue.get_nowait()
                    img = item['frame']
                    x1, y1, x2, y2 = item['bbox']
                    
                    # Vẽ khung đỏ cảnh báo khẩn cấp lên ảnh
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(img, f"ALERT: {item['event_type']} - ID:{item['vehicle_id']}", 
                                (max(0, x1), max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    filename = f"{item['event_type']}_ID{item['vehicle_id']}_{item['timestamp']}.jpg"
                    filepath = self.evidence_dir / filename
                    cv2.imwrite(str(filepath), img)
                    
                    self.evidence_queue.task_done()
                except Empty: break
                except Exception as e: print(f"Lỗi lưu ảnh: {e}")
            
            # Tạm nghỉ 10ms để nhường CPU cho AI chạy
            threading.Event().wait(0.01)

    def stop(self):
        """Dừng an toàn trước khi tắt chương trình"""
        self._running = False
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=3.0)