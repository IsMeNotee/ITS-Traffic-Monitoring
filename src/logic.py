import yaml
import cv2
import numpy as np
from .datastruct import FrameState

class TrafficLightController:
    def __init__(self, config_path: str = "config/system_config.yaml", video_fps: int = 30):
        print("🚦 Khởi tạo Module Logic (Bản Phục Hồi Code Gốc)")
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        rules = config['traffic_logic']
        
        self.active_area = np.array(config['roi']['active_area'], np.int32)
        self.violation_area = np.array(config['roi']['intersection_area'], np.int32)
        
        self.fps = video_fps
        self.green_duration = rules['default_green_time'] * self.fps
        self.yellow_duration = rules['yellow_clearance_time'] * self.fps
        self.red_duration = rules['all_red_clearance_time'] * self.fps
        
        # Đồng bộ trạng thái ban đầu: Đèn Đỏ (22s)
        self.current_color = "RED" 
        self.timer_frames = 22 * self.fps

        # --- KHỞI TẠO BIẾN ĐẾM XE THỰC TẾ ---
        self.tracked_vehicle_ids = set() # Lưu danh sách ID duy nhất để không đếm trùng
        self.total_vehicles_count = 0

    def process(self, frame_state: FrameState) -> FrameState:
        # 1. Lọc xe & Bắt lỗi
        valid_vehicles = []
        
        # SỬA ĐÚNG 1 DÒNG Ở ĐÂY: Kế thừa danh sách vi phạm từ counter.py để không bị ghi đè
        violator_ids = getattr(frame_state, 'all_violator_ids', set())

        for vehicle in frame_state.vehicles:
            x1, y1, x2, y2 = vehicle.bbox
            cx, cy = int((x1 + x2) / 2), int(y2) - 5
            
            # Chỉ xử lý xe nằm trong vùng xanh (Active Area)
            if cv2.pointPolygonTest(self.active_area, (cx, cy), False) >= 0:
                valid_vehicles.append(vehicle)
                
                # --- CẬP NHẬT THỐNG KÊ SỐ LƯỢNG XE ĐÃ XỬ LÝ ---
                if vehicle.track_id not in self.tracked_vehicle_ids:
                    self.tracked_vehicle_ids.add(vehicle.track_id)
                    self.total_vehicles_count = len(self.tracked_vehicle_ids)

            # Nếu đèn ĐỎ, xe nào đè vào vùng vàng -> Vi phạm
            if self.current_color == "RED":
                if cv2.pointPolygonTest(self.violation_area, (cx, cy), False) >= 0:
                    violator_ids.add(vehicle.track_id)

        frame_state.vehicles = valid_vehicles
        frame_state.all_violator_ids = violator_ids

        # 2. Đếm ngược đèn (Logic thuần túy)
        self.timer_frames -= 1
        if self.timer_frames <= 0:
            if self.current_color == "GREEN":
                self.current_color = "YELLOW"; self.timer_frames = self.yellow_duration
            elif self.current_color == "YELLOW":
                self.current_color = "RED"; self.timer_frames = self.red_duration
            elif self.current_color == "RED":
                self.current_color = "GREEN"; self.timer_frames = self.green_duration

        setattr(frame_state, 'light_color', self.current_color)
        setattr(frame_state, 'light_time', max(1, self.timer_frames // self.fps))

        # Lấy tín hiệu ưu tiên từ counter, không dùng để đổi đèn, chỉ để báo UI
        is_emergency = getattr(frame_state, 'emergency_alert', False)
        setattr(frame_state, 'is_emergency', is_emergency)

        return frame_state

    def get_stats(self):
        # Trả về số lượng xe đã đếm được trong phiên chạy
        return self.total_vehicles_count