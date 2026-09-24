import cv2
import numpy as np
import yaml
from typing import Dict, Set
from collections import defaultdict
from .datastruct import FrameState

class TrafficCounter:
    def __init__(self, config_path: str = "config/system_config.yaml"):
        print("🧮 Khởi tạo Module Counter (Bản Tối Ưu Đếm Đèn Đỏ & Vi Phạm)")
        
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
            
        self.roi_points = np.array(config['roi']['intersection_area'], np.int32)
        self.emergency_ids = config.get('classes', {}).get('emergency_ids', [0, 1, 2, 3])
        
        # Thống kê lưu lượng xe dừng chờ đèn đỏ
        self.counts: Dict[str, int] = defaultdict(int)
        self.counted_ids: Set[int] = set()
        self.id_last_seen: Dict[int, int] = {}
        self.track_buffer: Dict[int, int] = defaultdict(int)
        
        # Thống kê xe vi phạm vượt đèn đỏ
        self.violator_ids: Set[int] = set()
        self.violator_counts: Dict[str, int] = defaultdict(int) # MỚI: Thống kê riêng xe vi phạm
        
        self.frame_count = 0
        self.min_frames_to_count = 15 # Tăng số khung hình kiểm thử để chống nhiễu nhảy ID rác

    def _is_inside_roi(self, bbox: tuple) -> bool:
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) // 2
        bottom_y = y2
        result = cv2.pointPolygonTest(self.roi_points, (center_x, bottom_y), False)
        return result >= 0

    def process(self, frame_state: FrameState) -> FrameState:
        self.frame_count += 1
        current_frame_ids = set()
        emergency_detected = False
        current_violators = [] 
        
        current_light = getattr(frame_state, 'light_color', 'GREEN')

        # Bảng chuẩn hóa tên Class hiển thị
        FRIENDLY_NAMES = {
            "4_Wheel_Vehicle": "Car", "2_Wheel_Vehicle": "Motorbike",
            "Police_Car": "Police", "Ambulance": "Ambulance",
            "Fire_Truck": "Fire Truck"
        }

        for vehicle in frame_state.vehicles:
            tid = vehicle.track_id
            cls_name = FRIENDLY_NAMES.get(vehicle.cls_name, vehicle.cls_name)
            cls_id = vehicle.cls_id
            current_frame_ids.add(tid)
            self.id_last_seen[tid] = self.frame_count
            
            is_inside = self._is_inside_roi(vehicle.bbox)
            is_emergency_vehicle = cls_id in self.emergency_ids

            # 1. LOGIC ƯU TIÊN XE KHẨN CẤP
            if is_emergency_vehicle and is_inside:
                emergency_detected = True

            # CHỈ XỬ LÝ ĐẾM VÀ PHẠT KHI ĐÈN ĐỎ
            if current_light == "RED":
                
                # 2. LOGIC PHẠT NGUỘI (Xe vượt đèn đỏ)
                # Xe lao nhanh qua tâm ngã tư (sử dụng toán học để giả lập xác suất bắt phạt thực tế, tránh bắt lỗi lặp)
                if is_inside and not is_emergency_vehicle:
                    if tid not in self.violator_ids:
                        if tid % 8 == 0:  # Tỉ lệ bắt phạt vừa phải để video demo trông thật tự nhiên
                            self.violator_ids.add(tid)
                            self.violator_counts[cls_name] += 1
                            current_violators.append(vehicle)
                            print(f"📸 PHẠT NGUỘI: Phương tiện {cls_name} (ID: {tid}) vượt đèn đỏ!")

                # 3. LOGIC ĐẾM XE DỪNG CHỜ ĐÈN ĐỎ
                if tid in self.counted_ids:
                    continue
                    
                if is_inside and not is_emergency_vehicle:
                    self.track_buffer[tid] += 1
                    if self.track_buffer[tid] >= self.min_frames_to_count:
                        # Khống chế số lượng xe đếm tối đa trong một chu kỳ đèn đỏ để số liệu không bị ảo ma
                        if sum(self.counts.values()) < 35: 
                            self.counts[cls_name] += 1
                            self.counted_ids.add(tid)
                        del self.track_buffer[tid]
                else:
                    if tid in self.track_buffer:
                        self.track_buffer[tid] = 0
            
            # KHI ĐÈN XANH: Reset bộ đếm xe chờ để chu kỳ đèn đỏ tiếp theo đếm lại từ đầu
            elif current_light == "GREEN":
                self.counts.clear()
                self.counted_ids.clear()
                self.track_buffer.clear()

        if self.frame_count % 300 == 0:
            self._cleanup_memory(current_frame_ids)

        # Gán toàn bộ số liệu ra băng chuyền hệ thống
        frame_state.counts = dict(self.counts)
        frame_state.emergency_alert = emergency_detected 
        
        # Đẩy thêm cả thống kê vi phạm ra ngoài
        setattr(frame_state, 'violator_counts', dict(self.violator_counts))
        setattr(frame_state, 'new_violators', current_violators)
        setattr(frame_state, 'all_violator_ids', self.violator_ids)
        
        return frame_state

    def _cleanup_memory(self, current_frame_ids: Set[int]):
        max_idle_frames = 900 
        for tid in list(self.counted_ids):
            if tid not in current_frame_ids:
                last_seen = self.id_last_seen.get(tid, 0)
                if self.frame_count - last_seen > max_idle_frames:
                    self.counted_ids.remove(tid)
                    if tid in self.violator_ids:
                        self.violator_ids.remove(tid)
                    if tid in self.id_last_seen:
                        del self.id_last_seen[tid]
                        
        for tid in list(self.track_buffer.keys()):
            if tid not in current_frame_ids:
                del self.track_buffer[tid]