import cv2
import numpy as np
import yaml
import time 

class SystemVisualizer:
    def __init__(self, config_path="config/system_config.yaml"):
        print("🎨 Khởi tạo Module Visualizer (Giao diện UI Chuẩn Hóa)")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        self.active_area = np.array(config['roi']['active_area'], np.int32)
        self.violation_area = np.array(config['roi']['intersection_area'], np.int32)
        
        self.prev_time = time.time()
        
    def draw(self, state) -> np.ndarray:
        frame = state.original_frame.copy()
        width = frame.shape[1]
        height = frame.shape[0]
        
        # 1. Vẽ 2 Vùng ROI
        cv2.polylines(frame, [self.active_area], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.polylines(frame, [self.violation_area], isClosed=True, color=(0, 255, 255), thickness=2)
        
        # 2. Đọc thông tin Đèn giao thông ảo
        light_color = getattr(state, 'light_color', 'GREEN')
        light_time = getattr(state, 'light_time', 0)
        is_emergency = getattr(state, 'is_emergency', False)
        frame_id = getattr(state, 'frame_id', 0)
        
        # 3. Vẽ Bảng LED Trạng thái Đèn lên góc phải
        color_map = {"GREEN": (0, 255, 0), "YELLOW": (0, 255, 255), "RED": (0, 0, 255)}
        box_color = color_map.get(light_color, (255, 255, 255))
        
        cv2.rectangle(frame, (950, 20), (1250, 80), (0, 0, 0), -1)
        cv2.putText(frame, f"LIGHT: {light_color} ({light_time}s)", (970, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2)
        
        # 4. Hiệu ứng báo động chớp tắt của AI (Giữ lại vì đẹp)
        if is_emergency:
            if frame_id % 30 < 15:
                cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), 12)
                cv2.putText(frame, "[!] DETECTED EMERGENCY VEHICLE [!]", (300, 110), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 4)

        # 5. Vẽ Bounding Box và Khôi phục bộ đếm xe thời gian thực độc lập
        violator_ids = getattr(state, 'all_violator_ids', set())
        
        DISPLAY_NAMES = {
            "4_Wheel_Vehicle": "Car", "2_Wheel_Vehicle": "Motorbike",
            "Police_Car": "Police", "Ambulance": "Ambulance",
            "Fire_Truck": "Fire Truck", "Warning_Light": "Siren"
        }
        
        CLASS_COLORS = {
            "Car": (255, 255, 0),        # Cyan
            "Motorbike": (0, 165, 255),  # Cam
            "Police": (0, 0, 255),       # Đỏ
            "Ambulance": (255, 0, 0),    # Xanh dương
            "Fire Truck": (0, 0, 255),   # Đỏ
        }
        
        # KHÔI PHỤC LOGIC TỰ ĐẾM XE TRÊN KHUNG HÌNH (Tránh lỗi sập hệ thống)
        vehicle_counts = {}
        
        for vehicle in state.vehicles:
            x1, y1, x2, y2 = vehicle.bbox
            tid = vehicle.track_id
            
            friendly_name = DISPLAY_NAMES.get(vehicle.cls_name, vehicle.cls_name)
            vehicle_counts[friendly_name] = vehicle_counts.get(friendly_name, 0) + 1
            
            bbox_color = CLASS_COLORS.get(friendly_name, (0, 255, 0)) 
            text_color = (0, 0, 0) 
            label = f"{friendly_name} #{tid}"
            box_thickness = 2
            
            if tid in violator_ids:
                bbox_color = (0, 0, 255) 
                text_color = (255, 255, 255) 
                label = f"VIOLATOR #{tid}"
                box_thickness = 3

            cv2.rectangle(frame, (x1, y1), (x2, y2), bbox_color, box_thickness)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 2
            
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
            cv2.rectangle(frame, (x1, y1 - text_height - baseline - 5), (x1 + text_width, y1), bbox_color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), font, font_scale, text_color, font_thickness)

        # 6. Hiển thị Model text góc trái
        cv2.putText(frame, "Model: Proposed (YOLOv8-Optimized)", (20, 690), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    
        # 7. FPS VÀ THỐNG KÊ SỐ LƯỢNG XE (ĐÃ ĐƯỢC CHUẨN HÓA AN TOÀN)
        current_time = time.time()
        fps = 1 / (current_time - self.prev_time) if (current_time - self.prev_time) > 0 else 0
        self.prev_time = current_time
        
        cv2.putText(frame, f"FPS: {int(fps)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # --- BẢNG 1: SỐ LƯỢNG XE HIỆN TẠI TRÊN FRAME (Đã sửa lại tiêu đề đúng thực tế)
        y_offset = 90
        cv2.putText(frame, "--- CURRENT VEHICLES ---", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 35 
        
        if not vehicle_counts:
            cv2.putText(frame, "No vehicles in scene.", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 2)
            y_offset += 35
        else:
            for v_name, count in vehicle_counts.items():
                cv2.putText(frame, f"{v_name}: {count}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
                y_offset += 30 
                
        # --- BẢNG 2: SỐ LƯỢNG XE VI PHẠM (Chỉ hiện khi có xe vi phạm thực tế)
        if len(violator_ids) > 0:
            y_offset += 20 
            cv2.putText(frame, "--- VIOLATIONS ---", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            y_offset += 35
            cv2.putText(frame, f"Total Violators: {len(violator_ids)}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

        return frame