from dataclasses import dataclass, field
from typing import List, Tuple, Dict
import numpy as np

# 1. Khuôn mẫu cho một vật thể vừa được tìm thấy (Chưa có ID)
@dataclass
class Detection:
    bbox: Tuple[int, int, int, int]  # Tọa độ khung chữ nhật: x1, y1, x2, y2
    conf: float                      # Độ tự tin (%)
    cls_id: int                      # Mã loại xe (0, 1, 2...)
    cls_name: str                    # Tên loại xe (Fire_Truck, Ambulance...)

# 2. Khuôn mẫu cho một vật thể đang được theo dõi (Đã được cấp ID)
@dataclass
class TrackedVehicle(Detection):
    track_id: int = -1               # ID duy nhất của xe, mặc định là -1 cho đến khi Tracker cấp số

# 3. Khuôn mẫu tổng chứa trạng thái của 1 khung hình (Cái hộp truyền giữa các module)
@dataclass
class FrameState:
    frame_id: int                    # Khung hình số mấy trong video
    original_frame: np.ndarray       # Dữ liệu ảnh gốc
    detections: List[Detection] = field(default_factory=list)          # Danh sách xe (Detector đổ vào đây)
    vehicles: List[TrackedVehicle] = field(default_factory=list)       # Danh sách xe có ID (Tracker đổ vào đây)
    counts: Dict[str, int] = field(default_factory=dict)               # Bảng thống kê (Counter đổ vào đây)
    emergency_alert: bool = False