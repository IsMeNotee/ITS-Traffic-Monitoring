import cv2
import numpy as np
from ultralytics import YOLO
from .datastruct import Detection, FrameState

class VehicleDetector:
    def __init__(self, model_path: str, conf_thresh: float = 0.5):
        print(f"🔧 Khởi tạo Detector với mô hình: {model_path}")
        self.model = YOLO(model_path)
        self.conf_thresh = conf_thresh
        
        # Kế thừa ý tưởng hay: Dùng CLAHE để tự động làm sáng ảnh nếu video quay lúc chạng vạng
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Tự động cân bằng sáng mà không làm biến dạng màu sơn xe"""
        # SỬA LỖI: Bọc list() ra ngoài để có thể thay đổi giá trị
        lab = list(cv2.split(cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)))
        lab[0] = self.clahe.apply(lab[0])
        return cv2.cvtColor(cv2.merge(lab), cv2.COLOR_LAB2BGR)

    def process(self, frame_state: FrameState) -> FrameState:
        """Thực thi nhận diện (Chỉ Detect, KHÔNG Track)"""
        
        # 1. Tiền xử lý (Làm rõ ảnh)
        # enhanced_frame = self._preprocess(frame_state.original_frame)

        enhanced_frame = frame_state.original_frame.copy()
        
        # 2. Dùng mô hình YOLO để tìm xe (Sử dụng hàm predict thay vì track)
        results = self.model.predict(
            enhanced_frame, 
            conf=self.conf_thresh, 
            verbose=False # Tắt log cho đỡ rác Terminal
        )
        
        # 3. Đóng gói kết quả vào chuẩn dữ liệu đã định nghĩa
        frame_state.detections = []
        
        if len(results) > 0:
            result = results[0]
            boxes = result.boxes
            
            if boxes is not None and len(boxes) > 0:
                # Chuyển dữ liệu từ Card đồ họa (GPU) về RAM (CPU) để tính toán
                xyxy = getattr(boxes.xyxy, "cpu")().numpy()
                confs = getattr(boxes.conf, "cpu")().numpy()
                clss = getattr(boxes.cls, "cpu")().numpy()
                names = result.names
                
                for box, conf, cls in zip(xyxy, confs, clss):
                    cls_id = int(cls)
                    
                    # Tạo một đối tượng Detection mới
                    det = Detection(
                        # SỬA LỖI PYLANCE: Bóc tách chính xác 4 tọa độ thay vì dùng map
                        bbox=(int(box[0]), int(box[1]), int(box[2]), int(box[3])),
                        conf=float(conf),
                        cls_id=cls_id,
                        cls_name=names[cls_id]
                    )
                    
                    # Đẩy vào danh sách
                    frame_state.detections.append(det)
                    
        return frame_state