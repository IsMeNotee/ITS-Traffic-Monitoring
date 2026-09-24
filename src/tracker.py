from deep_sort_realtime.deepsort_tracker import DeepSort
from .datastruct import TrackedVehicle, FrameState

class VehicleTracker:
    def __init__(self, max_age: int = 30):
        print("👁️ Khởi tạo Module Tracker (DeepSORT độc lập)")
        # Khởi tạo thuật toán DeepSORT
        # max_age: Số khung hình tối đa Tracker sẽ "nhớ" một chiếc xe nếu nó vô tình bị xe khác che khuất (occlusion).
        # n_init: Số khung hình liên tiếp xe phải xuất hiện thì mới chính thức được cấp ID (giảm nhiễu bóng ma).
        self.tracker = DeepSort(max_age=max_age, n_init=3, nn_budget=100)

    def process(self, frame_state: FrameState) -> FrameState:
        """Thực thi gán ID cho các xe (Chỉ Track, KHÔNG Detect)"""
        
        # 1. Chuyển đổi định dạng dữ liệu
        # Băng chuyền Detector gửi tới [x1, y1, x2, y2], nhưng DeepSORT lại đòi hỏi [x, y, width, height]
        bbs = []
        for det in frame_state.detections:
            x1, y1, x2, y2 = det.bbox
            width = x2 - x1
            height = y2 - y1
            # Đóng gói theo chuẩn DeepSORT: (Tọa độ, Độ tự tin, ID loại xe)
            bbs.append(([x1, y1, width, height], det.conf, det.cls_id))

        # 2. Đưa vào Tracker xử lý (Nó sẽ so khớp xe của frame hiện tại với frame trước đó)
        # Hàm này trả về danh sách các xe đã được gắn ID số hiệu
        tracks = self.tracker.update_tracks(bbs, frame=frame_state.original_frame)

        # 3. Đóng gói lại vào chiếc hộp FrameState để truyền đi tiếp
        frame_state.vehicles = []
        
        for track in tracks:
            # Chỉ lấy những xe đã được xác nhận (bỏ qua nhiễu)
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            ltrb = track.to_ltrb() # Chuyển ngược lại thành [x1, y1, x2, y2]
            cls_id = track.get_det_class()
            
            # Tìm lại tên class từ danh sách ban đầu
            cls_name = "Unknown"
            for det in frame_state.detections:
                if det.cls_id == cls_id:
                    cls_name = det.cls_name
                    break

            # Tạo Căn cước công dân (TrackedVehicle) cho chiếc xe
            vehicle = TrackedVehicle(
                bbox=(int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])),
                conf=track.det_conf if track.det_conf else 0.0,
                cls_id=cls_id,
                cls_name=cls_name,
                track_id=int(track_id) # ĐÂY LÀ ĐIỂM ĂN TIỀN CỦA MODULE NÀY
            )
            
            frame_state.vehicles.append(vehicle)

        return frame_state