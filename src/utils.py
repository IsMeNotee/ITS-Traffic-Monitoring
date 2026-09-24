"""
utils.py: Cung cấp các hàm toán học và tiện ích xử lý hình học không gian.
Các hàm này tách biệt với luồng logic chính để giữ cho hệ thống Clean Code.
"""
import cv2
import numpy as np

def get_bottom_center(bbox: tuple) -> tuple:
    """
    Toán học Không gian: Lấy tọa độ tâm dưới cùng (điểm chạm đất) của Bounding Box.
    Giúp xác định chính xác vị trí bánh xe trên mặt đường thay vì dùng tâm (center) của xe.
    """
    x1, y1, x2, y2 = bbox
    center_x = int((x1 + x2) / 2)
    bottom_y = int(y2)
    return (center_x, bottom_y)

def calculate_iou(box1: tuple, box2: tuple) -> float:
    """
    Tính chỉ số IoU (Intersection over Union) giữa 2 Bounding Box.
    Công thức cốt lõi để so khớp các vật thể chồng chéo lên nhau.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Diện tích phần giao nhau
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Diện tích từng Box
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Công thức IoU
    iou = intersection / float(area1 + area2 - intersection) if (area1 + area2 - intersection) > 0 else 0
    return iou

def draw_text_with_bg(img: np.ndarray, text: str, position: tuple, 
                      font=cv2.FONT_HERSHEY_SIMPLEX, scale=0.6, 
                      text_color=(255, 255, 255), bg_color=(0, 0, 0), thickness=2):
    """
    Vẽ chữ có viền và nền đằng sau để chống lóa khi camera bị chói nắng.
    """
    x, y = position
    text_size, _ = cv2.getTextSize(text, font, scale, thickness)
    
    # Vẽ hộp nền đen mờ
    cv2.rectangle(img, (x, y - text_size[1] - 5), (x + text_size[0], y + 5), bg_color, -1)
    # Ghi chữ lên trên
    cv2.putText(img, text, (x, y), font, scale, text_color, thickness, cv2.LINE_AA)
    
    return img