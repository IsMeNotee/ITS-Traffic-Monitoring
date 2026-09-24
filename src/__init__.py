"""
Smart Traffic Monitoring System - Core Modules
Version: 1.0.0
Author: [Tên_Của_Bạn]
Description: Hệ thống nhận diện, theo dõi và cảnh báo giao thông thông minh sử dụng YOLOv8.
"""

# Khai báo sẵn các khuôn mẫu dữ liệu để bên ngoài import dễ dàng hơn
from .datastruct import FrameState, Detection, TrackedVehicle

__all__ = ['FrameState', 'Detection', 'TrackedVehicle']

print("📦 Đã nạp thành công lõi hệ thống (Core Engine v1.0)")