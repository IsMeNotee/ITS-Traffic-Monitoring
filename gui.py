import customtkinter as ctk
from tkinter import filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import os

# Cấu hình giao diện tổng thể
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class YouTubeStyleITS(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("🚦 HỆ THỐNG GIÁM SÁT GIAO THÔNG THÔNG MINH (ITS)")
        self.geometry("1000x700")
        self.minsize(800, 600)

        # Trạng thái hệ thống
        self.video_path = None
        self.cap = None
        self.is_playing = False
        self.total_frames = 0
        self.current_frame = 0
        self.fps = 30
        
        self.setup_ui()

    def setup_ui(self):
        # --- HEADER ---
        self.header_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.header_frame.pack(fill="x", padx=20, pady=(15, 0))
        
        ctk.CTkLabel(
            self.header_frame, 
            text="📺 CAMERA GIÁM SÁT AI", 
            font=ctk.CTkFont(family="Arial", size=20, weight="bold"),
            text_color="#e0e0e0"
        ).pack(side="left")

        # --- 1. MÀN HÌNH CHÍNH (Có viền bo góc) ---
        self.video_container = ctk.CTkFrame(self, corner_radius=15, border_width=2, border_color="#333333", fg_color="#000000")
        self.video_container.pack(pady=15, padx=20, fill="both", expand=True)

        self.lbl_video = ctk.CTkLabel(self.video_container, text="Vui lòng tải video lên hệ thống...", font=("Arial", 16, "italic"), text_color="#555555")
        self.lbl_video.pack(expand=True, fill="both", padx=5, pady=5)

        # --- 2. BẢNG ĐIỀU KHIỂN BÊN DƯỚI (Controls Panel) ---
        self.bottom_panel = ctk.CTkFrame(self, fg_color="#1a1a1a", corner_radius=15)
        self.bottom_panel.pack(fill="x", padx=20, pady=(0, 20), ipady=10)

        # 2.1 Thanh tua (Timeline)
        self.slider_var = ctk.DoubleVar(value=0)
        self.slider = ctk.CTkSlider(
            self.bottom_panel, 
            from_=0, to=100, 
            variable=self.slider_var, 
            command=self.seek_video,
            progress_color="#ff0000", # Đỏ YouTube
            button_color="#ff0000",
            button_hover_color="#cc0000",
            height=15
        )
        self.slider.pack(fill="x", padx=25, pady=(15, 10))
        self.slider.configure(state="disabled")

        # 2.2 Các nút bấm điều khiển
        self.controls = ctk.CTkFrame(self.bottom_panel, fg_color="transparent")
        self.controls.pack(fill="x", padx=20)

        # Cụm điều khiển bên trái
        self.left_controls = ctk.CTkFrame(self.controls, fg_color="transparent")
        self.left_controls.pack(side="left")

        # Nút Play/Pause
        self.btn_play = ctk.CTkButton(
            self.left_controls, text="▶ Phát", width=80, height=35,
            command=self.toggle_play_pause, state="disabled",
            fg_color="#333333", hover_color="#444444", font=ctk.CTkFont(weight="bold")
        )
        self.btn_play.pack(side="left", padx=(0, 5))

        # Lùi 5s
        self.btn_rewind = ctk.CTkButton(
            self.left_controls, text="⏪ 5s", width=60, height=35,
            command=lambda: self.skip_video(-5), state="disabled",
            fg_color="#333333", hover_color="#444444"
        )
        self.btn_rewind.pack(side="left", padx=5)

        # Tua 5s
        self.btn_forward = ctk.CTkButton(
            self.left_controls, text="5s ⏩", width=60, height=35,
            command=lambda: self.skip_video(5), state="disabled",
            fg_color="#333333", hover_color="#444444"
        )
        self.btn_forward.pack(side="left", padx=5)

        # Thời gian
        self.lbl_time = ctk.CTkLabel(self.left_controls, text="00:00 / 00:00", font=("Arial", 14), text_color="#aaaaaa")
        self.lbl_time.pack(side="left", padx=15)

        # Cụm công cụ bên phải
        self.right_controls = ctk.CTkFrame(self.controls, fg_color="transparent")
        self.right_controls.pack(side="right")

        self.btn_open = ctk.CTkButton(
            self.right_controls, text="📁 Chọn Video", width=120, height=35,
            command=self.open_file,
            fg_color="#10B981", hover_color="#059669", font=ctk.CTkFont(weight="bold")
        )
        self.btn_open.pack(side="right", padx=5)

    def open_file(self):
        file_path = filedialog.askopenfilename(
            title="Chọn Video", 
            filetypes=[("Video Files", "*.mp4 *.avi *.mkv")]
        )
        if not file_path:
            return

        self.video_path = file_path
        if self.cap:
            self.cap.release()

        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            messagebox.showerror("Lỗi", "Không thể mở video này!")
            return

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        
        # Mở khóa các nút
        self.slider.configure(to=self.total_frames, state="normal")
        for btn in [self.btn_play, self.btn_rewind, self.btn_forward]:
            btn.configure(state="normal")
            
        self.btn_play.configure(text="⏸ Tạm dừng")
        self.is_playing = True
        self.update_frame()

    def toggle_play_pause(self):
        if not self.cap or not self.cap.isOpened():
            return
            
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.btn_play.configure(text="⏸ Tạm dừng")
            self.update_frame()
        else:
            self.btn_play.configure(text="▶ Phát")

    def skip_video(self, seconds):
        if not self.cap or not self.cap.isOpened():
            return
            
        # Tính toán frame đích
        target_frame = self.current_frame + (seconds * self.fps)
        # Ép khung hình không vượt quá giới hạn video
        target_frame = max(0, min(target_frame, self.total_frames - 1))
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        self.current_frame = target_frame
        
        # Cập nhật UI ngay lập tức nếu đang tạm dừng
        if not self.is_playing:
            ret, frame = self.cap.read()
            if ret:
                self.slider_var.set(self.current_frame)
                self.update_time_label()
                self.display_frame(frame)

    def seek_video(self, value):
        if self.cap and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(value))
            self.current_frame = int(value)
            if not self.is_playing:
                ret, frame = self.cap.read()
                if ret:
                    self.update_time_label()
                    self.display_frame(frame)

    def update_time_label(self):
        current_sec = int(self.current_frame / self.fps)
        total_sec = int(self.total_frames / self.fps)
        self.lbl_time.configure(text=f"{self.format_time(current_sec)} / {self.format_time(total_sec)}")

    def update_frame(self):
        if self.cap and self.cap.isOpened() and self.is_playing:
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                self.slider_var.set(self.current_frame)
                self.update_time_label()
                self.display_frame(frame)
                
                # Logic độ trễ (delay) chuẩn theo FPS video
                delay = int(1000 / self.fps)
                self.after(delay, self.update_frame)
            else:
                self.is_playing = False
                self.btn_play.configure(text="▶ Phát lại")
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.slider_var.set(0)

    def display_frame(self, frame):
        # Lấy kích thước thực của label chứa video
        label_width = self.lbl_video.winfo_width()
        label_height = self.lbl_video.winfo_height()
        
        if label_width < 10 or label_height < 10: 
            return 
            
        # Dùng cv2.resize để thu phóng ảnh cực mượt, không bị lag như PIL
        frame_resized = cv2.resize(frame, (label_width, label_height))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        
        image = Image.fromarray(frame_rgb)
        photo = ImageTk.PhotoImage(image=image)
        self.lbl_video.configure(image=photo, text="")
        self.lbl_video.image = photo

    def format_time(self, seconds):
        mins, secs = divmod(seconds, 60)
        return f"{mins:02d}:{secs:02d}"

if __name__ == "__main__":
    app = YouTubeStyleITS()
    app.mainloop()