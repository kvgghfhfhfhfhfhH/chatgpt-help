# camera_view.py
import cv2

class CameraView:
    def __init__(self, camera_index=0):
        self.cap = cv2.VideoCapture(camera_index)
        self.opened = self.cap.isOpened()
        if not self.opened:
            print(f"[ERROR] Cannot open camera at index {camera_index}")

    def get_frame(self):
        if not self.opened:
            return None
        ret, frame = self.cap.read()
        if ret:
            return frame
        return None

    def release(self):
        if self.cap.isOpened():
            self.cap.release()

def start_camera_view(camera_index=0):
    camera = CameraView(camera_index)
    if not camera.opened:
        return

    while True:
        frame = camera.get_frame()
        if frame is not None:
            print("[INFO] Camera frame captured.")
