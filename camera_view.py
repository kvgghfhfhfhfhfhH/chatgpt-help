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
            return cv2.flip(frame, 1)  # Mirror horizontally
        return None

    def show_frame(self, frame):
        cv2.imshow("Camera View", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.release()
            exit()

    def release(self):
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
