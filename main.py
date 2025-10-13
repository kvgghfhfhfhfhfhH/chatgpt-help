# main.py
from camera_view import CameraView
from audio_stream import AudioStream
import time

def main():
    camera = CameraView(camera_index=0)
    audio = AudioStream(duration=1, threshold=0.02)

    print("[INFO] Starting detection. Press Ctrl+C to quit.")
    try:
        while True:
            # Camera
            frame = camera.get_frame()
            if frame is not None:
                camera.detect_human(frame)
                camera.detect_color(frame)

            # Audio
            audio.detect_sound()

            time.sleep(0.5)
    except KeyboardInterrupt:
        print("[INFO] Exiting...")
    finally:
        camera.release()
        audio.close()

if __name__ == "__main__":
    main()
