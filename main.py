# main.py
from camera_view import CameraView
from audio_stream import AudioStream
import sounddevice as sd
import time
import os
from dotenv import load_dotenv
import openai

# Load API key from .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_default_camera_index():
    import cv2
    index = 0
    while True:
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            cap.release()
            return index
        index += 1
        if index > 10:
            raise RuntimeError("No camera detected!")

def get_default_mic_device():
    devices = sd.query_devices()
    for i, d in enumerate(devices):
        if d['max_input_channels'] > 0:
            return i
    raise RuntimeError("No microphone detected!")

def main():
    cam_index = get_default_camera_index()
    print(f"[INFO] Using camera index: {cam_index}")
    camera = CameraView(camera_index=cam_index)

    mic_device = get_default_mic_device()
    print(f"[INFO] Using microphone device: {mic_device}")
    audio = AudioStream(device=mic_device, duration=3)

    try:
        print("[INFO] Starting audio loop. Press Ctrl+C to stop.")
        while True:
            # Record audio
            clip = audio.record_short_clip()
            if clip is not None:
                print("[INFO] Audio clip recorded.")

                # Convert audio to text via OpenAI
                # Placeholder: simulate transcription
                transcript = "hello"  # You can integrate Whisper here
                print(f"You said: {transcript}")

                # GPT response
                if "hello" in transcript.lower() or "hi" in transcript.lower():
                    response = "Yes, sir."
                else:
                    response = "I heard you."

                print(f"GPT: {response}")

            # Capture frame without GUI
            frame = camera.get_frame()
            if frame is not None:
                print("[INFO] Frame captured.")
            
            time.sleep(1)

    except KeyboardInterrupt:
        print("[INFO] Exiting...")
    finally:
        camera.release()
        audio.close()

if __name__ == "__main__":
    main()
