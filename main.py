# main.py (Headless with ChatGPT API integration)
import os
import time
import sounddevice as sd
import numpy as np
import openai
from camera_view import CameraView
from audio_stream import AudioStream

# Load OpenAI key from .env
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Simple function to send prompt to ChatGPT
def ask_chatgpt(prompt):
    if any(greeting in prompt.lower() for greeting in ["hello", "hi", "hey"]):
        return "... sir"
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[ERROR] ChatGPT API call failed: {e}")
        return None

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
        print("[INFO] Starting headless camera and audio test. Press Ctrl+C to quit.")
        while True:
            # Capture camera frame (for detection or processing)
            frame = camera.get_frame()
            if frame is not None:
                print("[INFO] Frame captured.")

            # Capture audio clip
            clip = audio.record_short_clip()
            if clip is not None:
                # Convert audio to 16-bit PCM
                audio_data = (clip * 32767).astype(np.int16)

                # Save temporary audio file
                tmp_file = "tmp_audio.wav"
                import soundfile as sf
                sf.write(tmp_file, audio_data, audio.samplerate)

                # Send audio to OpenAI Whisper API
                try:
                    with open(tmp_file, "rb") as f:
                        transcript = openai.Audio.transcriptions.create(
                            file=f,
                            model="whisper-1"
                        )
                    text = transcript["text"]
                    print(f"[AUDIO] Heard: {text}")
                    
                    # Ask ChatGPT for a response
                    reply = ask_chatgpt(text)
                    if reply:
                        print(f"[GPT] {reply}")
                except Exception as e:
                    print(f"[ERROR] Failed transcription/ChatGPT: {e}")
                finally:
                    if os.path.exists(tmp_file):
                        os.remove(tmp_file)

            time.sleep(0.5)
    except KeyboardInterrupt:
        print("[INFO] Exiting...")
    finally:
        camera.release()
        audio.close()

if __name__ == "__main__":
    main()
