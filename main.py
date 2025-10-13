# main.py
import os
import time
import numpy as np
import sounddevice as sd
import cv2
from openai import OpenAI
from dotenv import load_dotenv
from audio_stream import AudioStream
from camera_view import CameraView

# Load .env file (API key)
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_default_camera_index():
    """Find the first available USB camera."""
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cap.release()
            return i
    raise RuntimeError("No camera detected!")

def get_default_mic_device():
    """Find first available microphone."""
    devices = sd.query_devices()
    for i, d in enumerate(devices):
        if d["max_input_channels"] > 0:
            return i
    raise RuntimeError("No microphone detected!")

def speech_to_text(audio_data, samplerate):
    """Transcribe short audio using Whisper."""
    try:
        import tempfile, wave
        temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        with wave.open(temp_wav.name, "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(samplerate)
            f.writeframes((audio_data * 32767).astype(np.int16).tobytes())

        with open(temp_wav.name, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        os.remove(temp_wav.name)
        return transcript.text.strip()
    except Exception as e:
        print(f"[ERROR] Speech recognition failed: {e}")
        return ""

def ask_gpt(prompt):
    """Send prompt to GPT and return response."""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an assistant that responds briefly and politely."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[ERROR] GPT API failed: {e}")
        return "[No response]"

def main():
    # Setup devices
    cam_index = get_default_camera_index()
    print(f"[INFO] Using camera index: {cam_index}")
    camera = CameraView(camera_index=cam_index)

    mic_device = get_default_mic_device()
    print(f"[INFO] Using microphone device: {mic_device}")
    audio = AudioStream(device=mic_device, duration=3)

    print("[INFO] System ready. Listening and observing...")

    try:
        while True:
            # Capture frame for context/logging
            frame = camera.get_frame()
            if frame is not None:
                print("[INFO] Frame captured.")

            # Record audio
            clip = audio.record_short_clip()
            if clip is None or np.max(np.abs(clip)) < 0.01:
                # Skip silence
                continue

            print("[INFO] Audio clip recorded. Processing...")

            text = speech_to_text(clip.flatten(), audio.samplerate)
            if not text:
                continue

            print(f"You said: {text}")

            # Simple greeting trigger
            if any(word in text.lower() for word in ["hi", "hello", "hey", "greetings"]):
                gpt_response = "Yes, sir."
            else:
                gpt_response = ask_gpt(text)

            print(f"GPT: {gpt_response}")
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("[INFO] Exiting...")
    finally:
        camera.release()
        audio.close()

if __name__ == "__main__":
    main()
