# main.py
import os
import openai
import sounddevice as sd
import numpy as np
import cv2
import threading
import queue
import time
from dotenv import load_dotenv

# Load API key
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("[ERROR] OpenAI API key not found in .env!")

openai.api_key = OPENAI_API_KEY

# Audio settings
SAMPLE_RATE = 16000
DURATION = 5  # seconds

# Thread-safe queue for audio
audio_queue = queue.Queue()

# Camera class
class CameraView:
    def __init__(self, camera_index=0):
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            print(f"[ERROR] Cannot open camera at index {camera_index}")
            self.cap = None

    def get_frame(self):
        if not self.cap:
            return None
        ret, frame = self.cap.read()
        if ret:
            return cv2.flip(frame, 1)  # mirror horizontally
        return None

    def release(self):
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

# Function to detect gender from text
def infer_gender(text):
    male_words = ['he', 'him', 'sir', 'his', 'man']
    female_words = ['she', 'her', 'maam', 'woman', 'hers']
    text_lower = text.lower()
    for w in male_words:
        if w in text_lower:
            return "Male (he/him/sir)"
    for w in female_words:
        if w in text_lower:
            return "Female (she/her/maam)"
    return "Unknown"

# Record audio continuously
def record_audio():
    while True:
        try:
            data = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
            sd.wait()
            audio_queue.put(data)
        except Exception as e:
            print(f"[ERROR] Audio recording failed: {e}")
            time.sleep(1)

# Transcribe audio using OpenAI (replace if using newer OpenAI audio API)
def transcribe_audio(audio_data):
    audio_list = audio_data.flatten().tolist()
    try:
        transcription = openai.Audio.transcriptions(file=audio_list, model="whisper-1")
        return transcription["text"]
    except Exception as e:
        print(f"[ERROR] Transcription failed: {e}")
        return None

# GPT response
def gpt_response(prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[ERROR] GPT request failed: {e}")
        return None

# Analyze frame (simple placeholder detection)
def analyze_frame(frame):
    # Convert to grayscale and show
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Nova Camera View (Press q to close)", gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()

# Main listener
def listener(camera_index=0):
    camera = CameraView(camera_index)
    while True:
        if not audio_queue.empty():
            audio_data = audio_queue.get()
            transcription = transcribe_audio(audio_data)
            if transcription and transcription.lower().startswith("hey nova"):
                question = transcription[9:].strip()
                print(f"[User] {question}")
                
                # Capture frame while user speaks
                frame = camera.get_frame()
                if frame is not None:
                    analyze_frame(frame)
                
                answer = gpt_response(question)
                if answer:
                    gender = infer_gender(answer)
                    print(f"[Nova] ({gender}): {answer}")

# Main
def main():
    print("[INFO] Nova is listening. Say 'Hey Nova' to interact.")
    threading.Thread(target=record_audio, daemon=True).start()
    listener()

if __name__ == "__main__":
    main()
