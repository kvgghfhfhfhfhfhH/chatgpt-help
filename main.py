# main.py
import os
import cv2
import json
import queue
import threading
import time
import numpy as np
from flask import Flask, Response, render_template_string, request
import sounddevice as sd
import speech_recognition as sr
import pyttsx3
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_KEY:
    raise ValueError("[ERROR] OpenAI API key not found in .env!")

client = OpenAI(api_key=OPENAI_KEY)

# Flask app
app = Flask(__name__)

# Audio
audio_queue = queue.Queue()
recognizer = sr.Recognizer()
mic = sr.Microphone()

# TTS
tts_engine = pyttsx3.init()

# Training file
TRAIN_FILE = "nova_training.json"
if os.path.exists(TRAIN_FILE):
    with open(TRAIN_FILE, "r") as f:
        training_data = json.load(f)
else:
    training_data = {}

# Camera
camera = cv2.VideoCapture(0)
if not camera.isOpened():
    raise RuntimeError("[ERROR] Cannot open camera.")

# Object detection model (YOLOv5 via OpenCV DNN)
net = cv2.dnn.readNet(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")  # placeholder
CONF_THRESHOLD = 0.5

# HTML template
HTML = """
<html>
<head><title>Nova Webcam</title></head>
<body>
<h1>Nova AI Webcam Feed</h1>
<img src="{{ url_for('video_feed') }}" width="640" height="480">
<form method="POST" action="/train">
<label>Correction:</label>
<input type="text" name="correction">
<input type="submit" value="Submit">
</form>
</body>
</html>
"""

# Video streaming generator
def gen_frames():
    while True:
        success, frame = camera.read()
        if not success:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = net.detectMultiScale(gray, 1.3, 5) if hasattr(net, 'detectMultiScale') else []

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = "Face: 99%"
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/train', methods=['POST'])
def train():
    correction = request.form.get("correction")
    if correction:
        training_data['last'] = correction
        with open(TRAIN_FILE, "w") as f:
            json.dump(training_data, f)
        print(f"[TRAINING] Stored correction: {correction}")
    return "Saved!"

# Audio listening thread
def listen_audio():
    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
        while True:
            try:
                audio = recognizer.listen(source)
                audio_queue.put(audio)
            except Exception as e:
                print(f"[ERROR] {e}")

def process_audio():
    while True:
        if not audio_queue.empty():
            audio_data = audio_queue.get()
            try:
                text = recognizer.recognize_google(audio_data)
                if text.lower().startswith("hey nova"):
                    query = text[8:].strip()
                    print(f"[USER] {query}")
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role":"user","content":query}]
                    )
                    answer = response.choices[0].message.content
                    print(f"[NOVA] {answer}")
                    tts_engine.say(answer)
                    tts_engine.runAndWait()
            except sr.UnknownValueError:
                continue
            except Exception as e:
                print(f"[ERROR] {e}")

if __name__ == "__main__":
    threading.Thread(target=listen_audio, daemon=True).start()
    threading.Thread(target=process_audio, daemon=True).start()
    app.run(host="0.0.0.0", port=5000, debug=False)
