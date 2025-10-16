import cv2
from flask import Flask, Response, render_template_string
import threading
import os
import sounddevice as sd
import numpy as np
import io
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = Flask(__name__)

# Simple HTML page
PAGE = """
<!DOCTYPE html>
<html>
  <head>
    <title>Nova Interface</title>
    <style>
      body { background: #111; color: #eee; text-align: center; font-family: Arial; }
      video { border-radius: 8px; border: 2px solid #00ff99; }
      #log { width: 80%; margin: auto; text-align: left; background: #222; padding: 10px; border-radius: 8px; }
    </style>
  </head>
  <body>
    <h1>👁️ Nova Interface</h1>
    <video id="camera" autoplay playsinline width="640" height="480"></video>
    <div id="log"></div>
    <script>
      const video = document.getElementById('camera');
      video.src = "/video_feed";
    </script>
  </body>
</html>
"""

# Flask routes
@app.route('/')
def index():
    return render_template_string(PAGE)

def gen_frames():
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def audio_listener():
    print("[INFO] Nova is listening for voice input…")
    while True:
        duration = 5  # seconds
        sr = 16000
        audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
        sd.wait()
        audio_bytes = io.BytesIO(np.int16(audio * 32767).tobytes())
        try:
            transcript = client.audio.transcriptions.create(
                model="gpt-4o-mini-transcribe",
                file=("input.wav", audio_bytes, "audio/wav")
            )
            text = transcript.text.strip().lower()
            if text:
                print(f"[USER]: {text}")
                response = client.responses.create(
                    model="gpt-4o-mini",
                    input=f"The user said: '{text}'. Respond naturally as Nova."
                )
                print(f"[NOVA]: {response.output_text}")
        except Exception as e:
            print("[ERROR]", e)

# Run both web server and listener
if __name__ == '__main__':
    threading.Thread(target=audio_listener, daemon=True).start()
    app.run(host='0.0.0.0', port=8080, debug=False)
