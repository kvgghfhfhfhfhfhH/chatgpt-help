# audio_stream.py
import sounddevice as sd
import numpy as np

class AudioStream:
    def __init__(self, device=None, duration=1, threshold=0.02):
        self.device = device
        self.duration = duration
        self.threshold = threshold

    def detect_sound(self):
        data = sd.rec(int(self.duration * 16000), samplerate=16000, channels=1, device=self.device)
        sd.wait()
        volume = np.abs(data).mean()
        if volume > self.threshold:
            print("[AUDIO] Sound detected!")
        else:
            print("[AUDIO] Silence")

    def close(self):
        pass
