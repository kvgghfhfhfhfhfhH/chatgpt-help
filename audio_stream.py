# audio_stream.py
import sounddevice as sd
import numpy as np

class AudioStream:
    def __init__(self, input_device=1, output_device=2, samplerate=16000, duration=1.5):
        self.input_device = input_device
        self.output_device = output_device
        self.samplerate = samplerate
        self.duration = duration

    def record_short_clip(self):
        """Record a short audio clip and skip silence."""
        try:
            print("[DEBUG] Listening...")
            audio = sd.rec(
                int(self.duration * self.samplerate),
                samplerate=self.samplerate,
                channels=1,
                dtype="float32",
                device=self.input_device
            )
            sd.wait()

            volume = np.max(np.abs(audio))
            if volume < 0.03:  # silence threshold
                print("[DEBUG] Silence or unclear speech detected.")
                return None

            # Optional quick speech clarity check
            energy = np.sum(np.abs(audio))
            if energy < 0.2:
                print("[DEBUG] Mumbled or unclear speech — ignored.")
                return None

            return audio

        except Exception as e:
            print(f"[ERROR] Audio recording failed: {e}")
            return None

    def play_audio(self, data):
        """Play audio through USB speakers if available."""
        try:
            sd.play(data, samplerate=self.samplerate, device=self.output_device)
            sd.wait()
        except Exception as e:
            print(f"[ERROR] Playback failed: {e}")
