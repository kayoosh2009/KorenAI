import os
import numpy as np
import sounddevice as sd
from pydub import AudioSegment
from TTS.api import TTS

VOICE_FILE = "voice/main.ogg"
VOICE_WAV = "voice/main.wav"

print("[Voice] Initializing XTTS v2 model...")
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
print("[Voice] Model loaded successfully")

def convert_ogg_to_wav():
    if not os.path.exists(VOICE_WAV):
        print(f"[Voice] Converting {VOICE_FILE} to WAV...")
        audio = AudioSegment.from_ogg(VOICE_FILE)
        audio.export(VOICE_WAV, format="wav")
        print("[Voice] Conversion complete")

def speak(text: str, speaker_wav: str = VOICE_WAV) -> np.ndarray:
    convert_ogg_to_wav()
    
    print(f"[Voice] Generating speech for: {text[:50]}...")
    wav_path = "temp_output.wav"
    
    tts.tts_to_file(
        text=text,
        speaker_wav=speaker_wav,
        language="en",
        file_path=wav_path
    )
    
    audio = AudioSegment.from_wav(wav_path)
    samples = np.array(audio.get_array_of_samples())

    if samples.dtype == np.int16:
        samples = samples.astype(np.float32) / 32768.0

    print("[Voice] Playing audio...")
    sd.play(samples, samplerate=audio.frame_rate)
    sd.wait()

    if os.path.exists(wav_path):
        os.remove(wav_path)
    
    return samples

def get_audio_levels(samples: np.ndarray, chunk_size: int = 1024) -> list:

    levels = []

    for i in range(0, len(samples), chunk_size):
        chunk = samples[i:i + chunk_size]
        rms = np.sqrt(np.mean(chunk ** 2))
        level = min(1.0, rms * 10) 
        levels.append(level)
    
    return levels

def speak_with_levels(text: str, speaker_wav: str = VOICE_WAV) -> tuple:
    convert_ogg_to_wav()

    wav_path = "temp_output.wav"
    
    tts.tts_to_file(
        text=text,
        speaker_wav=speaker_wav,
        language="en",
        file_path=wav_path
    )
    
    audio = AudioSegment.from_wav(wav_path)
    samples = np.array(audio.get_array_of_samples())
    
    if samples.dtype == np.int16:
        samples = samples.astype(np.float32) / 32768.0
    
    levels = get_audio_levels(samples)

    if os.path.exists(wav_path):
        os.remove(wav_path)
    
    return samples, levels, audio.frame_rate

if __name__ == "__main__":
    test_text = "Hello! I'm Koren, your AI assistant. How can I help you today?"
    print(f"Testing voice with: {test_text}")
    speak(test_text)
    print("Test complete!")