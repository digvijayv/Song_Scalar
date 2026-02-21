import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

# Load audio file
y, sr = librosa.load("file.mp3")

# --- Step 1: Detect pitches and chroma features ---
chroma = librosa.feature.chroma_stft(y=y, sr=sr)
chroma_mean = chroma.mean(axis=1)

# Map chroma to note names
note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 
              'F#', 'G', 'G#', 'A', 'A#', 'B']

# Find the most dominant pitch class
dominant_note = note_names[np.argmax(chroma_mean)]
print(f"Detected dominant note: {dominant_note}")

# --- Step 2: Estimate key (major/minor) ---
# Librosa has a built-in key estimation in v0.10+
try:
    key = librosa.key(y, sr=sr)
    print(f"Estimated key: {key}")
except:
    print("Key estimation requires Librosa v0.10+; upgrade if needed.")

# --- Step 3: Shift scale (transpose) ---
# Example: shift up by 2 semitones
y_shifted = librosa.effects.pitch_shift(y, sr, n_steps=2)

# Save shifted audio
librosa.output.write_wav("song_shifted.wav", y_shifted, sr)

print("Shifted audio saved as song_shifted.wav")