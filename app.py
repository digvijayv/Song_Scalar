
#importing libraries

import streamlit as st
import librosa
import soundfile as sf
import numpy as np


st.title("🎶 Pitch & Scale Adjuster")

# Upload audio file
uploaded_file = st.file_uploader("file1.mp3", type=["mp3", "wav"])

if uploaded_file is not None:
    # Load audio
    y, sr = librosa.load(uploaded_file, sr=None)

    # --- Detect Key ---
    try:
        key = librosa.key(y, sr=sr)
        st.write(f"**Detected Key/Scale:** {key}")
    except:
        st.write("Upgrade to Librosa v0.10+ for key detection")

    # --- Pitch Shift Control ---
    semitone_shift = st.slider("Shift Pitch (Semitones)", -12, 12, 0)

    if semitone_shift != 0:
        y_shifted = librosa.effects.pitch_shift(y, sr, n_steps=semitone_shift)
        sf.write("shifted_song.wav", y_shifted, sr)
        st.audio("shifted_song.wav")
        st.success(f"Song shifted by {semitone_shift} semitones!")
    else:
        st.audio(uploaded_file)
        st.info("Playing original song")
