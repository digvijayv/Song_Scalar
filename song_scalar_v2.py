import streamlit as st
import librosa
import librosa.display
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import tempfile

st.title("🎶 Pitch & Scale Adjuster with Visualization + Lyrics")

# Upload audio file OR record from mic
uploaded_file = st.file_uploader("Upload a song (MP3/WAV)", type=["mp3", "wav"])
recorded_file = st.audio_input("Record your voice")

audio_source = uploaded_file or recorded_file

if audio_source is not None:
    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
        tmpfile.write(audio_source.read())
        file_path = tmpfile.name

    # Load audio
    y, sr = librosa.load(file_path, sr=None)

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
        sf.write("shifted_output.wav", y_shifted, sr)
        st.audio("shifted_output.wav")
        st.success(f"Audio shifted by {semitone_shift} semitones!")
    else:
        st.audio(file_path)
        st.info("Playing original audio")

    # --- Visualization Section ---
    st.subheader("🎼 Pitch & Scale Visualization")

    # Spectrogram
    st.write("**Spectrogram**")
    fig, ax = plt.subplots()
    S = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    img = librosa.display.specshow(S, sr=sr, x_axis='time', y_axis='log', ax=ax)
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    st.pyplot(fig)

    # Chroma Plot
    st.write("**Chroma Features (Pitch Classes)**")
    fig, ax = plt.subplots()
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', ax=ax)
    st.pyplot(fig)

# --- Lyrics Section ---
st.subheader("📖 Song Lyrics")
lyrics_input = st.text_area("Paste song lyrics here:")
if lyrics_input:
    st.write("### Lyrics Preview")
    st.write(lyrics_input)