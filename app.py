import streamlit as st
import yt_dlp
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os

DOWNLOAD_DIR = "downloads"
if not os.path.exists(DOWNLOAD_DIR):
    os.makedirs(DOWNLOAD_DIR)

def download_audio(url):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(DOWNLOAD_DIR, 'audio_track.%(ext)s'),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'quiet': True,
        'noplaylist': True,
        'extractor_args': {'youtube': {'player_client': ['default,-android_sdkless']}},
        'nocheckcertificate': True
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
        
    return os.path.join(DOWNLOAD_DIR, 'audio_track.wav')

def analyze_audio(file_path):
    y, sr = librosa.load(file_path, sr=None)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    S_dB = librosa.power_to_db(S, ref=np.max)
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000, ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title='Mel-frequency Spectrogram')
    return tempo[0] if isinstance(tempo, np.ndarray) else tempo, fig

st.set_page_config(page_title="Music Analyzer", layout="centered")
st.title("🎵 Music Information Retrieval Tool")
st.write("Enter a URL to download the audio and analyze its composition.")

url_input = st.text_input("Enter Audio/Video URL (e.g., YouTube, SoundCloud):")

if st.button("Analyze Track"):
    if url_input:
        with st.spinner("Downloading audio..."):
            try:
                audio_path = download_audio(url_input)
                st.success("Audio downloaded successfully!")
                st.audio(audio_path, format="audio/wav")
                
                with st.spinner("Analyzing audio data..."):
                    bpm, spectrogram_fig = analyze_audio(audio_path)
                    st.subheader("Analysis Results")
                    st.metric(label="Estimated Tempo", value=f"{round(bpm)} BPM")
                    st.write("**Spectrogram Visual**")
                    st.pyplot(spectrogram_fig)
                    
                    if os.path.exists(audio_path):
                        os.remove(audio_path)
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a valid URL.")
