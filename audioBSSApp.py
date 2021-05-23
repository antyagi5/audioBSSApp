import streamlit as st
import io
import numpy as np
from sklearn.decomposition import FastICA
import matplotlib.pyplot as plt
from scipy.io.wavfile import read, write


st.title("Audio File Source Separation")

file_wav = st.file_uploader("Upload a .wav file", type="wav")

if file_wav:
    samplerate, data = read(file_wav)
    audio_bytes = file_wav.read()
    'Audio sample rate = ' + str(samplerate)
    st.audio(audio_bytes, format="audio/wav")
    ica = FastICA(n_components=2)
    S_ = ica.fit_transform(data)  # Reconstruct signals


    amp = np.max(np.abs(data))
    S_ = S_/np.max(np.abs(S_))
    S_amped = amp*S_

    Ch1 = amp*S_[:,0]
    Ch2 = amp*S_[:,1]
    write("Ch1.wav", samplerate, Ch1.astype(np.int16))
    write("Ch2.wav", samplerate, Ch2.astype(np.int16))

    audio_file1 = open("Ch1.wav",'rb')
    audio_bytes1 = audio_file1.read()
    audio_file2 = open("Ch2.wav",'rb')
    audio_bytes2 = audio_file2.read()

    'Separated 1'
    st.audio(audio_bytes1, format="audio/wav")
    'Separated 2'
    st.audio(audio_bytes2, format="audio/wav")
