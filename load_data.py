import os
import scipy.io.wavfile as wav
import pandas as pd
from python_speech_features import mfcc

def load_audio_data(path):
    emotion_data = pd.DataFrame()

    for root, dirs, files in os.walk(path):
        for clip_file in files:
            (rate,sig) = wav.read(os.path.join(root, clip_file))
            row = pd.DataFrame([[mfcc(sig,rate,nfft=1024), os.path.basename(root)]])
            emotion_data = emotion_data.append(row, ignore_index=True)

    emotion_data.columns = ['Features', "Class"]    
    return emotion_data

def load_one_file(path):
    return wav.read(path)