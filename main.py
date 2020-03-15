import sounddevice as sd
from scipy.io.wavfile import write
from learn_features import cat_file
import pickle
import numpy as np
import threading

(scaler, model) = pickle.load(open('trained_model', 'rb'))

fs = 24414  # Sample rate
frame_time = 1  # Duration of recording
duration = 10
sd.default.samplerate = fs
sd.default.channels = 1
sd.default.dtype = 'int16'

chunks = []
buffer = np.array([])

def process_chunk(chunk):
    global chunks
    pred = cat_file(chunk, fs, scaler, model)
    amp = np.mean(np.absolute(chunk))
    chunks.append((chunk.astype('int16'), amp, pred))
    

def listen_chunk(indata, frames, time, status):
    global buffer
    indata = indata.flatten()
    chunk_size = frame_time * fs
    if len(buffer) + frames <= chunk_size:
        buffer = np.append(buffer,indata)
    else:
        print('CHUNK')
        delta = chunk_size - len(buffer)
        buffer = np.append(buffer,indata[:delta])
        p = threading.Thread(target=process_chunk, args=[buffer[:]])
        p.start()
        buffer = np.array([])
        buffer = np.append(buffer, indata[delta:])
    

print('Listening...')
#myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
#sd.wait()  # Wait until recording is finished

with sd.InputStream(callback=listen_chunk):
    sd.sleep(duration * 1000)

# print('You can shut up now.')
# print(cat_file(myrecording,fs,scaler,model))
# #write('output.wav', fs, myrecording)  # Save as WAV file 

# for chunk,_,_ in chunks:
#     sd.play(chunk, fs)
#     sd.wait()
#     sd.sleep(1000)

# %%

def get_me(emotions, chunks, continuous=True):
    wav = []
    if continuous:
        data = np.array([])
        for chunk,_,emo in chunks:
            if emo in emotions:
                data = np.append(data,chunk)
            elif len(data) > 0:
                wav.append(data.astype('int16'))
                data = np.array([])
        if len(data) > 0:
            wav.append(data.astype('int16'))
    else:
        wav = np.array([chunk[0] for chunk in chunks if (chunk[2] in emotions)]).flatten()
    return wav
                

print([chunk[1] for chunk in chunks])
print([chunk[2] for chunk in chunks])