import pygame
import os
import sounddevice as sd
import scipy.io.wavfile as wav
import numpy as np
import threading
from learn_features import cat_file
import pickle
import os
import random
from scipy.io.wavfile import write
import csv

(scaler, model) = pickle.load(open('trained_model', 'rb'))

fs = 24414  # Sample rate
frame_time = 2  # Duration of recording
duration = 35
sd.default.samplerate = fs
sd.default.channels = 1
sd.default.dtype = 'int16'
black = (0, 0, 0)
notTalkingWait = 5
notTalkingCount = 0

strEmotion = 'Emotions'

#%%
soundbank = []
for root, dirs, files in os.walk('clips/'):
    for f in files:
        soundbank.append(wav.read(os.path.join(root, f)))
# %%
chunks = []
buffer = np.array([])
_image_library = {}
def get_image(path):
        global _image_library
        image = _image_library.get(path)
        if image == None:
                canonicalized_path = path.replace('/', os.sep).replace('', os.sep)
                image = pygame.image.load(path)
                image = pygame.transform.scale(image, (900, 1600))
                _image_library[path] = image
        return image

def process_chunk(chunk):
    global chunks
    next_emo = cat_file(chunk, fs, scaler, model)
    pred = strEmotion if next_emo == 'unknown' else next_emo
    amp = np.mean(np.absolute(chunk))
    chunks.append((chunk.astype('int16'), amp, pred))

def listen_chunk(indata, frames, time, status):
    global buffer
    global volume
    indata = indata.flatten()
    volume = np.mean(np.absolute(indata))/700
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

def differential_chunks(chunks):
    wav = []
    for i in range(0,len(chunks)-1):
        if chunks[i][2] != chunks[i+1][2]:
            pair = np.array([chunks[i][0], chunks[i+1][0]]).flatten()
            wav.append(pair)
    return wav

pygame.init()
screen = pygame.display.set_mode(( 900 , 1600))
pygame.display.set_caption('Talk Nerdy To Me') 
done = False
clock = pygame.time.Clock()

DeltaTime = 0

count = 1
pathBeg = 'Background/Untitled-Artwork-'
fileExt = '.png'
initial = True
font = pygame.font.Font('freesansbold.ttf', 42) 
text = font.render(strEmotion, True, black)
X = 170
Y = 660
textRect = text.get_rect()  
numChunksPrevious = 0
# set the center of the rectangular object. 
textRect.center = (X , Y) 
volume = 0
duckImg = pygame.image.load('duck.png')


    
with sd.InputStream(callback=listen_chunk):
        
        while not done:
                for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                                done = True
                
               

                DeltaTime = DeltaTime + clock.get_time()
                
                filePathStr = pathBeg + str(count) + fileExt
                screen.blit(get_image(filePathStr), (0, 0))
                if (initial == True):
                        count = count + 1
                        initial = False
                                
                
                if(DeltaTime>((duration/41)*1000)):
                        count = count + 1
                        DeltaTime = 0
                
                if (count > 41):
                        count = 1
                        DeltaTime = 0
                        done = True
                
                

                if (len(chunks)>0):
                        numChunks = len(chunks)
                        if (numChunks > numChunksPrevious):
                                numChunksPrevious = numChunks
                                strEmotion = chunks[-1][2]
                                
                                if (chunks[-1][1]<400):
                                        notTalkingCount = notTalkingCount + 1
                                        
                                else:
                                        notTalkingCount = 0
                                print(notTalkingCount)
                if(notTalkingCount>=notTalkingWait):
                        (rate,sig) = random.choice(soundbank)
                        sd.play(sig,rate)
                        notTalkingCount = 0
                        
                width = max(30 ,int(volume*20))
                height = max(30, int(volume*20))
                screen.blit(pygame.transform.scale(duckImg, (width, height)), (330 - (width//2),1100-(height//2)))
                text = font.render(strEmotion, True, black)
                screen.blit(text, textRect) 

                pygame.display.flip()
                clock.tick(60)


fullConvo = get_me(['anger','disgust','fear','happy','neutral','sad','surprise'],chunks,continuous = False)
write('output/transcript.wav', fs, fullConvo)

with open('output/emotion_volume_results.csv', mode='w') as results_file:
        results_writer = csv.writer(results_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        results_writer.writerow(['Emotion', 'Volume'])
        for _,vol,emo in chunks:
               results_writer.writerow([emo, int(vol)])

# Play back the changes
for transition in differential_chunks(chunks):
    sd.play(transition)
    sd.wait()