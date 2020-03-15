import pygame
import os
import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np


fs = 24414  # Sample rate
recordingLength = 5  # Duration of recording
CHUNK = 4096
chunkCount = 1
analyzeTime = 1000

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

def start_recording():
        myrecording = sd.rec(int(recordingLength * fs), samplerate=fs, channels=2)
        print('Recording')
        return myrecording

def isTalking(myrecording,chunkCount):

        chunkStart = (chunkCount-1)*fs
        chunkEnd = (chunkCount)*fs
        print(str(chunkStart) + " " + str(chunkEnd))
        data = myrecording[chunkStart:chunkEnd,0]
        data = data[np.isfinite(data)]
        
        data = np.absolute(data)
        print(data)
        meanData=np.mean(data)
        print(meanData)
        if (meanData < 0.001):
                print("not Talking")
                return False
        else:
                print("Is Talking")
                return True

pygame.init()
screen = pygame.display.set_mode(( 900 , 1600))
done = False
clock = pygame.time.Clock()
isTalkingBool = False
DeltaTime = 0
DeltaTimeAnalyze = 0
count = 1
pathBeg = 'Background/Untitled-Artwork-'
fileExt = '.png'
initial = True

while not done:
        for event in pygame.event.get():
                if event.type == pygame.QUIT:
                        done = True
        
        DeltaTime = DeltaTime + clock.get_time()
        DeltaTimeAnalyze = DeltaTimeAnalyze + clock.get_time()
        filePathStr = pathBeg + str(count) + fileExt
        
        if (initial == True):
            screen.blit(get_image(filePathStr), (0, 0))
            count = count + 1
            initial = False
            myrecording = start_recording()
            DeltaTimeAnalyze = 0
        if(DeltaTime>((recordingLength/41)*1000)):
            screen.blit(get_image(filePathStr), (0, 0))
            count = count + 1
            DeltaTime = 0
        
        if(DeltaTimeAnalyze>analyzeTime):
            isTalkingBool = isTalking(myrecording,chunkCount)
            chunkCount = chunkCount + 1
            DeltaTimeAnalyze = 0

        if (count > 41):
            count = 1
            DeltaTime = 0
            sd.wait()
            write('output.wav', fs, myrecording) 
            
            sd.play(myrecording, fs)
            sd.wait()
            done = True
        
        
        pygame.display.flip()
        clock.tick(60)