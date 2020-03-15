from load_data import load_one_file
from learn_features import try_to_train, cat_file
import numpy as np
import pickle
import os
import scipy.io.wavfile as wav

(scaler, model) = pickle.load(open('trained_model', 'rb'))

path = 'Data/'

# for root, dirs, files in os.walk(path):
#     for clip_file in files:
#         (rate,sig) = wav.read(os.path.join(root, clip_file))
#         pred = cat_file(sig, rate, scaler, model)
#         print(os.path.basename(root) + ' = ' + pred)

sig_lens = []

r = 24414

for root, dirs, files in os.walk(path):
    for clip_file in files:
        (rate,sig) = wav.read(os.path.join(root, clip_file))
        sig_lens.append([sig])

print(sig_lens)

# (rate, sig) = load_one_file('Data/anger/OAF_bean_angry.wav')
# print(sig)
