from load_data import load_audio_data, load_one_file
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras.utils as tfutils 
from sklearn.model_selection import train_test_split
from python_speech_features import mfcc
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import accuracy_score
import pickle

class_map = { 'anger': 0, 'disgust': 1, 'fear': 2,
              'happy': 3, 'neutral': 4, 'sad': 5,
              'surprise': 6 }

map_class = { v: k for k, v in class_map.items() }
# %%

def average_features(data_column):
    return [np.mean(frame, axis=0) for frame in data_column]

def map_classes(data_column,reverse=False):
    local_map = class_map
    if reverse:
        local_map = {v: k for k, v in local_map.items()}
    return [local_map[c] for c in data_column]

# def cat_sample(sig, rate, scaler, model):
#     features = mfcc(sig, rate, nfft=1024)
#     squished = np.mean(features, axis = 0)
#     x_data = scaler.transform(squished.reshape(1, -1))
#     predictions = model.predict(x_data)
#     return predictions

# %%
# def get_hot()    

def try_to_train():
    emotion_data = load_audio_data('Data/')
    
    emotion_data['Features'] = average_features(emotion_data['Features'])
    emotion_data['Class'] = map_classes(emotion_data['Class'])
    
    one_hot = tfutils.to_categorical(emotion_data['Class'])
    one_hot_df = pd.DataFrame(one_hot, columns=class_map.keys())
    emotion_data = emotion_data.drop('Class', 1).join(one_hot_df)
    
    x_data = np.vstack(emotion_data['Features'])
    y_data = emotion_data.drop('Features', 1)

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data)
    
    scaler = MinMaxScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    
    knn_model = KNeighborsClassifier(n_neighbors=7, algorithm='brute').fit(np.vstack(x_train), y_train) 
    knn_predictions = knn_model.predict(x_test)
    
    pred_df = pd.DataFrame(knn_predictions)
    pred_df.columns = map_classes(pred_df.columns, reverse=True)
    
    # Calculate accuracy
    acc = accuracy_score(y_test, knn_predictions)
    return (scaler, knn_model, acc)

# %%
    
# tensor_data = load_audio_data('Data/')
# tf.keras.preprocessing.sequence.pad_sequences(tensor_data['Features'].tolist(), padding='post', maxlen=200).shape
# feats = []
# for ident,row in tensor_data['Features'].iteritems():
#     feats.append(row[0])
#     padded = tf.keras.preprocessing.sequence.pad_sequences(row)
#     print(str(len(row)) + '->' + str(len(padded)))

# %%

# n = 100
# s = 0 
# for x in range(1, n+1):
#     print(str(x) + '/' + str(n))
#     s += try_to_train()[2]

# avg_acc = s / n

# %%

def save_model(path):
    (scaler, model, _) = try_to_train()
    pickle.dump((scaler, model), open(path, 'wb'))

# (rate, sig) = load_one_file('mad_brooks.wav')

def cat_file(sig, rate, scaler, model):
    features = mfcc(sig, rate, nfft=1024)
    squished = np.mean(features, axis = 0)
    one_x_data = scaler.transform(squished.reshape(1, -1))
    hot = model.predict(one_x_data)[0]
    pred = np.where(hot == 1.0)[0]
    return (map_class[pred[0]] if len(pred) > 0  else 'unknown')

    # pred_df = pd.DataFrame(model.predict(one_x_data))
    # pred_df.columns = map_classes(pred_df.columns, reverse=True)
    # return pred_df

#print(np.mean([x for x in [cat_file(sig,rate,scaler,model) for _ in range(50)] if np.mean(x) > 0], axis=0))

#result = cat_sample(sig, rate, scaler, model)

### Test a bunch of classifiers
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier

# names = ['K Nearest Neighbor', 'Descition Tree', 'Random Forest']
# classifiers = [KNeighborsClassifier(), DecisionTreeClassifier(), RandomForestClassifier()]


# for name, classifier in zip(names, classifiers):
#     classifier.fit(x_train, y_train)
#     pred = classifier.predict(x_test)
#     print(f'Accuracy for {name} is {accuracy_score(y_test, pred):.3f}')