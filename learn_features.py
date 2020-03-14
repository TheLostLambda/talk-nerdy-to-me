from load_data import load_audio_data
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras.utils as tfutils 
from sklearn.model_selection import train_test_split

emotion_data = load_audio_data('Data/')

class_map = { 'anger': 0, 'disgust': 1, 'fear': 2,
              'happy': 3, 'neutral': 4, 'sad': 5,
              'surprise': 6 }

def average_features(data_column):
    return [np.mean(frame, axis=0) for frame in data_column]

def map_classes(data_column,reverse=False):
    local_map = class_map
    if reverse:
        local_map = {v: k for k, v in local_map.items()}
    return [local_map[c] for c in data_column]

emotion_data['Features'] = average_features(emotion_data['Features'])
emotion_data['Class'] = map_classes(emotion_data['Class'])

one_hot = tfutils.to_categorical(emotion_data['Class'])
one_hot_df = pd.DataFrame(one_hot, columns=class_map.keys())
emotion_data = emotion_data.drop('Class', 1).join(one_hot_df)

x_data = np.vstack(emotion_data['Features'])
y_data = emotion_data.drop('Features', 1)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler().fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

from sklearn.neighbors import KNeighborsClassifier 
knn_model = KNeighborsClassifier().fit(np.vstack(x_train), y_train) 
knn_predictions = knn_model.predict(x_test) 

pred_df = pd.DataFrame(knn_predictions)
pred_df.columns = map_classes(pred_df.columns, reverse=True)

# Calculate accuracy
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, knn_predictions)

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