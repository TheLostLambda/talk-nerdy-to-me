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

def map_classes(data_column):
    return [class_map[c] for c in data_column]

emotion_data['Features'] = average_features(emotion_data['Features'])
emotion_data['Class'] = map_classes(emotion_data['Class'])

one_hot = tf.keras.utils.to_categorical(emotion_data['Class'])
one_hot_df = pd.DataFrame(one_hot, columns=class_map.keys())
emotion_data = emotion_data.drop('Class', 1).join(one_hot_df)

x_data = np.vstack(emotion_data['Features'])
y_data = emotion_data.drop('Features', 1)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data)

from sklearn.tree import DecisionTreeClassifier 
dtree_model = DecisionTreeClassifier(max_depth = 10).fit(np.vstack(x_train), y_train) 
dtree_predictions = dtree_model.predict(x_test) 
  
# creating a confusion matrix 
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, dtree_predictions)