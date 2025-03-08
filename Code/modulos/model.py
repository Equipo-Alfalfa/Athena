from tensorflow.keras import layers, models
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score, classification_report

def split_data(df):
    from sklearn.model_selection import train_test_split
    x = df['text'] 
    y = df['label'] 
    return train_test_split(x,y,test_size = 0.2,random_state=42)

def token_data(texts, tokenizer, max_length=512):
    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors='tf' 
    )
    return encodings

def performance(y_true,y_pred):
# y_pred son los valores predecidos mientras que y_true son los valores reales
    accuracy = accuracy_score(y_true,y_pred)
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    print(classification_report(y_true, y_pred))
    return{
        'accuracy':accuracy,
        'recall':recall,
        'f1':f1,
    }


def create_model(inputs, num):
    model = models.Sequential()
    model.add(layers.Conv1D(128, 5, activation='relu', input_shape=inputs))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Conv1D(128, 5, activation='relu', input_shape=inputs))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation = 'relu'))
    model.add(layers.Dense(num,activation = 'softmax'))
    return model
