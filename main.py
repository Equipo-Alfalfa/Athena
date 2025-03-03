 # Idontknowwhatimdoing.py
from transformers import BertTokenizer
import pandas as pd
import tensorflow as tf 
from tensorflo.keras import layers, models
# EL codigo esta modularizado en distintas funciones que me gustaria
# exportar a un modulo para tener un proyecto escalable, mantenible y legible

def load_data(file_path):
    return pd.read_csv(file_path)

def categorize(text):
    if "malware" in text.lower():
        return "Malware"
    elif "ransomware" in text.lower():
        return "Ransomware"
    elif "phishing" in text.lower():
        return "Phishing"
    else:
        return "Other"

def labeler(df):
    df["label"] = df["text"].apply(categorize)
    return df

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

def create_model(inputs, num):
    model = models.Sequential()
    model.add(layers.Conv1D(128, 5, activation='relu', input_shape=input))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Conv1D(128, 5, activation='relu', input_shape=input))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation = 'relu'))
    model.add(layers.Dense(num,activation = 'softmax'))
    return model

def main():
    # LOAD
    df = load_data('datasets/CyberBERT.csv')
    #LABEL
    df = labeler(df)
    # SPLIT
    x_train, x_test, y_train, y_test = split_data(df)
    # TOKENIZE
    train_texts = x_train.tolist()  
    val_texts = x_test.tolist()  
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_encodings = token_data(train_texts, tokenizer)
    val_encodings = token_data(val_texts,tokenizer)
    # Actually training the model
    y_train = pd.get_dummies(y_train).values
    y_test = pd.get_dummies(y_test).values
    inputs = (train_encodings['input_ids'].shape[1], 1) # no estoy seguro si esto sirve
    num = y_train.shape[1]
    model = create_model(inputs, num)

    model.compile(optimizer='adam', loss='categorical_crossentroppy', metrics=['accuracy'])
    model.fit(train_encodings['input_ids'],y_train,epochs=5, batch_size=32, validation_data=(val_encodings['input_ids'],y_test))

if __name__ == "__main__":
    main()