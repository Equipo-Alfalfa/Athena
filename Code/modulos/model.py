import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    stop_words = set(stopwords.words('english'))  # English stopwords
    words = [word for word in text.split() if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(words)

def preprocess_data(df, text_column='text'):
    df[text_column] = df[text_column].apply(preprocess_text)
    return df

def binarize(df,label_column = 'label'):
    mlb = MultiLabelBinarizer()
    labels = mlb.fit_transform(df[label_column])
    labels_df = pd.DataFrame(labels, columns=mlb.classes_)
    return labels_df, mlb

# RECORRER TEXTO PARA CREAR CATEGORIAS 
def categorize(text):
    # DICCIONARIO DE CATEGORIAS
    categories = {
        "malware": "Malware",
        "phishing": "Phishing",
        "ransomware": "Ransomware",
        "trojan": "Trojan",
        "worm": "Worm",
        "spyware": "Spyware",
        "ddos": "DDoS",
        "distributed denial of service": "DDoS",
        "zero day": "Zero Days",
        "data breach": "Data Breach",
        "social engineering": "Social Engineering",
        "payments": "Payments",
        "breach": "Data Breach",
        "suspicius": "Suspicious Software",
        "password": "Password",
        "scam": "Scam",
    }
    lower_text = text.lower()
    categ = []
    for keyword, category in categories.items():
        if keyword in lower_text:
            categ.append(category)
    if not categ:
        categ.append("Other")
    return categ


# PARAMETROS DE TOKENIZACION Y ETIQUETADO
def prepare_data(df, tokenizer, max_length=512):
    df["label"] = df["text"].apply(categorize)
    mlb = MultiLabelBinarizer()
    labels = mlb.fit_transform(df["label"])
    encodings = tokenizer(
        df["text"].tolist(),
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors='tf'
    )
    return encodings, labels, mlb

def tokenize_data(tokenizer, texts, max_length=128):
    encodings = tokenizer(
        texts.tolist(), padding=True, truncation=True, max_length=max_length, return_tensors='tf'
    )
    return encodings

def create_tf_datasets(train_encodings, y_train, val_encodings, y_test, batch_size=16):
    train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_encodings), y_train)).shuffle(1000).batch(batch_size)
    val_dataset = tf.data.Dataset.from_tensor_slices((dict(val_encodings), y_test)).batch(batch_size)
    return train_dataset, val_dataset

# CARGAR MODELO PRE ENTRENADO Y TOKENIZADOR
def create_bert(model_name= 'bert-base-uncased'):
    model = TFBertModel.from_pretrained(model_name)
    tokenizer = BertTokenizer.from_pretrained(model_name)
    return model, tokenizer

def split_data(df, labels_df, test_size=0.2, random_state=42):
    x_train, x_test, y_train, y_test = train_test_split(
        df['text'], labels_df, test_size=test_size, random_state=random_state
    )
    return x_train, x_test, y_train, y_test

def tune_bert(train_dataset, val_dataset, model_name='bert-base-uncased', num_labels=None, epochs=5):
    model = TFBertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    model.fit(train_dataset, validation_data=val_dataset, epochs=epochs, callbacks=[early_stopping])
    return model

# CREAR RED CONVOLUCIONADA
def create_cnn(bert_model, num_labels):
    input_ids = tf.keras.Input(shape=(None,), dtype=tf.int32, name='input_ids')
    attention_mask = tf.keras.Input(shape=(None,), dtype=tf.int32, name='attention_mask')
    bert_output = bert_model(input_ids, attention_mask=attention_mask)
    x = bert_output.last_hidden_state

    x = tf.keras.layers.Conv1D(filters=128,kernel_size=3, activation = 'relu')(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    x = tf.keras.layers.Conv1D(filters = 64, kernel_size=3, activation = 'relu')(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)

    x = tf.keras.layers.Dense(64, activation='relu')(x)
    outputs = tf.keras.layers.Dense(num_labels, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=outputs)
    return model

def train_mixed_model(model,train_dataset,val_dataset, epochs= 10, patience = 3):
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss',patience=patience)
    history = model.fit(train_dataset,validation_data = val_dataset, epochs=epochs, callbacks=[early_stopping])
    return history

#example of main.py
def main():
    df = pd.read_csv('datasets/labelled_data.csv')
    df = preprocess_data(df)
    labels_df, mlb = binarize(df)
    x_train, x_test, y_train, y_test = split_data(df, labels_df)
    bert_model, tokenizer = create_bert()
    train_encodings = tokenize_data(tokenizer, x_train)
    val_encodings = tokenize_data(tokenizer, x_test)

    train_dataset, val_dataset = create_tf_datasets(train_encodings, y_train, val_encodings, y_test)
    tuned_bert = tune_bert(train_dataset, val_dataset,num_labels=labels_df.shape[1])
    cnn_model = create_cnn(bert_model, num_labels=labels_df.shape[1])
    history = train_combined_model(cnn_model, train_dataset, val_dataset)
    cnn_model.save('bert_cnn_cybersecurity_model')
    print("mierda entrenada")
main()
