
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


def tune_bert():
    import pandas as pd
    from sklearn.preprocessing import MultiLabelBinarizer
    from sklearn.model_selection import train_test_split
    from transformers import BertTokenizer, TFBertForSequenceClassification
    import tensorflow as tf

    # Crear un dataset de ejemplo


    df = pd.read_csv('./datasets/clean_data.csv')

    # Convertir etiquetas a listas
    df['label'] = df['text'].apply(categorize)

    # Binarizar las etiquetas
    mlb = MultiLabelBinarizer()
    etiquetas_binarias = mlb.fit_transform(df['label'])

    # Dividir los datos en características (X) y etiquetas (y)
    X = df['text']
    y = etiquetas_binarias

    # Dividir los datos en entrenamiento y prueba
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Tokenizar el texto
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def tokenize_function(texts):
        return tokenizer(texts.tolist(), padding=True, truncation=True, max_length=128, return_tensors='tf')

    train_encodings = tokenize_function(x_train)
    val_encodings = tokenize_function(x_test)

    # Cargar el modelo BERT preentrenado
    model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=y.shape[1])

    # Crear un Dataset de TensorFlow
    train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_encodings), y_train)).shuffle(1000).batch(16)
    val_dataset = tf.data.Dataset.from_tensor_slices((dict(val_encodings), y_test)).batch(16)

    # Compilar el modelo con binary_crossentropy
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
                loss='binary_crossentropy',
                metrics=['accuracy'])

    # Entrenar el modelo
    model.fit(train_dataset, validation_data=val_dataset, epochs=3)

tune_bert()