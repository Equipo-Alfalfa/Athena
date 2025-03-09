import tensorflow as tf
from transformers import TFBertModel, BertTokenizer
import pandas as pd

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
    # APLICAR ETIQUETAS LLAMANDO A CATEGORIZE
    df["label"] = df["text"].apply(categorize) 
    # TOKENIZAR
    encodings = tokenizer(
        df,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors='tf' 
    )
    return encodings, df["label"].tolist()


# CARGAR MODELO PRE ENTRENADO Y TOKENIZADOR
def create_bert(model_name= 'bert-base-uncased'):
    model = TFBertModel.from_pretrained(model_name)
    tokenizer = BertTokenizer.from_pretrained(model_name)
    return model, tokenizer


def split_data(df, labels):
    from sklearn.model_selection import train_test_split
    import tensorflow as tf
    x = df['text'] 
    y = labels
    return train_test_split(x,y,test_size = 0.2,random_state=35)


def tune_bert(df, labels):
    from sklearn.model_selection import train_test_split
    from transformers import TFBertForSequenceClassification
    x_train, x_test, y_train, y_test = split_data(df, labels)
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def tokenize_function(text):
        return tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors='tf')
    
    train_encodings = tokenize_function(x_train)
    val_encodings = tokenize_function(x_test)

    model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')
    
    train_dataset = tf.data.Dataset.from_tensor_slices((dict(model(train_encodings)), y_train)).shuffle(100).batch(32)
    val_dataset = tf.data.Dataset.from_tensor_slices((dict(model(val_encodings)), y_test)).batch(16)
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    model.fit(train_dataset, validation_data=val_dataset, epochs=5)

    model.save_pretrained('tuned_bert_model')
    tokenizer.save_pretrained('tuned_bert_tokenizer')


# CREAR RED CONVOLUCIONADA
def create_cnn(bert_model, num_labels):
    input_ids = tf.keras.Input(shape=(None,), dtype=tf.int32, name='input_ids')
    attention_mask = tf.keras.Input(shape=(None,), dtype=tf.int32, name='attention_mask')
    bert_output = bert_model(input_ids, attention_mask=attention_mask)
    x = bert_output.last_hidden_state

    x = tf.keras.layers.Conv1D(filters=128,kernel_size=3, activation = 'relu')(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    x = tf.keras.layers.Conv1D(filers = 64, kernel_size=3, activation = 'relu')(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)

    x = tf.keras.layers.GlobalMaxPooling1D()(x)

    x = tf.keras.layers.Dense(64, activation='relu')(x)
    outputs = tf.keras.layers.Dense(num_labels, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=outputs)
    return model

    
#example of main
def main():
    # CARGAR DATA
    df = pd.read_csv('datasets/clean_data.csv')
    # CARGAR TOKENIZADOR Y MODELO PREENTRENADO
    model, tokenizer = create_bert(model_name='bert-base-uncased')
    # CANTIDAD DE ETIQUETAS QUE TIENE EL DF
    num_labels = 16 # this can change if you add or substract categories from categorize()
    # CREAR MODELO CONVOLUCIONADO
    cnn_model = create_cnn(model,num_labels)
    # EMPEZAR A TOKENIZAR UTILIZA TOKENIZADOR PRE ENTRENADO
    encodings,labels = prepare_data(df, tokenizer)
    # SE GUARDAN EN UN DATASET DE TENSORFLOW
    dataset = tf.data.Dataset.from_tensor_slices(({
        'input_ids': encodings['input_ids'],
        'attention_mask': encodings['attention_mask']    
    }, labels))
    
    dataset.batch(32)
    # COMPILAR MODELO    
    cnn_model.compile(optimizer= 'adam',loss= 'sparse_categorical_crossentropy', metrics= ['accuracy'])
    # INICIAR ENTRENAMIENTO DE MODELO
    cnn_model.fit(dataset, epochs=5)

# LLAMAR FUNCION MAIN
#if __name__ == "__main__":
#    main()
