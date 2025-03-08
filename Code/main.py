 # Idontknowwhatimdoing.py
from transformers import BertTokenizer
from modulos.utils import  labeler
from modulos.model import split_data, token_data, create_model, performance
import pandas as pd
import modulos.limpieza as clean_data
import modulos.load_data as ldata
def main():
    # LOAD
    #df = clean_data.limpieza(ldata.ldata1(), ldata.ldata2(), ldata.ldata3())
    df = pd.read_csv('./datasets/clean_data.csv')
    print("data loaded")
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
    inputs = (train_encodings['input_ids'].shape[1], 1) 
    num = y_train.shape[1]
    print("creando modelo")
    model = create_model(inputs, num)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_encodings['input_ids'],y_train,epochs=5, batch_size=32, validation_data=(val_encodings['input_ids'],y_test))
#    y_pred =[] # reemplazar
#    y_true = [] # reemplazar 
#    metrics = performance(y_true, y_pred)
#    print(metrics)


if __name__ == "__main__":
    main()