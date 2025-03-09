from transformers import BertTokenizer
from sklearn.preprocessing import MultiLabelBinarizer
from modulos.utils import  labeler
from modulos.model import split_data, token_data, create_model, performance
import pandas as pd

def main():

    # LOAD
    df = pd.read_csv('./datasets/clean_data.csv')
    print("data loaded")

    #LABEL
    df = labeler(df) 
    print("labelled")

    

    mlb = MultiLabelBinarizer()

    etiquetas_bin = mlb.fit_transform(df['label'])

    etiquetas_bin_df = pd.DataFrame(etiquetas_bin, columns=mlb.classes_)
    df = pd.concat([df, etiquetas_bin_df], axis=1)
    df = df.drop(columns=['label'])
    df = df.drop(columns=['Other'])

    df.to_csv('./datasets/labelled_data.csv', index=False)


    # SPLIT
    x_train, x_test, y_train, y_test = split_data(df, etiquetas_bin)
    print("splitted")

    # TOKENIZE
    train_texts = x_train.tolist()  
    val_texts = x_test.tolist()  
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_encodings = token_data(train_texts, tokenizer)
    val_encodings = token_data(val_texts,tokenizer)
    print("tokenized")

    # Actually training the model
    #y_train = pd.get_dummies(y_train).values
    #y_test = pd.get_dummies(y_test).values
    inputs = (train_encodings['input_ids'].shape[1], 1) 
    num = y_train.shape[1]
    print(y_train)
    print("creando modelo")
    model = create_model(inputs, num)


    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(train_encodings['input_ids'],y_train,epochs=20, batch_size=20, validation_data=(val_encodings['input_ids'],y_test))

    #results = model.evaluate(val_encodings['input_ids'], y_test)
    #print(f"Loss: {results[0]}, Accuracy: {results[1]}")


if __name__ == "__main__":
    main()