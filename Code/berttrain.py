import modulos.model as model
import pandas as pd


def main():
    df = pd.read_csv('./datasets/clean_data.csv')
    df["label"] = df["text"].apply(model.categorize) 
    model.tune_bert(df, df['label'])

main()