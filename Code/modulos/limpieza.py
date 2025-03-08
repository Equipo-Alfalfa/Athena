def limpieza(df, df2, df3):
    import pandas as pd
    
    #Eliminar filas en blanco
    df_dropped = df.dropna().sum()
    df2_dropped = df2.dropna().sum()
    print(df_dropped, df2_dropped)
    

    #Verificar duplicados
    df_duplicated = df.duplicated().sum()
    df2_duplicated = df2.duplicated().sum()
    print(df_duplicated, df2_duplicated)
    
    #Eliminar duplicados 
    df = df.drop_duplicates()
    df2 = df2.drop_duplicates()
    df.duplicated().sum()
    df2.duplicated().sum()
    
    #Renombrar columnas
    df.rename(columns={"Article":"text"}, inplace=True)
    df2.rename(columns={"CleanText":"text"}, inplace= True)

    #eliminar columnas innecearias
    df = df.drop(columns=["Title", "Link", "Label"])
    df2 = df2.drop(columns=["RawText", "TTP", "URL"])
    print(df,df2)
    
    #Unir dfs
    Data_clean=pd.concat([df,df2,df3],axis=0)
    Data_clean =Data_clean.drop(columns=["Unnamed: 0"])

    #eliminar caracteres invisibles
    caracteres = ["\n", "\t", "\r", "@, #, $, %, ^, &, *, (, ), _, +, =, {, }, [, ], |, \, :, ;, <, >, /, ?, ., ,, !, ¡, ¿, ?, -, _, "]
    for caracter in caracteres:
        Data_clean["text"] = Data_clean["text"].str.replace(caracter, "")
    
    #regresa la data a la funcion que la necesita
    print(Data_clean.count())
    return Data_clean

    


