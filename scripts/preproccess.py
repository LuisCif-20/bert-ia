import os
import pandas as pd

def preprocess(data_path, output_path):
    # Implementa aquí el preprocesamiento
    df = pd.read_csv(data_path)
    # Ejemplo de preprocesamiento
    df['text'] = df['text'].str.lower() # Convertir a minúsculas
    # Guarda los datos preprocesados
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    preprocess('data/dialogues.csv', 'data/dialogues_processed.csv')
