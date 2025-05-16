import pandas as pd
import re 

df = pd.read_csv("candy_sales.csv")
df.columns = [re.sub(r'[^a-z0-9_]', '', col.lower().replace(' ', '_')) for col in df.columns] ##para normalizar los nombres de las columnas 
print("Primeras filas del dataset:")
print(df.head())
print("\nInformación del dataset:")
print(df.info())
print("\nValores nulos por columna:")
print(df.isnull().sum())
df_cleaned = df.dropna()
print("\nNúmero de filas duplicadas:", df_cleaned.duplicated().sum())
df_cleaned = df_cleaned.drop_duplicates()
df_cleaned['order_date'] = pd.to_datetime(df_cleaned['order_date'], errors='coerce')# Convertir fechas
df_cleaned['ship_date'] = pd.to_datetime(df_cleaned['ship_date'], errors='coerce')  # Convertir fechas
print("\nTipos de datos luego de convertir fechas:")
print(df_cleaned.dtypes)
df_cleaned.to_csv("candySalesCleaned.csv", index=False)
print("\nDataset limpio listo para análisis.")