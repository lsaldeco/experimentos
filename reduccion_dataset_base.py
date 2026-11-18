# -*- coding: utf-8 -*-
#Recorto el año 2019 para bajar un poco el tamaño del dataset
import pandas as pd

# Cargar el archivo original
dataset = "competencia_02.csv"
print("Cargando dataset...")
df = pd.read_csv(dataset)

# 1. Ver cuántos registros hay por cada mes para identificar la cobertura temporal
print("\n--- Conteo de registros por mes (foto_mes) ---")
print(df["foto_mes"].value_counts().sort_index())

# 2. Filtrar únicamente los meses del año 2019 (del 201901 al 201912)
df_2019 = df[
    (df["foto_mes"] >= 201901) & (df["foto_mes"] <= 201912)
].copy()

print(f"\nDimensiones originales: {df.shape}")
print(f"Dimensiones filtradas (solo 2019): {df_2019.shape}")

# 3. Guardar el nuevo archivo más liviano para las pruebas
output_path = "competencia_02_2019_crudo.parquet"
df_2019.to_parquet(output_path, index=False)
print(f"\n¡Archivo reducido guardado con éxito en:\n{output_path}")


