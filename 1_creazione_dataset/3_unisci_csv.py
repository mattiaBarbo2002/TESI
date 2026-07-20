# merge tra file S1_365.py e S2_365.py per avere csv unico, ogni riga contiene tutti i dati di una serie, sia opt che sar
# genera due csv, uno contenente serie complete (4 img S1 + 4 img S2 valide) e uno con serie non complete (una o più immagini non trovate)
#
# INPUT:
#   S1_365.csv
#   S2_365.csv
#
# OUTPUT:
#   complete_series.csv
#   incomplete_series.csv
#
#   complete_series_step_3.csv          copia dell'output originali
#   incomplete_series_step_3.csv
# 
# prossimo script 4_img_730_giorni.py

import pandas as pd
import numpy as np
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(script_dir)

file_s1 = os.path.join(PROJECT_ROOT, "csv", "org", "S1RTC_ORG.csv")
file_s2 = os.path.join(PROJECT_ROOT, "csv", "org", "S2L2A_ORG.csv")

csv_output = os.path.join(PROJECT_ROOT, "csv")
os.makedirs(csv_output, exist_ok=True)

complete_file = os.path.join(csv_output, "complete_series.csv")
incomplete_file = os.path.join(csv_output, "incomplete_series.csv")

csv_save_output = os.path.join(PROJECT_ROOT, "csv", "step_3")
os.makedirs(csv_save_output, exist_ok=True)

copy_complete_file = os.path.join(csv_save_output, "complete_series_step_3.csv")
copy_incomplete_file = os.path.join(csv_save_output, "incomplete_series_step_3.csv")

df_s1 = pd.read_csv(file_s1)
df_s2 = pd.read_csv(file_s2)

col_fisse_s1 = [col for col in df_s1.columns if df_s1[col].nunique() == 1]
col_fisse_s2 = [col for col in df_s2.columns if df_s2[col].nunique() == 1]

print("\n--- ANALISI COLONNE ORIGINALI ---")
print(f"Colonne in S1 con un valore costante fin dalla partenza: {col_fisse_s1}")
print(f"Colonne in S2 con un valore costante fin dalla partenza: {col_fisse_s2}")
print("---------------------------------\n")

df_s1['Nome_Serie'] = df_s1['Nome_Serie'].str.replace(r'_S1RTC$', '', regex=True)
df_s2['Nome_Serie'] = df_s2['Nome_Serie'].str.replace(r'_S2L2A$', '', regex=True)

print("\n--- TEST DI DIAGNOSTICA ---")
print(f"Valori univoci in S1: {df_s1['Nome_Serie'].nunique()} su {len(df_s1)} righe totali")
print(f"Valori univoci in S2: {df_s2['Nome_Serie'].nunique()} su {len(df_s2)} righe totali")

nomi_comuni = set(df_s1['Nome_Serie']).intersection(set(df_s2['Nome_Serie']))
print(f"Corrispondenze esatte trovate tra i due file: {len(nomi_comuni)}")
print("---------------------------\n")

df_merged = pd.merge(df_s1, df_s2, on='Nome_Serie', how='outer')

df_merged = df_merged.replace(r'^\s*$', np.nan, regex=True)

df_complete = df_merged.dropna()
df_incomplete = df_merged[df_merged.isna().any(axis=1)]

df_complete.to_csv(complete_file, index=False)
df_incomplete.to_csv(incomplete_file, index=False)

df_complete.to_csv(copy_complete_file, index=False)
df_incomplete.to_csv(copy_incomplete_file, index=False)

print("Elaborazione terminata con successo!")
print(f"Righe TOTALI elaborate: {len(df_merged)}")
print(f" -> Righe COMPLETE salvate: {len(df_complete)}")
print(f" -> Righe INCOMPLETE salvate: {len(df_incomplete)}")

print("\n--- CONTEGIO RIGHE EFFETTIVE ---")
print(f"Righe totali nel file COMPLETI: {len(df_complete)}")
print(f"Righe totali nel file INCOMPLETI: {len(df_incomplete)}")

print("\n--- VALORI UNIVOCI DOPO LA MERGE (df_merged) ---")
print(df_merged.nunique())

print("\n--- ANTEPRIMA DEI DATI (Prime 3 righe) ---")
print(df_merged.head(3))