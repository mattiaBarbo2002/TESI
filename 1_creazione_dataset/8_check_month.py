# controllo occorrenze mesi nel nuovo complete_series.csv
#
# INPUT:
#   complete_series.csv


import pandas as pd
import os 

script_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(script_dir) # /workspace/1_creazione_dataset

path_complete = os.path.join(PROJECT_ROOT, "csv", "complete_series.csv")


df = pd.read_csv(path_complete)

# conversione formato data
df['Data_3_OPT'] = pd.to_datetime(df['Data_3_OPT'], errors='coerce')

# conto occorrenze mese
conteggio_mesi = df['Data_3_OPT'].dt.month.value_counts()

conteggio_mesi = conteggio_mesi.sort_index()

print("Occorrenze per mese:")
print(conteggio_mesi)