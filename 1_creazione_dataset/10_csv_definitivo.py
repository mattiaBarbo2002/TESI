import pandas as pd
import os

# percorso /workspace
script_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(script_dir) # /workspace/1_creazione_dataset

# percorsi csv
file_completi = os.path.join(PROJECT_ROOT, "csv", "new_complete_series_gee_COMPLETO.csv")
file_da_controllare = os.path.join(PROJECT_ROOT, "csv", "da_controllare.csv")
file_definitivo = os.path.join(PROJECT_ROOT, "csv", "definitivo.csv")

df_completo = pd.read_csv(file_completi)
df_da_controllare = pd.read_csv(file_da_controllare)

# Filtra il primo dataframe: 
# Mantiene solo le righe dove 'Nome_serie' NON è presente nella lista del secondo file
df_risultato = df_completo[~df_completo['Nome_Serie'].isin(df_da_controllare['Nome_Serie'])]

# Salva il risultato in un nuovo CSV
df_risultato.to_csv(file_definitivo, index=False)

print("Operazione completata! file 'definitivo.csv' creato.")