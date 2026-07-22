# conta quante serie sono presenti in complete_series.csv per mese (data_3_opt)
# conta numero serie totali di incomplete_series.csv
# assegna un mese random ad ogni serie di incomplete_series.csv per bilanciare il dataset (colonna target_month)
#
# INPUT:
#   complete_series.csv
#   incomplete_series.csv
#
# OUTPUT:
#   incomplete_series.csv (aggiunta/aggiornata colonna target_month)

import pandas as pd
import random
import os 
import shutil

# percorso workspace
script_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(script_dir) # /workspace/1_creazione_dataset

# percorso csv input
path_complete = os.path.join(PROJECT_ROOT, "csv", "complete_series.csv")
path_incomplete = os.path.join(PROJECT_ROOT, "csv", "incomplete_series.csv")

df_complete = pd.read_csv(path_complete)
df_incomplete = pd.read_csv(path_incomplete)


# conteggio mesi complete_series.csv
df_complete['Data_3_OPT'] = pd.to_datetime(df_complete['Data_3_OPT'], errors='coerce')
conteggi_estratti = df_complete['Data_3_OPT'].dt.month.dropna().astype(int).value_counts().to_dict()
conteggio_attuale = {mese: conteggi_estratti.get(mese, 0) for mese in range(1, 13)}

# calcolo bilanciamento (greedy secchi)
righe_da_assegnare = len(df_incomplete)
nuovi_mesi_assegnati = {mese: 0 for mese in range(1, 13)}

for _ in range(righe_da_assegnare):
    mese_minore = min(conteggio_attuale.keys(), key=lambda m: conteggio_attuale[m] + nuovi_mesi_assegnati[m])
    nuovi_mesi_assegnati[mese_minore] += 1

# assegnazione casuale mesi target a incomplete_series.csv
lista_mesi = []
for m, quantita in nuovi_mesi_assegnati.items():
    lista_mesi.extend([m] * quantita)

random.seed(42) 
random.shuffle(lista_mesi)

# --- VERIFICA E AGGIORNAMENTO COLONNA ---
if 'target_month' in df_incomplete.columns:
    print("🔄 La colonna 'target_month' esiste già: i valori verranno sovrascritti.")
else:
    print("🆕 Creazione della nuova colonna 'target_month'...")

# Assegnazione dei valori (crea o sovrascrive automaticamente)
df_incomplete['target_month'] = lista_mesi
# ----------------------------------------

# salvataggio
df_incomplete.to_csv(path_incomplete, index=False)

# stampe
mesi_ordinati = sorted(nuovi_mesi_assegnati.items(), key=lambda x: x[1], reverse=True)

risultati = []
for m in range(1, 13):
    risultati.append({
        'Mese': m,
        'Dati Iniziali': conteggio_attuale[m],
        'Dati Aggiunti': nuovi_mesi_assegnati[m],
        'Totale Finale': conteggio_attuale[m] + nuovi_mesi_assegnati[m]
    })

df_riepilogo = pd.DataFrame(risultati).set_index('Mese')
print("\n--- RIEPILOGO BILANCIAMENTO ---")
print(df_riepilogo.to_string())



