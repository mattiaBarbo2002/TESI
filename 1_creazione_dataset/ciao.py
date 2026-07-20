import pandas as pd
import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(script_dir) # /workspace/1_creazione_dataset

file_s1 = os.path.join(PROJECT_ROOT, "csv", "S2_365.csv")
file_s2 = os.path.join(PROJECT_ROOT, "csv", "vecchi2", "S2_365.csv")
output_path = os.path.join(PROJECT_ROOT, "csv", "nuovo.csv")

def aggiorna_csv(file1_path, file2_path, output_path=None):
    if output_path is None:
        # Sovrascrive il file 1 se non viene specificato un output
        output_path = file1_path
        
    print(f"📥 Caricamento del primo CSV: {file1_path}")
    try:
        df1 = pd.read_csv(file1_path)
    except Exception as e:
        print(f"❌ Errore nella lettura di {file1_path}: {e}")
        return

    print(f"📥 Caricamento del secondo CSV: {file2_path}")
    try:
        df2 = pd.read_csv(file2_path)
    except Exception as e:
        print(f"❌ Errore nella lettura di {file2_path}: {e}")
        return

    # Verifica della presenza delle colonne necessarie
    if 'Nome_Serie' not in df1.columns or 'Nome_Serie' not in df2.columns:
        print("❌ Errore: Entrambi i file CSV devono contenere la colonna 'Nome_Serie'")
        return
        
    if 'MGRS_Tile' not in df2.columns:
        print("❌ Errore: Il secondo CSV deve contenere la colonna 'MGRS_Tile'")
        return

    print("⚙️ Associazione dei valori MGRS_Tile...")
    # Crea un dizionario di mappatura dal secondo file: Nome_Serie -> MGRS_Tile
    mapping = dict(zip(df2['Nome_Serie'], df2['MGRS_Tile']))

    # Se la colonna MGRS_Tile_OPT non esiste nel primo file, la creiamo vuota
    if 'MGRS_Tile_OPT' not in df1.columns:
        df1['MGRS_Tile_OPT'] = None

    # Aggiorna 'MGRS_Tile_OPT' in df1 con i valori trovati in df2
    # La funzione map() incrocia i dati. Usiamo combine_first() per far sì che, 
    # se per caso non c'è corrispondenza per una riga, venga mantenuto il valore preesistente in df1.
    nuovi_valori = df1['Nome_Serie'].map(mapping)
    df1['MGRS_Tile_OPT'] = nuovi_valori.combine_first(df1['MGRS_Tile_OPT'])

    print(f"💾 Salvataggio in corso su: {output_path}")
    df1.to_csv(output_path, index=False)
    print("✅ Operazione completata con successo!")

if __name__ == "__main__":

    
    aggiorna_csv(file_s1, file_s2, output_path)