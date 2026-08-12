import pandas as pd
import os

# percorso worksapce
script_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(script_dir) # /workspace/1_creazione_dataset

# percorsi csv 
csv_output_dir = os.path.join(PROJECT_ROOT, "csv")
file_csv = os.path.join(csv_output_dir, "complete_series.csv")

def rimuovi_duplicati_sovrascrivendo(percorso_csv):
    colonna_chiave = 'Nome_Serie'
    try:
        
        df = pd.read_csv(percorso_csv)
        
        if colonna_chiave not in df.columns:
            print(f"Errore: La colonna '{colonna_chiave}' non esiste in {percorso_csv}")
            return
            
        righe_iniziali = len(df)
    
        df_pulito = df.drop_duplicates(subset=[colonna_chiave], keep='first')
        
        righe_finali = len(df_pulito)
        righe_eliminate = righe_iniziali - righe_finali
        
        df_pulito.to_csv(percorso_csv, index=False)
        
        print("Elaborazione completata con successo!")
        print(f"Righe presenti inizialmente: {righe_iniziali}")
        print(f"Righe duplicate eliminate: {righe_eliminate}")
        print(f"Righe mantenute: {righe_finali}")
        print(f"Il file '{percorso_csv}' è stato aggiornato e sovrascritto.")
        
    except FileNotFoundError:
        print(f"Errore: File {percorso_csv} non trovato.")
    except PermissionError:
        print(f"Errore: Permesso negato. Assicurati che il file '{percorso_csv}' non sia aperto in un altro programma.")
    except Exception as e:
        print(f"Si è verificato un errore inaspettato: {e}")

if __name__ == "__main__":
            
    rimuovi_duplicati_sovrascrivendo(file_csv)