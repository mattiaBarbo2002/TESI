import pandas as pd
import sys
import os

# percorso worksapce
script_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(script_dir) # /workspace/1_creazione_dataset

# percorsi csv 
csv_output_dir = os.path.join(PROJECT_ROOT, "csv")
file_csv1 = os.path.join(csv_output_dir, "complete_series.csv")
file_csv2 = os.path.join(csv_output_dir, "incomplete_series.csv")

def filtra_e_sovrascrivi(csv1_path, csv2_path):
    colonna_chiave = 'Nome_Serie'
    
    try:
        # Caricamento dei file in memoria
        df1 = pd.read_csv(csv1_path)
        df2 = pd.read_csv(csv2_path)
        
        # Verifica della presenza della colonna in entrambi i file
        if colonna_chiave not in df1.columns:
            print(f"Errore: La colonna '{colonna_chiave}' non esiste in {csv1_path}")
            return
        if colonna_chiave not in df2.columns:
            print(f"Errore: La colonna '{colonna_chiave}' non esiste in {csv2_path}")
            return
            
        righe_iniziali = len(df2)
        
        # Filtro: mantiene in df2 solo le righe in cui Nome_Serie NON è presente nel df1
        df2_filtrato = df2[~df2[colonna_chiave].isin(df1[colonna_chiave])]
        
        righe_finali = len(df2_filtrato)
        righe_eliminate = righe_iniziali - righe_finali
        
        # Sovrascrive esattamente il secondo file
        df2_filtrato.to_csv(csv2_path, index=False)
        
        print("Elaborazione completata con successo!")
        print(f"Righe presenti inizialmente in {csv2_path}: {righe_iniziali}")
        print(f"Righe eliminate: {righe_eliminate}")
        print(f"Righe mantenute: {righe_finali}")
        print(f"Il file '{csv2_path}' è stato aggiornato e sovrascritto.")
        
    except FileNotFoundError as e:
        print(f"Errore: File non trovato. Dettagli: {e}")
    except PermissionError:
        print(f"Errore: Permesso negato. Assicurati che '{csv2_path}' non sia aperto in un altro programma (es. Excel).")
    except Exception as e:
        print(f"Si è verificato un errore inaspettato: {e}")

if __name__ == "__main__":
            
    filtra_e_sovrascrivi(file_csv1, file_csv2)

