import pandas as pd
import os

# percorso worksapce
script_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(script_dir) # /workspace/1_creazione_dataset

# percorsi csv 
csv_output_dir = os.path.join(PROJECT_ROOT, "csv")
file_csv = os.path.join(csv_output_dir, "complete_series.csv")

def filtra_date_distanti(percorso_csv):
    try:
        # Carica il file in memoria
        df = pd.read_csv(percorso_csv)
        righe_iniziali = len(df)
        
        # 1. Convertiamo le colonne in formato datetime
        for t in range(1, 5):
            col_sar = f'Data_{t}_SAR'
            col_opt = f'Data_{t}_OPT'
            
            # Verifica che le colonne esistano prima di procedere
            if col_sar not in df.columns or col_opt not in df.columns:
                print(f"Errore: Le colonne '{col_sar}' o '{col_opt}' non sono presenti nel CSV.")
                return
            
            # errors='coerce' trasforma eventuali date malformate o vuote in NaT (Not a Time)
            df[col_sar] = pd.to_datetime(df[col_sar], errors='coerce')
            df[col_opt] = pd.to_datetime(df[col_opt], errors='coerce')

        # 2. Creiamo una maschera booleana (inizialmente tutte le righe sono valide: True)
        da_mantenere = pd.Series(True, index=df.index)

        # 3. Controlliamo per ogni t se la differenza supera i 7 giorni
        for t in range(1, 5):
            col_sar = f'Data_{t}_SAR'
            col_opt = f'Data_{t}_OPT'
            
            # Calcola la differenza assoluta in giorni
            diff_giorni = (df[col_sar] - df[col_opt]).dt.days.abs()
            
            # Aggiorniamo la maschera: ~(diff_giorni > 7) scarta le righe oltre i 7 giorni.
            # I valori NaT (date vuote) daranno esito False al controllo > 7, quindi verranno mantenuti.
            da_mantenere = da_mantenere & ~(diff_giorni > 14)
            
        # 4. Applichiamo il filtro al DataFrame
        df_pulito = df[da_mantenere]
        
        righe_finali = len(df_pulito)
        righe_eliminate = righe_iniziali - righe_finali
        
        # 5. Sovrascrive esattamente lo stesso file originale
        df_pulito.to_csv(percorso_csv, index=False)
        
        print("Elaborazione completata con successo!")
        print(f"Righe presenti inizialmente: {righe_iniziali}")
        print(f"Righe eliminate (scarto > 14 giorni): {righe_eliminate}")
        print(f"Righe mantenute: {righe_finali}")
        print(f"Il file '{percorso_csv}' è stato aggiornato e sovrascritto.")
        
    except FileNotFoundError:
        print(f"Errore: File '{percorso_csv}' non trovato.")
    except PermissionError:
        print(f"Errore: Permesso negato. Assicurati che '{percorso_csv}' non sia aperto in un altro programma (es. Excel).")
    except Exception as e:
        print(f"Si è verificato un errore inaspettato: {e}")

if __name__ == "__main__":
            
    filtra_date_distanti(file_csv)