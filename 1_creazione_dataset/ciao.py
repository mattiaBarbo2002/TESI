import pandas as pd
import os

# Impostazione percorsi
script_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(script_dir) 

file_input = os.path.join(PROJECT_ROOT, "csv", "S1_365.csv")
file_output = os.path.join(PROJECT_ROOT, "csv", "date_duplicate_sar.csv")

print(f"Lettura file: {file_input}")
df = pd.read_csv(file_input)

righe_anomale = []

# Scorro tutte le righe del dataframe
for index, row in df.iterrows():
    motivi = []
    
    # Estraggo le date SAR e OPT
    date = [row.get('Data_1_org_SAR'), row.get('Data_2_org_SAR'), row.get('Data_3_org_SAR'), row.get('Data_4_org_SAR')]
    #date = [row.get('Data_1_org_OPT'), row.get('Data_2_org_OPT'), row.get('Data_3_org_OPT'), row.get('Data_4_org_OPT')]
    
    # Pulisco le liste ignorando i valori vuoti o 'NaN'
    valide = [d for d in date if pd.notna(d) and str(d).strip() not in ('nan', 'NaN', '')]
    #opt_valide = [d for d in date_opt if pd.notna(d) and str(d).strip() not in ('nan', 'NaN', '')]
    
    # Controllo duplicati confrontando la lunghezza della lista con quella del set (che elimina i doppioni)
    if len(valide) != len(set(valide)):
        motivi.append("Date duplicate")
        
    # Se ho trovato dei problemi, salvo la riga
    if motivi:
        riga_dict = row.to_dict()
        riga_dict['Problema'] = " | ".join(motivi)
        righe_anomale.append(riga_dict)

# Creazione del nuovo file CSV con gli errori
df_out = pd.DataFrame(righe_anomale)

if not df_out.empty:
    # Riordino le colonne per avere 'Problema' all'inizio
    colonne_ordinate = ['Nome_Serie', 'Problema'] + [c for c in df.columns if c != 'Nome_Serie']
    df_out = df_out[colonne_ordinate]

    df_out.to_csv(file_output, index=False)
    print(f"\n⚠️ Trovate {len(df_out)} serie con date duplicate!")
    print(f"File salvato: {file_output}")
else:
    print("\n✅ OK, nessun duplicato trovato. Tutte le date sono uniche per ogni serie.")