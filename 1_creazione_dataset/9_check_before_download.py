# controllo errori nel complete_series.csv:
#   serie con stesso nome
#   serie con date orginali entro +- 6 mesi dalle date trovate
#  
# inserisce serie da controllare in file da_controllare.csv
# rieseguire 9_check_month per controllare serie e mesi cancellati
# 
# INPUT:
#   complete_series.csv
#
# OUTPUT:
#   
#   da_controllare.csv


import pandas as pd
import os

# percorso /workspace
script_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(script_dir) # /workspace/1_creazione_dataset

# percorsi csv
file_completi = os.path.join(PROJECT_ROOT, "csv", "new_complete_series_gee_COMPLETO.csv")
file_da_controllare = os.path.join(PROJECT_ROOT, "csv", "da_controllare.csv")

df = pd.read_csv(file_completi)

# ricerca duplicati
duplicati_mask = df.duplicated(subset=['Nome_Serie'], keep=False).values

# conversione formato date per controllo
d3_org_sar = pd.to_datetime(df['Data_3_org_SAR'], errors='coerce')
d3_sar = pd.to_datetime(df['Data_3_SAR'], errors='coerce')
d3_org_opt = pd.to_datetime(df['Data_3_org_OPT'], errors='coerce')
d3_opt = pd.to_datetime(df['Data_3_OPT'], errors='coerce')

# converione righe in liste
nomi = df['Nome_Serie'].values
righe_anomale = []

# scorrimento righe
for idx, (nome, is_dup, org_s, s, org_o, o) in enumerate(zip(nomi, duplicati_mask, d3_org_sar, d3_sar, d3_org_opt, d3_opt)):
    motivi = []
    
    # duplicato
    if is_dup:
        motivi.append("Nome_Serie duplicato")
        
    # data target sar <= 6 mesi
    if pd.notna(org_s) and pd.notna(s):
        
        diff_sar = abs((org_s.year - s.year) * 12 + org_s.month - s.month)
        if diff_sar <= 6:
            motivi.append(f"SAR troppo vicino ({diff_sar} mesi di distanza)")
            
    # data target opt <= 6 mesi
    if pd.notna(org_o) and pd.notna(o):
        diff_opt = abs((org_o.year - o.year) * 12 + org_o.month - o.month)
        if diff_opt <= 6:
            motivi.append(f"OPT troppo vicino ({diff_opt} mesi di distanza)")

    # data target meno recente opt
    if pd.notna(org_o) and pd.notna(o):
        diff_opt = (org_o.year <= o.year) 
        if diff_opt:
            motivi.append(f"ERRORE")

    # data target meno recente sar
    if pd.notna(org_s) and pd.notna(s):
        diff_opt = (org_s.year <= s.year) 
        if diff_opt:
            motivi.append(f"ERRORE")
            
    # salvataggio righe anomale
    if motivi:
        riga_dict = df.iloc[idx].to_dict()
        
        
        riga_dict['Problema'] = " | ".join(motivi)
        righe_anomale.append(riga_dict)


# salvataggio da_controllare.csv
df_out = pd.DataFrame(righe_anomale)

if not df_out.empty:
    
    colonne_ordinate = ['Nome_Serie', 'Problema'] + [c for c in df.columns if c != 'Nome_Serie']
    df_out = df_out[colonne_ordinate]

    df_out.to_csv(file_da_controllare, index=False)
    print(f"\nTrovate {len(df_out)} serie anomale")
    print(f"File salvato: {file_da_controllare}")
else:
    print("\nOK, nessun conflitto")