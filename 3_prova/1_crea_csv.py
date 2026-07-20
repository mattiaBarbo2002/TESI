# estrazione dati da cartelle tar del dataset ImpactMesh e generazione di un UNICO file csv.
# Mantiene SOLO le serie che hanno:
#   - 4 date valide e non duplicate
#   - Coordinate del centro (Lat e Lon) valide
# (Il controllo viene fatto in modo indipendente per SAR e OTTICO prima del merge).
# Non genera file intermedi.
#
# INPUT:
#   cartelle tar in \data\ImpactMesh\...
#
# OUTPUT:
#   merged_series.csv

import tarfile
import zipfile
import io
import json
import os
import pandas as pd
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(script_dir)

# paths tar ImpactMesh
tar_s1_train = os.path.join(PROJECT_ROOT, "data", "ImpactMesh-Flood", "train", "S1RTC.tar")
tar_s2_train_1 = os.path.join(PROJECT_ROOT, "data", "ImpactMesh-Flood", "train", "S2L2A_1.tar")
tar_s2_train_2 = os.path.join(PROJECT_ROOT, "data", "ImpactMesh-Flood", "train", "S2L2A_2.tar")
tar_s2_train_3 = os.path.join(PROJECT_ROOT, "data", "ImpactMesh-Flood", "train", "S2L2A_3.tar")

tar_s1_val = os.path.join(PROJECT_ROOT, "data", "ImpactMesh-Flood", "val", "S1RTC.tar")
tar_s2_val = os.path.join(PROJECT_ROOT, "data", "ImpactMesh-Flood", "val", "S2L2A.tar")

tar_s1_test = os.path.join(PROJECT_ROOT, "data", "ImpactMesh-Flood", "test", "S1RTC.tar")
tar_s2_test = os.path.join(PROJECT_ROOT, "data", "ImpactMesh-Flood", "test", "S2L2A.tar")

# liste per divisione S1 e S2
lista_tar_s1 = [tar_s1_train, tar_s1_val, tar_s1_test]
lista_tar_s2 = [tar_s2_train_1, tar_s2_train_2, tar_s2_train_3, tar_s2_val, tar_s2_test]

# path e nome output
cartella_output_csv = os.path.join(PROJECT_ROOT, "csv")
os.makedirs(cartella_output_csv, exist_ok=True) 

csv_merged = os.path.join(cartella_output_csv, "anomaly_ds.csv")

# Valori considerati "vuoti" per i controlli di completezza
VALORI_VUOTI = {"", "n/a", "nan", "none", "null"}

def estrai_dati(lista_percorsi_tar, tipo_sensore):
    dati_estratti = []
    conteggio_scartate_incomplete = 0
    conteggio_scartate_dup = 0

    print(f"\nEstrazione {tipo_sensore}")

    for percorso_tar in lista_percorsi_tar:
        if not os.path.exists(percorso_tar):
            print(f"File non trovato: {os.path.basename(percorso_tar)}.")
            continue
            
        print(f"Scansione: {os.path.basename(percorso_tar)}")
        
        try:
            with tarfile.open(percorso_tar, 'r') as tar:
                lista_zip = [m for m in tar.getmembers() if m.name.endswith('.zip')]
                print(f"Totale serie trovate: {len(lista_zip)}")

                for membro_zip in lista_zip:
                    # Estraiamo il nome e rimuoviamo direttamente il suffisso per il merge successivo
                    nome_serie = os.path.basename(membro_zip.name).replace('.zarr.zip', '')
                    nome_serie_pulito = nome_serie.replace('_S1RTC', '').replace('_S2L2A', '')

                    try:
                        file_estratto = tar.extractfile(membro_zip)
                        if file_estratto is None: continue
                        
                        with zipfile.ZipFile(io.BytesIO(file_estratto.read())) as inner_zip:
                            zattrs_files = [f for f in inner_zip.namelist() if f.endswith('.zattrs')]
                            
                            if zattrs_files:
                                dati_json = json.loads(inner_zip.read(zattrs_files[0]))
                                
                                # dati comuni
                                date_raw = dati_json.get("datetime", "").split(';')
                                date_immagini = (date_raw + ["N/A"] * 4)[:4] 
                                
                                lat_centro = dati_json.get("center_lat", "N/A")
                                lon_centro = dati_json.get("center_lon", "N/A")

                                proj_code_raw = str(dati_json.get("proj_code", "N/A")).split(';')
                                codice_epsg = proj_code_raw[0] if proj_code_raw[0] else "N/A"
                                
                                # 1. CONTROLLO COMPLETEZZA (4 date valide + coordinate centro)
                                date_valide = [d for d in date_immagini if str(d).strip().lower() not in VALORI_VUOTI]
                                lat_valida = str(lat_centro).strip().lower() not in VALORI_VUOTI
                                lon_valida = str(lon_centro).strip().lower() not in VALORI_VUOTI
                                
                                if len(date_valide) < 4 or not lat_valida or not lon_valida:
                                    conteggio_scartate_incomplete += 1
                                    continue # Salta alla prossima serie

                                # 2. CONTROLLO DATE DUPLICATE
                                if len(date_valide) != len(set(date_valide)):
                                    conteggio_scartate_dup += 1
                                    continue # Salta alla prossima serie
                                
                                # Creazione della riga come dizionario per Pandas
                                if tipo_sensore == "SAR":
                                    bbox_raw = dati_json.get("proj_bbox", "").split(';')
                                    bounding_box = (bbox_raw + ["N/A"] * 4)[:4]
                                    
                                    riga = {
                                        'Nome_Serie': nome_serie_pulito,
                                        'Data_1_org_SAR': date_immagini[0], 'Data_2_org_SAR': date_immagini[1],
                                        'Data_3_org_SAR': date_immagini[2], 'Data_4_org_SAR': date_immagini[3],
                                        'Bbox_1_SAR': bounding_box[0], 'Bbox_2_SAR': bounding_box[1],
                                        'Bbox_3_SAR': bounding_box[2], 'Bbox_4_SAR': bounding_box[3],
                                        'Lat_Centro_SAR': lat_centro, 'Lon_Centro_SAR': lon_centro, 'EPSG_SAR': codice_epsg
                                    }
                                    
                                elif tipo_sensore == "OTTICO":
                                    cloud_raw = str(dati_json.get("eo_cloud_cover", "")).split(';')
                                    cloud_cover = (cloud_raw + ["N/A"] * 4)[:4]
                                    mgrs_tile = str(dati_json.get("s2_mgrs_tile", "")).split(';')[0]
                                    
                                    riga = {
                                        'Nome_Serie': nome_serie_pulito,
                                        'Data_1_org_OPT': date_immagini[0], 'Data_2_org_OPT': date_immagini[1],
                                        'Data_3_org_OPT': date_immagini[2], 'Data_4_org_OPT': date_immagini[3],
                                        'Cloud_1_OPT': cloud_cover[0], 'Cloud_2_OPT': cloud_cover[1],
                                        'Cloud_3_OPT': cloud_cover[2], 'Cloud_4_OPT': cloud_cover[3],
                                        'Lat_Centro_OPT': lat_centro, 'Lon_Centro_OPT': lon_centro,
                                        'MGRS_Tile_OPT': mgrs_tile, 'EPSG_OPT': codice_epsg
                                    }

                                dati_estratti.append(riga)
                                
                    except Exception as e:
                        print(f"❌ Errore lettura di {nome_serie}: {e}")

        except Exception as e:
            print(f"❌ Errore apertura del TAR {os.path.basename(percorso_tar)}: {e}")
            
    print(f" -> Righe estratte con successo: {len(dati_estratti)}")
    print(f" -> Righe scartate (mancano date o coordinate): {conteggio_scartate_incomplete}")
    print(f" -> Righe scartate (date duplicate): {conteggio_scartate_dup}")
    
    return dati_estratti

# main
print("--- ESTRAZIONE DATI SAR (S1) ---")
dati_s1 = estrai_dati(lista_tar_s1, "SAR")

print("\n--- ESTRAZIONE DATI OTTICI (S2) ---")
dati_s2 = estrai_dati(lista_tar_s2, "OTTICO")

# --- MERGE CON PANDAS ---
print("\n--- AVVIO MERGE DEI DATI ---")

# Creazione Dataframes
df_s1 = pd.DataFrame(dati_s1)
df_s2 = pd.DataFrame(dati_s2)

# Unione dei dataframe (Outer join)
# Anche se facciamo l'outer join, sappiamo che ogni riga inserita ha superato 
# i test di completezza per il rispettivo sensore.
df_merged = pd.merge(df_s1, df_s2, on='Nome_Serie', how='outer')

# Sostituzione delle stringhe vuote o di soli spazi con NaN per pulizia formato
df_merged = df_merged.replace(r'^\s*$', np.nan, regex=True)

# Salvataggio del file finale unito
df_merged.to_csv(csv_merged, index=False)

print(f"🔵 File unificato salvato in: {csv_merged}")
print(f"Righe TOTALI nel file unito: {len(df_merged)}")
print("\n✔️ FINITO")