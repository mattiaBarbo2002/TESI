# estrazione dati da cartelle tar del dataset ImpactMesh e generazione file csv contenenti metadati e pulizia righe con valori nan
#
# INPUT:
#   cartelle tar in \data\ImpactMesh\...
#
# OUTPUT:
#   S1RTC_ORG.csv
#   S2L2A_ORG.csv
#
# prossimo 2_pulisci_csv

import tarfile
import zipfile
import io
import json
import csv
import os

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
cartella_output_csv = os.path.join(PROJECT_ROOT, "csv", "org")
os.makedirs(cartella_output_csv, exist_ok=True) 

csv_s1_all = os.path.join(cartella_output_csv, "S1RTC_ORG.csv")
csv_s2_all = os.path.join(cartella_output_csv, "S2L2A_ORG.csv")

VALORI_INVALIDI = {"", "n/a", "nan", "none", "null"}


def estrai_csv(lista_percorsi_tar, percorso_csv, tipo_sensore):
    conteggio_righe = 0
    conteggio_scartate = 0

    print(f"\nEstrazione {tipo_sensore}")
    
    with open(percorso_csv, mode='w', newline='', encoding='utf-8') as file_csv:
        writer = csv.writer(file_csv)
        
        # intestazione csv
        if tipo_sensore == "SAR":
            writer.writerow(['Nome_Serie', 'Data_1_org_SAR', 'Data_2_org_SAR', 'Data_3_org_SAR', 'Data_4_org_SAR', 
                             'Lat_Centro_SAR', 'Lon_Centro_SAR', 'EPSG_SAR'])
        elif tipo_sensore == "OTTICO":
            writer.writerow(['Nome_Serie', 'Data_1_org_OPT', 'Data_2_org_OPT', 'Data_3_org_OPT', 'Data_4_org_OPT', 
                             'Lat_Centro_OPT', 'Lon_Centro_OPT', 'MGRS_Tile_OPT', 'EPSG_OPT'])


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
                        nome_serie = os.path.basename(membro_zip.name).replace('.zarr.zip', '')

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
                                    
                                    # dati specifici per sensore
                                    if tipo_sensore == "SAR":
                                        bbox_raw = dati_json.get("proj_bbox", "").split(';')
                                        bounding_box = (bbox_raw + ["N/A"] * 4)[:4]
                                        riga_csv = [nome_serie] + date_immagini + bounding_box + [lat_centro, lon_centro, codice_epsg]
                                    
                                    elif tipo_sensore == "OTTICO":
                                        cloud_raw = str(dati_json.get("eo_cloud_cover", "")).split(';')
                                        cloud_cover = (cloud_raw + ["N/A"] * 4)[:4]
                                        mgrs_tile = str(dati_json.get("s2_mgrs_tile", "")).split(';')[0]
                                        riga_csv = [nome_serie] + date_immagini + cloud_cover + [lat_centro, lon_centro, mgrs_tile, codice_epsg]

                                    # controllo valori non validi
                                    riga_valida = True
                                    for elemento in riga_csv:
                                        valore_pulito = str(elemento).strip().lower()
                                        if valore_pulito in VALORI_INVALIDI:
                                            riga_valida = False
                                            break
                                    
                                    # scrittura riga nel csv
                                    if riga_valida:
                                        writer.writerow(riga_csv)
                                        conteggio_righe += 1
                                    else:
                                        conteggio_scartate += 1
                                    
                        except Exception as e:
                            print(f"❌ Errore lettura di {nome_serie}: {e}")

            except Exception as e:
                print(f"❌ Errore apertura del TAR {os.path.basename(percorso_tar)}: {e}")
                
    print(f"🔵 File salvato in: {percorso_csv}")
    print(f"Totale righe {tipo_sensore}: {conteggio_righe + conteggio_scartate}")
    print(f"Totale righe estratte {tipo_sensore}: {conteggio_righe}")
    print(f"Totale righe scartate {tipo_sensore}: {conteggio_scartate}")
    

# main
print("--- ESTRAZIONE DATI SAR (S1) ---")
estrai_csv(lista_tar_s1, csv_s1_all, "SAR")
print("\n")

print("--- ESTRAZIONE DATI OTTICI (S2) ---")
estrai_csv(lista_tar_s2, csv_s2_all, "OTTICO")
print("\n")

print("\n✔️ FINITO")