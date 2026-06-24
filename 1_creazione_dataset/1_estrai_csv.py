import tarfile
import zipfile
import io
import json
import csv
import os

# percorsi cartella dataset
tar_s1_train = "/workspace/data/ImpactMesh-Flood/train/S1RTC.tar"
tar_s2_train_1 = "/workspace/data/ImpactMesh-Flood/train/S2L2A_1.tar"
tar_s2_train_2 = "/workspace/data/ImpactMesh-Flood/train/S2L2A_2.tar"
tar_s2_train_3 = "/workspace/data/ImpactMesh-Flood/train/S2L2A_3.tar"

tar_s1_val = "/workspace/data/ImpactMesh-Flood/val/S1RTC.tar"
tar_s2_val = "/workspace/data/ImpactMesh-Flood/val/S2L2A.tar"

tar_s1_test = "/workspace/data/ImpactMesh-Flood/test/S1RTC.tar"
tar_s2_test = "/workspace/data/ImpactMesh-Flood/test/S2L2A.tar"

# file csv output  
csv_s1_train = "/workspace/csv/train/org/S1RTC_train.csv"
csv_s2_train_1 = "/workspace/csv/train/org/S2L2A_train_1.csv"
csv_s2_train_2 = "/workspace/csv/train/org/S2L2A_train_2.csv"
csv_s2_train_3 = "/workspace/csv/train/org/S2L2A_train_3.csv"

csv_s1_val = "/workspace/csv/val/org/S1RTC_val.csv"
csv_s2_val = "/workspace/csv/val/org/S2L2A_val.csv"

csv_s1_test = "/workspace/csv/test/org/S1RTC_test.csv"
csv_s2_test = "/workspace/csv/test/org/S2L2A_test.csv"


def estrai_tar(percorso_tar, percorso_csv, tipo_sensore):

    print(f"\nScansione {tipo_sensore} da file: {os.path.basename(percorso_tar)}")
    
    with open(percorso_csv, mode='w', newline='', encoding='utf-8') as file_csv:
        writer = csv.writer(file_csv)
        
        # Scriviamo l'intestazione in base al sensore
        if tipo_sensore == "SAR":
            writer.writerow(['Nome_Serie', 'Data_1', 'Data_2', 'Data_3', 'Data_4', 
                             'BBox_1', 'BBox_2', 'BBox_3', 'BBox_4', 
                             'Lat_Centro', 'Lon_Centro', 'EPSG'])
        elif tipo_sensore == "OTTICO":
            writer.writerow(['Nome_Serie', 'Data_1', 'Data_2', 'Data_3', 'Data_4', 
                             'CloudCover_1', 'CloudCover_2', 'CloudCover_3', 'CloudCover_4', 
                             'Lat_Centro', 'Lon_Centro', 'MGRS_Tile', 'EPSG'])

        try:
            with tarfile.open(percorso_tar, 'r') as tar:
                lista_zip = [m for m in tar.getmembers() if m.name.endswith('.zip')]
                print(f"Totale serie: {len(lista_zip)}")

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
                                date_immagini = (date_raw + ["N/A"] * 4)[:4] # Pad a 4
                                
                                lat_centro = dati_json.get("center_lat", "N/A")
                                lon_centro = dati_json.get("center_lon", "N/A")

                                proj_code_raw = str(dati_json.get("proj_code", "N/A")).split(';')
                                codice_epsg = proj_code_raw[0] if proj_code_raw[0] else "N/A"
                                
                                # specifici per sensore
                                if tipo_sensore == "SAR":
                                    bbox_raw = dati_json.get("proj_bbox", "").split(';')
                                    bounding_box = (bbox_raw + ["N/A"] * 4)[:4]
                                    riga_csv = [nome_serie] + date_immagini + bounding_box + [lat_centro, lon_centro, codice_epsg]
                                
                                elif tipo_sensore == "OTTICO":
                                    cloud_raw = str(dati_json.get("eo_cloud_cover", "")).split(';')
                                    cloud_cover = (cloud_raw + ["N/A"] * 4)[:4]
                                    mgrs_tile = str(dati_json.get("s2_mgrs_tile", "")).split(';')[0]
                                    riga_csv = [nome_serie] + date_immagini + cloud_cover + [lat_centro, lon_centro, mgrs_tile, codice_epsg]
                                
                                # riga in csv
                                writer.writerow(riga_csv)
                                
                    except Exception as e:
                        print(f"❌ Errore nella lettura di {nome_serie}: {e}")

            print(f"🔵 File salvato in: {percorso_csv}")

        except Exception as e:
            print(f"❌ Errore apertura del TAR: {e}")

# main

print(f"---------- TRAIN SET ----------")
estrai_tar(tar_s1_train, csv_s1_train, "SAR")
estrai_tar(tar_s2_train_1, csv_s2_train_1, "OTTICO")
estrai_tar(tar_s2_train_2, csv_s2_train_2, "OTTICO")
estrai_tar(tar_s2_train_3, csv_s2_train_3, "OTTICO")

print(f"\n---------- VAL SET ----------")
estrai_tar(tar_s1_val, csv_s1_val, "SAR")
estrai_tar(tar_s2_val, csv_s2_val, "OTTICO")

print(f"\n---------- TEST SET ----------")
estrai_tar(tar_s1_test, csv_s1_test, "SAR")
estrai_tar(tar_s2_test, csv_s2_test, "OTTICO")
print("\n✔️ FINITO")


