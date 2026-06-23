import tarfile
import zipfile
import io
import json
import csv
import os

# Percorsi dei file
percorso_tar = "/workspace/data/ImpactMesh-Flood/val/S1RTC.tar"
percorso_csv = "/workspace/csv/val/S1RTC_val.csv"

print(f"Apro il file TAR: {percorso_tar}")
print("Inizio l'estrazione e formattazione in colonne separate...")

# Creazione del CSV
with open(percorso_csv, mode='w', newline='', encoding='utf-8') as file_csv:
    writer = csv.writer(file_csv)
    
    # Intestazione aggiornata con colonne separate per i 4 scatti
    writer.writerow([
        'Nome_Serie', 
        'Data_1', 'Data_2', 'Data_3', 'Data_4', 
        'BBox_1', 'BBox_2', 'BBox_3', 'BBox_4', 
        'Latitudine_Centro', 'Longitudine_Centro'
    ])

    try:
        with tarfile.open(percorso_tar, 'r') as tar:
            
            lista_zip = [m for m in tar.getmembers() if m.name.endswith('.zip')]
            print(f"Trovate {len(lista_zip)} serie temporali da analizzare.")

            for membro_zip in lista_zip:
                nome_serie = os.path.basename(membro_zip.name).replace('.zarr.zip', '')

                try:
                    file_estratto = tar.extractfile(membro_zip)
                    if file_estratto is None:
                        continue
                    
                    with zipfile.ZipFile(io.BytesIO(file_estratto.read())) as inner_zip:
                        
                        zattrs_files = [f for f in inner_zip.namelist() if f.endswith('.zattrs')]
                        
                        if zattrs_files:
                            contenuto = inner_zip.read(zattrs_files[0])
                            dati_json = json.loads(contenuto)
                            
                            # 1. Estrazione e split dei dati basato sul punto e virgola
                            date_raw = dati_json.get("datetime", "").split(';')
                            bbox_raw = dati_json.get("proj_bbox", "").split(';')
                            
                            # 2. Imposizione di esattamente 4 elementi (padding con "N/A" se mancanti)
                            date_immagini = (date_raw + ["N/A"] * 4)[:4]
                            bounding_box = (bbox_raw + ["N/A"] * 4)[:4]
                            
                            lat_centro = dati_json.get("center_lat", "N/A")
                            lon_centro = dati_json.get("center_lon", "N/A")
                            
                            # 3. Concatenazione della riga finale espansa
                            riga_csv = [nome_serie] + date_immagini + bounding_box + [lat_centro, lon_centro]
                            writer.writerow(riga_csv)
                            
                except Exception as e:
                    print(f"Errore minore durante la lettura di {nome_serie}: {e}")

        print(f"\nOperazione conclusa con successo!")
        print(f"Il tuo file è pronto in: {percorso_csv}")

    except FileNotFoundError:
        print(f"ERRORE: Il file {percorso_tar} non è stato trovato.")