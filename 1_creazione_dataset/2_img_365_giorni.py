# da i csv estratti da impact mesh verifica disponibilità di immagini sar e ottiche -1 anno (365 giorni)
# genera csv con date immagini disponibili che rispettano requisiti  (25% nuvole, 5% pixel Nan)
# se immagine non trovata scrive NaN
#
# INPUT:
#   S1RTC_ORG.csv
#   S2L2A_ORG.csv
#
# OUTPUT:
#   S1_365.csv
#   S2_365.csv
#
# prossimo 3_unisci_csv.py

import ee
import csv
import os
import requests
import zipfile
import io
import json
from datetime import datetime, timedelta, timezone
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# configurazione percorsi
script_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(script_dir) # /workspace/1_creazione_dataset

csv_lock = threading.Lock()

# costanti 
CONTAINER = False       # true se lo lancio da container, false in locale
MAX_WORKERS = 15

PIXEL_PERCENT = 5       # % pixel Nan in img SAR e OTTICHE
CLOUDY_PERCENTAGE = 25  # % copertura nuvolosa
TARGET_SCALE = 10       # risoluzione 10 metri 
RAGGIO_METRI = 1280     # raggio di 1280m (2560m/2), box 256x256 (256 px * risoluzione 10m = 2560)
RANGE_GIORNI = 10       # +- giorni intorno alla data
ANNO_PRIMA = 365        # ricerca tot anni prima della data originale (in giorni)


# percorsi csv
input_csv_s1 = os.path.join(PROJECT_ROOT, "csv", "org", "S1RTC_ORG.csv")
input_csv_s2 = os.path.join(PROJECT_ROOT, "csv", "org", "S2L2A_ORG.csv")

# output dataset
S1_folder = os.path.join(PROJECT_ROOT, "data", "dataset", "S1")
S2_folder = os.path.join(PROJECT_ROOT, "data", "dataset", "S2")

os.makedirs(S1_folder, exist_ok=True)
os.makedirs(S2_folder, exist_ok=True)

# output csv
csv_output = os.path.join(PROJECT_ROOT, "csv")
os.makedirs(csv_output, exist_ok=True)

csv_master_s1 = os.path.join(csv_output, "S1_365.csv")
csv_master_s2 = os.path.join(csv_output, "S2_365.csv")


# inizializzazione API
try:
    print("Connessione a GEE...")
    ee.Initialize(project='impactmeshprova-498219') 
    print("🟢 Connessione a GEE riuscita")
except Exception as e:
    print("❌ ERRORE: richiesta autenticazione GEE. comando 'earthengine authenticate'")
    exit()

def scarica_geotiff(immagine_ee, bbox_ee, epsg, nome_file_out, cartella_dest):
    try:
        url = immagine_ee.getDownloadURL({
            'scale': TARGET_SCALE,
            'crs': epsg,
            'region': bbox_ee,
            'format': 'GEO_TIFF'
        })
        
        print(f"Download da GEE...")
        response = requests.get(url)
        
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            for file_name in z.namelist():
                z.extract(file_name, cartella_dest)
                
                vecchio_path = os.path.join(cartella_dest, file_name)
                nuovo_path = os.path.join(cartella_dest, f"{nome_file_out}_{file_name}")
                os.rename(vecchio_path, nuovo_path)
                
        print(f"🔵 Salvato in: {cartella_dest}")
    except Exception as e:
        print(f"❌ Errore durante il download: {e}")

def elabora_dataset(lista_csv_destinazioni, tipo_sensore, csv_output, max_workers=MAX_WORKERS):
    global contatore_righe
    serie_gia_fatte = set()
    file_esiste = os.path.exists(csv_output)
    
    if file_esiste:
        with open(csv_output, mode='r', encoding='utf-8') as f_in:
            reader = csv.reader(f_in)
            next(reader, None)
            for riga in reader:
                if riga: serie_gia_fatte.add(riga[0])
        print(f"🔵 Trovate {len(serie_gia_fatte)} serie già elaborate.")

    with open(csv_output, mode='a', encoding='utf-8', newline='') as f_out:
        writer = csv.writer(f_out)

        # intestazione
        if not file_esiste:
            if tipo_sensore == "SAR":
                writer.writerow(['Nome_Serie', 
                                 'Data_1_org_SAR', 'Data_2_org_SAR', 'Data_3_org_SAR', 'Data_4_org_SAR', 
                                 'Data_1_SAR', 'Data_2_SAR', 'Data_3_SAR', 'Data_4_SAR', 
                                 'ID_1_SAR', 'ID_2_SAR', 'ID_3_SAR', 'ID_4_SAR',
                                 'Lat_Centro_SAR', 'Lon_Centro_SAR', 'EPSG_SAR'])
            else:
                writer.writerow(['Nome_Serie', 
                                 'Data_1_org_OPT', 'Data_2_org_OPT', 'Data_3_org_OPT', 'Data_4_org_OPT', 
                                 'Data_1_OPT', 'Data_2_OPT', 'Data_3_OPT', 'Data_4_OPT', 
                                 'ID_1_OPT', 'ID_2_OPT', 'ID_3_OPT', 'ID_4_OPT',
                                 'Lat_Centro_OPT', 'Lon_Centro_OPT', 'MGRS_Tile_OPT', 'EPSG_OPT'])
    
        # funzione singolo thread
        def processa_singola_serie(riga, cartella_out, riga_num, totale):
            nome_serie = riga['Nome_Serie']
            suffix = "SAR" if tipo_sensore == "SAR" else "OPT"
            
            # centro per ritaglio
            lat, lon = float(riga[f'Lat_Centro_{suffix}']), float(riga[f'Lon_Centro_{suffix}'])
            punto_centro = ee.Geometry.Point([lon, lat])
            
            # lettura EPSG 
            epsg = riga.get(f'EPSG_{suffix}', 'EPSG:32632')
            if not epsg or epsg == "N/A":
                epsg = 'EPSG:32632'
                print(f"🔴 ERRORE: epsg non trovato")

            date_org_list = []
            date_effettive = ['NaN', 'NaN', 'NaN', 'NaN']
            id_effettivi = ['NaN', 'NaN', 'NaN', 'NaN']
                                
            print(f"[{tipo_sensore}] Riga {riga_num}/{totale} | Serie: {nome_serie}")
            
            # BBox 256x256
            bbox_esatto = punto_centro.buffer(RAGGIO_METRI).bounds()
            
            # ciclo sulle 4 date
            for step_temporale in range(1, 5):
                colonna_data = f'Data_{step_temporale}_org_{suffix}'
                data_stringa = riga.get(colonna_data, 'N/A')

                date_org_list.append(data_stringa)
                
                if data_stringa == 'N/A': continue
                                            
                # calcolo data target
                data_originale = datetime.strptime(data_stringa[:10], '%Y-%m-%d')
                data_target = data_originale - timedelta(days=ANNO_PRIMA)
                data_inizio = (data_target - timedelta(days=RANGE_GIORNI)).strftime('%Y-%m-%d')
                data_fine = (data_target + timedelta(days=RANGE_GIORNI)).strftime('%Y-%m-%d')
                
                nome_file_out = f"{nome_serie}_t{step_temporale}"
                                                                
                # SAR
                if tipo_sensore == "SAR":
                    s1_col = ee.ImageCollection('COPERNICUS/S1_GRD') \
                        .filterBounds(punto_centro) \
                        .filterDate(data_inizio, data_fine) \
                        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
                        .filter(ee.Filter.eq('instrumentMode', 'IW'))
                        
                    def calcola_nan(img):
                        mask = img.select(0).mask()
                        stats = mask.reduceRegion(reducer=ee.Reducer.mean(), geometry=bbox_esatto, scale=100, maxPixels=1e9)
                        valid_pct = ee.Number(stats.get(mask.bandNames().get(0))).multiply(100)
                        nan_pct = ee.Number(100).subtract(valid_pct)
                        return img.set('nan_percent', nan_pct)
                        
                    s1_filtrata = s1_col.map(calcola_nan).filter(ee.Filter.lt('nan_percent', PIXEL_PERCENT)).sort('nan_percent')
                    
                    try:
                        migliore_s1 = ee.Image(s1_filtrata.first())
                        id_img = migliore_s1.get('system:id').getInfo()
                        if id_img:
                            id_effettivi[step_temporale - 1] = id_img
                            nan_val = migliore_s1.get('nan_percent').getInfo()

                            timestamp = migliore_s1.get('system:time_start').getInfo()
                            if timestamp:
                                date_effettive[step_temporale - 1] = datetime.fromtimestamp(timestamp / 1000.0, tz=timezone.utc).strftime('%Y-%m-%d')
                            # print(f"🔵 Immagine t{step_temporale} trovata | Data: {date_effettive[step_temporale - 1]} | ID: {id_img[:15]}")
                            
                    except Exception as e:
                        # print(f"🔴 Nessuna immagine valida trovata | Data target: {data_target.strftime('%Y-%m-%d')}")
                        pass
                
                # OTTICO
                elif tipo_sensore == "OTTICO":
                    s2_col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
                        .filterBounds(punto_centro) \
                        .filterDate(data_inizio, data_fine) 
                    
                    def analizza_pixel_ottici(img):

                        # calcolo pixel non validi con banda rosso
                        mask_dati = img.select('B4').mask()
                        stats_nodata = mask_dati.reduceRegion(
                            reducer=ee.Reducer.mean(), 
                            geometry=bbox_esatto, 
                            scale=10, 
                            maxPixels=1e9
                        )
                        valid_pct = ee.Number(stats_nodata.get('B4')).multiply(100)
                        nan_pct = ee.Number(100).subtract(valid_pct)
                        
                        # calcolo copertura nuvolosa con Scene Classification Layer
                        scl = img.select('SCL')
                        # maschera nuvole
                        mask_nuvole = scl.eq(8).Or(scl.eq(9)).Or(scl.eq(10))
                        
                        # calcolo percentuale copertura maschera. Scale = 20 perche SCL ha risoluzione 20 metri
                        stats_nuvole = mask_nuvole.reduceRegion(
                            reducer=ee.Reducer.mean(),
                            geometry=bbox_esatto,
                            scale=20, 
                            maxPixels=1e9
                        )
                        cloud_pct = ee.Number(stats_nuvole.get('SCL')).multiply(100)
                        
                        return img.set('nan_percent', nan_pct).set('roi_cloud_percent', cloud_pct)
                        
                    # ordine per copertura nuvolosa
                    s2_filtrata = s2_col.map(analizza_pixel_ottici)\
                        .filter(ee.Filter.lt('nan_percent', PIXEL_PERCENT))\
                        .filter(ee.Filter.lt('roi_cloud_percent', CLOUDY_PERCENTAGE))\
                        .sort('roi_cloud_percent')
                        
                    try:
                        migliore_s2 = ee.Image(s2_filtrata.first())
                        id_img = migliore_s2.get('system:id').getInfo()
                        if id_img:
                            id_effettivi[step_temporale - 1] = id_img
                            cloud_val = migliore_s2.get('CLOUDY_PIXEL_PERCENTAGE').getInfo()

                            timestamp = migliore_s2.get('system:time_start').getInfo()
                            if timestamp:
                                date_effettive[step_temporale - 1] = datetime.fromtimestamp(timestamp / 1000.0, tz=timezone.utc).strftime('%Y-%m-%d')
                            # print(f"🔵 Immagine t{step_temporale} trovata | Data: {date_effettive[step_temporale - 1]} | ID: {id_img[:15]}")
                            
                    except Exception as e:
                        # print(f"🔴 Nessuna immagine valida trovata | Data target: {data_target.strftime('%Y-%m-%d')}")
                        pass

            if tipo_sensore == "SAR":
                nuova_riga = [nome_serie] + date_org_list + date_effettive + id_effettivi + [lat, lon, epsg]
            else:
                mgrs_tile = riga.get('MGRS_Tile', 'N/A')
                nuova_riga = [nome_serie] + date_org_list + date_effettive + id_effettivi + [lat, lon, mgrs_tile, epsg]
            
            return nuova_riga

        # raccolta task
        tasks_da_fare = []
        for csv_path, cartella_out in lista_csv_destinazioni:
            if not os.path.exists(csv_path):
                print(f"\n❌ ATTENZIONE: file {csv_path} non esiste")
                continue
                
            with open(csv_path, mode='r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                # Uso enumerate partendo da 1 per contare le righe
                for riga_num, riga in enumerate(reader, start=1):
                    if riga['Nome_Serie'] not in serie_gia_fatte:
                        # Aggiungo riga_num come terzo elemento della tupla
                        tasks_da_fare.append((riga, cartella_out, riga_num))
                        
        totale_tasks = len(tasks_da_fare)

        # esecuzione parallela
        print(f"\nAvvio esecuzione in parallelo per {len(tasks_da_fare)} serie...")
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # invio dati ai thread: passo task[0] (riga), task[1] (cartella), task[2] (riga_num) e totale_tasks
            futures = {
                executor.submit(processa_singola_serie, task[0], task[1], task[2], totale_tasks): task 
                for task in tasks_da_fare
            }
            
            # risultati
            for future in as_completed(futures):
                try:
                    risultato_riga = future.result()
                    
                    # scrive un thread alla volta
                    with csv_lock:
                        writer.writerow(risultato_riga)  
                        if not CONTAINER:
                            f_out.flush() 
                            os.fsync(f_out.fileno()) 
                except Exception as e:
                    print(f"🔴 Errore in un thread: {e}")
    
    print(f"\n🟢 Finito. Salvato CSV in: {csv_output}")


# --- LISTE DI ESECUZIONE ---
# Mappiamo i file CSV alle loro cartelle di destinazione
task_sar = [
    (input_csv_s1, S1_folder)
]

task_opt = [
    (input_csv_s2, S2_folder)
]

print("AVVIO PROCESSO DI RICERCA SAR E OTTICO (-365 giorni)")


elabora_dataset(task_sar, "SAR", csv_master_s1)
elabora_dataset(task_opt, "OTTICO", csv_master_s2)

print("\n✔️ FINITO")