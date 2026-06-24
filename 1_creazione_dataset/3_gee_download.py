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
SOLO_VERIFICA = True    # se true non fa download img ma solo ricerca img valide
CONTAINER = False       # true se lo lancio da container, false in locale
MAX_WORKERS = 15

PIXEL_PERCENT = 5       # % pixel Nan in img SAR e OTTICHE
CLOUDY_PERCENTAGE = 20  # % copertura nuvolosa
TARGET_SCALE = 10       # risoluzione 10 metri 
RAGGIO_METRI = 1280     # raggio di 1280m (2560m/2), box 256x256 (256 px * risoluzione 10m = 2560)
RANGE_GIORNI = 10       # +- giorni intorno alla data
ANNO_PRIMA = 365        # ricerca tot anni prima della data originale (in giorni)


# percorsi csv
clean_s1_train = os.path.join(PROJECT_ROOT, "csv", "train", "clean", "S1RTC_train.csv")
clean_s2_train_1 = os.path.join(PROJECT_ROOT, "csv", "train", "clean", "S2L2A_train_1.csv")
clean_s2_train_2 = os.path.join(PROJECT_ROOT, "csv", "train", "clean", "S2L2A_train_2.csv")
clean_s2_train_3 = os.path.join(PROJECT_ROOT, "csv", "train", "clean", "S2L2A_train_3.csv")

clean_s1_val = os.path.join(PROJECT_ROOT, "csv", "val", "clean", "S1RTC_val.csv")
clean_s2_val = os.path.join(PROJECT_ROOT, "csv", "val", "clean", "S2L2A_val.csv")

clean_s1_test = os.path.join(PROJECT_ROOT, "csv", "test", "clean", "S1RTC_test.csv")
clean_s2_test = os.path.join(PROJECT_ROOT, "csv", "test", "clean", "S2L2A_test.csv")

# output dataset
train_S1_folder = os.path.join(PROJECT_ROOT, "data", "GEE_download", "train", "S1")
train_S2_folder = os.path.join(PROJECT_ROOT, "data", "GEE_download", "train", "S2")

val_S1_folder = os.path.join(PROJECT_ROOT, "data", "GEE_download", "val", "S1")
val_S2_folder = os.path.join(PROJECT_ROOT, "data", "GEE_download", "val", "S2")

test_S1_folder = os.path.join(PROJECT_ROOT, "data", "GEE_download", "test", "S1")
test_S2_folder = os.path.join(PROJECT_ROOT, "data", "GEE_download", "test", "S2")

# output csv
csv_output = os.path.join(PROJECT_ROOT, "csv")
os.makedirs(csv_output, exist_ok=True)

csv_master_s1 = os.path.join(csv_output, "series_gee_S1.csv")
csv_master_s2 = os.path.join(csv_output, "series_gee_S2.csv")


os.makedirs(train_S1_folder, exist_ok=True)
os.makedirs(train_S2_folder, exist_ok=True)
os.makedirs(val_S1_folder, exist_ok=True)
os.makedirs(val_S2_folder, exist_ok=True)
os.makedirs(test_S1_folder, exist_ok=True)
os.makedirs(test_S2_folder, exist_ok=True)



# inizializzazione API
try:
    print("Connessione a GEE...")
    ee.Initialize(project='impactmeshprova-498219') 
    print("🟢 Connessione a GEE riuscita")
except Exception as e:
    print("❌ ERRORE: richiesta autenticazione GEE. comando 'earthengine authenticate'")
    exit()

if(SOLO_VERIFICA):
    print(f"Solo verifica disponibilità immagini, no download")
else:
    print(f"Download delle immagini")

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

        # Scrivi l'intestazione SOLO se il file è nuovo
        if not file_esiste:
            if tipo_sensore == "SAR":
                writer.writerow(['Nome_Serie', 'Data_1', 'Data_2', 'Data_3', 'Data_4', 'Lat_Centro', 'Lon_Centro', 'EPSG'])
            else:
                writer.writerow(['Nome_Serie', 'Data_1', 'Data_2', 'Data_3', 'Data_4', 'Lat_Centro', 'Lon_Centro', 'MGRS_Tile', 'EPSG'])
    
        # 1. CREIAMO LA FUNZIONE WORKER PER IL SINGOLO THREAD
        def processa_singola_serie(riga, cartella_out):
            nome_serie = riga['Nome_Serie']
            
            # centro per ritaglio
            lat, lon = float(riga['Lat_Centro']), float(riga['Lon_Centro'])
            punto_centro = ee.Geometry.Point([lon, lat])
            
            # lettura EPSG 
            epsg = riga.get('EPSG', 'EPSG:32632')
            if not epsg or epsg == "N/A":
                epsg = 'EPSG:32632'
                print(f"🔴 ERRORE: epsg non trovato")

            date_effettive = ['NaN', 'NaN', 'NaN', 'NaN']
                                
            print(f"[{tipo_sensore}] Processo Serie: {nome_serie}")
            
            # BBox 256x256
            bbox_esatto = punto_centro.buffer(RAGGIO_METRI).bounds()
            
            # CICLO SULLE 4 DATE POSSIBILI (Data_1, Data_2, Data_3, Data_4)
            for step_temporale in range(1, 5):
                colonna_data = f'Data_{step_temporale}'
                data_stringa = riga.get(colonna_data, 'N/A')
                
                if data_stringa == 'N/A': continue
                                            
                # Calcolo data storica (365 giorni prima)
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
                        id_img = migliore_s1.id().getInfo()
                        if id_img:
                            nan_val = migliore_s1.get('nan_percent').getInfo()

                            timestamp = migliore_s1.get('system:time_start').getInfo()
                            if timestamp:
                                date_effettive[step_temporale - 1] = datetime.fromtimestamp(timestamp / 1000.0, tz=timezone.utc).strftime('%Y-%m-%d')
                            print(f"🔵 Immagine t{step_temporale} trovata | Data: {date_effettive[step_temporale - 1]} | ID: {id_img[:15]}")
                            
                            if not SOLO_VERIFICA:
                                img_da_scaricare = migliore_s1.select(['VV', 'VH'])
                                scarica_geotiff(img_da_scaricare, bbox_esatto, epsg, nome_file_out, cartella_out)
                    except Exception as e:
                        print(f"🔴 Nessuna immagine valida trovata | Data target: {data_target.strftime('%Y-%m-%d')}")
                
                # OTTICO
                elif tipo_sensore == "OTTICO":
                    s2_col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
                        .filterBounds(punto_centro) \
                        .filterDate(data_inizio, data_fine) 
                    
                    def analizza_pixel_ottici(img):
                        # 1. CALCOLO PIXEL NERI (NoData)
                        # Usiamo la banda B4 (Rosso) per vedere dove ci sono dati validi
                        mask_dati = img.select('B4').mask()
                        stats_nodata = mask_dati.reduceRegion(
                            reducer=ee.Reducer.mean(), 
                            geometry=bbox_esatto, 
                            scale=10, 
                            maxPixels=1e9
                        )
                        valid_pct = ee.Number(stats_nodata.get('B4')).multiply(100)
                        nan_pct = ee.Number(100).subtract(valid_pct)
                        
                        # 2. CALCOLO NUVOLOSITA' LOCALE (nel BBox)
                        # Selezioniamo la banda di classificazione SCL
                        scl = img.select('SCL')
                        # Creiamo una maschera dove i pixel valgono 1 se sono nuvole (8, 9 o 10), altrimenti 0
                        mask_nuvole = scl.eq(8).Or(scl.eq(9)).Or(scl.eq(10))
                        
                        # Calcoliamo la media di questi 1 e 0 nel nostro BBox per avere la percentuale
                        # Nota: la banda SCL è nativamente a 20m di risoluzione, quindi usiamo scale=20
                        stats_nuvole = mask_nuvole.reduceRegion(
                            reducer=ee.Reducer.mean(),
                            geometry=bbox_esatto,
                            scale=20, 
                            maxPixels=1e9
                        )
                        cloud_pct = ee.Number(stats_nuvole.get('SCL')).multiply(100)
                        
                        # Salviamo entrambi i valori come proprietà dell'immagine
                        return img.set('nan_percent', nan_pct).set('roi_cloud_percent', cloud_pct)
                        
                    # Applichiamo l'analisi, filtriamo per nero, filtriamo per nuvole e ordiniamo per nuvole
                    s2_filtrata = s2_col.map(analizza_pixel_ottici)\
                        .filter(ee.Filter.lt('nan_percent', PIXEL_PERCENT))\
                        .filter(ee.Filter.lt('roi_cloud_percent', CLOUDY_PERCENTAGE))\
                        .sort('roi_cloud_percent')
                        
                    try:
                        migliore_s2 = ee.Image(s2_col.first())
                        id_img = migliore_s2.id().getInfo()
                        if id_img:
                            cloud_val = migliore_s2.get('CLOUDY_PIXEL_PERCENTAGE').getInfo()

                            timestamp = migliore_s2.get('system:time_start').getInfo()
                            if timestamp:
                                date_effettive[step_temporale - 1] = datetime.fromtimestamp(timestamp / 1000.0, tz=timezone.utc).strftime('%Y-%m-%d')
                            print(f"🔵 Immagine t{step_temporale} trovata | Data: {date_effettive[step_temporale - 1]} | ID: {id_img[:15]}")
                            
                            if not SOLO_VERIFICA:
                                img_da_scaricare = migliore_s2.select(['B4', 'B3', 'B2', 'B8'])
                                scarica_geotiff(img_da_scaricare, bbox_esatto, epsg, nome_file_out, cartella_out)
                            else:
                                pass # Tolto il print di saltato per non sporcare i log del multithread
                    except Exception as e:
                        print(f"🔴 Nessuna immagine valida trovata | Data target: {data_target.strftime('%Y-%m-%d')}")

            if tipo_sensore == "SAR":
                nuova_riga = [nome_serie] + date_effettive + [lat, lon, epsg]
            else:
                mgrs_tile = riga.get('MGRS_Tile', 'N/A')
                nuova_riga = [nome_serie] + date_effettive + [lat, lon, mgrs_tile, epsg]
            
            return nuova_riga

        # 2. RACCOLTA TASK DAI CSV
        tasks_da_fare = []
        for csv_path, cartella_out in lista_csv_destinazioni:
            if not os.path.exists(csv_path):
                print(f"\n❌ ATTENZIONE: Il file {csv_path} non esiste")
                continue
                
            with open(csv_path, mode='r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for riga in reader:
                    if riga['Nome_Serie'] not in serie_gia_fatte:
                        tasks_da_fare.append((riga, cartella_out))

        # 3. ESECUZIONE MULTITHREAD (15 Lavoratori in parallelo)
        print(f"\n🚀 Avvio elaborazione in parallelo per {len(tasks_da_fare)} serie...")
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Invio tutto ai thread
            futures = {executor.submit(processa_singola_serie, task[0], task[1]): task for task in tasks_da_fare}
            
            # Catturo i risultati man mano che finiscono
            for future in as_completed(futures):
                try:
                    risultato_riga = future.result()
                    
                    # PROTEZIONE: Solo 1 thread alla volta scrive e forza il disco
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
    (clean_s1_train, train_S1_folder),
    (clean_s1_val, val_S1_folder),
    (clean_s1_test, test_S1_folder)
]

task_opt = [
    (clean_s2_train_1, train_S2_folder),
    (clean_s2_train_2, train_S2_folder),
    (clean_s2_train_3, train_S2_folder),
    (clean_s2_val, val_S2_folder),
    (clean_s2_test, test_S2_folder)
]

print("AVVIO PROCESSO DI RICERCA SAR E OTTICO")


elabora_dataset(task_sar, "SAR", csv_master_s1)
elabora_dataset(task_opt, "OTTICO", csv_master_s2)

print("\n✔️ FINITO")