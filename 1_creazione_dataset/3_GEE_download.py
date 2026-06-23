import ee
import csv
import os
import requests
import zipfile
import io
import json
from datetime import datetime, timedelta, timezone

# configurazione percorsi
script_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(script_dir) # /workspace/1_creazione_dataset


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

csv_master_s1 = os.path.join(csv_output, "GEE_S1.csv")
csv_master_s2 = os.path.join(csv_output, "GEE_S2.csv")

# TRUE per verifica disponibilità immagini senza download
SOLO_VERIFICA = True 
SAR_PIXEL_PERCENT = 5
CLOUDY_PERCENTAGE = 15

os.makedirs(train_S1_folder, exist_ok=True)
os.makedirs(train_S2_folder, exist_ok=True)
os.makedirs(val_S1_folder, exist_ok=True)
os.makedirs(val_S2_folder, exist_ok=True)
os.makedirs(test_S1_folder, exist_ok=True)
os.makedirs(test_S2_folder, exist_ok=True)

TARGET_SCALE = 10 # risoluzione 10 metri 
RAGGIO_METRI = 1280 # raggio di 1280m, box 256x256

# inizializzazione API
try:
    print("🟡 Connessione a GEE...")
    ee.Initialize(project='impactmeshprova-498219') 
    print("🟢 Connessione a GEE riuscita")
except Exception as e:
    print("🔴 ERRORE: richiesta autenticazione GEE. Esegui 'earthengine authenticate'")
    exit()

if(SOLO_VERIFICA):
    print(f"🟡 Solo verifica disponibilità immagini, no download")
else:
    print(f"🟢 Download delle immagini")

def scarica_geotiff(immagine_ee, bbox_ee, epsg, nome_file_out, cartella_dest):
    try:
        url = immagine_ee.getDownloadURL({
            'scale': TARGET_SCALE,
            'crs': epsg,
            'region': bbox_ee,
            'format': 'GEO_TIFF'
        })
        
        print(f"🟡 Download da GEE...")
        response = requests.get(url)
        
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            for file_name in z.namelist():
                z.extract(file_name, cartella_dest)
                
                vecchio_path = os.path.join(cartella_dest, file_name)
                nuovo_path = os.path.join(cartella_dest, f"{nome_file_out}_{file_name}")
                os.rename(vecchio_path, nuovo_path)
                
        print(f"🔵 Salvato in: {cartella_dest}")
    except Exception as e:
        print(f"🔴 Errore durante il download: {e}")

def elabora_dataset(lista_csv_destinazioni, tipo_sensore, csv_output):

    serie_gia_fatte = set()
    file_esiste = os.path.exists(csv_output)

    if file_esiste:
        print(f"🟡 Trovato file esistente {csv_output}, conto serie salvate")
        with open(csv_output, mode='r', encoding='utf-8') as f_in:
            reader = csv.reader(f_in)
            next(reader, None)  # salta intestazione
            for riga in reader:
                if len(riga) > 0:
                    serie_gia_fatte.add(riga[0]) 
        print(f"🔵 Trovate {len(serie_gia_fatte)} serie già elaborate")

    with open(csv_output, mode='w', encoding='utf-8', newline='') as f_out:
        writer = csv.writer(f_out)
        
        # intestazione sar
        if tipo_sensore == "SAR":
            writer.writerow(['Nome_Serie', 'Data_1', 'Data_2', 'Data_3', 'Data_4', 'Lat_Centro', 'Lon_Centro', 'EPSG'])
        # intestazione ottico
        else:
            writer.writerow(['Nome_Serie', 'Data_1', 'Data_2', 'Data_3', 'Data_4', 'Lat_Centro', 'Lon_Centro', 'MGRS_Tile', 'EPSG'])
    
        # scorro tutti i csv per sensore
        for csv_path, cartella_out in lista_csv_destinazioni:
            if not os.path.exists(csv_path):
                print(f"\n🔴 ATTENZIONE: Il file {csv_path} non esiste")
                continue
                
            print(f"\napertura file: {os.path.basename(csv_path)}")
            
            with open(csv_path, mode='r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for riga in reader:
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
                                       
                    print(f"\n[{tipo_sensore}] Serie: {nome_serie}")
                    
                    # BBox 256x256
                    bbox_esatto = punto_centro.buffer(RAGGIO_METRI).bounds()
                    
                    # CICLO SULLE 4 DATE POSSIBILI (Data_1, Data_2, Data_3, Data_4)
                    for step_temporale in range(1, 5):
                        colonna_data = f'Data_{step_temporale}'
                        data_stringa = riga.get(colonna_data, 'N/A')
                                            
                        # Calcolo data storica (365 giorni prima)
                        data_originale = datetime.strptime(data_stringa[:10], '%Y-%m-%d')
                        data_target = data_originale - timedelta(days=365)
                        data_inizio = (data_target - timedelta(days=7)).strftime('%Y-%m-%d')
                        data_fine = (data_target + timedelta(days=7)).strftime('%Y-%m-%d')
                        
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
                                
                            s1_filtrata = s1_col.map(calcola_nan).filter(ee.Filter.lt('nan_percent', 5)).sort('nan_percent')
                            
                            try:
                                migliore_s1 = ee.Image(s1_filtrata.first())
                                id_img = migliore_s1.id().getInfo()
                                if id_img:
                                    nan_val = migliore_s1.get('nan_percent').getInfo()

                                    timestamp = migliore_s1.get('system:time_start').getInfo()
                                    if timestamp:
                                        date_effettive[step_temporale - 1] = datetime.fromtimestamp(timestamp / 1000.0, tz=timezone.utc).strftime('%Y-%m-%d')
                                    print(f"🔵 Immagine t{step_temporale} trovata | Data target: {data_target.strftime('%Y-%m-%d')} | Data: {date_effettive[step_temporale - 1]} | ID: {id_img[:15]}")
                                    
                                    if not SOLO_VERIFICA:
                                        img_da_scaricare = migliore_s1.select(['VV', 'VH'])
                                        scarica_geotiff(img_da_scaricare, bbox_esatto, epsg, nome_file_out, cartella_out)
                            except Exception as e:
                                print(f"🔴 Nessuna immagine valida trovata | Data target: {data_target.strftime('%Y-%m-%d')}")
                        
                        # OTTICO
                        elif tipo_sensore == "OTTICO":
                            s2_col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
                                .filterBounds(punto_centro) \
                                .filterDate(data_inizio, data_fine) \
                                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 15)) \
                                .sort('CLOUDY_PIXEL_PERCENTAGE')
                                
                            try:
                                migliore_s2 = ee.Image(s2_col.first())
                                id_img = migliore_s2.id().getInfo()
                                if id_img:
                                    cloud_val = migliore_s2.get('CLOUDY_PIXEL_PERCENTAGE').getInfo()

                                    timestamp = migliore_s2.get('system:time_start').getInfo()
                                    if timestamp:
                                        date_effettive[step_temporale - 1] = datetime.fromtimestamp(timestamp / 1000.0, tz=timezone.utc).strftime('%Y-%m-%d')
                                    print(f"🔵 Immagine t{step_temporale} trovata | Data target: {data_target.strftime('%Y-%m-%d')} | Data: {date_effettive[step_temporale - 1]} | ID: {id_img[:15]}")
                                    
                                    if not SOLO_VERIFICA:
                                        img_da_scaricare = migliore_s2.select(['B4', 'B3', 'B2', 'B8'])
                                        scarica_geotiff(img_da_scaricare, bbox_esatto, epsg, nome_file_out, cartella_out)
                                    else:
                                        print("🟡 Modalità SOLO VERIFICA: Download saltato.")
                            except Exception as e:
                                print(f"🔴 Nessuna immagine valida trovata | Data target: {data_target.strftime('%Y-%m-%d')}")

                    if tipo_sensore == "SAR":
                        nuova_riga = [nome_serie] + date_effettive + [lat, lon, epsg]
                    else:
                        mgrs_tile = riga.get('MGRS_Tile', 'N/A')
                        nuova_riga = [nome_serie] + date_effettive + [lat, lon, mgrs_tile, epsg]
                    
                    writer.writerow(nuova_riga)            
    
    print(f"\n🟢 Terminato! Salvato CSV in: {csv_output}")


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

print("\n🟢 TUTTE LE OPERAZIONI SONO CONCLUSE!")