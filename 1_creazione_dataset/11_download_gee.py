# download delle immagini sar e ottiche da gee tramite complete_series.csv in cartella /data/dataset/ S1 o S2
# no controlli su copertura nuvolosa e pixel perche csv contiene solo date valide
#  
# intervallo di download di serie in complete_series.csv per controllo a campione delle immagini scaricate
# salva sempre versione png della prima serie dell'intervallo (sia sar che ottica)
# cartelle PNG_intervallo in /data/dataset/
# 
# INPUT:
#   complete_series.csv
#
# OUTPUT:
#   
#   immagini.tiff
#   file.png


import ee
import pandas as pd
import os
import requests
import zipfile
import io
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import rasterio
import matplotlib.pyplot as plt
import numpy as np

# no apertura finestre grafiche
plt.switch_backend('Agg')

# percorso workspace
script_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(script_dir) 

# intervallo righe complete.series.csv (salta di default intestazione, riga 1 = prima serie valida, riga 10 compresa)
ROW_START = 11
ROW_END = 20

MAX_WORKERS = 15
TARGET_SCALE = 10
RAGGIO_METRI = 1280  

CSV_PATH = os.path.join(PROJECT_ROOT, "csv", "definitivo.csv")

# cartelle
S1_FOLDER = os.path.join(PROJECT_ROOT, "data", "dataset", "S1")
S2_FOLDER = os.path.join(PROJECT_ROOT, "data", "dataset", "S2")

png_folder_name = f"PNG_{ROW_START}_{ROW_END}"
BASE_PNG_FOLDER = os.path.join(PROJECT_ROOT, "data", "dataset", png_folder_name)

os.makedirs(S1_FOLDER, exist_ok=True)
os.makedirs(S2_FOLDER, exist_ok=True)
os.makedirs(BASE_PNG_FOLDER, exist_ok=True)

# codice simile a 4_img_730_giorni.py e 7_img_balance.py
try:
    print("Connessione a GEE in corso...")
    ee.Initialize(project='impactmeshprova-498219') 
    print("🟢 Connessione a GEE riuscita!\n")
except Exception as e:
    print("❌ ERRORE: impossibile autenticarsi a GEE.")
    exit()

# normalizzazione colori per visualizzazione png
def normalize(array):
    array_valido = array[~np.isnan(array)]
    if array_valido.size == 0:
        return np.zeros_like(array)
        
    p_min, p_max = np.percentile(array_valido, 2), np.percentile(array_valido, 98)
    if p_max - p_min == 0:
        return np.zeros_like(array)
        
    array_tagliato = np.clip(array, p_min, p_max)
    return (array_tagliato - p_min) / (p_max - p_min)

def visualizza_serie_completa(paths_sar, paths_opt, nome_serie, nome_output):
    try:
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle(f"Serie: {nome_serie}", fontsize=16)

        for step in range(4):
            # SAR (Riga 0)
            ax_sar = axes[0, step]
            if paths_sar[step]:
                with rasterio.open(paths_sar[step]) as src:
                    img = src.read(1).astype(float)
                    if src.nodata is not None: img[img == src.nodata] = np.nan
                    ax_sar.imshow(normalize(img), cmap='gray')
            ax_sar.set_title(f"SAR t{step+1}")
            ax_sar.axis('off')

            # OTTICO (Riga 1)
            ax_opt = axes[1, step]
            if paths_opt[step]:
                with rasterio.open(paths_opt[step]) as src:
                    # 1. Lettura dati grezzi
                    r_raw = src.read(3).astype(float)
                    g_raw = src.read(2).astype(float)
                    b_raw = src.read(1).astype(float)
                    
                    # 2. Gestione NoData (stessa di prima)
                    if src.nodata is not None:
                        r_raw[r_raw == src.nodata] = np.nan
                        g_raw[g_raw == src.nodata] = np.nan
                        b_raw[b_raw == src.nodata] = np.nan
                    else:
                        r_raw[r_raw == 0] = np.nan
                        g_raw[g_raw == 0] = np.nan
                        b_raw[b_raw == 0] = np.nan

                    # 3. Uniamo i canali nel formato corretto per l'immagine
                    rgb_stack = np.dstack((r_raw, g_raw, b_raw))
                    
                    # 4. Normalizzazione FISSA per Sentinel-2 (Colori Naturali)
                    # Tagliamo i valori estremi a 3000 e dividiamo per 3000 per avere la scala 0.0 - 1.0
                    rgb_norm = np.clip(rgb_stack, 0, 3000) / 3000.0
                    
                    ax_opt.imshow(rgb_norm)
            ax_opt.set_title(f"OPT t{step+1}")
            ax_opt.axis('off')

        plt.tight_layout()
        plt.savefig(nome_output, bbox_inches='tight', dpi=150) # dpi 150 per non fare file enormi
        plt.close()
    except Exception as e:
        print(f"🔴 Errore creazione PNG riassuntivo per {nome_serie}: {e}")

def scarica_geotiff(immagine_ee, bbox_ee, epsg, nome_file_out, cartella_dest):
    percorso_finale_zip = os.path.join(cartella_dest, f"{nome_file_out}.zip")
    
    # controllo esistenza file zip
    if os.path.exists(percorso_finale_zip) and os.path.getsize(percorso_finale_zip) > 0:
        print(f"⏭️  File già presente, download saltato: {nome_file_out}.zip")
        return percorso_finale_zip

    try:
        url = immagine_ee.getDownloadURL({
            'scale': TARGET_SCALE,
            'crs': epsg,
            'region': bbox_ee,
            'format': 'GEO_TIFF'
        })
        
        response = requests.get(url, timeout=60)
        if response.status_code != 200: return None
            
        file_bytes = io.BytesIO(response.content)
        
        # Scriviamo direttamente un file .zip compresso sul disco
        with zipfile.ZipFile(percorso_finale_zip, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z_out:
            if zipfile.is_zipfile(file_bytes):
                with zipfile.ZipFile(file_bytes) as z_in:
                    for file_name in z_in.namelist():
                        if file_name.endswith('.tif'):
                            tif_data = z_in.read(file_name)
                            z_out.writestr(f"{nome_file_out}.tif", tif_data)
                            return percorso_finale_zip
            else:
                z_out.writestr(f"{nome_file_out}.tif", response.content)
                return percorso_finale_zip
                
    except Exception as e:
        if os.path.exists(percorso_finale_zip):
            os.remove(percorso_finale_zip)
        return None

def processa_singola_serie(riga, genera_png, riga_num):
    nome_serie = riga['Nome_Serie']
    
    # box SAR
    lat_sar, lon_sar = float(riga['Lat_Centro_SAR']), float(riga['Lon_Centro_SAR'])
    epsg_sar = riga.get('EPSG_SAR', 'EPSG:32632')
    bbox_sar = ee.Geometry.Point([lon_sar, lat_sar]).buffer(RAGGIO_METRI).bounds()

    # box OTTICO
    lat_opt, lon_opt = float(riga['Lat_Centro_OPT']), float(riga['Lon_Centro_OPT'])
    epsg_opt = riga.get('EPSG_OPT', 'EPSG:32632')
    bbox_opt = ee.Geometry.Point([lon_opt, lat_opt]).buffer(RAGGIO_METRI).bounds()
    
    # liste per creazione immagine png
    paths_per_png_sar = []
    paths_per_png_opt = []
    
    for step in range(1, 5):
        
        # SAR 
        path_zip_sar = None
        nome_out_sar = f"{nome_serie}_SAR_t{step}"
        data_str_sar = str(riga.get(f'Data_{step}_SAR', '')).strip()[:10]
        
        if data_str_sar and data_str_sar != 'nan':
            data_esatta = datetime.strptime(data_str_sar, '%Y-%m-%d')
            d_inizio = (data_esatta - timedelta(days=1)).strftime('%Y-%m-%d')
            d_fine = (data_esatta + timedelta(days=1)).strftime('%Y-%m-%d')
            
            s1_img = ee.ImageCollection('COPERNICUS/S1_GRD') \
                .filterBounds(bbox_sar).filterDate(d_inizio, d_fine) \
                .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
                .filter(ee.Filter.eq('instrumentMode', 'IW')) \
                .mosaic().select(['VV', 'VH'])
                
            path_zip_sar = scarica_geotiff(s1_img, bbox_sar, epsg_sar, nome_out_sar, S1_FOLDER)
            
        if genera_png:
            paths_per_png_sar.append(f"zip://{path_zip_sar}!{nome_out_sar}.tif" if path_zip_sar else None)

        # OTTICO
        path_zip_opt = None
        nome_out_opt = f"{nome_serie}_OPT_t{step}"
        data_str_opt = str(riga.get(f'Data_{step}_OPT', '')).strip()[:10]
        
        if data_str_opt and data_str_opt != 'nan':
            data_esatta = datetime.strptime(data_str_opt, '%Y-%m-%d')
            d_inizio = (data_esatta - timedelta(days=1)).strftime('%Y-%m-%d')
            d_fine = (data_esatta + timedelta(days=1)).strftime('%Y-%m-%d')
            
            s2_img = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
                .filterBounds(bbox_opt) \
                .filterDate(d_inizio, d_fine) \
                .first() \
                .select(['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12'])
                
            path_zip_opt = scarica_geotiff(s2_img, bbox_opt, epsg_opt, nome_out_opt, S2_FOLDER)
            
        if genera_png:
            paths_per_png_opt.append(f"zip://{path_zip_opt}!{nome_out_opt}.tif" if path_zip_opt else None)      

    if genera_png:
        nome_output = os.path.join(BASE_PNG_FOLDER, f"{nome_serie}_riassunto.png")
        print(f"Generazione PNG per: {nome_serie}")
        visualizza_serie_completa(paths_per_png_sar, paths_per_png_opt, nome_serie, nome_output)

    print(f"Completata: {nome_serie} (Serie: {nome_serie})")
    return True

if __name__ == "__main__":
    if not os.path.exists(CSV_PATH):
        print(f"❌ File CSV non trovato: {CSV_PATH}")
        exit()
        
    df = pd.read_csv(CSV_PATH)
    
    # filtro intervallo
    df_subset = df.iloc[ROW_START-1 : ROW_END]
    tasks = df_subset.to_dict('records')
    
    if len(tasks) == 0:
        print("Intervallo di righe non valido o vuoto. Controlla ROW_START e ROW_END.")
        exit()
    
    print(f"Download da riga {ROW_START} alla {ROW_END}")
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # true solo al primo task per stampare png
        futures = []
        for i, task in enumerate(tasks):
            riga_assoluta = ROW_START + i
            fare_png = (i == 0) or (riga_assoluta % 100 == 0)
            futures.append(executor.submit(processa_singola_serie, task, fare_png, riga_assoluta))
        
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Errore in un thread: {e}")
                
    print(f"\nFINE (Righe {ROW_START}-{ROW_END}). PNG salvate in '{png_folder_name}'!")