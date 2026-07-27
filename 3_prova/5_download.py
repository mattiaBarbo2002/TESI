# download delle immagini sar e ottiche da gee tramite complete_series.csv in cartella /data/dataset/ S1 o S2
# Usa ID specifici validati e una singola Bounding Box comune per allineamento spaziale perfetto.
#
# INPUT:
#   complete_series.csv
#
# OUTPUT:
#   immagini .tif compresse in .zip
#   file .png riassuntivi a campione

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
import time

# no apertura finestre grafiche
plt.switch_backend('Agg')

# percorso workspace
script_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(script_dir) 

# intervallo righe
ROW_START = 5860
ROW_END = 5860
NR_PNG = 1             # numero totale righe / NR_PNG

MAX_WORKERS = 10          
TARGET_SCALE = 10
RAGGIO_METRI = 1280  

CSV_PATH = os.path.join(PROJECT_ROOT, "csv", "complete_series.csv")

# cartelle
S1_FOLDER = os.path.join(PROJECT_ROOT, "data", "dataset", "S1")
S2_FOLDER = os.path.join(PROJECT_ROOT, "data", "dataset", "S2")

png_folder_name = f"PNG_{ROW_START}_{ROW_END}"
BASE_PNG_FOLDER = os.path.join(PROJECT_ROOT, "data", "dataset", png_folder_name)

os.makedirs(S1_FOLDER, exist_ok=True)
os.makedirs(S2_FOLDER, exist_ok=True)
os.makedirs(BASE_PNG_FOLDER, exist_ok=True)

try:
    print("Connessione a GEE in corso...")
    ee.Initialize(project='impactmeshprova-498219') 
    print("🟢 Connessione a GEE riuscita!\n")
except Exception as e:
    print("❌ ERRORE: impossibile autenticarsi a GEE.")
    exit()

def normalize(array):
    array_valido = array[~np.isnan(array)]
    if array_valido.size == 0:
        return np.zeros_like(array)
        
    p_min, p_max = np.percentile(array_valido, 2), np.percentile(array_valido, 98)
    if p_max - p_min == 0:
        return np.zeros_like(array)
        
    array_tagliato = np.clip(array, p_min, p_max)
    return (array_tagliato - p_min) / (p_max - p_min)

def visualizza_serie_completa(paths_sar, paths_opt, nome_serie, date_sar, date_opt, nome_output):
    try:
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle(f"Serie: {nome_serie}", fontsize=16)

        for step in range(4):
            # SAR (Riga 0)
            ax_sar = axes[0, step]
            if paths_sar[step]:
                with rasterio.open(paths_sar[step]) as src:
                    img = src.read(1).astype(float)
                    
                    # Creazione maschera per i NaN del SAR
                    mask_nan_sar = np.isnan(img)
                    if src.nodata is not None: 
                        mask_nan_sar |= (img == src.nodata)
                    else:
                        mask_nan_sar |= (img == 0)
                        
                    img[mask_nan_sar] = np.nan
                    
                    # Diciamo a matplotlib di disegnare i NaN del SAR in Magenta
                    cmap_custom = plt.cm.gray.copy()
                    cmap_custom.set_bad(color='#ff00ff')
                    
                    ax_sar.imshow(normalize(img), cmap=cmap_custom)
            ax_sar.set_title(f"SAR t{step+1}\n{date_sar[step]}")
            ax_sar.axis('off')

            # OTTICO (Riga 1)
            ax_opt = axes[1, step]
            if paths_opt[step]:
                with rasterio.open(paths_opt[step]) as src:
                    r_raw = src.read(3).astype(float)
                    g_raw = src.read(2).astype(float)
                    b_raw = src.read(1).astype(float)
                    
                    # Creazione maschera per l'Ottico (veri NaN o zeri assoluti)
                    mask_nan_opt = np.isnan(r_raw) | np.isnan(g_raw) | np.isnan(b_raw)
                    if src.nodata is not None:
                        mask_nan_opt |= (r_raw == src.nodata) | (g_raw == src.nodata) | (b_raw == src.nodata)

                    # Resettiamo temporaneamente i valori a 0 per non far sballare la normalizzazione
                    r_raw[mask_nan_opt] = 0
                    g_raw[mask_nan_opt] = 0
                    b_raw[mask_nan_opt] = 0

                    rgb_stack = np.dstack((r_raw, g_raw, b_raw))
                    rgb_norm = np.clip(rgb_stack, 0, 3000) / 3000.0
                    rgb_gamma = np.power(rgb_norm, 0.75)
                    
                    # --- IL TRUCCO ---
                    # Dipingiamo fisicamente di Magenta i pixel mascherati nell'array finale
                    rgb_gamma[mask_nan_opt] = [1.0, 0.0, 1.0] 
                    
                    ax_opt.imshow(rgb_gamma)
            ax_opt.set_title(f"OPT t{step+1}\n{date_opt[step]}")
            ax_opt.axis('off')

        plt.tight_layout()
        plt.savefig(nome_output, bbox_inches='tight', dpi=150)
        plt.close(fig) # Chiude esplicitamente la figura corrente per pulire la RAM
    except Exception as e:
        print(f"🔴 Errore PNG: {nome_serie}: {e}")

def scarica_geotiff(immagine_ee, bbox_ee, epsg, nome_file_out, cartella_dest):
    percorso_finale_zip = os.path.join(cartella_dest, f"{nome_file_out}.zip")
    
    if os.path.exists(percorso_finale_zip) and os.path.getsize(percorso_finale_zip) > 0:
        return percorso_finale_zip

    try:
        immagine_sicura = immagine_ee.unmask(-9999, False)
        url = immagine_sicura.getDownloadURL({
            'scale': TARGET_SCALE,
            'crs': 'EPSG:3857',
            'region': bbox_ee,
            'format': 'GEO_TIFF'
        })
        
        # Aumentiamo i tentativi a 5
        max_tentativi = 5
        for tentativo in range(max_tentativi):
            try:
                response = requests.get(url, timeout=120) 
                
                if response.status_code == 200:
                    break  # Successo! Usciamo dal ciclo.
                
                elif response.status_code == 429:
                    # GEE ci sta bloccando per troppe richieste.
                    # Applichiamo un'attesa crescente: 15s, poi 20s, poi 25s...
                    attesa = 15 + (tentativo * 5)
                    print(f"🟡 {nome_file_out}. ({tentativo + 1}/{max_tentativi})")
                    time.sleep(attesa)
                
                else:
                    print(f"🟡 Errore {response.status_code}. | {nome_file_out}. ({tentativo + 1}/{max_tentativi})")
                    time.sleep(5)
                    
            except requests.exceptions.RequestException as e:
                print(f"Timeout rete per {nome_file_out}. ({tentativo + 1}/{max_tentativi})")
                time.sleep(10)
                
        else:
            # Scatta solo se il ciclo finisce esaurendo tutti e 5 i tentativi senza successi
            print(f"❌ Impossibile scaricare {nome_file_out} dopo {max_tentativi} tentativi.")
            return None
            
        # --- SALVATAGGIO DEL FILE ---
        file_bytes = io.BytesIO(response.content)
        
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
    date_sar = [str(riga.get('Data_1_SAR', '')), str(riga.get('Data_2_SAR', '')), str(riga.get('Data_3_SAR', '')), str(riga.get('Data_4_SAR', ''))]
    date_opt = [str(riga.get('Data_1_OPT', '')), str(riga.get('Data_2_OPT', '')), str(riga.get('Data_3_OPT', '')), str(riga.get('Data_4_OPT', ''))]
    
    # --- CREAZIONE DI UN'UNICA BOUNDING BOX COMUNE ---
    # Usiamo il centro ottico come riferimento (e l'EPSG ottico), fallback al SAR se manca
    try:
        if not pd.isna(riga.get('Lat_Centro_OPT')) and str(riga.get('Lat_Centro_OPT')).lower() != 'nan':
            lat_centro = float(riga['Lat_Centro_OPT'])
            lon_centro = float(riga['Lon_Centro_OPT'])
            epsg_comune = riga.get('EPSG_OPT', 'EPSG:32632')
        else:
            lat_centro = float(riga['Lat_Centro_SAR'])
            lon_centro = float(riga['Lon_Centro_SAR'])
            epsg_comune = riga.get('EPSG_SAR', 'EPSG:32632')
    except:
        return {'successo': False}
        
    punto_comune = ee.Geometry.Point([lon_centro, lat_centro])
    bbox_comune = punto_comune.buffer(RAGGIO_METRI).bounds()
    
    paths_per_png_sar = []
    paths_per_png_opt = []
    
    for step in range(1, 5):
        
        # --- SAR ---
        path_zip_sar = None
        nome_out_sar = f"{nome_serie}_SAR_t{step}"
        id_sar = str(riga.get(f'ID_{step}_SAR', '')).strip()
        
        if id_sar and id_sar != 'nan' and id_sar != 'NaN':
            s1_img = ee.Image(id_sar).select(['VV', 'VH'])
            path_zip_sar = scarica_geotiff(s1_img, bbox_comune, epsg_comune, nome_out_sar, S1_FOLDER)
            
        if genera_png:
            paths_per_png_sar.append(f"zip://{path_zip_sar}!{nome_out_sar}.tif" if path_zip_sar else None)

        # --- OTTICO ---
        path_zip_opt = None
        nome_out_opt = f"{nome_serie}_OPT_t{step}"
        id_opt = str(riga.get(f'ID_{step}_OPT', '')).strip()
        
        if id_opt and id_opt != 'nan' and id_opt != 'NaN':
            s2_img = ee.Image(id_opt).select(['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12'])
            path_zip_opt = scarica_geotiff(s2_img, bbox_comune, epsg_comune, nome_out_opt, S2_FOLDER)
            
        if genera_png:
            paths_per_png_opt.append(f"zip://{path_zip_opt}!{nome_out_opt}.tif" if path_zip_opt else None)  

    nome_output_png = None
    if genera_png:
        nome_output_png = os.path.join(BASE_PNG_FOLDER, f"{nome_serie}_riassunto.png")

    print(f"🟢 Download completato: {nome_serie} [{riga_num}/]")
    
    # dati restituiti
    return {
        'successo': True,
        'genera_png': genera_png,
        'nome_serie': nome_serie,
        'paths_sar': paths_per_png_sar,
        'paths_opt': paths_per_png_opt,
        'date_sar': date_sar,
        'date_opt': date_opt,
        'nome_output': nome_output_png
    }

if __name__ == "__main__":
    if not os.path.exists(CSV_PATH):
        print(f"❌ File CSV non trovato: {CSV_PATH}")
        exit()
        
    df = pd.read_csv(CSV_PATH)
    
    inizio_idx = max(0, ROW_START - 1)
    fine_idx = min(len(df), ROW_END)
    df_subset = df.iloc[inizio_idx : fine_idx]
    tasks = df_subset.to_dict('records')
    
    if len(tasks) == 0:
        print("Intervallo di righe non valido o vuoto. Controlla ROW_START e ROW_END.")
        exit()
    
    print(f"Download in parallelo da riga {inizio_idx + 1} alla {fine_idx} (Totale: {len(tasks)})")
    
    risultati_per_png = []
    
    # download multi-thread
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for i, task in enumerate(tasks):
            riga_assoluta = inizio_idx + 1 + i
            fare_png = (i == 0) or (riga_assoluta % NR_PNG == 0)
            futures.append(executor.submit(processa_singola_serie, task, fare_png, riga_assoluta))
        
        for future in as_completed(futures):
            try:
                dati = future.result()
                # Raccogliamo solo quelli per cui era richiesto di generare il PNG e che hanno avuto successo
                if dati and dati.get('successo') and dati.get('genera_png'):
                    risultati_per_png.append(dati)
            except Exception as e:
                print(f"Errore in un thread: {e}")
                
    print(f"\ndownload completati. Inizio FASE 2: Generazione di {len(risultati_per_png)} PNG in singolo thread...")

    # visualizza png (solo un thread)
    for dati in risultati_per_png:
        print(f"🖌️  Disegno PNG per: {dati['nome_serie']}")
        visualizza_serie_completa(
            dati['paths_sar'], 
            dati['paths_opt'], 
            dati['nome_serie'], 
            dati['date_sar'], 
            dati['date_opt'], 
            dati['nome_output']
        )
                
    print(f"\n FINITO (Righe {ROW_START}-{ROW_END}). PNG salvate in '{png_folder_name}'!")