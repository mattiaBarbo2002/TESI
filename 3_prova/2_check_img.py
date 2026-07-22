# Cerca immagini SAR e OTTICHE andando indietro di 1 anno alla volta fino al 2015.
# L'intervallo di ricerca nominale è +- 10 giorni. Se due intervalli sono troppo vicini
# (meno di 7 giorni tra fine di uno e inizio dell'altro), le finestre vengono
# ridotte in modo equo e simmetrico (es: [-10, +5] e [-5, +10]).
# Le colonne Bbox e Cloud vengono omesse nei file di output.
#
# INPUT:
#   merged_series.csv
#
# OUTPUT:
#   complete_series.csv
#   incomplete_series.csv (aggiunge ai file esistenti se si processa a blocchi)

import ee
import pandas as pd
import os
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

# configurazione percorsi
script_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(script_dir) 

# --- COSTANTI E CONFIGURAZIONI ---
MAX_WORKERS = 10
PIXEL_PERCENT = 5       # % max pixel Nan
CLOUDY_PERCENTAGE = 25  # % max copertura nuvolosa
RAGGIO_METRI = 1280     # raggio di 1280m
RANGE_GIORNI = 10       # +- giorni intorno alla data
GAP_MINIMO_GIORNI = 7   # giorni minimi di differenza tra fine intervallo precedente e inizio successivo

# --- IMPOSTAZIONE INTERVALLO RIGHE ---
# none tutto il file
# RIGA_INIZIO = 25001
# RIGA_FINE = 30000

RIGA_INIZIO = 79001
RIGA_FINE = 80297

# percorsi
input_csv = os.path.join(PROJECT_ROOT, "csv", "anomaly_ds.csv")
csv_output_dir = os.path.join(PROJECT_ROOT, "csv")
os.makedirs(csv_output_dir, exist_ok=True)

csv_complete = os.path.join(csv_output_dir, "complete_series.csv")
csv_incomplete = os.path.join(csv_output_dir, "incomplete_series.csv")

# inizializzazione API
try:
    print("Connessione a GEE in corso...")
    ee.Initialize(project='impactmeshprova-498219') 
    print("🟢 Connessione a GEE riuscita!\n")
except Exception as e:
    print("❌ ERRORE: impossibile autenticarsi a GEE.")
    exit()

def bilancia_intervalli(dt_targets, range_giorni, gap_minimo):

    starts = [d - timedelta(days=range_giorni) for d in dt_targets]
    ends = [d + timedelta(days=range_giorni) for d in dt_targets]
    
    for i in range(len(dt_targets) - 1):
        gap = (starts[i+1] - ends[i]).days
        
        if gap < gap_minimo:
            # Calcoliamo quanti giorni mancano per rispettare il gap minimo
            deficit = gap_minimo - gap
            
            # Distribuiamo il taglio a metà tra i due intervalli
            shift_sinistra = deficit // 2
            shift_destra = deficit - shift_sinistra
            
            ends[i] -= timedelta(days=shift_sinistra)
            starts[i+1] += timedelta(days=shift_destra)
            
    # Controllo finale: se l'inizio ha superato la fine, l'intervallo è collassato
    for s, e in zip(starts, ends):
        if s > e:
            return None, None
            
    return starts, ends

def processa_singola_serie(riga, riga_num, totale):
    nome_serie = riga.get('Nome_Serie', 'Sconosciuta')
    
    # --- CREAZIONE DI UN'UNICA BOUNDING BOX COMUNE (Centro OPT prioritario) ---
    try:
        if not pd.isna(riga.get('Lat_Centro_OPT')) and str(riga.get('Lat_Centro_OPT')).lower() != 'nan':
            lat_centro = float(riga['Lat_Centro_OPT'])
            lon_centro = float(riga['Lon_Centro_OPT'])
        else:
            lat_centro = float(riga['Lat_Centro_SAR'])
            lon_centro = float(riga['Lon_Centro_SAR'])
    except (ValueError, TypeError):
        return ('incomplete', riga, "Coordinate mancanti o malformate")

    # Geometria COMUNE per GEE
    punto_comune = ee.Geometry.Point([lon_centro, lat_centro])
    bbox_comune = punto_comune.buffer(RAGGIO_METRI).bounds()

    # Lettura e conversione delle date originali
    try:
        dt_org_sar = [datetime.strptime(str(riga[f'Data_{i}_org_SAR'])[:10], '%Y-%m-%d') for i in range(1, 5)]
        dt_org_opt = [datetime.strptime(str(riga[f'Data_{i}_org_OPT'])[:10], '%Y-%m-%d') for i in range(1, 5)]
    except Exception as e:
        return ('incomplete', riga, "Date originali mancanti o malformate")

    # Funzioni di riduzione GEE basate ESCLUSIVAMENTE sulla bbox_comune
    def calcola_nan_sar(img):
        # unmask(-9999, False) rende l'immagine infinita, superando il taglio del satellite
        img_globale = img.select(0).unmask(-9999, False).clip(bbox_comune)
        
        mask_validi = img_globale.gt(-9999)
        stats = mask_validi.reduceRegion(reducer=ee.Reducer.mean(), geometry=bbox_comune, scale=100, maxPixels=1e9)
        
        valid_pct = ee.Number(stats.get(img.select(0).bandNames().get(0))).multiply(100)
        nan_pct = ee.Number(100).subtract(valid_pct)
        return img.set('nan_percent', nan_pct)

    def analizza_pixel_ottici(img):
        # unmask(-9999, False) per espandere anche l'ottico
        img_globale = img.select('B4').unmask(-9999, False).clip(bbox_comune)
        mask_validi = img_globale.neq(-9999).And(img_globale.gt(0))
        
        stats_nodata = mask_validi.reduceRegion(reducer=ee.Reducer.mean(), geometry=bbox_comune, scale=10, maxPixels=1e9)
        valid_pct = ee.Number(stats_nodata.get('B4')).multiply(100)
        nan_pct = ee.Number(100).subtract(valid_pct)
        
        # Stessa cosa per la maschera nuvole
        scl_globale = img.select('SCL').unmask(-9999, False).clip(bbox_comune)
        mask_nuvole = scl_globale.eq(8).Or(scl_globale.eq(9)).Or(scl_globale.eq(10))
        
        stats_nuvole = mask_nuvole.reduceRegion(reducer=ee.Reducer.mean(), geometry=bbox_comune, scale=20, maxPixels=1e9)
        cloud_pct = ee.Number(stats_nuvole.get('SCL')).multiply(100)
        
        return img.set('nan_percent', nan_pct).set('roi_cloud_percent', cloud_pct)

    anno_offset = 1

    # Inizia il back-off iterativo
    while True:
        # Calcolo date target arretrate
        dt_target_sar = [d - timedelta(days=365 * anno_offset) for d in dt_org_sar]
        dt_target_opt = [d - timedelta(days=365 * anno_offset) for d in dt_org_opt]
        
        # Se anche una sola data target scende sotto il 2015, ci arrendiamo
        if any(d.year < 2015 for d in dt_target_sar) or any(d.year < 2015 for d in dt_target_opt):
            return ('incomplete', riga, f"< 2015")
            
        # Calcolo intervalli bilanciati
        starts_sar, ends_sar = bilancia_intervalli(dt_target_sar, RANGE_GIORNI, GAP_MINIMO_GIORNI)
        starts_opt, ends_opt = bilancia_intervalli(dt_target_opt, RANGE_GIORNI, GAP_MINIMO_GIORNI)
        
        # Se il bilanciamento è impossibile (start > end), saltiamo direttamente questo anno
        if starts_sar is None or starts_opt is None:
            anno_offset += 1
            continue

        successo_anno = True
        
        # --- RICERCA SAR ---
        date_effettive_sar = []
        id_effettivi_sar = []
        
        # Variabile per forzare la stessa orbita per tutte e 4 le immagini di questo anno
        orbita_fissata = None 
        
        for i in range(4):
            # Interrogazione GEE usando gli intervalli pre-bilanciati
            start_str = starts_sar[i].strftime('%Y-%m-%d')
            end_str = (ends_sar[i] + timedelta(days=1)).strftime('%Y-%m-%d') # +1 perchè GEE esclude il giorno finale
            
            s1_col = ee.ImageCollection('COPERNICUS/S1_GRD') \
                .filterBounds(punto_comune) \
                .filterDate(start_str, end_str) \
                .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
                .filter(ee.Filter.eq('instrumentMode', 'IW'))
            
            # Se abbiamo già trovato la prima orbita utile, la blocchiamo per le successive
            if orbita_fissata:
                s1_col = s1_col.filter(ee.Filter.eq('orbitProperties_pass', orbita_fissata))
                
            s1_filtrata = s1_col.map(calcola_nan_sar).filter(ee.Filter.lt('nan_percent', PIXEL_PERCENT)).sort('nan_percent')
            
            try:
                migliore_s1 = ee.Image(s1_filtrata.first())
                id_img = migliore_s1.get('system:id').getInfo()
                
                if id_img:
                    # Registriamo l'orbita alla prima immagine valida che troviamo
                    if orbita_fissata is None:
                        orbita_fissata = migliore_s1.get('orbitProperties_pass').getInfo()
                        
                    id_effettivi_sar.append(id_img)
                    timestamp = migliore_s1.get('system:time_start').getInfo()
                    date_effettive_sar.append(datetime.fromtimestamp(timestamp / 1000.0, tz=timezone.utc).strftime('%Y-%m-%d'))
                else:
                    successo_anno = False
                    break
            except Exception:
                successo_anno = False
                break
            
        if not successo_anno:
            anno_offset += 1
            continue
            
        # --- RICERCA OTTICO ---
        date_effettive_opt = []
        id_effettivi_opt = []
        
        for i in range(4):
            start_str = starts_opt[i].strftime('%Y-%m-%d')
            end_str = (ends_opt[i] + timedelta(days=1)).strftime('%Y-%m-%d')
            
            s2_col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
                .filterBounds(punto_comune) \
                .filterDate(start_str, end_str)
                
            s2_filtrata = s2_col.map(analizza_pixel_ottici) \
                .filter(ee.Filter.lt('nan_percent', PIXEL_PERCENT)) \
                .filter(ee.Filter.lt('roi_cloud_percent', CLOUDY_PERCENTAGE)) \
                .sort('roi_cloud_percent')
                
            try:
                migliore_s2 = ee.Image(s2_filtrata.first())
                id_img = migliore_s2.get('system:id').getInfo()
                if id_img:
                    id_effettivi_opt.append(id_img)
                    timestamp = migliore_s2.get('system:time_start').getInfo()
                    date_effettive_opt.append(datetime.fromtimestamp(timestamp / 1000.0, tz=timezone.utc).strftime('%Y-%m-%d'))
                else:
                    successo_anno = False
                    break
            except Exception:
                successo_anno = False
                break

        if not successo_anno:
            anno_offset += 1
            continue
            
        # SE ARRIVIAMO QUI: Trovate 4 SAR coerenti per orbita e 4 OPT valide per questo anno!
        nuova_riga = riga.copy()
        
        for i in range(4):
            nuova_riga[f'Data_{i+1}_SAR'] = date_effettive_sar[i]
            nuova_riga[f'ID_{i+1}_SAR'] = id_effettivi_sar[i]
            
            nuova_riga[f'Data_{i+1}_OPT'] = date_effettive_opt[i]
            nuova_riga[f'ID_{i+1}_OPT'] = id_effettivi_opt[i]
            
        nuova_riga['Anni_Offset'] = anno_offset
        
        print(f"🟢 [{riga_num}/{totale}] {nome_serie}: COMPLETA (-{anno_offset} anni)")
        return ('complete', nuova_riga, "OK")


if __name__ == "__main__":
    if not os.path.exists(input_csv):
        print(f"❌ File input non trovato: {input_csv}")
        exit()
        
    print(f"Lettura di {input_csv}...")
    df_input = pd.read_csv(input_csv)
    
    # --- FILTRAGGIO RIGHE DA ELABORARE ---
    if RIGA_INIZIO is not None and RIGA_FINE is not None:
        inizio_idx = max(0, RIGA_INIZIO - 1) # Python è 0-indexed
        fine_idx = min(len(df_input), RIGA_FINE)
        df_input = df_input.iloc[inizio_idx:fine_idx]
        print(f"IRGHE SELEZIONATE: {inizio_idx + 1} a {fine_idx} (Totale righe: {len(df_input)})")
    
    # --- RIMOZIONE COLONNE BBOX E CLOUD ---
    colonne_da_rimuovere = [col for col in df_input.columns if 'Bbox' in col or 'Cloud' in col]
    if colonne_da_rimuovere:
        df_input.drop(columns=colonne_da_rimuovere, inplace=True)
    
    tasks = df_input.to_dict('records')
    totale_tasks = len(tasks)
    
    risultati_completi = []
    risultati_incompleti = []

    print(f"\nAvvio elaborazione in parallelo per {totale_tasks} serie...")
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Salvo sia il task che il numero riga (i+1) come valore nel dizionario
        futures = {
            executor.submit(processa_singola_serie, task, i+1, totale_tasks): (task, i+1) 
            for i, task in enumerate(tasks)
        }
        
        for future in as_completed(futures):
            # Recupero il task originale e il numero di riga associati a questo future
            task_originale, riga_num = futures[future]
            nome_serie = task_originale.get('Nome_Serie', 'Sconosciuta')
            
            try:
                status, riga_dict, msg = future.result()
                if status == 'complete':
                    risultati_completi.append(riga_dict)
                else:
                    riga_dict['Motivo_Scarto'] = msg
                    risultati_incompleti.append(riga_dict)
                    print(f"🟡 [{riga_num}/{totale_tasks}] {nome_serie}: Scartata: {msg}")
            except Exception as e:
                print(f"🔴 [{riga_num}/{totale_tasks}] {nome_serie}: Errore irreversibile in un thread: {e}")

    # --- FUNZIONE SALVATAGGIO CON LOGICA APPEND ---
    def salva_csv_append(df, filepath):
        if df.empty:
            return 0
        file_esiste = os.path.exists(filepath)
        # Se il file esiste già, appendiamo (mode='a') e NON mettiamo l'header
        # Se non esiste, creiamo nuovo (mode='w') E mettiamo l'header
        df.to_csv(filepath, mode='a' if file_esiste else 'w', 
                  header=not file_esiste, index=False)
        return len(df)

    # --- SALVATAGGIO DEI RISULTATI ---
    df_completi = pd.DataFrame(risultati_completi)
    df_incompleti = pd.DataFrame(risultati_incompleti)
    
    righe_c = salva_csv_append(df_completi, csv_complete)
    righe_inc = salva_csv_append(df_incompleti, csv_incomplete)
    
    print("\n--- RIEPILOGO SALVATAGGIO ---")
    if righe_c > 0:
        print(f"✔️ Aggiunte {righe_c} serie a {os.path.basename(csv_complete)}")
    else:
        print(f"⚠️ Nessuna serie completa da salvare.")
        
    if righe_inc > 0:
        print(f"✔️ Aggiunte {righe_inc} serie a {os.path.basename(csv_incomplete)}")
    else:
        print(f"⚠️ Nessuna serie incompleta da salvare.")
        
    print("\nPROCESSO TERMINATO")