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
#   incomplete_series.csv

import ee
import pandas as pd
import os
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

# configurazione percorsi
script_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(script_dir) 

# costanti 
MAX_WORKERS = 15
PIXEL_PERCENT = 5       # % max pixel Nan
CLOUDY_PERCENTAGE = 25  # % max copertura nuvolosa
RAGGIO_METRI = 1280     # raggio di 1280m
RANGE_GIORNI = 10       # +- giorni intorno alla data
GAP_MINIMO_GIORNI = 7   # giorni minimi di differenza tra fine intervallo precedente e inizio successivo

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
    """
    Calcola gli intervalli di ricerca per una lista di date.
    Se due intervalli sono troppo vicini, li accorcia in modo simmetrico/equo.
    Restituisce None, None se la compressione rende un intervallo impossibile.
    """
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
    
    # Estrazione coordinate
    try:
        lat_sar, lon_sar = float(riga['Lat_Centro_SAR']), float(riga['Lon_Centro_SAR'])
        lat_opt, lon_opt = float(riga['Lat_Centro_OPT']), float(riga['Lon_Centro_OPT'])
    except:
        return ('incomplete', riga, "Coordinate mancanti")

    # Geometrie per GEE
    punto_sar = ee.Geometry.Point([lon_sar, lat_sar])
    bbox_sar = punto_sar.buffer(RAGGIO_METRI).bounds()
    
    punto_opt = ee.Geometry.Point([lon_opt, lat_opt])
    bbox_opt = punto_opt.buffer(RAGGIO_METRI).bounds()

    # Lettura e conversione delle date originali
    try:
        dt_org_sar = [datetime.strptime(str(riga[f'Data_{i}_org_SAR'])[:10], '%Y-%m-%d') for i in range(1, 5)]
        dt_org_opt = [datetime.strptime(str(riga[f'Data_{i}_org_OPT'])[:10], '%Y-%m-%d') for i in range(1, 5)]
    except Exception as e:
        return ('incomplete', riga, "Date originali mancanti o malformate")

    # Funzioni di riduzione GEE
    def calcola_nan_sar(img):
        mask = img.select(0).mask()
        stats = mask.reduceRegion(reducer=ee.Reducer.mean(), geometry=bbox_sar, scale=100, maxPixels=1e9)
        valid_pct = ee.Number(stats.get(mask.bandNames().get(0))).multiply(100)
        nan_pct = ee.Number(100).subtract(valid_pct)
        return img.set('nan_percent', nan_pct)

    def analizza_pixel_ottici(img):
        mask_dati = img.select('B4').mask()
        stats_nodata = mask_dati.reduceRegion(reducer=ee.Reducer.mean(), geometry=bbox_opt, scale=10, maxPixels=1e9)
        valid_pct = ee.Number(stats_nodata.get('B4')).multiply(100)
        nan_pct = ee.Number(100).subtract(valid_pct)
        
        scl = img.select('SCL')
        mask_nuvole = scl.eq(8).Or(scl.eq(9)).Or(scl.eq(10))
        stats_nuvole = mask_nuvole.reduceRegion(reducer=ee.Reducer.mean(), geometry=bbox_opt, scale=20, maxPixels=1e9)
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
            return ('incomplete', riga, f"Raggiunto 2015 senza successo (offset -{anno_offset} anni)")
            
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
        
        for i in range(4):
            # Interrogazione GEE usando gli intervalli pre-bilanciati
            start_str = starts_sar[i].strftime('%Y-%m-%d')
            end_str = (ends_sar[i] + timedelta(days=1)).strftime('%Y-%m-%d') # +1 perchè GEE esclude il giorno finale
            
            s1_col = ee.ImageCollection('COPERNICUS/S1_GRD') \
                .filterBounds(punto_sar) \
                .filterDate(start_str, end_str) \
                .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
                .filter(ee.Filter.eq('instrumentMode', 'IW'))
                
            s1_filtrata = s1_col.map(calcola_nan_sar).filter(ee.Filter.lt('nan_percent', PIXEL_PERCENT)).sort('nan_percent')
            
            try:
                migliore_s1 = ee.Image(s1_filtrata.first())
                id_img = migliore_s1.get('system:id').getInfo()
                if id_img:
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
                .filterBounds(punto_opt) \
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
            
        # SE ARRIVIAMO QUI: Trovate 4 SAR e 4 OPT valide per questo anno!
        nuova_riga = riga.copy()
        
        for i in range(4):
            nuova_riga[f'Data_{i+1}_SAR'] = date_effettive_sar[i]
            nuova_riga[f'ID_{i+1}_SAR'] = id_effettivi_sar[i]
            
            nuova_riga[f'Data_{i+1}_OPT'] = date_effettive_opt[i]
            nuova_riga[f'ID_{i+1}_OPT'] = id_effettivi_opt[i]
            
        nuova_riga['Anni_Offset'] = anno_offset
        
        print(f"🟢 [{riga_num}/{totale}] {nome_serie}: COMPLETA a -{anno_offset} anni!")
        return ('complete', nuova_riga, "OK")


if __name__ == "__main__":
    if not os.path.exists(input_csv):
        print(f"❌ File input non trovato: {input_csv}")
        exit()
        
    print(f"Lettura di {input_csv}...")
    df_input = pd.read_csv(input_csv)
    
    # --- RIMOZIONE COLONNE BBOX E CLOUD ---
    colonne_da_rimuovere = [col for col in df_input.columns if 'Bbox' in col or 'Cloud' in col]
    if colonne_da_rimuovere:
        df_input.drop(columns=colonne_da_rimuovere, inplace=True)
        print(f"🗑️ Rimosse le seguenti colonne non necessarie: {colonne_da_rimuovere}")
    
    tasks = df_input.to_dict('records')
    totale_tasks = len(tasks)
    
    risultati_completi = []
    risultati_incompleti = []

    print(f"\nAvvio elaborazione in parallelo per {totale_tasks} serie...")
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(processa_singola_serie, task, i+1, totale_tasks): task for i, task in enumerate(tasks)}
        
        for future in as_completed(futures):
            try:
                status, riga_dict, msg = future.result()
                if status == 'complete':
                    risultati_completi.append(riga_dict)
                else:
                    riga_dict['Motivo_Scarto'] = msg
                    risultati_incompleti.append(riga_dict)
                    print(f"🟡 Scartata: {riga_dict.get('Nome_Serie', 'N/A')} - {msg}")
            except Exception as e:
                print(f"🔴 Errore irreversibile in un thread: {e}")

    # --- SALVATAGGIO DEI RISULTATI ---
    df_completi = pd.DataFrame(risultati_completi)
    df_incompleti = pd.DataFrame(risultati_incompleti)
    
    if not df_completi.empty:
        df_completi.to_csv(csv_complete, index=False)
        print(f"\n✔️ Salvato {csv_complete} con {len(df_completi)} serie COMPLETE.")
    else:
        print("\n⚠️ Nessuna serie completa trovata.")

    if not df_incompleti.empty:
        df_incompleti.to_csv(csv_incomplete, index=False)
        print(f"✔️ Salvato {csv_incomplete} con {len(df_incompleti)} serie INCOMPLETE.")
        
    print("\nPROCESSO TERMINATO")