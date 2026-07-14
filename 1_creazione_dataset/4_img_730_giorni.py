# prende in input le serie incomplete incomplete_series.csv, per cui la ricerca della serie completa è fallita cercando un anno prima dell'evento anomalo
# si prova a cercare serie completa (sar + opt) 2 anni prima dell'evento anomalo
# se si trova si rimuove riga da incomplete_series.csv e si aggiunge riga a complete_series.csv con nuove date
#
# NOTA:
#   se si interrompe l'esecuzione eseguire 5_interrupt.py per avere csv consistenti
#
# INPUT:
#   complete_series.csv
#   incomplete_series.csv
#
# OUTPUT:
#   complete_series.csv (modificato)
#   incomplete_series.csv (modificato)
# 
# prossimo script 6_balance_ds.py

import ee
import csv
import os
from datetime import datetime, timedelta, timezone
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# percorso workspace
script_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(script_dir) # /workspace/1_creazione_dataset

csv_lock = threading.Lock()

# costanti
CONTAINER = False
MAX_WORKERS = 15
PIXEL_PERCENT = 5       # % pixel Nan in img SAR e OTTICHE
CLOUDY_PERCENTAGE = 25  # % copertura nuvolosa
RAGGIO_METRI = 1280     # raggio di 1280m (2560m/2), box 256x256
RANGE_GIORNI = 10       # +- giorni intorno alla data
ANNO_PRIMA = 730        # ricerca 2 anni prima della data originale

# percorsi csv
csv_output_dir = os.path.join(PROJECT_ROOT, "csv")
incomplete_csv = os.path.join(csv_output_dir, "new_incomplete_series_gee_COMPLETO.csv")
complete_csv = os.path.join(csv_output_dir, "new_complete_series_gee_COMPLETO.csv")


# inizializzazione api gee
try:
    print("Connessione a GEE...")
    ee.Initialize(project='impactmeshprova-498219') 
    print("🟢 Connessione a GEE riuscita")
except Exception as e:
    print("❌ ERRORE: richiesta autenticazione GEE. comando 'earthengine authenticate'")
    exit()

# conteggio righe
def conta_righe_csv(filepath):
    if not os.path.exists(filepath):
        return 0
    with open(filepath, mode='r', encoding='utf-8') as f:
        reader = csv.reader(f)
        try:
            next(reader) # salta l'intestazione
            return sum(1 for row in reader)
        except StopIteration:
            return 0 # file vuoto o solo intestazione


def elabora_recupero():
    if not os.path.exists(incomplete_csv):
        print(f"❌ File incompleti non trovato: {incomplete_csv}")
        return

    # conteggio iniziale 
    righe_complete_start = conta_righe_csv(complete_csv)
    righe_incomplete_start = conta_righe_csv(incomplete_csv)
    totale_start = righe_complete_start + righe_incomplete_start


    # legge serie incomplete
    tasks_da_fare = []
    with open(incomplete_csv, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for riga in reader:
            tasks_da_fare.append(riga)

    # apertura file complete per appendere nuove serie trovate
    file_completi_esiste = os.path.exists(complete_csv)
    
    # set con nuove serie complete
    serie_recuperate_set = set()

    with open(complete_csv, mode='a', encoding='utf-8', newline='') as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        if not file_completi_esiste:
            writer.writeheader()

        # funzione singolo thread
        def processa_singola_serie(riga, riga_num):
            nome_serie = riga['Nome_Serie']
            
            print(f"Riga {riga_num} | Serie selezionata: {nome_serie}")
            
            # controllo presenza date originali
            for step in range(1, 5):
                val_org_sar = riga.get(f'Data_{step}_org_SAR', 'NaN')
                val_org_opt = riga.get(f'Data_{step}_org_OPT', 'NaN')
                
                if (not val_org_sar or val_org_sar == 'NaN' or str(val_org_sar).lower() == 'nan' or
                    not val_org_opt or val_org_opt == 'NaN' or str(val_org_opt).lower() == 'nan'):
                    return None
            
            # controllo presenza coordinate
            try:
                lat_sar, lon_sar = float(riga['Lat_Centro_SAR']), float(riga['Lon_Centro_SAR'])
                lat_opt, lon_opt = float(riga['Lat_Centro_OPT']), float(riga['Lon_Centro_OPT'])
            except (ValueError, TypeError):
                return None

            punto_centro_sar = ee.Geometry.Point([lon_sar, lat_sar])
            bbox_esatto_sar = punto_centro_sar.buffer(RAGGIO_METRI).bounds()

            punto_centro_opt = ee.Geometry.Point([lon_opt, lat_opt])
            bbox_esatto_opt = punto_centro_opt.buffer(RAGGIO_METRI).bounds()
            
            nuove_date_effettive = {}
            
            # ciclo sulle 4 date
            for step in range(1, 5):

                # date SAR
                data_stringa_sar = riga.get(f'Data_{step}_org_SAR')
                data_originale_sar = datetime.strptime(data_stringa_sar[:10], '%Y-%m-%d')
                data_target_sar = data_originale_sar - timedelta(days=ANNO_PRIMA)
                data_inizio_sar = (data_target_sar - timedelta(days=RANGE_GIORNI)).strftime('%Y-%m-%d')
                data_fine_sar = (data_target_sar + timedelta(days=RANGE_GIORNI)).strftime('%Y-%m-%d')

                # date OTTICO
                data_stringa_opt = riga.get(f'Data_{step}_org_OPT')
                data_originale_opt = datetime.strptime(data_stringa_opt[:10], '%Y-%m-%d')
                data_target_opt = data_originale_opt - timedelta(days=ANNO_PRIMA)
                data_inizio_opt = (data_target_opt - timedelta(days=RANGE_GIORNI)).strftime('%Y-%m-%d')
                data_fine_opt = (data_target_opt + timedelta(days=RANGE_GIORNI)).strftime('%Y-%m-%d')


                # OTTICO
                # stesso controllo requisiti immagini di 2_img_365.py
                s2_col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
                    .filterBounds(punto_centro_opt) \
                    .filterDate(data_inizio_opt, data_fine_opt) 
                
                def analizza_pixel_ottici(img):
                    mask_dati = img.select('B4').mask()
                    stats_nodata = mask_dati.reduceRegion(reducer=ee.Reducer.mean(), geometry=bbox_esatto_opt, scale=10, maxPixels=1e9)
                    nan_pct = ee.Number(100).subtract(ee.Number(stats_nodata.get('B4')).multiply(100))
                    
                    scl = img.select('SCL')
                    mask_nuvole = scl.eq(8).Or(scl.eq(9)).Or(scl.eq(10))
                    stats_nuvole = mask_nuvole.reduceRegion(reducer=ee.Reducer.mean(), geometry=bbox_esatto_opt, scale=20, maxPixels=1e9)
                    cloud_pct = ee.Number(stats_nuvole.get('SCL')).multiply(100)
                    
                    return img.set('nan_percent', nan_pct).set('roi_cloud_percent', cloud_pct)
                    
                s2_filtrata = s2_col.map(analizza_pixel_ottici)\
                    .filter(ee.Filter.lt('nan_percent', PIXEL_PERCENT))\
                    .filter(ee.Filter.lt('roi_cloud_percent', CLOUDY_PERCENTAGE))\
                    .sort('roi_cloud_percent')
                    
                try:
                    migliore_s2 = ee.Image(s2_filtrata.first())
                    id_img_s2 = migliore_s2.id().getInfo()
                    if not id_img_s2: raise Exception("No ID")
                    
                    timestamp_s2 = migliore_s2.get('system:time_start').getInfo()
                    data_eff_s2 = datetime.fromtimestamp(timestamp_s2 / 1000.0, tz=timezone.utc).strftime('%Y-%m-%d')
                    nuove_date_effettive[f'Data_{step}_OPT'] = data_eff_s2
                    
                    print(f"🔵 Immagine OTTICO t{step} trovata | Data: {data_eff_s2} | ID: {id_img_s2[:15]}")
                except Exception:
                    # se manca una img salto serie
                    print(f"🔴 Nessuna immagine valida trovata (OTTICO) | Data target: {data_target_opt.strftime('%Y-%m-%d')}")
                    return None 
                
                # SAR
                # stesso controllo requisiti immagini di 2_img_365.py
                s1_col = ee.ImageCollection('COPERNICUS/S1_GRD') \
                    .filterBounds(punto_centro_sar) \
                    .filterDate(data_inizio_sar, data_fine_sar) \
                    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
                    .filter(ee.Filter.eq('instrumentMode', 'IW'))
                    
                def calcola_nan_s1(img):
                    mask = img.select(0).mask()
                    stats = mask.reduceRegion(reducer=ee.Reducer.mean(), geometry=bbox_esatto_sar, scale=100, maxPixels=1e9)
                    valid_pct = ee.Number(stats.get(mask.bandNames().get(0))).multiply(100)
                    return img.set('nan_percent', ee.Number(100).subtract(valid_pct))
                    
                s1_filtrata = s1_col.map(calcola_nan_s1).filter(ee.Filter.lt('nan_percent', PIXEL_PERCENT)).sort('nan_percent')
                
                try:
                    migliore_s1 = ee.Image(s1_filtrata.first())
                    id_img_s1 = migliore_s1.id().getInfo()
                    if not id_img_s1: raise Exception("No ID")
                    
                    timestamp_s1 = migliore_s1.get('system:time_start').getInfo()
                    data_eff_s1 = datetime.fromtimestamp(timestamp_s1 / 1000.0, tz=timezone.utc).strftime('%Y-%m-%d')
                    nuove_date_effettive[f'Data_{step}_SAR'] = data_eff_s1
                    
                    print(f"🔵 Immagine SAR t{step} trovata | Data: {data_eff_s1} | ID: {id_img_s1[:15]}")
                except Exception:
                    # se manca una img salto serie
                    print(f"🔴 Nessuna immagine valida trovata (SAR) | Data target: {data_target_sar.strftime('%Y-%m-%d')}")
                    return None 

            # update riga con le nuove date valide
            for chiave, valore in nuove_date_effettive.items():
                riga[chiave] = valore
                
            return riga

        # esecuzione multithread
        print(f"\nVerifica in parallelo per {len(tasks_da_fare)} serie incomplete")
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # copia della riga
            futures = {executor.submit(processa_singola_serie, task.copy(), i + 1): task for i, task in enumerate(tasks_da_fare)}
            
            for future in as_completed(futures):
                try:
                    risultato_riga = future.result()
                    if risultato_riga is not None:
                        # scrittura solo 1 thread alla volta
                        with csv_lock:
                            writer.writerow(risultato_riga)  
                            f_out.flush() 
                            os.fsync(f_out.fileno()) 

                        # aggiunge nome serie a set
                        serie_recuperate_set.add(risultato_riga['Nome_Serie'])
                except Exception as e:
                    print(f"🔴 Errore in un thread: {e}")
                    
    print(f"\n✔️ FINE. Nuove serie recuperate: {len(serie_recuperate_set)}")

    # update incomplete_series.csv
    if len(serie_recuperate_set) > 0:
        print(f"\nUpdate incomplete_series.csv: {len(serie_recuperate_set)} serie")
        
        righe_rimanenti = [task for task in tasks_da_fare if task['Nome_Serie'] not in serie_recuperate_set]
        
        # update
        with open(incomplete_csv, mode='w', encoding='utf-8', newline='') as f_incomp:
            writer_inc = csv.DictWriter(f_incomp, fieldnames=fieldnames)
            writer_inc.writeheader()
            writer_inc.writerows(righe_rimanenti)
            
        print(f"File {incomplete_csv} aggiornato.")

    # conteggio finale
    righe_complete_end = conta_righe_csv(complete_csv)
    righe_incomplete_end = conta_righe_csv(incomplete_csv)
    totale_end = righe_complete_end + righe_incomplete_end

    print(f"\n--- STATO INIZIALE ---")
    print(f"Serie COMPLETE:   {righe_complete_start}")
    print(f"Serie INCOMPLETE: {righe_incomplete_start}")
    print(f"TOTALE SERIE:     {totale_start}")
    print(f"\n")

    print(f"\n--- STATO FINALE ---")
    print(f"Serie COMPLETE:   {righe_complete_end}")
    print(f"Serie INCOMPLETE: {righe_incomplete_end}")
    print(f"TOTALE SERIE:     {totale_end}")
    print(f"")

    if totale_start == totale_end:
        print("\nOK numero serie invariato.")
    else:
        print(f"\nATTENZIONE: numero totale cambiato (da {totale_start} a {totale_end})!")


if __name__ == "__main__":

    elabora_recupero()