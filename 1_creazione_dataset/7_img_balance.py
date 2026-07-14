# prende incomplete_series.csv con nuova colonna 'target_month'
# usa come data target (data_3_opt e sar) il 15 del mese target
# cerca serie complete intorno a quel mese partendo dal 2025
# se non la trova valida, prova 2024 e cosi via fino a 2018
# controlla di non selezionare l'anno della serie originale con anomalia
# se trova serie completa, aggiunge riga a complete_series.csv
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

import ee
import csv
import os
from datetime import datetime, timedelta, timezone
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# percorso worksapce
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

# percorsi csv 
csv_output_dir = os.path.join(PROJECT_ROOT, "csv")
complete_csv = os.path.join(csv_output_dir, "complete_series.csv")
incomplete_csv = os.path.join(csv_output_dir, "incomplete_series.csv")


# stesso codice di 4_img:730_giorni.py
# commenti solo su nuove funzioni
try:
    print("Connessione a GEE...")
    ee.Initialize(project='impactmeshprova-498219') 
    print("🟢 Connessione a GEE riuscita")
except Exception as e:
    print("❌ ERRORE: richiesta autenticazione GEE. comando 'earthengine authenticate'")
    exit()

def conta_righe_csv(filepath):
    if not os.path.exists(filepath):
        return 0
    with open(filepath, mode='r', encoding='utf-8') as f:
        reader = csv.reader(f)
        try:
            next(reader)
            return sum(1 for row in reader)
        except StopIteration:
            return 0 

# calcolo data (necessario per cambiare anno in caso di serie a cavallo tra due anni)
def calcola_data_target(anno, mese_base, offset_mesi):
    mese_calc = mese_base - 1 + offset_mesi
    nuovo_anno = anno + (mese_calc // 12)
    nuovo_mese = (mese_calc % 12) + 1
    return datetime(nuovo_anno, nuovo_mese, 15)


def elabora_recupero():
    if not os.path.exists(incomplete_csv):
        print(f"❌ File incompleti non trovato: {incomplete_csv}")
        return

    righe_complete_start = conta_righe_csv(complete_csv)
    righe_incomplete_start = conta_righe_csv(incomplete_csv)
    totale_start = righe_complete_start + righe_incomplete_start

    tasks_da_fare = []
    with open(incomplete_csv, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames_in = reader.fieldnames
        for riga in reader:
            tasks_da_fare.append(riga)

    # non includere in complete_series.csv la colonna 'target_month'
    fieldnames_out = [f for f in fieldnames_in if f.lower() not in ['target_month']]

    file_completi_esiste = os.path.exists(complete_csv)
    
    serie_recuperate_set = set()

    with open(complete_csv, mode='a', encoding='utf-8', newline='') as f_out:
        writer_out = csv.DictWriter(f_out, fieldnames=fieldnames_out)
        if not file_completi_esiste:
            writer_out.writeheader()

        def processa_singola_serie(riga, riga_num):
            nome_serie = riga['Nome_Serie']
            
            try:
                mese_assegnato = int(float(riga.get('mese', riga.get('month', 0))))
                if mese_assegnato == 0:
                    return None 
            except:
                return None
                
            try:
                anno_da_saltare = int(str(riga.get('Data_3_org_OPT', '')).split('-')[0])
            except:
                anno_da_saltare = -1 # se data orginale manca non salto nessun anno

            print(f"Riga {riga_num} | Processo Serie: {nome_serie} (Mese Target: {mese_assegnato}, Anno Skip: {anno_da_saltare})")
        
            try:
                lat_sar, lon_sar = float(riga['Lat_Centro_SAR']), float(riga['Lon_Centro_SAR'])
                lat_opt, lon_opt = float(riga['Lat_Centro_OPT']), float(riga['Lon_Centro_OPT'])
            except (ValueError, TypeError):
                
                return None

            punto_centro_sar = ee.Geometry.Point([lon_sar, lat_sar])
            bbox_esatto_sar = punto_centro_sar.buffer(RAGGIO_METRI).bounds()

            punto_centro_opt = ee.Geometry.Point([lon_opt, lat_opt])
            bbox_esatto_opt = punto_centro_opt.buffer(RAGGIO_METRI).bounds()
            
            # ricerca tra 2025 e 2018
            for anno_test in range(2025, 2017, -1):
                if anno_test == anno_da_saltare:
                    continue # salta l'anno corrispondente alla data originale
                
                anno_valido = True
                nuove_date_effettive = {}
                
                for step in range(1, 5):
                    # calcolo date target
                    offset_mesi = {1: -2, 2: -1, 3: 0, 4: 1}[step]
                    data_target = calcola_data_target(anno_test, mese_assegnato, offset_mesi)
                    
                    data_inizio = (data_target - timedelta(days=RANGE_GIORNI)).strftime('%Y-%m-%d')
                    data_fine = (data_target + timedelta(days=RANGE_GIORNI)).strftime('%Y-%m-%d')


                    # OTTICO
                    s2_col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
                        .filterBounds(punto_centro_opt) \
                        .filterDate(data_inizio, data_fine) 
                    
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
                        
                        print(f"🔵 OTTICO t{step} ({anno_test}) | Data: {data_eff_s2}")
                    except Exception:
                        # se non trovo serie valida, anno non valido
                        anno_valido = False
                        break 
                    
                    # SAR
                    s1_col = ee.ImageCollection('COPERNICUS/S1_GRD') \
                        .filterBounds(punto_centro_sar) \
                        .filterDate(data_inizio, data_fine) \
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
                        
                        print(f"🔵 SAR t{step} ({anno_test}) | Data: {data_eff_s1}")
                    except Exception:
                        # se non trovo serie valida, anno non valido
                        anno_valido = False
                        break

                # se anno è valido (4 img sar + 4 img opt trovate)
                if anno_valido:
                    print(f"🟢 Serie {nome_serie} completata con l'anno {anno_test}!")

                    # update riga con nuove date 
                    for chiave, valore in nuove_date_effettive.items():
                        riga[chiave] = valore
                    
                    # tolgo colonna
                    riga.pop('target_month', None)
                    
                    return riga

            print(f"🔴 Serie {nome_serie}: Impossibile completare (nessun anno valido trovato)")
            return None

        # esecuzione multithread
        print(f"\nVerifica in parallelo per {len(tasks_da_fare)} serie incomplete")
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            
            futures = {executor.submit(processa_singola_serie, task.copy(), i + 1): task for i, task in enumerate(tasks_da_fare)}
            
            for future in as_completed(futures):
                try:
                    risultato_riga = future.result()
                    if risultato_riga is not None:
                        
                        with csv_lock:
                            writer_out.writerow(risultato_riga)  
                            f_out.flush() 
                            os.fsync(f_out.fileno()) 
                        
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
            writer_inc = csv.DictWriter(f_incomp, fieldnames=fieldnames_in)
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