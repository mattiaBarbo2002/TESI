# prende incomplete_series.csv con nuova colonna 'target_month'
# usa come data target (data_3_opt e sar) il 15 del mese target
# cerca serie complete intorno a quel mese partendo dall'anno prima di data_3_org_opt
# se non la trova valida, prova gli anni precedenti fino al 2015
# se l'anno prima e < del 2015 salta tutta la serie
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
import shutil
from datetime import datetime, timedelta, timezone
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# percorso worksapce
script_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(script_dir) # /workspace/1_creazione_dataset

csv_lock = threading.Lock()

# --- IMPOSTAZIONE INTERVALLO RIGHE ---
# Imposta i valori da 1 in poi.
# Se vuoi fare tutto il file in un colpo solo, imposta entrambi a None.
RIGA_INIZIO = 10001
RIGA_FINE = 15000

# costanti
CONTAINER = False
MAX_WORKERS = 15        # Ripristinato a 15 come da originale
PIXEL_PERCENT = 5       # % pixel Nan in img SAR e OTTICHE
CLOUDY_PERCENTAGE = 25  # % copertura nuvolosa
RAGGIO_METRI = 1280     # raggio di 1280m (2560m/2), box 256x256
RANGE_GIORNI = 10       # +- giorni intorno alla data
GAP_MINIMO_GIORNI = 7   # giorni minimi di differenza tra fine intervallo precedente e inizio successivo

# percorsi csv 
csv_output_dir = os.path.join(PROJECT_ROOT, "csv")
complete_csv = os.path.join(csv_output_dir, "complete_series.csv")
incomplete_csv = os.path.join(csv_output_dir, "incomplete_series.csv")

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

def leggi_header_csv(filepath):
    """Legge la prima riga (header) di un CSV già esistente. None se il file non esiste o è vuoto."""
    if not os.path.exists(filepath):
        return None
    with open(filepath, mode='r', encoding='utf-8') as f:
        reader = csv.reader(f)
        try:
            return next(reader)
        except StopIteration:
            return None

def costruisci_header_completo(fieldnames_in):
    """Ricostruisce l'header di complete_series.csv (colonne base + Data/ID SAR-OPT + Anni_Offset),
    usato solo come fallback se complete_series.csv non esiste ancora."""
    base = [f for f in fieldnames_in if f.lower() not in ['target_month', 'motivo_scarto']]
    extra = []
    for i in range(1, 5):
        extra.append(f'Data_{i}_SAR')
        extra.append(f'ID_{i}_SAR')
        extra.append(f'Data_{i}_OPT')
        extra.append(f'ID_{i}_OPT')
    extra.append('Anni_Offset')
    return base + extra

# calcolo data (necessario per cambiare anno in caso di serie a cavallo tra due anni)
def calcola_data_target(anno, mese_base, offset_mesi):
    mese_calc = mese_base - 1 + offset_mesi
    nuovo_anno = anno + (mese_calc // 12)
    nuovo_mese = (mese_calc % 12) + 1
    return datetime(nuovo_anno, nuovo_mese, 15)

def bilancia_intervalli(dt_targets, range_giorni, gap_minimo):
    
    starts = [d - timedelta(days=range_giorni) for d in dt_targets]
    ends = [d + timedelta(days=range_giorni) for d in dt_targets]
    
    for i in range(len(dt_targets) - 1):
        gap = (starts[i+1] - ends[i]).days
        
        if gap < gap_minimo:
            deficit = gap_minimo - gap
            shift_sinistra = deficit // 2
            shift_destra = deficit - shift_sinistra
            
            ends[i] -= timedelta(days=shift_sinistra)
            starts[i+1] += timedelta(days=shift_destra)
            
    for s, e in zip(starts, ends):
        if s > e:
            return None, None
            
    return starts, ends

def elabora_recupero():
    if not os.path.exists(incomplete_csv):
        print(f"❌ File incompleti non trovato: {incomplete_csv}")
        return

    righe_complete_start = conta_righe_csv(complete_csv)
    righe_incomplete_start = conta_righe_csv(incomplete_csv)
    totale_start = righe_complete_start + righe_incomplete_start

    # Legge TUTTO il file per non perdere le righe non processate
    tutte_le_righe = []
    with open(incomplete_csv, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames_in = reader.fieldnames
        for riga in reader:
            tutte_le_righe.append(riga)
            
    # --- FILTRAGGIO RIGHE DA ELABORARE ---
    if RIGA_INIZIO is not None and RIGA_FINE is not None:
        inizio_idx = max(0, RIGA_INIZIO - 1) 
        fine_idx = min(len(tutte_le_righe), RIGA_FINE)
        tasks_da_processare = tutte_le_righe[inizio_idx:fine_idx]
        print(f"✅ Selezionate righe da {inizio_idx + 1} a {fine_idx} per l'elaborazione (Totale: {len(tasks_da_processare)})")
    else:
        tasks_da_processare = tutte_le_righe
        print(f"✅ Selezionate tutte le {len(tutte_le_righe)} righe per l'elaborazione")

    # Le colonne di complete_series.csv sono diverse da quelle di incomplete_series.csv
    # (contengono Data_X_SAR/OPT, ID_X_SAR/OPT, Anni_Offset e NON Motivo_Scarto/target_month).
    # Se il file esiste già (caso normale), usiamo il suo header reale per allineare la scrittura.
    file_completi_esiste = os.path.exists(complete_csv)

    header_esistente = leggi_header_csv(complete_csv)
    if header_esistente:
        fieldnames_out = header_esistente
    else:
        # fallback: file assente o vuoto, ricostruiamo l'header come farebbe lo script 1
        fieldnames_out = costruisci_header_completo(fieldnames_in)

    serie_recuperate_set = set()

    with open(complete_csv, mode='a', encoding='utf-8', newline='') as f_out:
        writer_out = csv.DictWriter(f_out, fieldnames=fieldnames_out)
        if not file_completi_esiste:
            writer_out.writeheader()

        def processa_singola_serie(riga, riga_num):
            nome_serie = riga['Nome_Serie']
            
            try:
                mese_assegnato = int(float(riga.get('target_month', 0)))
                if mese_assegnato == 0:
                    return None 
            except:
                return None
                
            # Estrazione anno originale (prima da OPT, poi da SAR come fallback)
            data_org_opt = str(riga.get('Data_3_org_OPT', '')).strip()
            data_org_sar = str(riga.get('Data_3_org_SAR', '')).strip()
            
            anno_originale = None
            
            # Controllo OPT
            if data_org_opt and data_org_opt.lower() != 'nan':
                try:
                    anno_originale = int(data_org_opt.split('-')[0])
                except ValueError:
                    pass
                    
            # Se OPT è fallito o assente, provo con SAR
            if anno_originale is None and data_org_sar and data_org_sar.lower() != 'nan':
                try:
                    anno_originale = int(data_org_sar.split('-')[0])
                    print(f"Riga {riga_num} | 🔴 Serie {nome_serie}: Data_3_org_OPT mancante, uso Data_3_org_SAR per anno.")
                except ValueError:
                    pass

            # Se entrambi sono falliti o assenti, salto la serie
            if anno_originale is None:
                print(f"Riga {riga_num} | 🔴 Serie {nome_serie}: Data_3_org_OPT e Data_3_org_SAR mancanti. Impossibile calcolare anno.")
                return None

            # --- NUOVO CONTROLLO DEI 6 MESI ---
            # Estraiamo la data originale esatta (usiamo OPT, se fallisce SAR) come oggetto datetime
            try:
                data_str_valida = data_org_opt if data_org_opt and data_org_opt.lower() != 'nan' else data_org_sar
                data_evento_dt = datetime.strptime(data_str_valida[:10], '%Y-%m-%d')
                
                # Anno di partenza predefinito
                anno_partenza = anno_originale - 1
                
                # Creiamo la data target fittizia per il controllo
                data_target_test = datetime(anno_partenza, mese_assegnato, 15)
                
                # Se la distanza tra l'evento e il target è minore di 92 giorni (circa 3 mesi)
                if (data_evento_dt - data_target_test).days < 92:
                    anno_partenza -= 1  # Scendiamo di un altro anno
                    print(f"Riga {riga_num} | ⚠️ Mese target ({mese_assegnato}) troppo vicino all'evento. Anno di partenza scalato a {anno_partenza}.")
                    
            except Exception as e:
                # Fallback di sicurezza se la data è malformata
                anno_partenza = anno_originale - 1
            # ----------------------------------

            if anno_partenza < 2015:
                print(f"Riga {riga_num} | 🔴 Serie {nome_serie}: Anno di partenza ({anno_partenza}) < 2015. Serie saltata.")
                return None

            print(f"Riga {riga_num} | Processo Serie: {nome_serie} (Mese Target: {mese_assegnato}, Partenza: {anno_partenza})")
        
            # --- CREAZIONE DI UN'UNICA BOUNDING BOX COMUNE (Centro OPT prioritario) ---
            try:
                import pandas as pd # Assicuriamoci che pandas sia accessibile per il check isna
                if not pd.isna(riga.get('Lat_Centro_OPT')) and str(riga.get('Lat_Centro_OPT')).lower() != 'nan':
                    lat_centro = float(riga['Lat_Centro_OPT'])
                    lon_centro = float(riga['Lon_Centro_OPT'])
                else:
                    lat_centro = float(riga['Lat_Centro_SAR'])
                    lon_centro = float(riga['Lon_Centro_SAR'])
            except (ValueError, TypeError, ImportError):
                return None

            punto_comune = ee.Geometry.Point([lon_centro, lat_centro])
            bbox_comune = punto_comune.buffer(RAGGIO_METRI).bounds()

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

            # Ricerca dall'anno prima dell'evento fino al 2015 compreso
            for anno_test in range(anno_partenza, 2014, -1):
                
                # --- CALCOLO E BILANCIAMENTO DATE TARGET PER L'ANNO CORRENTE ---
                date_target_anno = []
                for step in range(1, 5):
                    offset_mesi = {1: -2, 2: -1, 3: 0, 4: 1}[step]
                    date_target_anno.append(calcola_data_target(anno_test, mese_assegnato, offset_mesi))
                
                # Calcolo intervalli bilanciati
                starts, ends = bilancia_intervalli(date_target_anno, RANGE_GIORNI, GAP_MINIMO_GIORNI)
                
                # Se il bilanciamento è impossibile (start > end), saltiamo direttamente questo anno
                if starts is None:
                    continue
                
                anno_valido = True
                nuove_date_effettive = {}
                orbita_fissata = None  # Per garantire la coerenza geometrica del SAR nell'anno
                
                for step in range(1, 5):
                    idx = step - 1
                    
                    data_inizio = starts[idx].strftime('%Y-%m-%d')
                    # +1 giorno perchè .filterDate in GEE è esclusivo per la data finale
                    data_fine = (ends[idx] + timedelta(days=1)).strftime('%Y-%m-%d')

                    # --- SAR ---
                    s1_col = ee.ImageCollection('COPERNICUS/S1_GRD') \
                        .filterBounds(punto_comune) \
                        .filterDate(data_inizio, data_fine) \
                        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
                        .filter(ee.Filter.eq('instrumentMode', 'IW'))
                        
                    # Se abbiamo già fissato l'orbita per la prima immagine di questa serie, la imponiamo alle successive
                    if orbita_fissata:
                        s1_col = s1_col.filter(ee.Filter.eq('orbitProperties_pass', orbita_fissata))
                        
                    s1_filtrata = s1_col.map(calcola_nan_sar).filter(ee.Filter.lt('nan_percent', PIXEL_PERCENT)).sort('nan_percent')
                    
                    try:
                        migliore_s1 = ee.Image(s1_filtrata.first())
                        id_img_s1 = migliore_s1.get('system:id').getInfo()
                        if not id_img_s1: raise Exception("No ID")
                        
                        # Fissiamo l'orbita alla prima immagine SAR valida trovata
                        if orbita_fissata is None:
                            orbita_fissata = migliore_s1.get('orbitProperties_pass').getInfo()
                        
                        timestamp_s1 = migliore_s1.get('system:time_start').getInfo()
                        data_eff_s1 = datetime.fromtimestamp(timestamp_s1 / 1000.0, tz=timezone.utc).strftime('%Y-%m-%d')
                        
                        nuove_date_effettive[f'Data_{step}_SAR'] = data_eff_s1
                        nuove_date_effettive[f'ID_{step}_SAR'] = id_img_s1 
                        
                        print(f"🔵 SAR t{step} ({anno_test}) | Data: {data_eff_s1}")
                    except Exception:
                        anno_valido = False
                        break

                    # --- OTTICO ---
                    s2_col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
                        .filterBounds(punto_comune) \
                        .filterDate(data_inizio, data_fine) 
                    
                    s2_filtrata = s2_col.map(analizza_pixel_ottici)\
                        .filter(ee.Filter.lt('nan_percent', PIXEL_PERCENT))\
                        .filter(ee.Filter.lt('roi_cloud_percent', CLOUDY_PERCENTAGE))\
                        .sort('roi_cloud_percent')
                        
                    try:
                        migliore_s2 = ee.Image(s2_filtrata.first())
                        id_img_s2 = migliore_s2.get('system:id').getInfo()
                        if not id_img_s2: raise Exception("No ID")
                        
                        timestamp_s2 = migliore_s2.get('system:time_start').getInfo()
                        data_eff_s2 = datetime.fromtimestamp(timestamp_s2 / 1000.0, tz=timezone.utc).strftime('%Y-%m-%d')
                        
                        nuove_date_effettive[f'Data_{step}_OPT'] = data_eff_s2
                        nuove_date_effettive[f'ID_{step}_OPT'] = id_img_s2 
                        
                        print(f"🔵 OTTICO t{step} ({anno_test}) | Data: {data_eff_s2}")
                    except Exception:
                        anno_valido = False
                        break 

                # se anno e valido (4 img sar + 4 img opt trovate)
                if anno_valido:
                    print(f"🟢 Serie {nome_serie} completata con l'anno {anno_test}!")

                    for chiave, valore in nuove_date_effettive.items():
                        riga[chiave] = valore

                    riga['Anni_Offset'] = anno_originale - anno_test

                    riga.pop('target_month', None)
                    riga.pop('Motivo_Scarto', None)

                    return riga

            print(f"🔴 Serie {nome_serie}: Impossibile completare (nessun anno valido trovato fino al 2015)")
            return None

        print(f"\nVerifica in parallelo per {len(tasks_da_processare)} serie incomplete")
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            
            futures = {executor.submit(processa_singola_serie, task.copy(), i + 1): task for i, task in enumerate(tasks_da_processare)}
            
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
                    
    print(f"\n✔️ FINE. Nuove serie recuperate in questo blocco: {len(serie_recuperate_set)}")

    if len(serie_recuperate_set) > 0:
        print(f"\nUpdate incomplete_series.csv: rimozione di {len(serie_recuperate_set)} serie recuperate")
        
        righe_rimanenti = [task for task in tutte_le_righe if task['Nome_Serie'] not in serie_recuperate_set]
        
        with open(incomplete_csv, mode='w', encoding='utf-8', newline='') as f_incomp:
            writer_inc = csv.DictWriter(f_incomp, fieldnames=fieldnames_in)
            writer_inc.writeheader()
            writer_inc.writerows(righe_rimanenti)
            
        print(f"File {incomplete_csv} aggiornato.")

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