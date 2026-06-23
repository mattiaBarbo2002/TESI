import tarfile
import zipfile
import io
import json

# Percorso del file TAR di Sentinel-2 (Ottico)
percorso_tar = "/workspace/data/ImpactMesh-Flood/val/S1RTC.tar"

print(f"Apro il file principale (TAR): {percorso_tar}")

try:
    with tarfile.open(percorso_tar, 'r') as tar:
        
        # Trova gli zip interni
        inner_zips = [m for m in tar.getmembers() if m.name.endswith('.zip')]
        
        if not inner_zips:
            print("Nessun file .zip trovato nel file TAR!")
        else:
            primo_inner_zip = inner_zips[0]
            print(f"Leggo il primo zip interno: {primo_inner_zip.name} ...")
            
            # Estrae in RAM
            extracted_file = tar.extractfile(primo_inner_zip)
            
            if extracted_file is not None:
                dati_inner_zip = extracted_file.read()
                
                with zipfile.ZipFile(io.BytesIO(dati_inner_zip)) as inner_zip:
                    
                    zattrs_files = [f for f in inner_zip.namelist() if f.endswith('.zattrs')]
                    
                    if zattrs_files:
                        file_target = zattrs_files[0]
                        print(f"\nTrovato file: {file_target}")
                        
                        contenuto = inner_zip.read(file_target)
                        dati_json = json.loads(contenuto)
                        
                        print("\n--- CONTENUTO DI .ZATTRS (SENTINEL-2) ---")
                        print(json.dumps(dati_json, indent=4))
                    else:
                        print("File .zattrs non trovato!")
            else:
                print("Errore durante la lettura dello zip.")
except FileNotFoundError:
    print(f"ERRORE: Il file {percorso_tar} non esiste. Controlla il nome!")