import os
import shutil
import pandas as pd
import numpy as np
import rasterio
import digitalhub as dh
import sys
import matplotlib
import re
import xarray as xr
import zarr
import tarfile
import subprocess
from glob import glob
from digitalhub_runtime_python import handler


matplotlib.use("Agg")  


@handler()
def download_impactmesh_split(
    repo_id: str = "ibm-esa-geospatial/ImpactMesh-Flood",
    split: str = None,   # NUOVO: None = scarica l'intero dataset (train+val+test), non solo uno split
    output_name: str = "Floods_Anomalies_val",
    max_workers: int = 8,
    hf_token: str = ""
,   # dataset pubblico CC-BY 4.0, di norma non serve
):
    os.environ["HF_HUB_DISABLE_XET"] = "1"        # non basta da sola (bug noto, vedi sotto) ma la lascio comunque
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"  # default e' solo 10s, troppo poco per tar da svariati GB
 
    # NUOVO: HF_HUB_DISABLE_XET da sola e' inaffidabile su piu' versioni recenti di huggingface_hub
    # (bug noto e diffuso) - l'unico modo confermato di ricadere davvero sul percorso HTTP classico
    # e' disinstallare hf-xet, cosi' huggingface_hub non lo trova importabile e non puo' usarlo
    print("Disinstallo hf-xet per forzare il download HTTP classico...", flush=True)
    subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "-y", "hf-xet", "hf_xet"],
        check=False,   # non deve far fallire il job se il pacchetto non c'e' proprio o il nome non combacia
    )
 
    from huggingface_hub import snapshot_download
 
    project_data = dh.get_project("datasets")
 
    local_dir = "/data/impactmesh_download"
    os.makedirs(local_dir, exist_ok=True)
 
    allow_patterns = [f"{split}/*"] if split is not None else None   # NUOVO: None -> nessun filtro, scarica tutto
    print(f"Download {repo_id}" + (f" (solo '{split}/*')" if split else " (INTERO dataset, tutti gli split)") + "...", flush=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=allow_patterns,
        local_dir=local_dir,
        max_workers=max_workers,
        token=hf_token,
    )
    print("OK -> download terminato", flush=True)
 
    check_dir = os.path.join(local_dir, split) if split is not None else local_dir   # NUOVO
    if not os.path.isdir(check_dir):
        print(f"EXC -> '{check_dir}' non trovata dopo il download, controlla il nome dello split", flush=True)
        return "TERMINATO -> contenuto non trovato"
 
    for f in sorted(os.listdir(check_dir)):   # NUOVO: gestisce sia file sia sottocartelle (train/val/test)
        full_path = os.path.join(check_dir, f)
        if os.path.isfile(full_path):
            size_gb = os.path.getsize(full_path) / 1e9
            print(f"  {f} ({size_gb:.2f} GB)", flush=True)
        else:
            print(f"  {f}/ (cartella)", flush=True)
 
    print(f"Upload come artifact '{output_name}'...", flush=True)
    project_data.log_artifact(
        name=output_name,
        kind='artifact',
        source=local_dir,   # con split=None, local_dir contiene train/val/test; con uno split, solo quella sottocartella
    )
    print(f"OK -> artifact '{output_name}' caricato", flush=True)
 
    shutil.rmtree(local_dir, ignore_errors=True)
    return "TERMINATO -> contenuto scaricato e caricato come artifact"







@handler()
def inspect_mask_structure(
    anomalies_artifact: str = "Floods_Anomalies",
    split: str = "test",
    n_samples: int = 5,
):
    project_data = dh.get_project("datasets")
 
    print(f"Download {anomalies_artifact}...", flush=True)
    anom_path = project_data.get_artifact(anomalies_artifact).download("/data/anomalies_mask_check")
    print("OK -> download terminato", flush=True)
 
    mask_tar_path = os.path.join(anom_path, split, "MASK.tar")
    if not os.path.exists(mask_tar_path):
        print(f"EXC -> {mask_tar_path} non trovato", flush=True)
        return "TERMINATO -> MASK.tar non trovato"
 
    tf = tarfile.open(mask_tar_path, 'r:')
    names = tf.getnames()
    print(f"\ntotale membri in MASK.tar ({split}): {len(names)}", flush=True)
    print(f"primi 10 nomi:", flush=True)
    for n in names[:10]:
        print(f"  {n}", flush=True)
 
    # stessi segmenti di path distinti che avevamo gia' controllato per S1RTC/S2L2A
    top_level = sorted(set(n.split('/')[0] for n in names))
    print(f"\nprimi segmenti di path distinti: {top_level[:10]}", flush=True)
 
    # apro i primi n_samples membri .zip (presumo stesso formato .zarr.zip, da confermare)
    sample_names = [n for n in names if n.endswith('.zip')][:n_samples]
    if not sample_names:
        print("\nEXC -> nessun membro .zip trovato - il formato potrebbe essere diverso da .zarr.zip", flush=True)
        print("primi 10 nomi SENZA filtro per vedere l'estensione reale:", flush=True)
        for n in names[:10]:
            print(f"  {n}", flush=True)
        tf.close()
        return "TERMINATO -> formato inatteso, vedi log"
 
    extract_tmp = "/data/mask_extract_tmp"
    for name in sample_names:
        print(f"\n{'=' * 60}", flush=True)
        print(f"ispeziono: {name}", flush=True)
        try:
            os.makedirs(extract_tmp, exist_ok=True)
            tf.extract(name, extract_tmp)
            local_path = os.path.join(extract_tmp, name)
 
            store = zarr.storage.ZipStore(local_path, mode='r')
            ds = xr.open_zarr(store, consolidated=True)
 
            print(f"variabili: {list(ds.data_vars)}", flush=True)
            print(f"coordinate: {list(ds.coords)}", flush=True)
            print(f"dimensioni: {dict(ds.dims)}", flush=True)
 
            for var in ds.data_vars:
                arr = ds[var].values
                print(f"  variabile '{var}': shape={arr.shape}, dtype={arr.dtype}", flush=True)
                vals, counts = np.unique(arr[~np.isnan(arr)] if np.issubdtype(arr.dtype, np.floating) else arr, return_counts=True)
                print(f"    valori unici presenti: {dict(zip(vals.tolist(), counts.tolist()))}", flush=True)
 
            ds.close()
            store.close()
        except Exception as e:
            print(f"EXC -> {name}: {e}", flush=True)
        finally:
            import shutil
            shutil.rmtree(extract_tmp, ignore_errors=True)
 
    tf.close()
    return "TERMINATO -> ispezione MASK completata"


@handler()
def inspect_mask_tif(
    anomalies_artifact: str = "Floods_Anomalies_val",
    split: str = "val",
    n_samples: int = 5,
):
    project_data = dh.get_project("datasets")
 
    print(f"Download {anomalies_artifact}...", flush=True)
    anom_path = project_data.get_artifact(anomalies_artifact).download("/data/anomalies_mask_check")
    print("OK -> download terminato", flush=True)
 
    mask_tar_path = os.path.join(anom_path, split, "MASK.tar")
    if not os.path.exists(mask_tar_path):
        print(f"EXC -> {mask_tar_path} non trovato", flush=True)
        return "TERMINATO -> MASK.tar non trovato"
 
    tf = tarfile.open(mask_tar_path, 'r:')
    names = tf.getnames()
    print(f"\ntotale membri in MASK.tar ({split}): {len(names)}", flush=True)
 
    # controllo esplicito: esistono suffissi diversi da '_annotation_flood.tif'?
    # (es. un file separato '_annotation_water.tif' che non comparirebbe tra i primi 10)
    suffixes = set()
    for n in names:
        base = os.path.basename(n)
        if '_annotation_' in base:
            suffixes.add(base.split('_annotation_')[-1])
    print(f"suffissi '_annotation_*' distinti trovati in TUTTI i {len(names)} nomi: {suffixes}", flush=True)
 
    extract_tmp = "/data/mask_tif_extract_tmp"
    sample_names = names[:n_samples]
 
    for name in sample_names:
        print(f"\n{'=' * 60}", flush=True)
        print(f"ispeziono: {name}", flush=True)
        try:
            os.makedirs(extract_tmp, exist_ok=True)
            tf.extract(name, extract_tmp)
            local_path = os.path.join(extract_tmp, name)
 
            with rasterio.open(local_path) as src:
                print(f"  count bande: {src.count}, dtype: {src.dtypes}, shape: {src.shape} (H,W)", flush=True)
                print(f"  CRS: {src.crs}", flush=True)
                print(f"  nodata dichiarato: {src.nodata}", flush=True)
                data = src.read()   # (bande, H, W)
                print(f"  shape array letto: {data.shape}", flush=True)
 
                vals, counts = np.unique(data, return_counts=True)
                print(f"  valori unici presenti: {dict(zip(vals.tolist(), counts.tolist()))}", flush=True)
        except Exception as e:
            print(f"EXC -> {name}: {e}", flush=True)
        finally:
            shutil.rmtree(extract_tmp, ignore_errors=True)
 
    tf.close()
    return "TERMINATO -> ispezione MASK (.tif) completata"



@handler()
def inspect_full_metadata(
    anomalies_artifact: str = "Floods_Anomalies_val",
    split: str = "val",
    n_samples: int = 2,
):
    project_data = dh.get_project("datasets")
 
    print(f"Download {anomalies_artifact}...", flush=True)
    anom_path = project_data.get_artifact(anomalies_artifact).download("/data/full_metadata_check")
    print("OK -> download terminato\n", flush=True)
 
    split_dir = os.path.join(anom_path, split)
    extract_tmp = "/data/full_metadata_extract_tmp"
 
    for modality, suffix in [("S1RTC", "S1RTC"), ("S2L2A", "S2L2A")]:
        tar_paths = sorted(set(
            glob.glob(os.path.join(split_dir, f"{modality}.tar")) +
            glob.glob(os.path.join(split_dir, f"{modality}_*.tar"))
        ))
        if not tar_paths:
            print(f"EXC -> nessun tar trovato per {modality}", flush=True)
            continue
 
        pattern = re.compile(r"^[^/]+/(.*?)_" + re.escape(suffix) + r"\.zarr\.zip$")
 
        for tar_path in tar_paths:
            tf = tarfile.open(tar_path, 'r:')
            sample_members = [n for n in tf.getnames() if pattern.match(n)][:n_samples]
 
            for name in sample_members:
                print(f"\n{'=' * 70}", flush=True)
                print(f"{modality} - {name}", flush=True)
                try:
                    os.makedirs(extract_tmp, exist_ok=True)
                    tf.extract(name, extract_tmp)
                    local_path = os.path.join(extract_tmp, name)
 
                    store = zarr.storage.ZipStore(local_path, mode='r')
                    ds = xr.open_zarr(store, consolidated=True)
 
                    print("ATTRIBUTI GLOBALI DEL DATASET (ds.attrs):", flush=True)
                    for k, v in ds.attrs.items():
                        print(f"  {k} = {v}", flush=True)
 
                    print("\nCOORDINATE (ds.coords):", flush=True)
                    for c in ds.coords:
                        vals = ds.coords[c].values
                        preview = vals if vals.size <= 10 else f"{vals[:5]} ... ({vals.size} valori totali)"
                        print(f"  {c}: {preview}", flush=True)
 
                    print("\nATTRIBUTI PER VARIABILE:", flush=True)
                    for var in ds.data_vars:
                        if ds[var].attrs:
                            print(f"  {var}: {dict(ds[var].attrs)}", flush=True)
 
                    ds.close()
                    store.close()
                except Exception as e:
                    print(f"EXC -> {name}: {e}", flush=True)
                finally:
                    shutil.rmtree(extract_tmp, ignore_errors=True)
 
            tf.close()
            break   # un solo file tar per modalita' e' sufficiente per questa ispezione
 
    return "TERMINATO -> ispezione metadati completa"


@handler()
def extract_mask_pixel_counts(
    input_csv_artifact: str = "anomaly_ds_filtrato",   # il CSV che hai gia', con Nome_Serie e le altre colonne
    anomalies_artifact: str = "Floods_Anomalies",       # o Floods_Anomalies_v2 se nel frattempo e' pronto
    split: list = None,                                # None -> ["train", "val", "test"]
    output_name: str = "anomaly_ds_with_mask",
):
    project_data = dh.get_project("datasets")

    if split is None:
        split = ["train", "val", "test"]

    print(f"Lettura CSV di input: {input_csv_artifact}", flush=True)
    csv_path = project_data.get_artifact(input_csv_artifact).download("/data/input_mask_extract.csv", overwrite=True)
    df = pd.read_csv(csv_path)
    print(f"OK -> {len(df)} righe", flush=True)

    print(f"Download {anomalies_artifact}...", flush=True)
    anom_path = project_data.get_artifact(anomalies_artifact).download("/data/anomalies_mask_extract")
    print("OK -> download terminato", flush=True)

    # ------------------------------------------------------------------
    # indicizzo MASK.tar per ogni split - stesso schema 'cerca singolo o
    # diviso in parti' gia' usato per S1RTC/S2L2A, per sicurezza/uniformita'
    # anche se qui non ci aspettiamo file divisi
    # ------------------------------------------------------------------
    def _list_mask_members(split_dir):
        tar_paths = sorted(set(
            glob.glob(os.path.join(split_dir, "MASK.tar")) +
            glob.glob(os.path.join(split_dir, "MASK_*.tar"))
        ))
        handles = []
        members = {}   # ID -> (tar_handle, nome membro)
        for tar_path in tar_paths:
            tf = tarfile.open(tar_path, 'r:')
            handles.append(tf)
            for name in tf.getnames():
                base = os.path.basename(name)
                if base.endswith('_annotation_flood.tif'):
                    ID = base[:-len('_annotation_flood.tif')]
                    members[ID] = (tf, name)
        return handles, members

    all_handles = []
    id_to_member = {}
    for split in split:
        split_dir = os.path.join(anom_path, split)
        if not os.path.isdir(split_dir):
            print(f"EXC -> split '{split}' non trovato in {anom_path}, salto", flush=True)
            continue
        handles, members = _list_mask_members(split_dir)
        all_handles.extend(handles)
        print(f"OK -> '{split}': {len(members)} maschere indicizzate", flush=True)
        for ID, v in members.items():
            if ID not in id_to_member:   # non dovrebbero sovrapporsi tra split, ma il primo trovato vince
                id_to_member[ID] = v

    print(f"OK -> {len(id_to_member)} ID totali indicizzati su tutti gli split\n", flush=True)

    # ------------------------------------------------------------------
    # per ogni riga del CSV, estraggo e leggo la maschera corrispondente
    # ------------------------------------------------------------------
    mask_water, mask_flood = [], []
    n_ok, n_missing, n_failed = 0, 0, 0
    extract_tmp = "/data/mask_extract_tmp"
    total = len(df)

    for i, ID in enumerate(df['Nome_Serie']):
        if i % 1000 == 0:
            print(f"  {i}/{total} ({n_ok} ok, {n_missing} non trovate, {n_failed} errori)", flush=True)

        if ID not in id_to_member:
            mask_water.append(np.nan)
            mask_flood.append(np.nan)
            n_missing += 1
            continue

        tf, member_name = id_to_member[ID]
        try:
            os.makedirs(extract_tmp, exist_ok=True)
            tf.extract(member_name, extract_tmp)
            local_path = os.path.join(extract_tmp, member_name)
            with rasterio.open(local_path) as src:
                data = src.read(1)
            mask_water.append(int((data == 1).sum()))
            mask_flood.append(int((data == 2).sum()))
            n_ok += 1
        except Exception as e:
            print(f"EXC -> {ID}: {e}", flush=True)
            mask_water.append(np.nan)
            mask_flood.append(np.nan)
            n_failed += 1
        finally:
            shutil.rmtree(extract_tmp, ignore_errors=True)

    for tf in all_handles:
        tf.close()

    df['Mask_water'] = mask_water
    df['Mask_flood'] = mask_flood

    print(f"\nOK -> {n_ok} maschere lette, {n_missing} ID non trovati, {n_failed} errori di lettura", flush=True)

    out_path = f"/data/{output_name}.csv"
    df.to_csv(out_path, index=False)

    project_data.log_artifact(name=output_name, kind='artifact', source=out_path)
    print(f"OK -> '{output_name}' caricato come artifact", flush=True)

    return f"TERMINATO -> {n_ok} maschere estratte, {n_missing} mancanti, {n_failed} errori"    





@handler()
def build_full_metadata_csv(
    anomalies_artifact: str = "Floods_Anomalies_v2",
    splits: list = None,            # None -> ["train", "val", "test"]
    output_name: str = "anomaly_ds_v2",
    max_series: int = 0,            # 0 = tutte; usa un numero piccolo per un primo test
):
    project_data = dh.get_project("datasets")

    if splits is None:
        splits = ["train", "val", "test"]

    print(f"Download {anomalies_artifact}...", flush=True)
    anom_path = project_data.get_artifact(anomalies_artifact).download("/data/full_meta_dl")
    print("OK -> download terminato\n", flush=True)

    # ------------------------------------------------------------------
    # indicizzazione tar - stesso schema 'singolo o diviso in parti'
    # gia' usato per crop_norm_artifact_anomalies / extract_mask_pixel_counts
    # ------------------------------------------------------------------
    def _list_zarr_groups(split_dir, modality_label):
        tar_paths = sorted(set(
            glob.glob(os.path.join(split_dir, f"{modality_label}.tar")) +
            glob.glob(os.path.join(split_dir, f"{modality_label}_*.tar"))
        ))
        pattern = re.compile(r"^[^/]+/(.*?)_" + re.escape(modality_label) + r"\.zarr\.zip$")
        handles = []
        groups = {}   # ID -> (tar_handle, nome membro) - un solo file per ID, confermato in precedenza
        for tar_path in tar_paths:
            tf = tarfile.open(tar_path, 'r:')
            handles.append(tf)
            for name in tf.getnames():
                m = pattern.match(name)
                if m:
                    ID = m.group(1)
                    if ID not in groups:
                        groups[ID] = (tf, name)
        return handles, groups

    def _list_mask_members(split_dir):
        tar_paths = sorted(set(
            glob.glob(os.path.join(split_dir, "MASK.tar")) +
            glob.glob(os.path.join(split_dir, "MASK_*.tar"))
        ))
        handles = []
        members = {}
        for tar_path in tar_paths:
            tf = tarfile.open(tar_path, 'r:')
            handles.append(tf)
            for name in tf.getnames():
                base = os.path.basename(name)
                if base.endswith('_annotation_flood.tif'):
                    ID = base[:-len('_annotation_flood.tif')]
                    members[ID] = (tf, name)
        return handles, members

    rows = []
    extract_tmp = "/data/full_meta_extract_tmp"
    n_ok, n_failed = 0, 0
    stop = False

    for split in splits:
        if stop:
            break

        split_dir = os.path.join(anom_path, split)
        if not os.path.isdir(split_dir):
            print(f"EXC -> split '{split}' non trovato, salto", flush=True)
            continue

        print(f"Indicizzazione split '{split}'...", flush=True)
        handles_s1, groups_s1 = _list_zarr_groups(split_dir, "S1RTC")
        handles_s2, groups_s2 = _list_zarr_groups(split_dir, "S2L2A")
        handles_mask, members_mask = _list_mask_members(split_dir)

        common_ids = sorted(set(groups_s1) & set(groups_s2))
        print(f"OK -> '{split}': {len(common_ids)} serie con SAR+OTTICO, "
              f"{len(members_mask)} maschere disponibili", flush=True)

        for ID in common_ids:
            if max_series > 0 and n_ok >= max_series:
                stop = True
                break

            try:
                os.makedirs(extract_tmp, exist_ok=True)

                tf_s1, name_s1 = groups_s1[ID]
                tf_s1.extract(name_s1, extract_tmp)
                store_s1 = zarr.storage.ZipStore(os.path.join(extract_tmp, name_s1), mode='r')
                ds_s1 = xr.open_zarr(store_s1, consolidated=True)
                attrs_s1 = ds_s1.attrs

                tf_s2, name_s2 = groups_s2[ID]
                tf_s2.extract(name_s2, extract_tmp)
                store_s2 = zarr.storage.ZipStore(os.path.join(extract_tmp, name_s2), mode='r')
                ds_s2 = xr.open_zarr(store_s2, consolidated=True)
                attrs_s2 = ds_s2.attrs

                data_sar = attrs_s1['datetime'].split(';')
                bbox_sar = attrs_s1['proj_bbox'].split(';')
                epsg_sar = attrs_s1['proj_code'].split(';')[0]

                data_opt = attrs_s2['datetime'].split(';')
                cloud_opt = attrs_s2['eo_cloud_cover'].split(';')
                mgrs_opt = attrs_s2['s2_mgrs_tile'].split(';')[0]
                epsg_opt = attrs_s2['proj_code'].split(';')[0]

                row = {
                    'Nome_Serie': ID,
                    'Data_1_org_SAR': data_sar[0], 'Data_2_org_SAR': data_sar[1],
                    'Data_3_org_SAR': data_sar[2], 'Data_4_org_SAR': data_sar[3],
                    'Bbox_1_SAR': bbox_sar[0], 'Bbox_2_SAR': bbox_sar[1],
                    'Bbox_3_SAR': bbox_sar[2], 'Bbox_4_SAR': bbox_sar[3],
                    'Lat_Centro_SAR': float(attrs_s1['center_lat']),
                    'Lon_Centro_SAR': float(attrs_s1['center_lon']),
                    'EPSG_SAR': epsg_sar,
                    'Data_1_org_OPT': data_opt[0], 'Data_2_org_OPT': data_opt[1],
                    'Data_3_org_OPT': data_opt[2], 'Data_4_org_OPT': data_opt[3],
                    'Cloud_1_OPT': float(cloud_opt[0]), 'Cloud_2_OPT': float(cloud_opt[1]),
                    'Cloud_3_OPT': float(cloud_opt[2]), 'Cloud_4_OPT': float(cloud_opt[3]),
                    'Lat_Centro_OPT': float(attrs_s2['center_lat']),
                    'Lon_Centro_OPT': float(attrs_s2['center_lon']),
                    'MGRS_Tile_OPT': mgrs_opt,
                    'EPSG_OPT': epsg_opt,
                }

                ds_s1.close(); store_s1.close()
                ds_s2.close(); store_s2.close()

                if ID in members_mask:
                    tf_m, name_m = members_mask[ID]
                    tf_m.extract(name_m, extract_tmp)
                    with rasterio.open(os.path.join(extract_tmp, name_m)) as src:
                        mdata = src.read(1)
                    row['water_mask'] = int((mdata == 1).sum())
                    row['flood_mask'] = int((mdata == 2).sum())
                else:
                    row['water_mask'] = np.nan
                    row['flood_mask'] = np.nan

                rows.append(row)
                n_ok += 1
                if n_ok % 500 == 0:
                    print(f"  {n_ok} serie processate...", flush=True)

            except Exception as e:
                n_failed += 1
                print(f"EXC -> serie {ID} ({split}): {e}", flush=True)
            finally:
                shutil.rmtree(extract_tmp, ignore_errors=True)

        for tf in handles_s1 + handles_s2 + handles_mask:
            tf.close()

    print(f"\nOK -> {n_ok} serie processate, {n_failed} fallite", flush=True)

    if n_ok == 0:
        return "TERMINATO -> nessuna serie processata, controlla i log EXC sopra"

    df = pd.DataFrame(rows)
    out_path = f"/data/{output_name}.csv"
    df.to_csv(out_path, index=False)

    project_data.log_artifact(name=output_name, kind='artifact', source=out_path)
    print(f"OK -> '{output_name}' caricato come artifact ({len(df)} righe)", flush=True)

    return f"TERMINATO -> {n_ok} serie estratte, {n_failed} fallite"