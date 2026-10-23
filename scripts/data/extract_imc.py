"""Extract IMC acquisitions (ROIs) from .mcd files: keep channels shared with the TxCellNet panel + DNA, float16, compressed.
The four .mcd files (figshare 32671185) are read from env MCD_DIR (default data/crlm_xenium_imc/mcd).
usage: extract_imc.py OUT_DIR
"""
import os, re, json, sys
import numpy as np
from readimc import MCDFile
KEEP = {'CD20': 'CD20', 'panCK': 'PanCK', 'VISTA': 'VISTA', 'CD16': 'CD16', 'CD163': 'CD163', 'Vimentin': 'Vimentin',
        'Ki-67': 'Ki-67', 'Ki67': 'Ki-67', 'CD31': 'CD31', 'CD45': 'CD45', 'CD11c': 'CD11c', 'CD4': 'CD4', 'E-cad': 'E-Cadherin',
        'Ecad': 'E-Cadherin', 'CD68': 'CD68', 'CD8': 'CD8A', 'PD-1': 'PD-1', 'PD1': 'PD-1', 'PD-L1': 'PD-L1', 'PDL1': 'PD-L1',
        'B-Catenin': 'Beta-catenin', 'Bcatenin': 'Beta-catenin', 'CD3': 'CD3E', 'CD45RO': 'CD45RO', 'Keratin8_18': 'K8_18',
        'Keratin 818': 'K8_18', 'GranzymeB': 'GranzymeB', 'aSMA': 'alphaSMA', 'DNA1': 'DNA1', 'DNA2': 'DNA2'}
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
for f in ["tumor1_M242T.mcd", "tumor2_M131T.mcd", "AdjNormal_M131N.mcd", "postxnmIMC_tumor3_M397T.mcd"]:
    with MCDFile(os.path.join(os.environ.get('MCD_DIR', 'data/crlm_xenium_imc/mcd'), f)) as m:
        for a in m.slides[0].acquisitions:
            if a.description.endswith('_PM'): continue  # low-resolution overview acquisition
            lab = [re.sub(r'^\d+[A-Za-z]+_', '', l).strip() for l in a.channel_labels]
            lab = [re.sub(r'_[A-Z][a-z]?\d+$', '', l).strip() for l in lab]
            sel = [(i, KEEP[l]) for i, l in enumerate(lab) if l in KEEP]
            img = m.read_acquisition(a)
            arr = np.stack([img[i] for i, _ in sel]).astype(np.float16)
            key = f"{f.replace('.mcd', '')}__{a.description}"
            np.savez_compressed(f'{out}/{key}.npz', img=arr, channels=np.array([n for _, n in sel]),
                                meta=json.dumps(dict(file=f, acq=a.id, desc=a.description, width_um=a.width_um, height_um=a.height_um,
                                                     width_px=a.width_px, height_px=a.height_px,
                                                     roi_points_um=[list(map(float, p)) for p in (a.roi_points_um or [])])))
            print(key, arr.shape, 'channels', len(sel), 'MB', round(os.path.getsize(f'{out}/{key}.npz') / 1e6, 1), flush=True)
