"""Prepared benchmark cases (written by scripts/data/prep_case_*.py), one directory per section, on a 0.85 um/px grid:
  dapi.npy       float16 H x W   log1p(DAPI), z-scored over tissue pixels
  lab.npy        int32   H x W   cell labels (0 = background), label k <-> cells.parquet row with label == k
  tx.npz         gy, gx (int32 grid coordinates), gidx (index into genes.json), one row per molecule
  genes.json     genes of the case's panel that belong to the kidney training vocabulary
  cells.parquet  label, cell_id, area_px, x_um, y_um, truth_<marker> (raw per-cell mean intensity), ...
  counts.npz     CSR cell x gene counts (rows ordered like cells.parquet, columns like counts_genes.json)
  meta.json      dataset, platform, tissue, markers {name: {truth, components (kidney proteins), own (coding genes)}}
"""
import json

import numpy as np
import pandas as pd


def load_case_image(cd):
    """inputs of the image model for case directory `cd`"""
    T = np.load(f'{cd}/tx.npz')
    return dict(dir=cd, meta=json.load(open(f'{cd}/meta.json')), genes=json.load(open(f'{cd}/genes.json')),
                dapi=np.load(f'{cd}/dapi.npy').astype(np.float32), lab=np.load(f'{cd}/lab.npy'),
                gy=T['gy'].astype(np.int64), gx=T['gx'].astype(np.int64), gidx=T['gidx'].astype(np.int64),
                cells=pd.read_parquet(f'{cd}/cells.parquet'))


def label_counts(lab):
    """number of pixels of every label (index 0 = background)"""
    return np.bincount(lab.ravel(), minlength=int(lab.max()) + 1).astype(np.float64)
