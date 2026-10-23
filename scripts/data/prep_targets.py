"""Image targets and a spatial split for WITHIN-dataset training on a prepared benchmark case.

Writes into data/cases/<case>/:
  Y.npy        float16 (C, H, W): log1p protein image on the case grid, z-scored over the target mask
  Y.json       {"channels": [...], "markers": {marker key in meta.json -> channel}}
  ymask.npy    bool (H, W): pixels where the target image is defined (tissue / IMC footprint / COMET footprint)
  split.json   {"x_edges_px": [...], "test": "odd stripes (0-based) of 8 equal-width x-stripes over the evaluated cells"}
Sources: Xenium multimodal stains (morphology_focus channels 1 = boundary, 3 = interior; level 2 = the case grid),
IMC ROIs warped with the stored affine (CRLM), COMET channels warped with the authors' matrix (Duchini).
Xenium bundles: data/ext/<case>, or data/pre/<bundle> for the seven sections added after the configuration freeze (FRESH).
usage: prep_targets.py CASE [CASE ...]
"""
import glob, json, sys, time, xml.etree.ElementTree as ET
import numpy as np, pandas as pd, tifffile, cv2
from scipy.ndimage import map_coordinates, gaussian_filter

FRESH = {'xen_breast_5k': 'pre_breast_5k', 'xen_cervix_5k': 'pre_cervix_5k', 'xen_ovary_5k': 'pre_ovary_5k', 'xen_prostate_5k': 'pre_prostate_5k',
         'xen_skin_5k': 'pre_skin_5k', 'xen_gbm_v1': 'pre_gbm_v1', 'xen_ovary_v1ff': 'pre_ovary_v1_ff'}
IMC = {'crlm_tumor1': 'tumor1_M242T__Acquisition_ROI_1', 'crlm_tumor2': 'tumor2_M131T__Acquisition_ROI_1', 'crlm_adjnormal': 'AdjNormal_M131N__Acquisition_ROI_1',
       'crlm_tumor3_A1': 'postxnmIMC_tumor3_M397T__ROI_002-A1', 'crlm_tumor3_A2': 'postxnmIMC_tumor3_M397T__ROI_003-A2',
       'crlm_tumor3_A3': 'postxnmIMC_tumor3_M397T__ROI_008-A3'}
COMET = {'CD20': 'CD20 - TRITC', 'CD45': 'CD45 - TRITC', 'CD45RA': 'CD45RA - TRITC', 'E-Cadherin': 'E-Cadherin - TRITC',
         'Vimentin': 'vimentin - TRITC', 'alphaSMA': 'αSMA - TRITC', 'PanCK': 'CK8 - TRITC'}
UM, XPX = 0.85, 0.2125


def log(*a): print(time.strftime('%H:%M:%S'), *a, flush=True)


def read_level(path, level=2):
    with tifffile.TiffFile(path, is_ome=False) as tf: return tf.series[0].levels[level].asarray()


def zlog(a, mask):
    l = np.log1p(np.clip(a.astype(np.float32), 0, None)); v = l[mask]
    return ((l - v.mean()) / (v.std() + 1e-4)).astype(np.float16)


for case in sys.argv[1:]:
    cd = f'data/cases/{case}'; meta = json.load(open(f'{cd}/meta.json')); lab = np.load(f'{cd}/lab.npy'); H, W = lab.shape
    cells = pd.read_parquet(f'{cd}/cells.parquet')
    if case.startswith('xen_'):
        mask = np.load(f'{cd}/tissue.npy') if __import__('os').path.exists(f'{cd}/tissue.npy') else (gaussian_filter(np.load(f'{cd}/dapi.npy').astype(np.float32), 4) > -0.5)
        chans = ['boundary', 'interior']
        src = f'data/pre/{FRESH[case]}' if case in FRESH else f'data/ext/{case}'  # the case's Xenium bundle
        fs = lambda i: sorted(glob.glob(f'{src}/morphology_focus/*{i:04d}*.ome.tif'))[0]
        bnd, itr = read_level(fs(1)).astype(np.float32), read_level(fs(3)).astype(np.float32)
        Y = np.stack([zlog(bnd, mask), zlog(itr, mask)])
        assert Y.shape[1:] == (H, W), (case, Y.shape, (H, W))
        mk = {k: ('interior' if k.startswith('interior') else 'boundary') for k in meta['markers']}
    elif case.startswith('crlm_'):
        Z = np.load(f'data/crlm_xenium_imc/imc_npz/{IMC[case]}.npz'); names = [str(c) for c in Z['channels']]; img = Z['img'].astype(np.float32)
        M = np.array(meta['affine_imc_px_to_crop_px'])
        mask = cv2.warpAffine(np.ones(img.shape[1:], np.float32), M, (W, H), flags=cv2.INTER_NEAREST) > 0.5
        chans = [m for m in meta['markers'] if m in names]
        Y = np.stack([zlog(cv2.warpAffine(img[names.index(c)], M, (W, H), flags=cv2.INTER_LINEAR), mask) for c in chans])
        mk = {k: k for k in chans}
    elif case.startswith('duchini_'):
        core = case[len('duchini_'):]; DU = 'data/duchini_xenium_comet/ex'
        Mi = np.linalg.inv(np.loadtxt(f'{DU}/UnumLocalia/{core}/comet/matrix_comet.csv', delimiter=','))
        tx = pd.read_parquet(f'{DU}/UnumLocalia/{core}/xenium/transcripts.parquet', columns=['x_location', 'y_location', 'qv'])
        tx = tx[tx['qv'] >= 20]; x0, y0 = tx['x_location'].min() - 20, tx['y_location'].min() - 20  # same origin as prep_case_duchini.py
        with tifffile.TiffFile(f'{DU}/COMET_ome_tiff/{core[:6]}_comet.ome.tif') as tf:
            names = [e.attrib.get('Name', '') for e in ET.fromstring(tf.ome_metadata).iter() if e.tag.endswith('Channel')]
            arr = tf.series[0].levels[0].asarray()
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        px_x, px_y = (x0 + (xx + 0.5) * UM) / XPX, (y0 + (yy + 0.5) * UM) / XPX
        cxp = Mi[0, 0] * px_x + Mi[0, 1] * px_y + Mi[0, 2]; cyp = Mi[1, 0] * px_x + Mi[1, 1] * px_y + Mi[1, 2]
        mask = (cxp >= 0) & (cxp < arr.shape[2] - 1) & (cyp >= 0) & (cyp < arr.shape[1] - 1)
        chans = [m for m in meta['markers'] if m in COMET]
        def chan(nm):  # exact channel names; the alpha in 'αSMA' is stored as a broken entity (' #945;SMA - TRITC')
            i = [k for k, n in enumerate(names) if n.strip() == nm or (nm.endswith('SMA - TRITC') and n.strip().endswith('SMA - TRITC'))]
            assert len(i) == 1, (nm, names)
            return map_coordinates(gaussian_filter(arr[i[0]].astype(np.float32), 1.2), [cyp, cxp], order=1, mode='constant', cval=0)
        Y = np.stack([zlog(chan(COMET[c]), mask) for c in chans]); del arr
        mk = {k: k for k in chans}
    elif case.startswith('kidney_'):
        log('skip', case, '(kidney uses the training grid)'); continue
    else:
        log('skip', case); continue
    np.save(f'{cd}/Y.npy', Y); np.save(f'{cd}/ymask.npy', mask)
    json.dump(dict(channels=chans, markers=mk), open(f'{cd}/Y.json', 'w'))
    ev = cells[meta['eval_mask']].values.astype(bool) if 'eval_mask' in meta else np.ones(len(cells), bool)
    xs = cells['x_um'].values[ev] / UM if 'crop' not in meta else cells['x_um'].values[ev] / UM - meta['crop'][2]
    if case.startswith('duchini_'): xs = (cells['x_um'].values[ev] - x0) / UM
    edges = np.linspace(np.percentile(xs, 0.5), np.percentile(xs, 99.5), 9)[1:-1]
    json.dump(dict(x_edges_px=edges.tolist(), test='odd stripes (0-based) of 8 equal-width x-stripes over the evaluated cells'), open(f'{cd}/split.json', 'w'))
    log(case, 'Y', Y.shape, 'channels', chans, 'mask frac', round(float(mask.mean()), 3))
