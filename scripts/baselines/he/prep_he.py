"""Resample a same-section post-Xenium H&E image onto a 0.85 um/px Xenium grid (morphology_focus level 2).

Alignment: 10x he_imagealignment.csv, either a 3x3 affine (H&E px -> Xenium level-0 px) or a landmark table
(fixedX, fixedY = Xenium px; alignmentX, alignmentY = H&E px), in which case an affine is fitted by least squares.
The H&E pyramid level closest to (but finer than) 0.85 um is Gaussian-smoothed and sampled bilinearly.
Output: float16 (3, H, W) RGB in [0, 1] on the grid; pixels outside the H&E are set to 1 (white background).
usage: prep_he.py HE_DIR OUT_NPY GRID_H GRID_W
"""
import os, sys, time
import numpy as np, pandas as pd, tifffile, xml.etree.ElementTree as ET
from scipy.ndimage import map_coordinates, gaussian_filter

hd, out, H, W = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
XPX, UM = 0.2125, 0.85


def log(*a): print(time.strftime('%H:%M:%S'), *a, flush=True)


raw = pd.read_csv(f'{hd}/he_imagealignment.csv', header=None)
if isinstance(raw.iloc[0, 0], str) and 'fixed' in str(raw.iloc[0, 0]):
    L = pd.read_csv(f'{hd}/he_imagealignment.csv')
    A = np.c_[L[['alignmentX', 'alignmentY']].values, np.ones(len(L))]
    Bx = L[['fixedX', 'fixedY']].values
    sol = np.linalg.lstsq(A, Bx, rcond=None)[0]  # (3, 2): H&E px -> Xenium px
    M = np.r_[sol.T, [[0, 0, 1]]]
    res = np.abs(A @ sol - Bx); log('landmark affine fit, n', len(L), 'median |residual| px', np.round(np.median(res, 0), 2))
else:
    M = raw.values.astype(np.float64)
Mi = np.linalg.inv(M)  # Xenium px -> H&E px
with tifffile.TiffFile(f'{hd}/he_image.ome.tif') as tf:
    px = float([e.attrib for e in ET.fromstring(tf.ome_metadata).iter() if e.tag.endswith('Pixels')][0]['PhysicalSizeX'])
    levels = tf.series[0].levels
    lv = max(i for i in range(len(levels)) if px * 2 ** i <= UM)
    img = levels[lv].asarray()
log('H&E px', round(px, 4), 'um; using level', lv, img.shape, 'scale', round(px * 2 ** lv, 3), 'um/px')
if img.shape[0] == 3: img = np.moveaxis(img, 0, -1)
s = UM / (px * 2 ** lv)  # grid px / H&E-level px
yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
X0, Y0 = (xx + 0.5) * UM / XPX, (yy + 0.5) * UM / XPX  # Xenium level-0 px of grid centres
hx = (Mi[0, 0] * X0 + Mi[0, 1] * Y0 + Mi[0, 2]) / 2 ** lv
hy = (Mi[1, 0] * X0 + Mi[1, 1] * Y0 + Mi[1, 2]) / 2 ** lv
del xx, yy, X0, Y0
outa = np.empty((3, H, W), np.float16)
for c in range(3):
    ch = gaussian_filter(img[..., c].astype(np.float32) / 255.0, sigma=max(s / 2.5, 0.5))
    outa[c] = map_coordinates(ch, [hy, hx], order=1, mode='constant', cval=1.0).astype(np.float16)
os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
np.save(out, outa)
inside = ((hx >= 0) & (hx < img.shape[1]) & (hy >= 0) & (hy < img.shape[0])).mean()
log('saved', out, (3, H, W), 'fraction of grid inside the H&E', round(float(inside), 3))
