"""Shared helpers for the image panels (Figs 3, 4, S3): three-protein composites on black (cyan, magenta, yellow; screen
blend), scale bars and the per-column strip of Spearman values."""
import numpy as np
from matplotlib.patches import Rectangle
import figstyle as FS
from figstyle import axes_in, INK

UM = 0.85  # µm per pixel
CMY = [np.array([0.0, 0.78, 1.0]), np.array([1.0, 0.12, 0.82]), np.array([1.0, 0.85, 0.0])]  # cyan, magenta, yellow


MODE = {'mode': 'linear', 'gamma': 1.8}


def unit(a, mask, q=(50, 99.5)):
    """display scaling per channel and panel. 'linear' (default): median (black) to 99.5th percentile (full colour) over the
    panel's cell pixels. 'rank': a pixel's quantile among the panel's cell pixels, black below the median, (rescaled
    quantile)**gamma above it -- shows only the ordering (what the Spearman measures) and is robust to the heavy tails of
    some methods' outputs (used for the semi-synthetic figure)."""
    a = np.nan_to_num(np.asarray(a, np.float64), nan=-np.inf); v = np.sort(a[mask])
    if MODE['mode'] == 'rank':
        qq = np.searchsorted(v, a, side='right') / len(v)
        return np.clip((qq - q[0] / 100) / (1 - q[0] / 100), 0, 1) ** MODE['gamma']
    lo, hi = np.percentile(v, q[0]), np.percentile(v, q[1])
    return np.clip((a - lo) / (hi - lo + 1e-9), 0, 1)


def composite(chs, mask, outline=None, outside=None):
    """screen blend of three channels (cyan, magenta, yellow) on black; faint cell outlines underneath"""
    keep = np.ones(chs[0].shape + (3,))
    for a, c in zip(chs, CMY):
        keep *= 1 - unit(np.asarray(a, np.float32), mask)[..., None] * c
    rgb = 1 - keep
    if outline is not None: rgb = np.maximum(rgb, outline[..., None] * 0.2)
    if outside is not None: rgb[outside] = 0
    return rgb


def show(ax, rgb, interp='nearest'):
    ax.imshow(rgb, interpolation=interp); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)


def scalebar(ax, um, text=True):
    h, w = ax.get_images()[0].get_array().shape[:2]; px = um / UM
    x1 = w - 0.1 * w; ax.plot([x1 - px, x1], [h - 0.08 * h] * 2, color='white', lw=1.2, solid_capstyle='butt')
    if text: ax.text(x1 - px / 2, h - 0.12 * h, f'{um:g} µm', color='white', fontsize=FS.SMALL_FS, ha='center', va='bottom')


def values_strip(fig, x, y, w, vals, best):
    """three small colour keys with the Spearman values under a tile (bold: best in the row)"""
    ax = axes_in(fig, x, y, w, 0.12); ax.set_xlim(0, 3); ax.set_ylim(0, 1); ax.axis('off')
    for i in range(3):
        ax.add_patch(Rectangle((i + 0.03, 0.25), 0.13, 0.5, fc=CMY[i], ec='none'))
        ax.text(i + 0.21, 0.5, f'{vals[i]:.2f}' if np.isfinite(vals[i]) else '–', fontsize=FS.SMALL_FS, va='center', ha='left', color=INK,
                fontweight='bold' if best[i] else 'normal')
