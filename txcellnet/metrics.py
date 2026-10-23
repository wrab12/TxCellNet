"""Cell-level metrics of the benchmark: Spearman (primary), Pearson and the AUROC of Otsu-positive cells."""
import numpy as np
from scipy.stats import spearmanr, rankdata


def auroc(score, y):
    """Mann-Whitney AUROC of `score` for the binary labels `y`"""
    y = np.asarray(y, bool); n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0: return np.nan
    r = rankdata(score)
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def otsu(v, bins=256):
    h, e = np.histogram(v, bins=bins); c = (e[:-1] + e[1:]) / 2
    w0 = np.cumsum(h); w1 = w0[-1] - w0
    m0 = np.cumsum(h * c) / np.maximum(w0, 1); m1 = (np.sum(h * c) - np.cumsum(h * c)) / np.maximum(w1, 1)
    return c[np.argmax(w0 * w1 * (m0 - m1) ** 2)]


def metrics(score, truth):
    """truth: log1p per-cell measured intensity. Returns Pearson, Spearman, AUROC for Otsu-positive cells, positive fraction
    (all NaN with fewer than 50 valid cells or a constant prediction)."""
    ok = np.isfinite(score) & np.isfinite(truth)
    s, t = score[ok], truth[ok]
    if len(t) < 50 or np.std(s) == 0: return dict(pearson=np.nan, spearman=np.nan, auroc=np.nan, pos_frac=np.nan)
    pos = t > otsu(t)
    return dict(pearson=float(np.corrcoef(s, t)[0, 1]), spearman=float(spearmanr(s, t)[0]), auroc=float(auroc(s, pos)),
                pos_frac=float(pos.mean()))
