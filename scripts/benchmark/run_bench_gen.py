"""Score kidney-trained TxCellNet networks (train_gen.py) on prepared cases: whole-section prediction (txcellnet.predict),
per-cell means, Spearman / Pearson / AUROC per marker (stains: kidney-fitted weighted sum of the component proteins).
Per-cell predictions are cached in results/bench/predcache/<case>__<run>.npy.
ENS="name=run+run+...;name2=..." adds the mean of the members' per-cell predictions as method `name`.
Panel matching: a RUN_TAGS entry 'name=template' with '{panel}' in the template (e.g. pm_s1=pm_{panel}_s1) is resolved
per case to the model trained on that case's panel group (results/bench/panels.json; cases not listed -> 'full') and
scored under 'name'; a case whose group has no such model is not scored under 'name'.
usage: run_bench_gen.py OUT_DIR RUN_TAGS(comma) CASE_DIR [CASE_DIR ...]
"""
import json, os, sys
import numpy as np, pandas as pd, scipy.sparse as sp, torch

from txcellnet.bench import PROTS, OWN, combine, log
from txcellnet.io import label_counts
from txcellnet.metrics import metrics
from txcellnet.model import GenNet
from txcellnet.predict import predict_case
from txcellnet.utils import default_device

OUT, TAGS, CASES = sys.argv[1], sys.argv[2].split(','), sys.argv[3:]
os.makedirs(OUT, exist_ok=True)
R = 'data/xenium_protein_rcc/pilot_L2'
dev = default_device()
MIN_AREA = 10
CACHE = 'results/bench/predcache'; os.makedirs(CACHE, exist_ok=True)


def load_gen(tag):
    c = json.load(open(f'{R}/runs/{tag}/cfg.json'))
    net = GenNet(PROTS, OWN, c['cfg'], c['train_genes']).to(dev)
    net.load_state_dict(torch.load(f'{R}/runs/{tag}/model.pt', map_location=dev, weights_only=True), strict=False); net.eval()
    return net, c['cfg'], c['train_genes']


PANELS = json.load(open('results/bench/panels.json'))
CASE_PANEL = {c: p for p, v in PANELS.items() for c in v['cases']}
SPECS = [tuple(t.split('=', 1)) if '=' in t else (t, t) for t in TAGS]
_loaded = {}


def get_model(tag):
    if tag not in _loaded: _loaded[tag] = load_gen(tag)
    return _loaded[tag]


log('methods', SPECS)
rows = []
for cd in CASES:
    meta = json.load(open(f'{cd}/meta.json')); name = meta['dataset']
    pname = CASE_PANEL.get(name, CASE_PANEL.get(os.path.basename(cd.rstrip('/')), 'full'))
    models = {mn: (tm.format(panel=pname), get_model(tm.format(panel=pname))) for mn, tm in SPECS
              if '{panel}' not in tm or os.path.exists(f'{R}/runs/{tm.format(panel=pname)}/model.pt')}
    cells = pd.read_parquet(f'{cd}/cells.parquet'); labels = cells['label'].values
    lab = np.load(f'{cd}/lab.npy'); lc = label_counts(lab)
    dapi = np.load(f'{cd}/dapi.npy').astype(np.float32)
    Xc = sp.load_npz(f'{cd}/counts.npz').tocsr()
    ok = (cells['area_px'].values >= MIN_AREA) & (np.asarray(Xc.sum(1)).ravel() >= 1)
    if 'eval_mask' in meta: ok &= cells[meta['eval_mask']].values.astype(bool)
    T = np.load(f'{cd}/tx.npz')  # molecules of the kidney-vocabulary genes
    view = dict(dapi=dapi, lab=lab, genes=json.load(open(f'{cd}/genes.json')), gy=T['gy'].astype(np.int64), gx=T['gx'].astype(np.int64), gidx=T['gidx'].astype(np.int64))
    preds = {}
    for t, (rtag, (net, cfg, tg)) in models.items():
        cf = f'{CACHE}/{name}__{rtag}.npy'
        if os.path.exists(cf): preds[t] = np.load(cf).astype(np.float64); continue
        vocab = net.prepare(tg)
        with torch.no_grad():
            csum, kept, tot, _ = predict_case(net, vocab, view)
        preds[t] = (csum / np.maximum(lc, 1)[:, None])[labels]
        np.save(cf, preds[t].astype(np.float16))
        log(name, t, rtag, f'molecules read {kept}/{tot}')
    for spec in [e for e in os.environ.get('ENS', '').split(';') if e]:  # e.g. txcellnet=fin_clean_s1+fin_rand_s1
        en, mem = spec.split('='); mem = [m for m in mem.split('+') if m in preds]
        if mem: preds[en] = np.mean([preds[m] for m in mem], 0)
    for key, mk in meta['markers'].items():
        tt = cells[mk['truth']].values.astype(np.float64)
        if mk.get('log', True): tt = np.log1p(np.clip(tt - mk.get('offset', 0), 0, None))
        for mn, pc in preds.items():
            rows.append(dict(case=name, marker=key, method=mn, **metrics(combine(pc, mk['components'], key)[ok], tt[ok]), n=int(ok.sum())))
    df = pd.DataFrame(rows); df.to_csv(f'{OUT}/bench_rows.csv', index=False)
    log('\n' + df[df.case == name].pivot_table(index='marker', columns='method', values='spearman').round(3).to_string())
log('done')
