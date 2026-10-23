"""equivalence of the CUDA-graph replay with the eager official step (all dropout p=0 so no RNG is involved):
parameter differences after n steps for (a) graph vs eager and (b) eager vs eager with inputs perturbed by 1e-7 (rounding-level
control: how much Adam amplifies rounding differences)."""
import os, sys, copy
import torch, torch.nn as nn
sys.path.insert(0, os.environ.get('NICHETRANS_DIR', 'external/NicheTrans'))
from model.nicheTrans import NicheTrans
dev = 'cuda'; G, T, N = 300, 9, 20000
torch.manual_seed(0)
F = torch.rand(N, G, device=dev); Y = torch.rand(N, T, device=dev); nb = torch.randint(0, N, (N, 12), device=dev)
base = NicheTrans(source_length=G, target_length=T, noise_rate=0.0, dropout_rate=0.0).to(dev)
gen = torch.Generator(device=dev).manual_seed(1)
B = [torch.randint(0, N, (32,), device=dev, generator=gen) for _ in range(300)]
M = [torch.bernoulli(torch.full((32, 12, 1), 0.5, device=dev), generator=gen) if i % 3 == 0 else torch.ones(32, 12, 1, device=dev) for i in range(300)]
crit = nn.MSELoss()

def eager(n, Fx, k=None):
    m = copy.deepcopy(base); m.train(); o = torch.optim.Adam(m.parameters(), lr=3e-4, weight_decay=5e-4)
    for i, (b, mk) in enumerate(zip(B[:n], M[:n])):
        if i == k: o.param_groups[0]['lr'] = 3e-5
        xn = Fx[nb[b]]
        if mk.min() < 1: xn = xn * mk
        loss = crit(m(Fx[b], xn), Y[b]); o.zero_grad(); loss.backward(); o.step()
    return m

def graph(n, k=None):
    m = copy.deepcopy(base); m.train(); o = torch.optim.Adam(m.parameters(), lr=3e-4, weight_decay=5e-4, capturable=True)
    idx = torch.zeros(32, dtype=torch.long, device=dev); mask = torch.ones(32, 12, 1, device=dev)
    def body():
        loss = crit(m(F[idx], F[nb[idx]] * mask), Y[idx]); loss.backward(); o.step()
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for b, mk in zip(B[:3], M[:3]):
            idx.copy_(b); mask.copy_(mk); o.zero_grad(set_to_none=True); body()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph(); o.zero_grad(set_to_none=True)
    with torch.cuda.graph(g): body()
    for i, (b, mk) in enumerate(zip(B[3:n], M[3:n]), 3):
        if i == k:  # StepLR changed the learning rate: re-capture (as nichetrans_official.train_nt)
            g = None; o.param_groups[0]['lr'] = 3e-5; o.zero_grad(set_to_none=True); g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g): body()
        idx.copy_(b); mask.copy_(mk); g.replay()
    torch.cuda.synchronize(); return m

def diff(a, b):
    p = max((x - y).abs().max().item() for x, y in zip(a.parameters(), b.parameters()))
    bn = max((x - y).abs().max().item() for x, y in zip(a.buffers(), b.buffers()) if x.dtype.is_floating_point)
    return f'param {p:.1e} BN {bn:.1e}'

Fp = F * (1 + 1e-7)
for n, k in ((10, 6), (12, 6)):
    e = eager(n, F, k); print(f'lr change at step {k}, n={n}: graph(recapture) vs eager: {diff(graph(n, k), e)}   eager(perturbed 1e-7) vs eager: {diff(eager(n, Fp, k), e)}   graph WITHOUT recapture vs eager: {diff(graph(n), e)}', flush=True)
for n in (4, 10):
    e = eager(n, F)
    print(f'n={n:3d}  graph vs eager: {diff(graph(n), e)}   eager(perturbed 1e-7) vs eager: {diff(eager(n, Fp), e)}   eager vs eager: {diff(eager(n, F), e)}', flush=True)
