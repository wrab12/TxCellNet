"""Small helpers shared by the package and the scripts: device selection, mixed precision, logging."""
import contextlib
import time

import torch


def default_device():
    """CUDA when available, otherwise the CPU."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def amp(device):
    """bf16 autocast on CUDA (the setting of every experiment); full precision on other devices."""
    device = torch.device(device)
    if device.type == 'cuda':
        return torch.autocast('cuda', dtype=torch.bfloat16)
    return contextlib.nullcontext()


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)
