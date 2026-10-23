"""H&E -> protein baseline network: the TxCellNet backbone (U-Net + cell-niche transformer + 1x1 head) with a 3-channel
H&E input instead of transcripts + DAPI."""
import torch, torch.nn as nn
from txcellnet.model import UNet, CellNiche


class HENet(nn.Module):
    def __init__(s, nprot):
        super().__init__()
        w = (64, 128, 256, 512)
        s.unet = UNet(3, w); s.cell = CellNiche(w[0], w[0]); s.head = nn.Conv2d(w[0], nprot, 1)

    def forward(s, x, lab):
        h = s.unet(x)
        with torch.autocast(h.device.type, enabled=False):  # the transformer runs in fp32
            h = h.float() + s.cell(h.float(), lab)
        return s.head(h)
