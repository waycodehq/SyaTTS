import torch
from torch import nn


class SyaTiny(nn.Module):
    def __init__(self, vocab: int, n_mels: int, hidden: int):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden, padding_idx=0)
        self.encoder = nn.GRU(hidden, hidden, batch_first=True, bidirectional=True)
        self.decoder = nn.GRU(hidden * 2, hidden, batch_first=True)
        self.proj = nn.Linear(hidden, n_mels)

    def forward(self, text, frames: int):
        enc, _ = self.encoder(self.embed(text))
        # ponytail: naive uniform alignment; replace with duration/alignment model after baseline speaks.
        enc = enc.transpose(1, 2)
        enc = torch.nn.functional.interpolate(enc, size=frames, mode="linear", align_corners=False)
        enc = enc.transpose(1, 2)
        out, _ = self.decoder(enc)
        return self.proj(out).transpose(1, 2)

