from pathlib import Path
import math

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from .audio import wav_to_logmel
from .text import encode


class LJSpeech(Dataset):
    def __init__(self, root: str, cfg: dict, limit: int | None = None):
        self.root = Path(root)
        self.cfg = cfg
        rows = self.root.joinpath("metadata.csv").read_text(encoding="utf-8").splitlines()
        self.items = [line.split("|", 2) for line in rows if line.strip()]
        if limit:
            self.items = self.items[:limit]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        wav_id, _, text = self.items[i]
        x = torch.tensor(encode(text, self.cfg["max_text_len"]), dtype=torch.long)
        mel = wav_to_logmel(str(self.root / "wavs" / f"{wav_id}.wav"), self.cfg)
        mel = mel[:, : self.cfg["max_mel_frames"]]
        mel_len = mel.shape[1]
        pad = math.log(self.cfg.get("mel_floor", 1e-5))
        mel = F.pad(mel, (0, self.cfg["max_mel_frames"] - mel.shape[1]), value=pad)
        return x, mel, mel_len
