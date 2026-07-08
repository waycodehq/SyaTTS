import torch
import yaml

from syatts.model import SyaTiny
from syatts.text import encode, stoi


cfg = yaml.safe_load(open("configs/syatts_tiny.yaml", encoding="utf-8"))
model = SyaTiny(len(stoi) + 1, cfg["n_mels"], cfg["hidden"])
x = torch.tensor([encode("hello from syatts", cfg["max_text_len"])])
y = model(x, 32)
assert y.shape == (1, cfg["n_mels"], 32)
print("ok")

