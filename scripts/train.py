import argparse
from pathlib import Path

import torch
import yaml
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from syatts.dataset import LJSpeech
from syatts.model import SyaTiny
from syatts.text import stoi


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", default="runs/syatts-tiny")
    p.add_argument("--config", default="configs/syatts_tiny.yaml")
    p.add_argument("--limit", type=int)
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = LJSpeech(args.data, cfg, args.limit)
    dl = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=2, pin_memory=device == "cuda")
    model = SyaTiny(len(stoi) + 1, cfg["n_mels"], cfg["hidden"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])

    latest = out / "latest.pt"
    start = 0
    if latest.exists():
        ckpt = torch.load(latest, map_location=device)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        start = ckpt["epoch"] + 1

    for epoch in range(start, cfg["epochs"]):
        model.train()
        losses = []
        for text, mel in tqdm(dl, desc=f"epoch {epoch}"):
            text, mel = text.to(device), mel.to(device)
            pred = model(text, mel.shape[-1])
            loss = F.l1_loss(pred, mel)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch, "cfg": cfg}, latest)
        print(f"epoch={epoch} loss={sum(losses) / len(losses):.4f} saved={latest}")


if __name__ == "__main__":
    main()

