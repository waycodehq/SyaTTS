import argparse
from pathlib import Path

import torch
import yaml
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
    p.add_argument("--epochs", type=int)
    p.add_argument("--target-loss", type=float)
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

    epochs = args.epochs or cfg["epochs"]
    for epoch in range(start, epochs):
        model.train()
        losses = []
        for text, mel, mel_len in tqdm(dl, desc=f"epoch {epoch}"):
            text, mel, mel_len = text.to(device), mel.to(device), mel_len.to(device)
            pred = model(text, mel.shape[-1])
            mask = torch.arange(mel.shape[-1], device=device)[None, :] < mel_len[:, None]
            loss = ((pred - mel).abs() * mask[:, None, :]).sum() / (mask.sum() * mel.shape[1]).clamp_min(1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch, "cfg": cfg}, latest)
        avg_loss = sum(losses) / len(losses)
        print(f"epoch={epoch} masked_loss={avg_loss:.4f} saved={latest}")
        if args.target_loss and avg_loss <= args.target_loss:
            print(f"target_loss={args.target_loss} reached")
            break


if __name__ == "__main__":
    main()
