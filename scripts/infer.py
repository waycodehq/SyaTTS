import argparse
from pathlib import Path

import soundfile as sf
import torch

from syatts.audio import logmel_to_wav
from syatts.model import SyaTiny
from syatts.text import encode, stoi


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--text", required=True)
    p.add_argument("--out", default="sample.wav")
    p.add_argument("--frames", type=int, default=420)
    args = p.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = ckpt["cfg"]
    model = SyaTiny(len(stoi) + 1, cfg["n_mels"], cfg["hidden"])
    model.load_state_dict(ckpt["model"])
    model.eval()

    text = torch.tensor([encode(args.text, cfg["max_text_len"])])
    with torch.no_grad():
        mel = model(text, args.frames)[0]
    wav = logmel_to_wav(mel, cfg).numpy()
    sf.write(args.out, wav, cfg["sample_rate"])
    print(Path(args.out).resolve())


if __name__ == "__main__":
    main()

