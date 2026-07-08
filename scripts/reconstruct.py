import argparse
from pathlib import Path

import soundfile as sf
import yaml

from syatts.audio import logmel_to_wav, wav_to_logmel


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--config", default="configs/syatts_tiny.yaml")
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--out", default="reconstruct.wav")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    root = Path(args.data)
    row = root.joinpath("metadata.csv").read_text(encoding="utf-8").splitlines()[args.index]
    wav_id, _, text = row.split("|", 2)
    wav = logmel_to_wav(wav_to_logmel(str(root / "wavs" / f"{wav_id}.wav"), cfg), cfg).numpy()
    sf.write(args.out, wav, cfg["sample_rate"])
    print(f"{Path(args.out).resolve()} | {text}")


if __name__ == "__main__":
    main()
