# SyaTTS

Tiny from-scratch TTS baseline for Google Colab free tier.

This is not production quality yet. It is the smallest useful path to prove the
training loop, checkpoints, and inference path before spending bigger GPU time.

## Colab

Open `main.ipynb`, run cells top to bottom, and keep checkpoints in Google
Drive.

## Local smoke check

```bash
pip install -r requirements.txt
python scripts/smoke_test.py
```

## Data layout

LJSpeech is expected at:

```txt
data/LJSpeech-1.1/
  metadata.csv
  wavs/
```

## Train

```bash
python scripts/train.py --data data/LJSpeech-1.1 --out runs/syatts-tiny
```

## Infer

```bash
python scripts/infer.py --ckpt runs/syatts-tiny/latest.pt --text "hello from SyaTTS" --out sample.wav
```

