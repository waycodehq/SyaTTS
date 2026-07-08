import torch
import torchaudio


def mel_transform(cfg: dict):
    return torchaudio.transforms.MelSpectrogram(
        sample_rate=cfg["sample_rate"],
        n_fft=cfg["n_fft"],
        win_length=cfg["win_length"],
        hop_length=cfg["hop_length"],
        n_mels=cfg["n_mels"],
    )


def wav_to_logmel(path: str, cfg: dict) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    wav = wav.mean(0, keepdim=True)
    if sr != cfg["sample_rate"]:
        wav = torchaudio.functional.resample(wav, sr, cfg["sample_rate"])
    mel = mel_transform(cfg)(wav).squeeze(0)
    return torch.log(torch.clamp(mel, min=1e-5))


def logmel_to_wav(logmel: torch.Tensor, cfg: dict) -> torch.Tensor:
    mel = logmel.exp().cpu()
    inv_mel = torchaudio.transforms.InverseMelScale(
        n_stft=cfg["n_fft"] // 2 + 1,
        n_mels=cfg["n_mels"],
        sample_rate=cfg["sample_rate"],
    )
    griffin = torchaudio.transforms.GriffinLim(
        n_fft=cfg["n_fft"],
        win_length=cfg["win_length"],
        hop_length=cfg["hop_length"],
    )
    return griffin(inv_mel(mel)).clamp(-1, 1)

