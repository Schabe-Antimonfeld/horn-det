"""Independent Cnn6-compatible encoder and verified AudioSet weight loader.

Architecture source: qiuqiangkong/audioset_tagging_cnn, commit
d2f4b8c18eab44737fcc0de1248ae21eb43f6aa4, pytorch/models.py.
Weights: Kong et al., Zenodo record 3987831, CC BY 4.0.
"""

import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import torch
from torch import nn
from torch.nn import functional as functional
from scipy.signal import resample_poly


URL = "https://zenodo.org/api/records/3987831/files/Cnn6_mAP=0.343.pth/content"
CHECKSUM = "e25e26b84585b14c7754c91e48efc9be"
SIZE = 23698321


def fetch_weights(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with urlopen(URL, timeout=30) as response, path.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
    digest = hashlib.md5(path.read_bytes()).hexdigest()
    if path.stat().st_size != SIZE or digest != CHECKSUM:
        raise ValueError("Cnn6 checkpoint checksum mismatch; use a new path to retry")
    metadata = {"url": URL, "md5": CHECKSUM, "license": "CC-BY-4.0", "pretraining": "AudioSet",
                "citation": "Kong et al. 2020, DOI 10.1109/TASLP.2020.3030497",
                "source_commit": "d2f4b8c18eab44737fcc0de1248ae21eb43f6aa4"}
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


class AudioBlock(nn.Module):
    def __init__(self, incoming, outgoing):
        super().__init__()
        self.conv1 = nn.Conv2d(incoming, outgoing, 5, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(outgoing)

    def forward(self, values):
        return functional.avg_pool2d(functional.relu(self.bn1(self.conv1(values))), 2)


class Cnn6(nn.Module):
    """Input is checkpoint-compatible (batch, 1, time, 64) dB Mel, at least 16 frames."""

    def __init__(self):
        super().__init__()
        self.bn0 = nn.BatchNorm2d(64)
        for index, (incoming, outgoing) in enumerate(zip([1, 64, 128, 256], [64, 128, 256, 512]), 1):
            setattr(self, "conv_block" + str(index), AudioBlock(incoming, outgoing))
        self.fc1 = nn.Linear(512, 512)
        self.fc_audioset = nn.Linear(512, 527)
        self.register_buffer("mel_matrix", torch.empty(513, 64))
        self.pretrained = False

    def load_pretrained(self, path):
        if hashlib.md5(Path(path).read_bytes()).hexdigest() != CHECKSUM:
            raise ValueError("Expected the verified Cnn6 AudioSet checkpoint")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        state = checkpoint["model"]
        matrix = state["logmel_extractor.melW"]
        selected = {name: value for name, value in state.items()
                    if not name.startswith(("spectrogram_extractor.", "logmel_extractor."))}
        selected["mel_matrix"] = matrix
        self.load_state_dict(selected, strict=True)
        self.pretrained = True
        return self

    def frontend(self, waveform, sample_rate=48000):
        """Resample one completed past window; centered framing never reads beyond that window."""
        if not self.pretrained:
            raise ValueError("Load real pretrained weights before using the frontend")
        waveform = np.asarray(waveform, dtype=float)
        if waveform.ndim != 1 or not np.isfinite(waveform).all():
            raise ValueError("Pretrained input must be finite mono audio")
        from math import gcd
        divisor = gcd(sample_rate, 32000)
        waveform = resample_poly(waveform, 32000 // divisor, sample_rate // divisor)
        if len(waveform) <= 512:
            raise ValueError("Window is too short for reflect padding")
        audio = torch.as_tensor(waveform, dtype=torch.float32)
        spectrum = torch.stft(audio, 1024, hop_length=320,
                              window=torch.hann_window(1024), center=True,
                              pad_mode="reflect", return_complex=True)
        mel = spectrum.abs().square().T @ self.mel_matrix
        result = 10 * torch.log10(torch.clamp(mel, min=1e-10))
        if result.shape[0] < 16:
            raise ValueError("Cnn6 needs at least 16 Mel time steps")
        return result[None, None]

    def embedding(self, values):
        if values.ndim != 4 or values.shape[-1] != 64 or values.shape[-2] < 16:
            raise ValueError("Invalid Cnn6 input")
        output = self.bn0(values.transpose(1, 3)).transpose(1, 3)
        for index in range(1, 5):
            output = getattr(self, "conv_block" + str(index))(output)
            output = functional.dropout(output, 0.2, self.training)
        output = output.mean(dim=3)
        output = output.amax(dim=2) + output.mean(dim=2)
        output = functional.dropout(output, 0.5, self.training)
        return functional.relu(self.fc1(output))

    def forward(self, values):
        return torch.sigmoid(self.fc_audioset(self.embedding(values)))


class TransferCNN(nn.Module):
    def __init__(self, encoder, fine_tune=False):
        super().__init__()
        if not encoder.pretrained:
            raise ValueError("Transfer learning requires a real pretrained encoder")
        self.encoder = encoder
        self.fine_tune = fine_tune
        for parameter in encoder.parameters():
            parameter.requires_grad_(False)
        if fine_tune:
            for module in (encoder.conv_block4, encoder.fc1):
                for parameter in module.parameters():
                    parameter.requires_grad_(True)
        self.head = nn.Linear(512, 2)

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        if mode and self.fine_tune:
            self.encoder.conv_block4.train()
        return self

    def forward(self, values):
        return self.head(self.encoder.embedding(values))
