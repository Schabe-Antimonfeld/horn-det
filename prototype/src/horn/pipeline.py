"""One causal shared-spectrum pipeline from PCM to frame-aligned camera actions."""

from collections import deque
from dataclasses import asdict, dataclass
from time import perf_counter

import numpy as np

from .calibration import audio_array, remove_dc, StreamCalibration
from .camera import CameraPolicy
from .config import Config
from .events import Direction, EventFusion, Probability, State, encode
from .features import FFTGate, mel_energy, log_mel, mfcc, statistics, PCEN, Gate
from .frontend import STFT, power
from .geometry import circular_array, positions
from .localization import Estimate, gcc, locate, Search
from .weighting import NoisePSD, Weighting, fixed_band


class Detector:
    def __init__(self, model, config=Config(), frontend="logmel", recurrent=False):
        self.model, self.config, self.frontend = model, config, frontend
        self.context = deque(maxlen=config.context)
        self.pcen = PCEN()
        self.recurrent = recurrent
        self.state = None
        self.timings = {"features_s": 0.0, "inference_s": 0.0}

    def feed(self, spectrum, waveform=None):
        started = perf_counter()
        self.timings = {"features_s": 0.0, "inference_s": 0.0}
        if self.frontend == "panns":
            self.context.append(True)
            if len(self.context) < self.config.context or waveform is None:
                self.timings["features_s"] = perf_counter() - started
                return None
            feature = self.model.encoder.frontend(waveform, self.config.sample_rate).numpy()
        else:
            energy = mel_energy(power(spectrum)[None], self.config)
            value = self.pcen.feed(energy)[0] if self.frontend == "pcen" else log_mel(energy, self.config.epsilon)[0]
            self.context.append(value)
            if self.recurrent:
                import torch
                self.timings["features_s"] = perf_counter() - started
                inference_start = perf_counter()
                with torch.no_grad():
                    logits, self.state = self.model.sequence(torch.as_tensor(value[None, None, :, None], dtype=torch.float32), self.state)
                self.timings["inference_s"] = perf_counter() - inference_start
                if len(self.context) < self.config.context:
                    return None
                return float(torch.softmax(logits[0, -1], dim=-1)[1])
            if len(self.context) < self.config.context:
                self.timings["features_s"] = perf_counter() - started
                return None
            values = np.asarray(self.context)
            feature = statistics(mfcc(values))[None] if self.frontend == "mfcc" else values.T[None, None]
        self.timings["features_s"] = perf_counter() - started
        inference_start = perf_counter()
        if hasattr(self.model, "probability"):
            result = float(self.model.probability(feature)[0])
        else:
            import torch
            with torch.no_grad():
                logits = self.model(torch.as_tensor(feature, dtype=torch.float32))
            result = float(torch.softmax(logits, dim=-1)[0, 1])
        self.timings["inference_s"] = perf_counter() - inference_start
        return result


def load_detector(path):
    from pathlib import Path
    from .training import LinearSVM
    import json
    if Path(path).suffix == ".npz":
        model = LinearSVM.load(path)
        config = Config(**json.loads(str(model.values["config"]))) if "config" in model.values else Config()
        return Detector(model, config, "mfcc")
    import torch
    from .models import create_model
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["name"] == "transfer":
        from .pretrained import Cnn6, TransferCNN, CHECKSUM
        if checkpoint["metadata"].get("pretrained_md5") != CHECKSUM:
            raise ValueError("Transfer checkpoint lacks verified pretraining provenance")
        encoder = Cnn6()
        encoder.pretrained = True
        model = TransferCNN(encoder, checkpoint["metadata"].get("fine_tune", False))
    else:
        model = create_model(checkpoint["name"])
    model.load_state_dict(checkpoint["state"], strict=True)
    model.eval()
    return Detector(model, Config(**checkpoint["config"]), checkpoint["frontend"])


@dataclass(frozen=True)
class Output:
    decision: object
    actions: tuple
    timings: dict
    input_end: int
    emitted: float
    gate: dict
    numerical_direction: Estimate

    def dictionary(self):
        result = asdict(self)
        result["decision"]["state"] = self.decision.state.name
        result["decision"]["horn_valid"] = self.decision.horn_valid
        result["wire_hex"] = encode(self.decision).hex()
        return result


class Pipeline:
    """Each feed block is synchronous PCM; all emitted decisions use one exact frame."""

    def __init__(self, config=Config(), detector=None, search=Search(), fusion=None,
                 camera=None, gate=Gate(), weighting=Weighting(), prior=None,
                 mode="aware", fft_gate=False, background_samples=0,
                 gains=None, delays=None, channels=None, coordinates=None, origin=0, association_verified=False,
                 search_mode="sector"):
        if mode not in ("aware", "srp", "detect") or background_samples < 0:
            raise ValueError("Invalid pipeline mode or background prefix")
        if detector is not None:
            for name in ("sample_rate", "frame", "hop", "mel_bands", "context"):
                if getattr(config, name) != getattr(detector.config, name):
                    raise ValueError("Detector and pipeline frontend settings disagree")
        self.config, self.detector, self.search = config, detector, search
        if search_mode not in ("full", "coarse", "sector"):
            raise ValueError("Invalid localization search mode")
        self.search_mode = search_mode
        self.fusion = EventFusion() if fusion is None else fusion
        self.camera = CameraPolicy() if camera is None else camera
        self.gate = FFTGate(config, gate)
        self.fft_gate, self.mode = fft_gate, mode
        self.association_verified = association_verified
        self.noise, self.weighting = NoisePSD(), weighting
        self.prior = fixed_band(np.fft.rfftfreq(config.frame, 1 / config.sample_rate)) if prior is None else prior
        self.background_samples = background_samples
        self.backgrounds = []
        self.channels = np.ones(config.microphones, dtype=bool) if channels is None else np.asarray(channels, dtype=bool)
        if self.channels.shape != (config.microphones,) or not self.channels.any():
            raise ValueError("Channel mask must include at least one configured microphone")
        self.coordinates = circular_array(config) if coordinates is None else positions(coordinates)
        if len(self.coordinates) != config.microphones:
            raise ValueError("Coordinate count must match the array")
        self.origin = origin
        self.stft = STFT(config)
        self.dc_state = None
        self.calibration = None if gains is None and delays is None else StreamCalibration(
            np.ones(config.microphones) if gains is None else gains,
            np.zeros(config.microphones) if delays is None else delays)
        self.waveform = np.empty((0, config.microphones))
        self.wave_start = 0
        self.received = 0
        self.peak_waveform = 0

    def feed(self, audio):
        audio = audio_array(audio)
        if audio.shape[1] != self.config.microphones:
            raise ValueError("Input PCM channel count differs from configuration")
        if self.mode != "detect" and not self.channels.any():
            raise ValueError("No valid localization channels")
        if self.calibration is not None:
            audio = self.calibration.feed(audio)
        audio, self.dc_state = remove_dc(audio, state=self.dc_state)
        start = perf_counter()
        frames = self.stft.feed(audio)
        fft_time = perf_counter() - start
        self.waveform = np.concatenate((self.waveform, audio))
        self.peak_waveform = max(self.peak_waveform, len(self.waveform))
        self.received += len(audio)
        outputs = []
        for frame in frames:
            spectrum = frame.spectrum
            energy = power(spectrum, self.channels)
            rule, gate_values = self.gate.feed(energy)
            triggered = rule if self.fft_gate else gate_values["energy"] >= self.gate.settings.energy
            background = frame.end < self.background_samples
            if background:
                self.backgrounds.append(energy)
                triggered = False
            if self.noise.value is None and not background and self.backgrounds:
                self.noise.initialize(self.backgrounds, confirmed=True)
                self.backgrounds.clear()
            before = self.fusion.state
            begin = perf_counter()
            probability = None
            if self.detector is not None:
                window_start = frame.end + 1 - (self.config.frame + (self.config.context - 1) * self.config.hop)
                waveform = None
                if self.detector.frontend == "panns" and window_start >= self.wave_start:
                    waveform = self.waveform[window_start - self.wave_start:frame.end + 1 - self.wave_start, 0]
                probability = self.detector.feed(spectrum[:, self.channels], waveform)
            classifier_time = perf_counter() - begin
            direction, gcc_time, srp_time = Estimate(), 0.0, 0.0
            if self.mode != "detect" and (triggered or before in (State.CANDIDATE, State.CONFIRMED)):
                weight = None if self.mode == "srp" else self.noise.weights(energy, self.prior, self.weighting)
                begin = perf_counter()
                correlations = gcc(spectrum, self.config, weight, self.channels)
                gcc_time = perf_counter() - begin
                begin = perf_counter()
                direction = locate(correlations, self.coordinates, self.config, self.search, self.search_mode)
                srp_time = perf_counter() - begin
            numerical = direction
            if not self.association_verified:
                direction = Estimate(mode="association_unverified", fallback=numerical.fallback,
                                     pair_count=numerical.pair_count, evaluations=numerical.evaluations)
            decision = self.fusion.feed(frame.identity, frame.timestamp(self.config.sample_rate, self.origin), triggered,
                                        None if probability is None else Probability(frame.identity, probability),
                                        Direction(frame.identity, direction))
            self.noise.update(energy, before, triggered=triggered, channels_valid=self.channels.all())
            actions = tuple(self.camera.feed(decision))
            outputs.append(Output(decision, actions, {"fft_block_amortized_s": fft_time / max(1, len(frames)),
                                                      "classifier_branch_s": classifier_time,
                                                      "features_s": self.detector.timings["features_s"] if self.detector else 0.0,
                                                      "classifier_s": self.detector.timings["inference_s"] if self.detector else 0.0,
                                                      "gcc_s": gcc_time,
                                                      "srp_s": srp_time}, frame.end, perf_counter(), gate_values, numerical))
        keep = self.config.frame + (self.config.context - 1) * self.config.hop + self.config.frame
        if len(self.waveform) > keep:
            drop = len(self.waveform) - keep
            self.waveform = self.waveform[drop:]
            self.wave_start += drop
        return outputs
