"""Reproducible matched synthetic localization matrix and independent ablations."""

from dataclasses import asdict, replace
import platform
from time import perf_counter

import numpy as np
from scipy.signal import butter, sosfilt

from .config import Config
from .evaluation import latency_summary, localization_metrics
from .frontend import stft, power
from .localization import gcc, gcc_direction, locate, Search
from .simulation import far_field, noise_signal
from .weighting import fixed_band, learned_prior, frequency_weights, Weighting


def synthetic_horn(samples, sample_rate, generator):
    time = np.arange(samples) / sample_rate
    fundamental = generator.uniform(330, 500)
    tones = sum(np.sin(2 * np.pi * fundamental * harmonic * time + generator.uniform(0, 2 * np.pi))
                / harmonic for harmonic in range(1, 6))
    broadband = sosfilt(butter(3, [200, 3000], btype="bandpass", fs=sample_rate, output="sos"),
                        generator.normal(size=samples))
    return 0.3 * (tones + 0.8 * broadband)


def synthetic_prior(config=Config(), seed=17001):
    powers, labels = [], []
    generator = np.random.default_rng(seed)
    for index in range(24):
        source = (synthetic_horn(config.frame * 4, config.sample_rate, generator) if index % 2 else
                  noise_signal("traffic", config.frame * 4, config.sample_rate, generator))
        powers.append(np.mean([power(frame.spectrum) for frame in stft(source[:, None], config)], axis=0))
        labels.append(index % 2)
    fallback = fixed_band(np.fft.rfftfreq(config.frame, 1 / config.sample_rate))
    prior, metadata = learned_prior(powers, labels, ["train"] * len(labels), fallback)
    return prior, {**metadata, "source": "independent synthetic training sources", "seed": seed}


def run_benchmark(config=Config(), seed=0, quick=False, prior=None, search=Search(), weighting=Weighting()):
    if seed == 17001:
        raise ValueError("Test seed cannot equal the synthetic prior training seed")
    if prior is None:
        prior, prior_metadata = synthetic_prior(config)
    else:
        prior_metadata = {"source": "supplied prior; caller must verify training provenance"}
    angles = [-40, 0, 40] if quick else list(range(-80, 81, 20))
    snrs = [0, 10] if quick else [-5, 0, 5, 10, 20]
    kinds = ["white", "siren"] if quick else ["white", "engine", "traffic", "siren", "wind"]
    names = ["gcc", "srp", "fixed_band", "learned_prior", "prior_subtraction", "prior_snr",
             "horn_aware", "aware_integer", "aware_sector", "aware_pruned"]
    predictions = {name: [] for name in names}
    runtimes = {name: [] for name in names}
    samples = []
    fixed = fixed_band(np.fft.rfftfreq(config.frame, 1 / config.sample_rate))
    for angle in angles:
        for snr in snrs:
            for kind in kinds:
                identity = len(samples)
                generator = np.random.default_rng(np.random.SeedSequence([seed, identity, 9121]))
                source = synthetic_horn(config.frame * 6, config.sample_rate, generator)
                scene = far_field(source, angle, config)
                background = np.column_stack([noise_signal(kind, len(scene.audio), config.sample_rate, generator)
                                               for _ in range(config.microphones)])
                common = noise_signal(kind, len(scene.audio), config.sample_rate, generator)[:, None]
                background = 0.8 * background + 0.2 * common
                region = slice(int(scene.common_delay), int(scene.common_delay) + len(source))
                scale = np.sqrt(np.mean(scene.audio[region] ** 2) / np.mean(background[region] ** 2) / 10 ** (snr / 10))
                background *= scale
                audio = scene.audio + background
                spectrum = stft(audio, config)[3].spectrum
                noise = np.mean([power(frame.spectrum) for frame in stft(background, config)], axis=0)
                energy = power(spectrum)
                weights = {
                    "gcc": None, "srp": None, "fixed_band": fixed, "learned_prior": prior,
                    "prior_subtraction": frequency_weights(energy, noise, prior, weighting, use_snr=False),
                    "prior_snr": frequency_weights(energy, noise, prior, weighting, use_subtraction=False),
                    "horn_aware": frequency_weights(energy, noise, prior, weighting)}
                for name in ("aware_integer", "aware_sector", "aware_pruned"):
                    weights[name] = weights["horn_aware"]
                for name in names:
                    begin = perf_counter()
                    correlations = gcc(spectrum, config, weights[name])
                    selected = replace(search, interpolation=False) if name == "aware_integer" else search
                    if name == "aware_pruned":
                        selected = replace(selected, selected_pairs=8)
                    result = (gcc_direction(correlations, scene.coordinates, config, selected) if name == "gcc" else
                              locate(correlations, scene.coordinates, config, selected,
                                     mode="sector" if name in ("aware_sector", "aware_pruned") else "full"))
                    runtimes[name].append(perf_counter() - begin)
                    predictions[name].append({"sample_id": identity, **asdict(result)})
                samples.append({"sample_id": identity, "angle_deg": angle, "snr_db": snr,
                                "noise_proxy": kind, "source_group": f"synthetic-test-{seed}-{identity}", "split": "test"})
    truth = [sample["angle_deg"] for sample in samples]
    metrics = {}
    for name, estimates in predictions.items():
        metrics[name] = {**localization_metrics(truth, [item["angle"] for item in estimates], [item["valid"] for item in estimates]),
                         "cpu_localization_s": latency_summary(runtimes[name]),
                         "fallback_fraction": float(np.mean([item["fallback"] for item in estimates])),
                         "mean_pair_count": float(np.mean([item["pair_count"] for item in estimates]))}
    return {"evidence": "synthetic numeric reference; no road or board accuracy claim",
            "config": asdict(config), "search": asdict(search), "weighting": asdict(weighting),
            "seed": seed, "platform": platform.platform(), "prior": prior_metadata,
            "noise_psd": "controlled background-only spectra; does not validate online noise estimation",
            "samples": samples, "metrics": metrics, "predictions": predictions,
            "fpga_resources": None, "real_road_results": None}
