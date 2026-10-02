"""Executable workflows for simulation, training, evaluation and FPGA handoff."""

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
from time import perf_counter, sleep

import numpy as np
import soundfile as sf

from .config import Config
from .datasets import load_manifest, prepare, split_groups
from .evaluation import Event, choose_threshold, event_metrics, latency_summary, window_metrics


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def read_profile(path):
    return {} if path is None else json.loads(Path(path).read_text(encoding="utf-8"))


def configuration(profile):
    return Config(**profile.get("config", {}))


def load_dataset(path):
    with np.load(path, allow_pickle=False) as archive:
        return dict(archive)


def load_prior(path, config=None):
    if path is None:
        return None
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"]))
        if metadata.get("split") != "train":
            raise ValueError("Prior artifact must declare training-only provenance")
        if config is not None and any(metadata["config"][name] != getattr(config, name) for name in ("frame", "sample_rate")):
            raise ValueError("Prior and runtime frequency axes differ")
        return archive["weight"]


def encoder(path):
    from .pretrained import Cnn6
    if path is None:
        raise ValueError("This workflow requires --pretrained with a verified Cnn6 checkpoint")
    return Cnn6().load_pretrained(path).eval()


def prepare_command(arguments):
    profile = read_profile(arguments.profile)
    config = configuration(profile)
    records = load_manifest(arguments.manifest)
    if arguments.frontend == "panns":
        import torch
        torch.set_num_threads(2)
    features = prepare(records, config, arguments.frontend, arguments.stride,
                       encoder(arguments.pretrained) if arguments.frontend == "panns" else None)
    Path(arguments.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez(arguments.output, **features)
    return {"samples": len(features["labels"]), "shape": list(features["features"].shape), "output": arguments.output}


def split_command(arguments):
    records = split_groups(load_manifest(arguments.manifest), arguments.seed)
    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = [json.dumps({**asdict(record), "path": record.path}, ensure_ascii=False) for record in records]
    output.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return {split: sum(record.split == split for record in records) for split in ("train", "validation", "test")}


def prior_command(arguments):
    from .weighting import learned_prior, fixed_band
    dataset = load_dataset(arguments.dataset)
    config = Config(**json.loads(str(dataset["config"])))
    selected = dataset["splits"] == "train"
    fallback = fixed_band(np.fft.rfftfreq(config.frame, 1 / config.sample_rate))
    weight, metadata = learned_prior(dataset["powers"][selected], dataset["labels"][selected],
                                     dataset["splits"][selected], fallback)
    metadata.update({"split": "train", "groups": np.unique(dataset["groups"][selected]).tolist(),
                     "dataset_sha256": hashlib.sha256(Path(arguments.dataset).read_bytes()).hexdigest(),
                     "config": asdict(config)})
    Path(arguments.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez(arguments.output, weight=weight, metadata=np.asarray(json.dumps(metadata)))
    return metadata


def train_command(arguments):
    import torch
    from .models import create_model
    from .pipeline import load_detector
    from .training import LinearSVM, train_model, save_model, model_cost
    torch.set_num_threads(arguments.threads)
    torch.manual_seed(arguments.seed)
    dataset = load_dataset(arguments.dataset)
    config = Config(**json.loads(str(dataset["config"])))
    frontend = str(dataset["frontend"])
    metadata = {"seed": arguments.seed, "dataset_sha256": hashlib.sha256(Path(arguments.dataset).read_bytes()).hexdigest(),
                "groups": {split: np.unique(dataset["groups"][dataset["splits"] == split]).tolist()
                           for split in ("train", "validation", "test")}}
    if arguments.model == "svm":
        if frontend != "mfcc":
            raise ValueError("Linear SVM requires --frontend mfcc")
        LinearSVM().fit(dataset, arguments.seed).save(arguments.output)
        write_json(str(arguments.output) + ".json", metadata)
        return {"model": "svm", "output": arguments.output}
    if arguments.model == "transfer":
        from .pretrained import TransferCNN, CHECKSUM
        if frontend != "panns":
            raise ValueError("Transfer training requires --frontend panns")
        model = TransferCNN(encoder(arguments.pretrained), fine_tune=arguments.fine_tune)
        metadata.update({"pretrained_md5": CHECKSUM, "fine_tune": arguments.fine_tune, "pretraining": "AudioSet"})
    else:
        if (arguments.model == "mlp") != (frontend == "mfcc") or frontend == "panns":
            raise ValueError("The selected frontend does not match this model")
        model = create_model(arguments.model)
    teacher_logits = None
    if arguments.teacher:
        if arguments.teacher_dataset is None:
            raise ValueError("Distillation requires --teacher-dataset with aligned source-window identities")
        teacher_data = load_dataset(arguments.teacher_dataset)
        for key in ("identities", "groups", "splits", "labels"):
            if not np.array_equal(dataset[key], teacher_data[key]):
                raise ValueError("Teacher and student datasets are not aligned by source-window identity")
        teacher = load_detector(arguments.teacher)
        if hasattr(teacher.model, "probability"):
            raise ValueError("Use a trained neural teacher with two output logits")
        if teacher.frontend != str(teacher_data["frontend"]):
            raise ValueError("Teacher frontend and teacher dataset disagree")
        with torch.no_grad():
            teacher_logits = torch.cat([teacher.model(torch.as_tensor(teacher_data["features"][start:start + arguments.batch_size]))
                                        for start in range(0, len(dataset["labels"]), arguments.batch_size)])
        metadata["teacher_sha256"] = hashlib.sha256(Path(arguments.teacher).read_bytes()).hexdigest()
    history = train_model(model, dataset, arguments.epochs, arguments.batch_size,
                          arguments.learning_rate, arguments.seed, teacher_logits)
    save_model(model, arguments.output, arguments.model, config, frontend, history, metadata)
    cost = model_cost(model, torch.as_tensor(dataset["features"][:1]))
    write_json(str(arguments.output) + ".json", {"history": history, "cost": cost, **metadata})
    return {"model": arguments.model, "output": arguments.output, "cost": cost, "last_epoch": history[-1]}


def evaluate_command(arguments):
    from .pipeline import load_detector
    dataset = load_dataset(arguments.dataset)
    detector = load_detector(arguments.model)
    if detector.frontend != str(dataset["frontend"]) or asdict(detector.config) != json.loads(str(dataset["config"])):
        raise ValueError("Evaluation frontend or configuration differs from the trained model")
    timings, predictions = [], []
    quantized = None
    if arguments.quantize:
        if hasattr(detector.model, "probability"):
            raise ValueError("Weight quantization comparison currently supports neural models")
        from .fixed import quantize_model
        quantized, quantization = quantize_model(detector.model, arguments.quantize)
    for feature in dataset["features"]:
        begin = perf_counter()
        if hasattr(detector.model, "probability"):
            probability = float(detector.model.probability(feature[None])[0])
        else:
            import torch
            torch.set_num_threads(arguments.threads)
            with torch.no_grad():
                probability = float(torch.softmax(detector.model(torch.as_tensor(feature[None])), dim=-1)[0, 1])
        timings.append(perf_counter() - begin)
        predictions.append(probability)
    predictions = np.asarray(predictions)
    validation = dataset["splits"] == "validation"
    if arguments.threshold is None:
        selection = choose_threshold(dataset["labels"][validation], predictions[validation])
    else:
        selection = {"threshold": arguments.threshold, "source": "supplied frozen threshold"}
    result = {"scope": "window classification, not continuous event or road evaluation", "threshold": selection,
              "cpu_inference_s": latency_summary(timings),
              "first_call_s": timings[0] if timings else None,
              "steady_cpu_inference_s": latency_summary(timings[1:]),
              "partitions": {split: window_metrics(dataset["labels"][dataset["splits"] == split],
                                                    predictions[dataset["splits"] == split], selection["threshold"])
                             for split in ("train", "validation", "test")}}
    if quantized is not None:
        with torch.no_grad():
            actual = torch.cat([torch.softmax(quantized(torch.as_tensor(dataset["features"][start:start + 32])), dim=-1)[:, 1]
                                for start in range(0, len(predictions), 32)]).numpy()
        result["quantization"] = {"bits": arguments.quantize, "weight_errors": quantization,
                                   "probability_mae": float(np.mean(np.abs(actual - predictions))),
                                   "test": window_metrics(dataset["labels"][dataset["splits"] == "test"],
                                                          actual[dataset["splits"] == "test"], selection["threshold"]),
                                   "scope": "dequantized weight comparison; no integer activation/kernel claim"}
    write_json(arguments.output, result)
    prediction_path = Path(arguments.output).with_suffix(".predictions.npz")
    np.savez(prediction_path, probabilities=predictions, identities=dataset["identities"], labels=dataset["labels"], splits=dataset["splits"])
    return result


def simulate_command(arguments):
    from .simulation import far_field, randomized
    from .benchmark import synthetic_horn
    config = configuration(read_profile(arguments.profile))
    source = synthetic_horn(round(arguments.seconds * config.sample_rate), config.sample_rate, np.random.default_rng(arguments.seed))
    scene = (randomized(source, arguments.angle, config, arguments.seed, arguments.noise) if arguments.randomized else
             far_field(source, arguments.angle, config))
    Path(arguments.output).parent.mkdir(parents=True, exist_ok=True)
    sf.write(arguments.output, scene.audio, config.sample_rate, subtype="FLOAT")
    metadata = {**scene.metadata, "coordinates": scene.coordinates.tolist(), "angle_deg": scene.angle,
                "sound_speed": scene.speed, "common_delay_samples": scene.common_delay, "valid_channels": scene.valid.tolist()}
    write_json(str(arguments.output) + ".json", metadata)
    return metadata


def benchmark_command(arguments):
    from .benchmark import run_benchmark
    from .localization import Search
    from .weighting import Weighting
    profile = read_profile(arguments.profile)
    result = run_benchmark(configuration(profile), arguments.seed, arguments.quick,
                           load_prior(arguments.prior, configuration(profile)), Search(**profile.get("search", {})), Weighting(**profile.get("weighting", {})))
    output = Path(arguments.output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "report.json", result)
    rows = ["# 仿真 matched localization 结果", "", "合成声源/噪声代理；不是道路或 FPGA 板上结果。", "",
            "| Route | Valid/Total | MAE° | P90° | SR10 | CPU P90 ms |", "|---|---|---|---|---|---|"]
    for name, metrics in result["metrics"].items():
        number = lambda value: "undefined" if value is None else f"{value:.4f}"
        rows.append(f"| {name} | {metrics['valid']}/{metrics['total']} | {number(metrics['mae_deg'])} | {number(metrics['p90_deg'])} | {number(metrics['sr10'])} | {number(metrics['cpu_localization_s']['p90'] * 1000)} |")
    (output / "report.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return {"samples": len(result["samples"]), "report": str(output / "report.json"), "metrics": result["metrics"]}


def run_command(arguments):
    from .camera import Camera, CameraPolicy, CameraSimulation
    from .events import EventFusion, Fusion, State, encode
    from .frontend import read_audio
    from .features import Gate
    from .localization import Search
    from .pipeline import Pipeline, load_detector
    from .weighting import Weighting
    profile = read_profile(arguments.profile)
    if arguments.model and Path(arguments.model).suffix != ".npz":
        import torch
        torch.set_num_threads(2)
    detector = load_detector(arguments.model) if arguments.model else None
    config = detector.config if detector is not None and "config" not in profile else configuration(profile)
    audio = read_audio(arguments.input, config.sample_rate)
    if arguments.mode == "detect":
        config = replace(config, microphones=audio.shape[1])
    calibration = profile.get("calibration", {})
    start_time = perf_counter()
    origin = int(start_time * 1000000) if arguments.realtime else 0
    pipeline = Pipeline(config, detector, Search(**profile.get("search", {})),
                        EventFusion(Fusion(**profile.get("fusion", {}))), CameraPolicy(Camera(**profile.get("camera", {}))),
                        Gate(**profile.get("gate", {})), Weighting(**profile.get("weighting", {})),
                        load_prior(arguments.prior, config), arguments.mode, arguments.fft_gate,
                        round(arguments.background_prefix * config.sample_rate),
                        calibration.get("gains"), calibration.get("delays"), profile.get("channels"),
                        profile.get("coordinates"), origin, arguments.association_verified, profile.get("search_mode", "sector"))
    output = Path(arguments.output)
    output.mkdir(parents=True, exist_ok=True)
    outputs, events, active = [], [], None
    camera = CameraSimulation()
    with (output / "decisions.jsonl").open("w", encoding="utf-8") as rows, (output / "wire.bin").open("wb") as wire:
        for index in range(0, len(audio), config.hop):
            block = audio[index:index + config.hop]
            if arguments.realtime:
                target = start_time + (index + len(block) - 1) / config.sample_rate
                sleep(max(0, target - perf_counter()))
            for item in pipeline.feed(block):
                outputs.append(item)
                rows.write(json.dumps(item.dictionary(), ensure_ascii=False, allow_nan=False) + "\n")
                wire.write(encode(item.decision))
                decision = item.decision
                seconds = (decision.timestamp - origin) / 1000000
                if decision.state == State.CANDIDATE and active is None:
                    active = {"start": seconds, "end": seconds, "confirmed": None, "event_id": decision.event_id,
                              "output": None, "cold": decision.frame < config.context - 1}
                    active["candidate_output"] = item.emitted - origin / 1000000 if arguments.realtime else None
                if active is not None:
                    active["end"] = seconds
                    if decision.state == State.CONFIRMED and active["confirmed"] is None:
                        active["confirmed"] = seconds
                        active["output"] = item.emitted - origin / 1000000 if arguments.realtime else None
                    if decision.state == State.END:
                        events.append(Event(**active))
                        active = None
                for action in item.actions:
                    from dataclasses import replace as replace_action
                    relative = replace_action(action, timestamp=action.timestamp - origin)
                    received = item.emitted - origin / 1000000 if arguments.realtime else seconds
                    camera.feed(relative, received)
    if active is not None:
        events.append(Event(**active))
    camera.advance(max(camera.time, len(audio) / config.sample_rate) + 10)
    exposures = {exposure["event_id"]: exposure["exposure_s"] for exposure in camera.exposures}
    events = [replace(event, capture_simulated=exposures.get(event.event_id)) for event in events]
    write_json(output / "events.json", [asdict(event) for event in events])
    write_json(output / "camera.json", {"scope": "mechanical timing simulation", "exposures": camera.exposures})
    timings = {name: latency_summary([item.timings[name] for item in outputs])
               for name in ("fft_block_amortized_s", "features_s", "classifier_branch_s", "classifier_s", "gcc_s", "srp_s")}
    summary = {"scope": "paced WAV playback" if arguments.realtime else "offline WAV algorithm reference",
               "config": asdict(config), "profile": profile, "frames": len(outputs), "processing_wall_s": perf_counter() - start_time,
               "timings": timings, "background_prefix_s": arguments.background_prefix,
               "active_timings": {name: latency_summary([item.timings[name] for item in outputs if item.timings[name] > 0])
                                   for name in ("classifier_s", "gcc_s", "srp_s")},
               "classifier": arguments.model, "mode": arguments.mode,
               "association_verified": arguments.association_verified,
               "confirmed_events": sum(event.confirmed is not None for event in events),
               "cold_context_s": config.context_seconds, "calibration_delay_samples": pipeline.calibration.latency if pipeline.calibration else 0,
               "fallback_fraction": float(np.mean([item.decision.direction.fallback for item in outputs])) if outputs else None,
               "frame_drop_count": max(0, (max(0, len(audio) - config.frame) // config.hop + 1 if len(audio) >= config.frame else 0) - len(outputs)),
               "buffer_peak_samples": pipeline.stft.peak_buffer,
               "waveform_buffer_peak_samples": pipeline.peak_waveform,
               "input_duration_s": len(audio) / config.sample_rate,
               "processing_output_fps": len(outputs) / max(perf_counter() - start_time, 1e-12),
               "buffer_scope": "measured Python STFT and waveform context; excludes OS/device buffering",
               "latency_note": "FFT timings amortize block assembly/FFT; features and inference are separate; output is decision availability before disk serialization. Offline input/output clocks cannot be subtracted.",
               "fpga_resources": None}
    if arguments.labels:
        label_data = json.loads(Path(arguments.labels).read_text(encoding="utf-8"))
        labels = [Event(*interval) for interval in label_data]
        summary["event_metrics"] = event_metrics(labels, events, arguments.tolerance, duration=len(audio) / config.sample_rate if arguments.continuous else None)
    write_json(output / "summary.json", summary)
    return summary


def golden_command(arguments):
    from .benchmark import synthetic_horn
    from .fixed import golden_frame
    from .frontend import read_audio
    from .geometry import circular_array
    from .simulation import far_field
    config = configuration(read_profile(arguments.profile))
    coordinates = circular_array(config)
    audio = (read_audio(arguments.input, config.sample_rate) if arguments.input else
             far_field(synthetic_horn(config.frame * 5, config.sample_rate, np.random.default_rng(6)),
                       arguments.angle, config).audio)
    start = arguments.frame_index * config.hop
    return golden_frame(audio[start:start + config.frame], coordinates, arguments.output, config, load_prior(arguments.prior, config))


def demo_command(arguments):
    from .benchmark import synthetic_horn
    from .simulation import far_field
    output = Path(arguments.output)
    output.mkdir(parents=True, exist_ok=True)
    config = Config()
    records = []
    for index in range(18):
        generator = np.random.default_rng(5000 + index)
        waveform = generator.normal(0, 0.005, config.sample_rate)
        positive = index % 2
        if positive:
            waveform += synthetic_horn(len(waveform), config.sample_rate, generator)
        path = output / f"source-{index:02d}.wav"
        sf.write(path, waveform, config.sample_rate, subtype="FLOAT")
        records.append({"path": path.name, "group": f"synthetic-{index}", "split": "train" if index < 10 else "validation" if index < 14 else "test",
                        "label": positive, "intervals": [[0, 1]] if positive else [], "channel": 0})
    manifest = output / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
    base = {"manifest": str(manifest), "profile": None, "pretrained": None, "stride": 8}
    prepare_command(argparse.Namespace(**base, frontend="mfcc", output=str(output / "mfcc.npz")))
    prepare_command(argparse.Namespace(**base, frontend="logmel", output=str(output / "logmel.npz")))
    schedule = {"epochs": 20, "batch_size": 32, "learning_rate": 0.003, "seed": 7, "threads": 2,
                "pretrained": None, "fine_tune": False, "teacher": None, "teacher_dataset": None}
    train_command(argparse.Namespace(**schedule, dataset=str(output / "mfcc.npz"), model="svm", output=str(output / "svm.npz")))
    train_command(argparse.Namespace(**schedule, dataset=str(output / "logmel.npz"), model="tiny", output=str(output / "tiny.pt")))
    evaluate_command(argparse.Namespace(dataset=str(output / "logmel.npz"), model=str(output / "tiny.pt"),
                                         output=str(output / "classification.json"), threshold=None, quantize=8, threads=2))
    prior_command(argparse.Namespace(dataset=str(output / "logmel.npz"), output=str(output / "prior.npz")))
    generator = np.random.default_rng(777)
    source = np.zeros(config.sample_rate * 3)
    source[28800:62400] = synthetic_horn(33600, config.sample_rate, generator)
    scene = far_field(source, 30)
    audio = scene.audio + generator.normal(0, 0.005, scene.audio.shape)
    sf.write(output / "scene.wav", audio, config.sample_rate, subtype="FLOAT")
    write_json(output / "labels.json", [[0.6 + scene.common_delay / config.sample_rate, 1.3 + scene.common_delay / config.sample_rate]])
    write_json(output / "demo-profile.json", {"search_mode": "full", "evidence": "known single synthetic horn source"})
    run = run_command(argparse.Namespace(input=str(output / "scene.wav"), model=str(output / "tiny.pt"), profile=str(output / "demo-profile.json"),
                                         prior=str(output / "prior.npz"), mode="aware", background_prefix=0.3,
                                         fft_gate=False, realtime=False, labels=str(output / "labels.json"), tolerance=0.5,
                                         continuous=True, association_verified=True, output=str(output / "pipeline")))
    write_json(output / "evidence.json", {"scope": "synthetic workflow smoke exercise; no public-data or road performance claim",
                                           "groups": records, "pipeline_summary": run})
    return {"output": str(output), "confirmed_events": run["confirmed_events"], "evidence": "synthetic only"}


def parser():
    root = argparse.ArgumentParser(description="Document-aligned horn algorithm reference")
    commands = root.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("fetch-pretrained")
    fetch.add_argument("--output", default="weights/Cnn6.pth")
    split = commands.add_parser("split")
    split.add_argument("--manifest", required=True)
    split.add_argument("--output", required=True)
    split.add_argument("--seed", type=int, default=0)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--manifest", required=True)
    prepare_parser.add_argument("--output", required=True)
    prepare_parser.add_argument("--frontend", choices=["logmel", "pcen", "mfcc", "panns"], default="logmel")
    prepare_parser.add_argument("--stride", type=int, default=1)
    prepare_parser.add_argument("--pretrained")
    prior = commands.add_parser("learn-prior")
    prior.add_argument("--dataset", required=True)
    prior.add_argument("--output", required=True)
    train = commands.add_parser("train")
    train.add_argument("--dataset", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--model", choices=["svm", "tiny", "mlp", "crnn", "transfer"], default="tiny")
    train.add_argument("--epochs", type=int, default=10)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--learning-rate", type=float, default=0.001)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--pretrained")
    train.add_argument("--fine-tune", action="store_true")
    train.add_argument("--teacher")
    train.add_argument("--teacher-dataset")
    train.add_argument("--threads", type=int, default=2)
    evaluate = commands.add_parser("evaluate-model")
    evaluate.add_argument("--dataset", required=True)
    evaluate.add_argument("--model", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--threshold", type=float)
    evaluate.add_argument("--quantize", type=int)
    evaluate.add_argument("--threads", type=int, default=2)
    simulate = commands.add_parser("simulate")
    simulate.add_argument("--output", required=True)
    simulate.add_argument("--angle", type=float, default=30)
    simulate.add_argument("--seconds", type=float, default=1)
    simulate.add_argument("--seed", type=int, default=0)
    simulate.add_argument("--randomized", action="store_true")
    simulate.add_argument("--noise", choices=["white", "engine", "traffic", "siren", "wind"], default="white")
    benchmark = commands.add_parser("benchmark")
    benchmark.add_argument("--output", default="outputs/benchmark")
    benchmark.add_argument("--quick", action="store_true")
    benchmark.add_argument("--seed", type=int, default=0)
    benchmark.add_argument("--prior")
    run = commands.add_parser("run")
    run.add_argument("--input", required=True)
    run.add_argument("--model")
    run.add_argument("--output", default="outputs/run")
    run.add_argument("--mode", choices=["aware", "srp", "detect"], default="aware")
    run.add_argument("--prior")
    run.add_argument("--background-prefix", type=float, default=0)
    run.add_argument("--fft-gate", action="store_true")
    run.add_argument("--realtime", action="store_true")
    run.add_argument("--association-verified", action="store_true", help="Only for known single-source simulation or externally validated association conditions")
    run.add_argument("--labels")
    run.add_argument("--continuous", action="store_true")
    run.add_argument("--tolerance", type=float, default=0.5)
    golden = commands.add_parser("export-golden")
    golden.add_argument("--input")
    golden.add_argument("--output", default="outputs/golden")
    golden.add_argument("--angle", type=float, default=30)
    golden.add_argument("--frame-index", type=int, default=2)
    golden.add_argument("--prior")
    demo = commands.add_parser("demo")
    demo.add_argument("--output", default="outputs/demo")
    commands.add_parser("audit")
    for command in (prepare_parser, simulate, benchmark, run, golden):
        command.add_argument("--profile")
    return root


def main():
    arguments = parser().parse_args()
    functions = {"split": split_command, "prepare": prepare_command, "learn-prior": prior_command,
                 "train": train_command, "evaluate-model": evaluate_command, "simulate": simulate_command,
                 "benchmark": benchmark_command, "run": run_command, "export-golden": golden_command, "demo": demo_command}
    if arguments.command == "fetch-pretrained":
        from .pretrained import fetch_weights
        result = fetch_weights(arguments.output)
    elif arguments.command == "audit":
        from .audit import audit_repository
        result = audit_repository()
    else:
        result = functions[arguments.command](arguments)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
