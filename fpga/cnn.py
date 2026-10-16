"""Export TinyCNN weights, define integer inference and run RTL comparison."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LAYERS = (("conv", 1, 16, 3), ("depth", 16, 16, 3), ("point", 16, 24, 1),
          ("depth", 24, 24, 3), ("point", 24, 32, 1), ("linear", 32, 2, 1))
KEYS = ("encoder.0", "encoder.2.layers.0", "encoder.2.layers.2",
        "encoder.3.layers.0", "encoder.3.layers.2", "head")
WEIGHT_BASE = (0, 18, 36, 84, 111, 207)
BIAS_BASE = (0, 2, 4, 7, 10, 14)


def round_away(values):
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Quantization requires finite values")
    if np.any(np.abs(values) >= 1 << 62):
        raise ValueError("Quantization value exceeds the safe integer conversion range")
    return (np.sign(values) * np.floor(np.abs(values) + 0.5)).astype(np.int64)


def shift_round(values, shift):
    values = np.asarray(values, dtype=np.int64)
    if not 0 <= shift <= 35:
        raise ValueError("Hardware requantization shift must be in [0, 35]")
    if shift == 0:
        return values.copy()
    return np.sign(values) * ((np.abs(values) + (1 << (shift - 1))) >> shift)


def tensor_fraction(values, bits, maximum):
    peak = float(np.max(np.abs(values), initial=0))
    if not np.isfinite(peak):
        raise ValueError("Nonfinite calibration tensor")
    if peak == 0:
        return min(maximum, bits - 1)
    result = int(np.floor(np.log2(((1 << (bits - 1)) - 1) * 0.95 / peak)))
    return min(maximum, max(0, result))


def input_codes(features, fraction):
    features = np.asarray(features)
    if features.shape[-2:] != (40, 24) or not 0 <= fraction <= 15:
        raise ValueError("Require 40x24 features and a supported input fraction")
    codes = round_away(features * (1 << fraction))
    clipped = int(np.count_nonzero((codes < -32768) | (codes > 32767)))
    return np.clip(codes, -32768, 32767).astype(np.int16), clipped


def softmax_table():
    values = 1 / (1 + np.exp(-np.linspace(-8, 8, 1025)))
    return np.minimum(65535, round_away(values * 65536)).astype(np.uint16)


def probability_code(logits, fraction, table=None):
    difference = int(logits[1]) - int(logits[0])
    if fraction >= 6:
        offset = int(shift_round(difference, fraction - 6))
    else:
        offset = difference << (6 - fraction)
    return int((softmax_table() if table is None else table)[min(1024, max(0, 512 + offset))])


def integer_layer(values, weights, bias, kind, shift, relu):
    values, weights = np.asarray(values, dtype=np.int64), np.asarray(weights, dtype=np.int64)
    if kind == "linear":
        total = weights @ values + bias
    elif kind == "point":
        total = np.tensordot(weights[:, :, 0, 0], values, axes=(1, 0)) + bias[:, None, None]
    else:
        padded = np.pad(values, ((0, 0), (1, 1), (1, 1)))
        windows = np.lib.stride_tricks.sliding_window_view(padded, (3, 3), axis=(1, 2))
        if kind == "depth":
            total = np.sum(windows * weights[:, 0, None, None], axis=(-1, -2)) + bias[:, None, None]
        else:
            total = np.tensordot(weights, windows, axes=((1, 2, 3), (0, 3, 4))) + bias[:, None, None]
    if np.any(total < -(1 << 47)) or np.any(total >= 1 << 47):
        raise ValueError("Accumulator exceeds hardware 48-bit range")
    if relu:
        total = np.maximum(total, 0)
    rounded = shift_round(total, shift)
    clips = int(np.count_nonzero((rounded < -32768) | (rounded > 32767)))
    return np.clip(rounded, -32768, 32767).astype(np.int16), clips


def fixed_forward(model, codes):
    values = np.asarray(codes, dtype=np.int16)
    if values.shape != (40, 24):
        raise ValueError("One hardware input must have shape (40, 24)")
    values = values[None]
    clips, traces = [], []
    for index, layer in enumerate(model["layers"]):
        if index == 5:
            values = ((values.astype(np.int64).sum(axis=(1, 2)) + 480) // 960).astype(np.int16)
        values, count = integer_layer(values, layer["weights"], layer["bias"],
                                      LAYERS[index][0], layer["shift"], index != 5)
        clips.append(count)
        traces.append(values)
    probability = probability_code(values, model["logit_fraction"])
    return {"logits": values, "probability": probability, "clips": clips,
            "flags": int(any(clips)), "trace": traces}


def tiny_model():
    if str(ROOT / "prototype" / "src") not in sys.path:
        sys.path.insert(0, str(ROOT / "prototype" / "src"))
    from horn.models import TinyCNN
    return TinyCNN()


def calibrate(model, features):
    import torch
    torch.set_num_threads(2)
    model.eval()
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 3 or features.shape[1:] != (40, 24) or len(features) == 0:
        raise ValueError("Calibration requires nonempty N x 40 x 24 features")
    peaks = [0.0] * 6
    with torch.no_grad():
        for start in range(0, len(features), 16):
            values = torch.as_tensor(features[start:start + 16, None])
            modules = (model.encoder[0], model.encoder[2].layers[0], model.encoder[2].layers[2],
                       model.encoder[3].layers[0], model.encoder[3].layers[2], model.head)
            for index, module in enumerate(modules):
                if index == 5:
                    values = values.mean(dim=(2, 3))
                values = module(values)
                if index != 5:
                    values = torch.relu(values)
                peaks[index] = max(peaks[index], float(values.abs().max()))
    incoming = tensor_fraction(features, 16, 15)
    result = {"input_fraction": incoming, "layers": []}
    state = model.state_dict()
    for index, key in enumerate(KEYS):
        weights = state[key + ".weight"].detach().cpu().numpy()
        bias = state[key + ".bias"].detach().cpu().numpy()
        weight_fraction = tensor_fraction(weights, 8, 20)
        encoded = round_away(weights * (1 << weight_fraction))
        if np.max(np.abs(encoded), initial=0) > 127:
            raise ValueError("Weights exceed signed INT8 with supported scaling")
        outgoing = min(tensor_fraction([peaks[index]], 16, 15), incoming + weight_fraction)
        encoded_bias = round_away(bias * (1 << (incoming + weight_fraction)))
        rows = encoded.reshape(len(encoded), -1)
        bounds = np.abs(encoded_bias) + 32768 * np.abs(rows).sum(axis=1)
        if np.any(bounds >= 1 << 47):
            raise ValueError("Bias and worst-case dot product exceed 48 bits")
        result["layers"].append({"weights": encoded.astype(np.int8), "bias": encoded_bias,
                                 "shift": incoming + weight_fraction - outgoing,
                                 "input_fraction": incoming, "weight_fraction": weight_fraction,
                                 "output_fraction": outgoing})
        incoming = outgoing
    result["logit_fraction"] = incoming
    digest = hashlib.sha256()
    digest.update(bytes([result["input_fraction"], result["logit_fraction"]]))
    for layer in result["layers"]:
        digest.update(layer["weights"].tobytes())
        digest.update(layer["bias"].astype("<i8").tobytes())
        digest.update(bytes([layer["shift"]]))
    result["model_tag"] = int(digest.hexdigest()[:8], 16)
    return result


def write_memory(path, values, bits):
    mask = (1 << bits) - 1
    path.write_text("".join(f"{int(value) & mask:0{bits // 4}x}\n" for value in np.ravel(values)), encoding="ascii")


def export_model(model, calibration, directory, vectors=4, provenance=None):
    import torch
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    fixed = calibrate(model, calibration)
    weights = np.zeros((8, 256), dtype=np.int8)
    biases = np.zeros((8, 16), dtype=np.int64)
    for index, layer in enumerate(fixed["layers"]):
        rows = layer["weights"].reshape(len(layer["weights"]), -1)
        for channel, row in enumerate(rows):
            offset = WEIGHT_BASE[index] + (channel // 8) * len(row)
            weights[channel % 8, offset:offset + len(row)] = row
            biases[channel % 8, BIAS_BASE[index] + channel // 8] = layer["bias"][channel]
    for lane in range(8):
        write_memory(directory / f"weights{lane}.mem", weights[lane], 8)
        write_memory(directory / f"bias{lane}.mem", biases[lane], 48)
    configuration = [fixed["input_fraction"]] + [layer["shift"] for layer in fixed["layers"]]
    configuration += [fixed["logit_fraction"], fixed["model_tag"]]
    write_memory(directory / "config.mem", configuration, 32)
    write_memory(directory / "softmax.mem", softmax_table(), 16)
    count = min(vectors, len(calibration))
    if count < 1:
        raise ValueError("At least one verification vector is required")
    features = np.asarray(calibration[:count], dtype=np.float32)
    codes, input_clips = input_codes(features, fixed["input_fraction"])
    words, results, differences = [], [], []
    with torch.no_grad():
        floats = torch.softmax(model(torch.as_tensor(features[:, None])), dim=1)[:, 1].numpy()
    for index, item in enumerate(codes):
        output = fixed_forward(fixed, item)
        logit0, logit1 = map(int, output["logits"])
        results.extend([0x314e4e43, 0x00010008, index, logit0, logit1,
                        (output["flags"] << 16) | output["probability"], 0, fixed["model_tag"]])
        words.extend(item.astype(np.int32).ravel())
        differences.append(abs(output["probability"] / 65536 - float(floats[index])))
    write_memory(directory / "inputs.mem", words, 32)
    write_memory(directory / "expected.mem", results, 32)
    np.save(directory / "features.npy", features, allow_pickle=False)
    memories = sorted(directory.glob("*.mem"))
    metadata = {"format": 1, "architecture": "TinyCNN-1834", "input_shape": [40, 24],
                "input_fraction": fixed["input_fraction"], "logit_fraction": fixed["logit_fraction"],
                "model_tag": fixed["model_tag"], "vectors": count,
                "layers": [{key: value for key, value in layer.items() if key not in ("weights", "bias")}
                           for layer in fixed["layers"]],
                "input_clips": input_clips, "maximum_probability_error_on_vectors": max(differences),
                "provenance": provenance or {"evidence": "unspecified; no road performance claim"},
                "sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in memories}}
    (directory / "manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def check_export(directory):
    directory = Path(directory)
    metadata = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if metadata["format"] != 1 or metadata["architecture"] != "TinyCNN-1834" or metadata["input_shape"] != [40, 24]:
        raise ValueError("Unsupported FPGA model export")
    required = {f"weights{lane}.mem": (256, 2) for lane in range(8)}
    required.update({f"bias{lane}.mem": (16, 12) for lane in range(8)})
    required.update({"config.mem": (9, 8), "softmax.mem": (1025, 4),
                     "inputs.mem": (metadata["vectors"] * 960, 8),
                     "expected.mem": (metadata["vectors"] * 8, 8)})
    for name, (length, width) in required.items():
        data = (directory / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != metadata["sha256"].get(name):
            raise ValueError(f"Model file checksum mismatch: {name}")
        lines = data.decode("ascii").splitlines()
        if len(lines) != length or any(len(line) != width for line in lines):
            raise ValueError(f"Unexpected memory dimensions: {name}")
        for line in lines:
            int(line, 16)
    configuration = [int(line, 16) for line in (directory / "config.mem").read_text().splitlines()]
    if configuration[0] != metadata["input_fraction"] or configuration[7] != metadata["logit_fraction"] or configuration[8] != metadata["model_tag"]:
        raise ValueError("Manifest and hardware configuration disagree")
    if not 0 <= configuration[0] <= 15 or not 0 <= configuration[7] <= 15 or any(shift > 35 for shift in configuration[1:7]):
        raise ValueError("Unsupported hardware scaling")
    if len(metadata["layers"]) != 6 or any(layer["shift"] != configuration[index+1] for index, layer in enumerate(metadata["layers"])):
        raise ValueError("Manifest and hardware layer shifts disagree")
    return metadata


def read_memory(path, bits, signed=False):
    values = np.asarray([int(line, 16) for line in Path(path).read_text().splitlines()], dtype=np.int64)
    if signed:
        values = np.where(values >= 1 << (bits-1), values - (1 << bits), values)
    return values


def load_fixed(directory):
    directory = Path(directory)
    metadata = check_export(directory)
    weights = [read_memory(directory / f"weights{lane}.mem", 8, True) for lane in range(8)]
    biases = [read_memory(directory / f"bias{lane}.mem", 48, True) for lane in range(8)]
    result = {key: metadata[key] for key in ("input_fraction", "logit_fraction", "model_tag")}
    result["layers"] = []
    for index, (kind, incoming, outgoing, kernel) in enumerate(LAYERS):
        shape = (outgoing, incoming) if kind == "linear" else (outgoing, 1 if kind == "depth" else incoming, kernel, kernel)
        length = int(np.prod(shape[1:]))
        rows, bias = [], []
        for channel in range(outgoing):
            offset = WEIGHT_BASE[index] + channel // 8 * length
            rows.append(weights[channel % 8][offset:offset+length])
            bias.append(biases[channel % 8][BIAS_BASE[index]+channel//8])
        result["layers"].append({"weights": np.asarray(rows, dtype=np.int8).reshape(shape),
                                 "bias": np.asarray(bias, dtype=np.int64),
                                 "shift": metadata["layers"][index]["shift"]})
    return result


def verify_rtl(directory, executable=None):
    metadata = check_export(directory)
    directory = Path(directory).resolve()
    compiler, runtime = shutil.which("iverilog"), shutil.which("vvp")
    if not compiler or not runtime:
        raise RuntimeError("Icarus Verilog and vvp are required for RTL comparison")
    executable = Path(executable) if executable else directory / "cnn.vvp"
    command = [compiler, "-g2012", "-Wall", "-s", "tb_cnn",
               f'-Ptb_cnn.MODEL_DIR="{directory.as_posix()}"', f'-Ptb_cnn.FRAMES={metadata["vectors"]}',
               "-o", str(executable), str(ROOT / "fpga/rtl/tiny_cnn.v"), str(ROOT / "fpga/sim/tb_cnn.sv")]
    subprocess.run(command, check=True, timeout=120)
    result = subprocess.run([runtime, str(executable)], text=True, capture_output=True, timeout=300)
    print(result.stdout, end="", flush=True)
    if result.returncode:
        print(result.stderr, end="", file=sys.stderr)
        result.check_returncode()
    return result.stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("demo", "export"):
        command = commands.add_parser(name)
        command.add_argument("--output", required=True)
        command.add_argument("--vectors", type=int, default=4)
        if name == "export":
            command.add_argument("--checkpoint", required=True)
            command.add_argument("--dataset", required=True)
            command.add_argument("--calibration-split", choices=("train", "validation"), default="validation")
    for name in ("check", "verify"):
        command = commands.add_parser(name)
        command.add_argument("--model-dir", required=True)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--model-dir", required=True)
    evaluate.add_argument("--dataset", required=True)
    evaluate.add_argument("--split", choices=("validation", "test"), required=True)
    evaluate.add_argument("--threshold", type=float, required=True)
    evaluate.add_argument("--output", required=True)
    arguments = parser.parse_args()
    if arguments.command == "check":
        print(json.dumps(check_export(arguments.model_dir), indent=2))
    elif arguments.command == "verify":
        verify_rtl(arguments.model_dir)
    elif arguments.command == "evaluate":
        fixed = load_fixed(arguments.model_dir)
        with np.load(arguments.dataset, allow_pickle=False) as data:
            configuration = json.loads(str(data["config"]))
            expected = {"sample_rate": 48000, "frame": 1024, "hop": 512, "mel_bands": 40, "context": 24, "epsilon": 1e-12}
            if str(data["frontend"]) != "logmel" or any(configuration[key] != value for key, value in expected.items()):
                raise ValueError("Dataset frontend does not match the hardware contract")
            selected = data["splits"] == arguments.split
            features, labels = data["features"][selected, 0], data["labels"][selected]
        if len(features) == 0:
            raise ValueError("Requested evaluation split is empty")
        codes, input_clips = input_codes(features, fixed["input_fraction"])
        estimates = [fixed_forward(fixed, item) for item in codes]
        probabilities = np.asarray([item["probability"] / 65536 for item in estimates])
        sys.path.insert(0, str(ROOT / "prototype/src"))
        from horn.evaluation import window_metrics
        report = {"split": arguments.split, "threshold": arguments.threshold,
                  "metrics": window_metrics(labels, probabilities, arguments.threshold),
                  "input_clips": input_clips, "activation_clipped_windows": sum(item["flags"] for item in estimates),
                  "model_tag": fixed["model_tag"], "probabilities": probabilities.tolist(),
                  "scope": "integer window classification; no continuous event, road or board claim"}
        destination = Path(arguments.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report["metrics"], indent=2))
    else:
        import torch
        torch.manual_seed(7)
        model = tiny_model()
        if arguments.command == "demo":
            calibration = np.random.default_rng(19).normal(-7, 4, (8, 40, 24)).astype(np.float32)
            calibration[0] = 0
            calibration[1] = 0
            calibration[1, 0, 0] = 8
            calibration[1, -1, -1] = -8
            provenance = {"evidence": "random untrained weights and synthetic features; functional verification only"}
        else:
            checkpoint = torch.load(arguments.checkpoint, map_location="cpu", weights_only=True)
            expected = {"sample_rate": 48000, "frame": 1024, "hop": 512, "mel_bands": 40, "context": 24, "epsilon": 1e-12}
            if checkpoint["name"] != "tiny" or checkpoint["frontend"] != "logmel" or any(checkpoint["config"][key] != value for key, value in expected.items()):
                raise ValueError("Require the documented 48 kHz, 40x24 Log-Mel TinyCNN checkpoint")
            model.load_state_dict(checkpoint["state"], strict=True)
            with np.load(arguments.dataset, allow_pickle=False) as data:
                configuration = json.loads(str(data["config"]))
                if str(data["frontend"]) != "logmel" or any(configuration[key] != value for key, value in expected.items()):
                    raise ValueError("Calibration frontend differs from the checkpoint")
                selected = data["splits"] == arguments.calibration_split
                calibration = np.asarray(data["features"][selected, 0], dtype=np.float32)
            provenance = {"checkpoint_sha256": hashlib.sha256(Path(arguments.checkpoint).read_bytes()).hexdigest(),
                          "dataset_sha256": hashlib.sha256(Path(arguments.dataset).read_bytes()).hexdigest(),
                          "calibration_split": arguments.calibration_split,
                          "evidence": "exported reference; event and road performance require separate evaluation"}
        print(json.dumps(export_model(model, calibration, arguments.output, arguments.vectors, provenance), indent=2))


if __name__ == "__main__":
    main()
