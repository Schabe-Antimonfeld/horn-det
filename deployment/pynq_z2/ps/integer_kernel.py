"""Generated integer reference; regenerate with scripts/export_integer.py."""
import hashlib
import json
from pathlib import Path
import numpy as np

def sliding_windows(values, shape, axis):
    if tuple(axis) != (1, 2) or tuple(shape) != (3, 3):
        raise ValueError('Only the trained 3x3 convolution is supported')
    return np.lib.stride_tricks.as_strided(values,
        shape=(values.shape[0], values.shape[1]-2, values.shape[2]-2, 3, 3),
        strides=values.strides + values.strides[1:], writeable=False)

SOURCE_SHA256 = '705860976f30273344e0d4d500cc25f2f2ba6342ab3f27e6fbdb4e661474fbfe'

LAYERS = (('conv', 1, 16, 3), ('depth', 16, 16, 3), ('point', 16, 24, 1), ('depth', 24, 24, 3), ('point', 24, 32, 1), ('linear', 32, 2, 1))
WEIGHT_BASE = (0, 18, 36, 84, 111, 207)
BIAS_BASE = (0, 2, 4, 7, 10, 14)

def round_away(values):
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError('Quantization requires finite values')
    if np.any(np.abs(values) >= 1 << 62):
        raise ValueError('Quantization value exceeds the safe integer conversion range')
    return (np.sign(values) * np.floor(np.abs(values) + 0.5)).astype(np.int64)

def shift_round(values, shift):
    values = np.asarray(values, dtype=np.int64)
    if not 0 <= shift <= 35:
        raise ValueError('Hardware requantization shift must be in [0, 35]')
    if shift == 0:
        return values.copy()
    return np.sign(values) * (np.abs(values) + (1 << shift - 1) >> shift)

def input_codes(features, fraction):
    features = np.asarray(features)
    if features.shape[-2:] != (40, 24) or not 0 <= fraction <= 15:
        raise ValueError('Require 40x24 features and a supported input fraction')
    codes = round_away(features * (1 << fraction))
    clipped = int(np.count_nonzero((codes < -32768) | (codes > 32767)))
    return (np.clip(codes, -32768, 32767).astype(np.int16), clipped)

def softmax_table():
    values = 1 / (1 + np.exp(-np.linspace(-8, 8, 1025)))
    return np.minimum(65535, round_away(values * 65536)).astype(np.uint16)

def probability_code(logits, fraction, table=None):
    difference = int(logits[1]) - int(logits[0])
    if fraction >= 6:
        offset = int(shift_round(difference, fraction - 6))
    else:
        offset = difference << 6 - fraction
    return int((softmax_table() if table is None else table)[min(1024, max(0, 512 + offset))])

def integer_layer(values, weights, bias, kind, shift, relu):
    values, weights = (np.asarray(values, dtype=np.int64), np.asarray(weights, dtype=np.int64))
    if kind == 'linear':
        total = weights @ values + bias
    elif kind == 'point':
        total = np.tensordot(weights[:, :, 0, 0], values, axes=(1, 0)) + bias[:, None, None]
    else:
        padded = np.pad(values, ((0, 0), (1, 1), (1, 1)))
        windows = sliding_windows(padded, (3, 3), axis=(1, 2))
        if kind == 'depth':
            total = np.sum(windows * weights[:, 0, None, None], axis=(-1, -2)) + bias[:, None, None]
        else:
            total = np.tensordot(weights, windows, axes=((1, 2, 3), (0, 3, 4))) + bias[:, None, None]
    if np.any(total < -(1 << 47)) or np.any(total >= 1 << 47):
        raise ValueError('Accumulator exceeds hardware 48-bit range')
    if relu:
        total = np.maximum(total, 0)
    rounded = shift_round(total, shift)
    clips = int(np.count_nonzero((rounded < -32768) | (rounded > 32767)))
    return (np.clip(rounded, -32768, 32767).astype(np.int16), clips)

def fixed_forward(model, codes):
    values = np.asarray(codes, dtype=np.int16)
    if values.shape != (40, 24):
        raise ValueError('One hardware input must have shape (40, 24)')
    values = values[None]
    clips, traces = ([], [])
    for index, layer in enumerate(model['layers']):
        if index == 5:
            values = ((values.astype(np.int64).sum(axis=(1, 2)) + 480) // 960).astype(np.int16)
        values, count = integer_layer(values, layer['weights'], layer['bias'], LAYERS[index][0], layer['shift'], index != 5)
        clips.append(count)
        traces.append(values)
    probability = probability_code(values, model['logit_fraction'])
    return {'logits': values, 'probability': probability, 'clips': clips, 'flags': int(any(clips)), 'trace': traces}

def check_export(directory):
    directory = Path(directory)
    metadata = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    if metadata['format'] != 1 or metadata['architecture'] != 'TinyCNN-1834' or metadata['input_shape'] != [40, 24]:
        raise ValueError('Unsupported FPGA model export')
    required = {f'weights{lane}.mem': (256, 2) for lane in range(8)}
    required.update({f'bias{lane}.mem': (16, 12) for lane in range(8)})
    required.update({'config.mem': (9, 8), 'softmax.mem': (1025, 4), 'inputs.mem': (metadata['vectors'] * 960, 8), 'expected.mem': (metadata['vectors'] * 8, 8)})
    for name, (length, width) in required.items():
        data = (directory / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != metadata['sha256'].get(name):
            raise ValueError(f'Model file checksum mismatch: {name}')
        lines = data.decode('ascii').splitlines()
        if len(lines) != length or any((len(line) != width for line in lines)):
            raise ValueError(f'Unexpected memory dimensions: {name}')
        for line in lines:
            int(line, 16)
    configuration = [int(line, 16) for line in (directory / 'config.mem').read_text().splitlines()]
    if configuration[0] != metadata['input_fraction'] or configuration[7] != metadata['logit_fraction'] or configuration[8] != metadata['model_tag']:
        raise ValueError('Manifest and hardware configuration disagree')
    if not 0 <= configuration[0] <= 15 or not 0 <= configuration[7] <= 15 or any((shift > 35 for shift in configuration[1:7])):
        raise ValueError('Unsupported hardware scaling')
    if len(metadata['layers']) != 6 or any((layer['shift'] != configuration[index + 1] for index, layer in enumerate(metadata['layers']))):
        raise ValueError('Manifest and hardware layer shifts disagree')
    return metadata

def read_memory(path, bits, signed=False):
    values = np.asarray([int(line, 16) for line in Path(path).read_text().splitlines()], dtype=np.int64)
    if signed:
        values = np.where(values >= 1 << bits - 1, values - (1 << bits), values)
    return values

def load_fixed(directory):
    directory = Path(directory)
    metadata = check_export(directory)
    weights = [read_memory(directory / f'weights{lane}.mem', 8, True) for lane in range(8)]
    biases = [read_memory(directory / f'bias{lane}.mem', 48, True) for lane in range(8)]
    result = {key: metadata[key] for key in ('input_fraction', 'logit_fraction', 'model_tag')}
    result['layers'] = []
    for index, (kind, incoming, outgoing, kernel) in enumerate(LAYERS):
        shape = (outgoing, incoming) if kind == 'linear' else (outgoing, 1 if kind == 'depth' else incoming, kernel, kernel)
        length = int(np.prod(shape[1:]))
        rows, bias = ([], [])
        for channel in range(outgoing):
            offset = WEIGHT_BASE[index] + channel // 8 * length
            rows.append(weights[channel % 8][offset:offset + length])
            bias.append(biases[channel % 8][BIAS_BASE[index] + channel // 8])
        result['layers'].append({'weights': np.asarray(rows, dtype=np.int8).reshape(shape), 'bias': np.asarray(bias, dtype=np.int64), 'shift': metadata['layers'][index]['shift']})
    return result
