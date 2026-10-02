"""Saturating integer reference, radix-2 transforms and FPGA golden vectors."""

from dataclasses import dataclass, asdict
import json
import math
from pathlib import Path

import numpy as np

from .config import Config
from .features import mel_bank
from .geometry import delay_table, pairs, round_away


@dataclass(frozen=True)
class Format:
    bits: int
    fraction: int
    signed: bool = True

    def __post_init__(self):
        if not 1 <= self.bits <= 62 or not 0 <= self.fraction <= self.bits - int(self.signed):
            raise ValueError("Invalid fixed-point format")

    @property
    def minimum(self):
        return -(1 << (self.bits - 1)) if self.signed else 0

    @property
    def maximum(self):
        return (1 << (self.bits - int(self.signed))) - 1

    def saturate(self, codes):
        codes = np.asarray(codes)
        count = int(np.count_nonzero((codes < self.minimum) | (codes > self.maximum)))
        return np.asarray(np.clip(codes, self.minimum, self.maximum), dtype=np.int64), count

    def encode(self, values):
        values = np.asarray(values, dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Cannot encode nonfinite values")
        scaled = np.sign(values) * np.floor(np.abs(values) * (1 << self.fraction) + 0.5)
        return self.saturate(scaled)

    def decode(self, codes):
        return np.asarray(codes, dtype=float) / (1 << self.fraction)


PCM = Format(16, 15)
HANN = Format(16, 15)
TRANSFORM = Format(24, 15)
PHAT = Format(16, 15)
WEIGHT = Format(16, 16, False)
POWER = Format(40, 30, False)
ACCUMULATOR = Format(32, 15)


def shift_round(codes, shift):
    """Use Python-width intermediates so product rounding cannot silently overflow."""
    codes = np.asarray(codes, dtype=object)
    if shift <= 0:
        return codes * (1 << -shift)
    half = 1 << (shift - 1)
    return np.asarray([((int(value) + half) >> shift) if value >= 0 else
                       -((-int(value) + half) >> shift) for value in codes.flat], dtype=object).reshape(codes.shape)


def multiply(left, right, left_format, right_format, output_format):
    product = np.asarray(left, dtype=object) * np.asarray(right, dtype=object)
    shifted = shift_round(product, left_format.fraction + right_format.fraction - output_format.fraction)
    return output_format.saturate(shifted)


def integer_fft(real, imaginary=None, inverse=False, output_format=TRANSFORM, stage_shift=1):
    """Radix-2 DIT; quantize every twiddle product and butterfly, shift each stage."""
    real = np.asarray(real, dtype=np.int64)
    if real.ndim != 2 or len(real) < 2 or len(real) & (len(real) - 1):
        raise ValueError("FFT input must be (power-of-two samples, streams)")
    if stage_shift not in (0, 1):
        raise ValueError("Stage shift must be zero or one")
    imaginary = np.zeros_like(real) if imaginary is None else np.asarray(imaginary, dtype=np.int64)
    if imaginary.shape != real.shape:
        raise ValueError("FFT complex components must agree")
    stages = int(math.log2(len(real)))
    order = np.array([int(format(index, f"0{stages}b")[::-1], 2) for index in range(len(real))])
    real, imaginary = real[order].copy(), imaginary[order].copy()
    saturated = 0
    for stage in range(1, stages + 1):
        length = 1 << stage
        half = length // 2
        phase = (1 if inverse else -1) * 2 * np.pi * np.arange(half) / length
        cosine, _ = HANN.encode(np.cos(phase))
        sine, _ = HANN.encode(np.sin(phase))
        for start in range(0, len(real), length):
            lower = slice(start, start + half)
            upper = slice(start + half, start + length)
            top_real, top_imaginary = real[lower].copy(), imaginary[lower].copy()
            product_real = real[upper] * cosine[:, None] - imaginary[upper] * sine[:, None]
            product_imaginary = real[upper] * sine[:, None] + imaginary[upper] * cosine[:, None]
            bottom_real, count = output_format.saturate(shift_round(product_real, 15))
            saturated += count
            bottom_imaginary, count = output_format.saturate(shift_round(product_imaginary, 15))
            saturated += count
            for target, value in ((lower, top_real + bottom_real), (upper, top_real - bottom_real)):
                real[target], count = output_format.saturate(shift_round(value, stage_shift))
                saturated += count
            for target, value in ((lower, top_imaginary + bottom_imaginary), (upper, top_imaginary - bottom_imaginary)):
                imaginary[target], count = output_format.saturate(shift_round(value, stage_shift))
                saturated += count
    return real, imaginary, {"saturations": saturated, "scale": 2 ** (-stage_shift * stages)}


class Reciprocal:
    def __init__(self, address_bits=10, fraction=30):
        if not 2 <= address_bits <= 16 or not 16 <= fraction <= 40:
            raise ValueError("Invalid reciprocal LUT precision")
        self.address_bits, self.fraction = address_bits, fraction
        self.table = round_away(1 / (1 + np.arange(1 << address_bits) / (1 << address_bits)) * (1 << fraction))

    def lookup(self, magnitude):
        magnitude = int(magnitude)
        if magnitude <= 0:
            raise ValueError("Reciprocal requires a positive magnitude")
        exponent = magnitude.bit_length() - 1
        address = ((magnitude - (1 << exponent)) << self.address_bits) >> exponent
        return int(self.table[address]), exponent

    def approximate(self, magnitude):
        code, exponent = self.lookup(magnitude)
        return code / (1 << (self.fraction + exponent))


def integer_phat(real, imaginary, lut=None, floor=1):
    real, imaginary = np.asarray(real, dtype=np.int64), np.asarray(imaginary, dtype=np.int64)
    if real.shape != imaginary.shape or floor < 1:
        raise ValueError("Invalid PHAT components or floor")
    lut = Reciprocal() if lut is None else lut
    widened_real, widened_imaginary = np.abs(real.astype(object)), np.abs(imaginary.astype(object))
    maximum = np.maximum(widened_real, widened_imaginary)
    minimum = np.minimum(widened_real, widened_imaginary)
    magnitude = maximum + shift_round(minimum * 3, 3)
    output_real, output_imaginary = np.zeros_like(real), np.zeros_like(real)
    count = 0
    for index in np.ndindex(real.shape):
        if magnitude[index] < floor:
            continue
        code, exponent = lut.lookup(magnitude[index])
        shift = lut.fraction + exponent - PHAT.fraction
        output_real[index], saturated = PHAT.saturate(shift_round(int(real[index]) * code, shift))
        count += saturated
        output_imaginary[index], saturated = PHAT.saturate(shift_round(int(imaginary[index]) * code, shift))
        count += saturated
    return output_real, output_imaginary, count


def delay_codes(coordinates, angles, config=Config()):
    delays = delay_table(coordinates, angles, config.sample_rate, config.sound_speed)
    lower = np.floor(delays).astype(np.int64)
    fraction = round_away((delays - lower) * 65536)
    lower += fraction // 65536
    fraction %= 65536
    return {"integer_delay": lower, "fraction": fraction, "address": lower % config.frame,
            "next_address": (lower + 1) % config.frame, "rounded_delay": round_away(delays)}


def integer_srp(gcc_codes, lut, linear=True, mask=None):
    gcc_codes = np.asarray(gcc_codes, dtype=np.int64)
    rows = np.arange(len(gcc_codes))[:, None]
    if mask is None:
        mask = np.ones(len(gcc_codes), dtype=bool)
    if linear:
        first = gcc_codes[rows, lut["address"]]
        second = gcc_codes[rows, lut["next_address"]]
        interpolated = shift_round(first * (65536 - lut["fraction"]) + second * lut["fraction"], 16)
    else:
        interpolated = gcc_codes[rows, lut["rounded_delay"] % gcc_codes.shape[1]]
    return ACCUMULATOR.saturate(np.sum(interpolated * np.asarray(mask)[:, None], axis=0))


def divide_round(numerator, denominator):
    if denominator <= 0:
        raise ValueError("Integer division requires a positive denominator")
    sign = -1 if numerator < 0 else 1
    return sign * ((2 * abs(int(numerator)) + int(denominator)) // (2 * int(denominator)))


def noise_update(previous, current, alpha=0.98, enabled=True):
    previous, current = np.asarray(previous, dtype=np.int64), np.asarray(current, dtype=np.int64)
    if previous.shape != current.shape or np.any(previous < 0) or np.any(current < 0) or not 0 <= alpha < 1:
        raise ValueError("Invalid integer noise PSD update")
    if not enabled:
        return previous.copy(), 0
    code = int(WEIGHT.encode(alpha)[0])
    delta = shift_round((current.astype(object) - previous) * (65536 - code), 16)
    return POWER.saturate(previous.astype(object) + delta)


def integer_weights(power, noise, prior, subtraction=1.0, lower=1.0, upper=4.0,
                    use_prior=True, use_subtraction=True, use_snr=True):
    power, noise, prior = map(np.asarray, (power, noise, prior))
    if power.shape != noise.shape or prior.shape != power.shape or np.any(power < 0) or np.any(noise < 0):
        raise ValueError("Invalid integer frequency vectors")
    if not 0 <= subtraction <= 255 or not 0 <= lower < upper or not np.isfinite([lower, upper]).all():
        raise ValueError("Invalid integer frequency settings")
    scale = int(round_away(subtraction * 65536))
    low, high = int(round_away(lower * 65536)), int(round_away(upper * 65536))
    if high <= low:
        raise ValueError("SNR thresholds collapse after quantization")
    output = np.empty(power.shape, dtype=np.int64)
    saturated = 0
    for index in np.ndindex(power.shape):
        energy, background = int(power[index]), max(1, int(noise[index]))
        factors = []
        if use_prior:
            factors.append(int(prior[index]))
        if use_subtraction:
            remainder = max(0, energy - divide_round(scale * background, 65536))
            factors.append(min(65535, max(0, divide_round(remainder * 65536, max(1, energy)))))
        if use_snr:
            ratio = divide_round((energy * 65536 - low * background) * 65536, (high - low) * background)
            factors.append(min(65535, max(0, ratio)))
        value = 65536
        for factor in factors:
            value = divide_round(value * factor, 65536)
        output[index], clips = WEIGHT.saturate(value)
        saturated += clips
    return output, saturated


def log_codes(codes, input_format=POWER, address_bits=10, output_format=Format(24, 15)):
    table = np.log(1 + np.arange(1 << address_bits) / (1 << address_bits))
    output = np.empty(np.asarray(codes).shape)
    for index in np.ndindex(output.shape):
        value = max(1, int(codes[index]))
        exponent = value.bit_length() - 1
        address = ((value - (1 << exponent)) << address_bits) >> exponent
        output[index] = table[address] + (exponent - input_format.fraction) * np.log(2)
    return output_format.encode(output)


def write_memory(path, codes, bits):
    codes = np.asarray(codes)
    mask = (1 << bits) - 1
    text = "\n".join(format(int(code) & mask, f"0{(bits + 3) // 4}x") for code in codes.flat) + "\n"
    Path(path).write_text(text, encoding="ascii")


def golden_frame(audio, coordinates, output, config=Config(), weight=None):
    """Export bit-accurate reference stages; input is one already calibrated PCM frame."""
    audio = np.asarray(audio)
    if audio.shape != (config.frame, config.microphones):
        raise ValueError("Golden vector input must be one configured frame")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    pcm, clipped = PCM.encode(audio)
    hann, _ = HANN.encode(np.hanning(config.frame))
    windowed, count = multiply(pcm, hann[:, None], PCM, HANN, PCM)
    real, imaginary, forward = integer_fft(windowed)
    selected = pairs(config.microphones)
    cross_real = (real[:, selected[:, 0]] * real[:, selected[:, 1]]
                  + imaginary[:, selected[:, 0]] * imaginary[:, selected[:, 1]])
    cross_imaginary = (imaginary[:, selected[:, 0]] * real[:, selected[:, 1]]
                       - real[:, selected[:, 0]] * imaginary[:, selected[:, 1]])
    phat_real, phat_imaginary, saturated = integer_phat(cross_real, cross_imaginary)
    weight = np.ones(config.frame // 2 + 1) if weight is None else np.asarray(weight)
    if weight.shape != (config.frame // 2 + 1,) or np.any(weight < 0) or np.any(weight > 1):
        raise ValueError("Golden frequency weights must be a half spectrum in [0, 1]")
    weight_codes, _ = WEIGHT.encode(weight)
    full_weight = np.concatenate((weight_codes, weight_codes[-2:0:-1]))
    phat_real, real_clips = multiply(phat_real, full_weight[:, None], PHAT, WEIGHT, PHAT)
    phat_imaginary, imaginary_clips = multiply(phat_imaginary, full_weight[:, None], PHAT, WEIGHT, PHAT)
    gcc_real, gcc_imaginary, inverse = integer_fft(phat_real, phat_imaginary, inverse=True)
    angles = config.angles()
    delays = delay_codes(coordinates, angles, config)
    response, accumulator_clips = integer_srp(gcc_real.T, delays)
    power_codes = real ** 2 + imaginary ** 2
    bank_codes, _ = WEIGHT.encode(mel_bank(config))
    fused_power = np.asarray([divide_round(int(sum(row)), config.microphones) for row in power_codes], dtype=np.int64)
    mel_products = bank_codes.astype(object) @ fused_power[:config.frame // 2 + 1].astype(object)
    mel_codes, mel_clips = POWER.saturate(shift_round(mel_products, 16))
    logarithms, log_clips = log_codes(mel_codes)
    arrays = {"pcm": pcm, "hann": hann, "windowed": windowed, "fft_real": real,
              "fft_imaginary": imaginary, "power_wide": power_codes,
              "cross_real": cross_real, "cross_imaginary": cross_imaginary,
              "phat_real": phat_real, "phat_imaginary": phat_imaginary,
              "weight": weight_codes, "gcc_real": gcc_real, "gcc_imaginary": gcc_imaginary,
              "srp": response, "angles": angles, "mel_bank": bank_codes,
              "mel_energy": mel_codes, "log_mel": logarithms,
              "log_lut": Format(24, 15).encode(np.log(1 + np.arange(1024) / 1024))[0],
              "reciprocal": Reciprocal().table, **delays}
    np.savez(output / "golden.npz", **arrays)
    for name, bits in {"pcm": 16, "hann": 16, "fft_real": 24, "fft_imaginary": 24,
                       "phat_real": 16, "phat_imaginary": 16, "weight": 16,
                       "gcc_real": 24, "srp": 32, "integer_delay": 16,
                       "fraction": 16, "mel_bank": 16, "reciprocal": 32,
                       "mel_energy": 40, "log_mel": 24, "log_lut": 24}.items():
        write_memory(output / (name + ".mem"), arrays[name], bits)
    reference = np.fft.fft(audio * np.hanning(config.frame)[:, None], axis=0) / config.frame
    error = np.abs(TRANSFORM.decode(real) + 1j * TRANSFORM.decode(imaginary) - reference)
    metadata = {"config": asdict(config), "order": "C row-major; PCM/FFT sample/bin then channel; pairs i<j",
                "formats": {name: asdict(value) for name, value in {"pcm": PCM, "hann": HANN,
                            "fft_gcc": TRANSFORM, "phat": PHAT, "weight": WEIGHT, "srp": ACCUMULATOR}.items()},
                "cross_fraction": 30, "cross_signed_bits": 49, "power_unsigned_bits": 48,
                "forward": forward, "inverse": inverse, "rounding": "nearest, ties away from zero; saturation",
                "saturations": {"pcm": clipped, "window": count, "phat": saturated,
                                "weighting": real_clips + imaginary_clips, "srp": accumulator_clips,
                                "mel": mel_clips, "log": log_clips},
                "log_floor": "one POWER code (2^-30), not the floating epsilon",
                "gcc_imaginary_max_code": int(np.abs(gcc_imaginary).max()),
                "fft_max_error": float(error.max()), "estimated_angle": float(angles[np.argmax(response)]),
                "angle_validity": "argmax only; run floating validity checks separately",
                "boundary": "reference configuration, not a vendor FFT IP or board result"}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def quantize_model(model, bits=8):
    """Return dequantized per-tensor symmetric weights for accuracy comparison, not integer kernels."""
    import torch
    from copy import deepcopy
    if not 2 <= bits <= 16:
        raise ValueError("Invalid model quantization width")
    result, errors = deepcopy(model), {}
    maximum = (1 << (bits - 1)) - 1
    with torch.no_grad():
        for name, parameter in result.named_parameters():
            scale = float(parameter.abs().max()) / maximum
            if scale == 0:
                errors[name] = {"scale": 1.0, "maximum_error": 0.0}
                continue
            codes = torch.sign(parameter) * torch.floor(parameter.abs() / scale + 0.5)
            restored = torch.clamp(codes, -maximum, maximum) * scale
            errors[name] = {"scale": scale, "maximum_error": float((parameter - restored).abs().max())}
            parameter.copy_(restored)
    return result, errors
