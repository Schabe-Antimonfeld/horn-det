from pathlib import Path
import importlib.util
import json
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ps.configuration import load_config
from ps.dsp import coordinates, spatial_map
from ps.srp import GCC_SCALE, SrpDevice, decode_batch, encode_batch


def integer_output(packet):
    count = int(packet[0])
    triples = packet[1:].reshape(count, 91, 3)
    lower = triples[:, :, 0].copy().view(np.int32).astype(np.int64)
    upper = triples[:, :, 1].copy().view(np.int32).astype(np.int64)
    weight = triples[:, :, 2].astype(np.int64)
    totals = ((65536 - weight) * lower + weight * upper).sum(axis=1)
    result = np.zeros((count, 4), dtype=np.uint32)
    result[:, 0] = 0x31505253
    result[:, 1] = (totals & 0xffffffff).astype(np.uint32)
    result[:, 2] = (totals >> 32).astype(np.int32).view(np.uint32)
    return result.reshape(-1)


class Buffer(np.ndarray):
    def freebuffer(self):
        self.freed = True


class FakeDma:
    def __init__(self, mode='normal'):
        self.mode, self.reset = mode, False
        self.transfers = 0
        self.mmio = self
        self.sendchannel = SimpleNamespace(transfer=self.send, wait=lambda: None)
        self.recvchannel = SimpleNamespace(transfer=self.receive, wait=lambda: None)

    def receive(self, buffer, nbytes):
        self.output, self.length = buffer, nbytes

    def send(self, buffer, nbytes):
        self.transfers += 1
        result = integer_output(buffer[:nbytes // 4])
        self.output[:len(result)] = result
        if self.mode == 'packet':
            self.output[3] = 1

    def read(self, address):
        if address == 0:
            return 4 if self.mode == 'reset' else 0
        if address == 0x58:
            return self.length - 4 if self.mode == 'length' else self.length
        return 0x10 if self.mode == 'dma' else 0 if self.mode == 'timeout' else 2

    def write(self, address, value):
        assert (address, value) == (0, 4)
        self.reset = True


def make_device(mode='normal'):
    buffers = []
    dma = FakeDma(mode)
    def allocate(shape, dtype):
        value = np.zeros(shape, dtype=dtype).view(Buffer)
        value.freed = False
        buffers.append(value)
        return value
    return SrpDevice(SimpleNamespace(srp_dma=dma), allocate, 0.005), dma, buffers


def test_dma_batch_extremes_tail_blocks_and_lifecycle():
    device, dma, buffers = make_device()
    random = np.random.default_rng(92)
    for count in (1, 128, 76):
        lower = random.uniform(-1, 1, (91, count))
        upper = random.uniform(-1, 1, (91, count))
        fraction = random.uniform(0, 1, (91, count))
        actual = device.score(lower, upper, fraction)
        expected = ((1 - fraction) * lower + fraction * upper).sum(axis=0)
        bound = 91 * 0.5 / GCC_SCALE + np.abs(upper - lower).sum(axis=0) / 65536
        assert np.all(np.abs(actual - expected) <= bound)
    with pytest.raises(ValueError):
        device.score(lower * 100, upper, fraction)
    assert not device.faulted
    device.close()
    assert dma.reset and all(buffer.freed for buffer in buffers)
    device.close()
    with pytest.raises(RuntimeError, match='unavailable'):
        device.score(lower, upper, fraction)


@pytest.mark.parametrize('mode', ['dma', 'timeout', 'length', 'packet'])
def test_dma_failure_requires_reload_and_safe_cleanup(mode):
    device, dma, buffers = make_device(mode)
    values = np.zeros((91, 1))
    with pytest.raises((RuntimeError, TimeoutError)):
        device.score(values, values, values)
    assert device.faulted
    with pytest.raises(RuntimeError, match='reload'):
        device.score(values, values, values)
    device.close()
    assert dma.reset and all(buffer.freed for buffer in buffers)


def test_reset_timeout_preserves_dma_buffers():
    device, dma, buffers = make_device('reset')
    with pytest.raises(RuntimeError, match='remain allocated'):
        device.close()
    assert not any(buffer.freed for buffer in buffers)
    dma.mode = 'normal'
    device.close()


@pytest.mark.parametrize('problem', ['pairs', 'count', 'nan', 'fraction', 'overflow'])
def test_reject_invalid_inputs(problem):
    lower = np.zeros((91, 1))
    upper, fraction = lower.copy(), lower.copy()
    if problem == 'pairs': lower = lower[:90]
    if problem == 'count': lower = upper = fraction = np.empty((91, 0))
    if problem == 'nan': lower[0, 0] = np.nan
    if problem == 'fraction': fraction[0, 0] = 1.01
    if problem == 'overflow': lower[0, 0] = 2
    with pytest.raises(ValueError):
        encode_batch(lower, upper, fraction)


def test_output_rejects_bad_tags_and_sign_extension():
    words = integer_output(encode_batch(*[np.zeros((91, 1))] * 3))
    for position, value in ((0, 0), (2, 32768), (3, 2)):
        damaged = words.copy()
        damaged[position] = value
        with pytest.raises(RuntimeError):
            decode_batch(damaged, 1)


def test_fraction_rounded_to_one_preserves_the_upper_endpoint():
    lower, upper = np.zeros((91, 1)), np.ones((91, 1))
    packet = encode_batch(lower, upper, upper)
    np.testing.assert_array_equal(decode_batch(integer_output(packet), 1), [91])


def test_full_spatial_map_matches_numpy_and_keeps_peak():
    config = load_config()
    random = np.random.default_rng(23)
    source = random.normal(size=8192)
    vector = np.array([np.cos(np.deg2rad(30)), np.sin(np.deg2rad(30)), 0]) * np.cos(np.deg2rad(36))
    delays = -coordinates(config['audio']) @ vector * 48000 / 343
    audio = np.column_stack([np.interp(np.arange(4096) - delay + 40, np.arange(len(source)), source) for delay in delays])
    expected = spatial_map(audio, config['audio'], config['search'])
    device, _, _ = make_device()
    bounds = []
    def score(lower, upper, fraction):
        bounds.extend(91 * 0.5 / GCC_SCALE + np.abs(upper - lower).sum(axis=0) / 65536)
        return device.score(lower, upper, fraction)
    actual = spatial_map(audio, config['audio'], config['search'], SimpleNamespace(score=score))
    assert np.all(np.abs(actual[2] - expected[2]).ravel() <= np.asarray(bounds) + 1e-12)
    assert np.unravel_index(np.argmax(actual[2]), actual[2].shape) == np.unravel_index(np.argmax(expected[2]), expected[2].shape)
    device.close()


def test_batched_rtl_matches_exact_python_packets(tmp_path):
    if not shutil.which('iverilog') or not shutil.which('vvp'):
        pytest.skip('Icarus is not installed')
    random = np.random.default_rng(45)
    packets, expected, endings = [], [], []
    for count in (1, 3, 128, 2):
        lower = random.integers(-8388608, 8388608, (91, count)) / GCC_SCALE
        upper = random.integers(-8388608, 8388608, (91, count)) / GCC_SCALE
        fraction = random.integers(0, 65536, (91, count)) / 65536
        lower[0], upper[0], fraction[0] = -2, 8388607 / GCC_SCALE, 65535 / 65536
        packet = encode_batch(lower, upper, fraction)
        packets.extend(int(word) | ((1 << 32) if index == len(packet) - 1 else 0) for index, word in enumerate(packet))
        expected.extend(integer_output(packet))
        endings.extend([False] * (count * 4 - 1) + [True])
    # Bad headers, truncated batches, invalid TKEEP and missing final TLAST must fail.
    for packet in ([0 | (1 << 32)], [129 | (1 << 32)], [2, 100 | (1 << 32)],
                   [1] + [0 | (1 << 33)] + [0] * 271 + [(1 << 32)],
                   [1] + [0] * 273):
        packets.extend(packet)
        expected.extend([0x31505253, 0, 0, 1])
        endings.extend([False, False, False, True])
    input_path, output_path = tmp_path / 'input.mem', tmp_path / 'output.mem'
    input_path.write_text(''.join('{:09x}\n'.format(word) for word in packets))
    program = tmp_path / 'batch.vvp'
    subprocess.run(['iverilog', '-g2012', '-Wall', '-s', 'tb_srp_batch', '-o', str(program),
                    str(ROOT / 'rtl/srp_accumulator.v'), str(ROOT / 'rtl/srp_batch.v'),
                    str(ROOT / 'tests/tb_srp_batch.sv')], check=True, capture_output=True, text=True)
    result = subprocess.run(['vvp', str(program), '+INPUT=' + str(input_path), '+OUTPUT=' + str(output_path),
                             '+WORDS=' + str(len(packets)), '+EXPECTED=' + str(len(expected))],
                            check=True, capture_output=True, text=True, timeout=30)
    assert 'PASS BATCH SRP' in result.stdout
    received = [int(line, 16) for line in output_path.read_text().splitlines()]
    assert [word & 0xffffffff for word in received] == [int(word) for word in expected]
    assert [bool(word >> 32) for word in received] == endings


def test_live_entry_selects_srp_dma_and_closes_resources(tmp_path, monkeypatch):
    model = ROOT.parents[1] / 'prototype/outputs/horn_detector_v1/fpga_model'
    if not model.is_dir():
        pytest.skip('Local trained model absent')
    specification = importlib.util.spec_from_file_location('live_srp_service', ROOT / 'run.py')
    service = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(service)
    config = load_config()
    config['search']['backend'] = 'fpga'
    config['hardware']['overlay'] = str(tmp_path / 'test.bit')
    (tmp_path / 'test.bit').write_bytes(b'not-real-firmware')
    (tmp_path / 'test.hwh').write_text('test-only')
    config_path = tmp_path / 'config.json'
    config_path.write_text(json.dumps(config))
    dma, buffers, closed = FakeDma(), [], []
    def allocate(shape, dtype):
        buffer = np.zeros(shape, dtype=dtype).view(Buffer)
        buffer.freed = False
        buffers.append(buffer)
        return buffer
    monkeypatch.setitem(sys.modules, 'pynq', SimpleNamespace(
        Overlay=lambda path: SimpleNamespace(srp_dma=dma), allocate=allocate))
    monkeypatch.setattr(service, 'cnn_client', lambda: lambda *args: SimpleNamespace(close=lambda: closed.append('cnn')))
    monkeypatch.setattr(service, 'CaptureDevice', lambda *args: SimpleNamespace())
    snapshot = {'audio': np.zeros((48128, 14)), 'first': 0, 'end': 48128, 'epoch': 0, 'clipped': False}
    pump = SimpleNamespace(thread=SimpleNamespace(start=lambda: None), receive=lambda: snapshot,
                           close=lambda: closed.append('capture'), gaps=0, skipped=0)
    monkeypatch.setattr(service, 'CapturePump', lambda *args: pump)
    output = tmp_path / 'events'
    assert service.main(['--model', str(model), '--config', str(config_path),
                         '--output', str(output), '--limit', '1']) == 0
    event = json.loads((output / 'events.jsonl').read_text())
    assert event['srp_backend'] == 'fpga' and event['state'] == 'IDLE'
    assert dma.transfers == 44 and dma.reset and all(buffer.freed for buffer in buffers)
    assert closed == ['capture', 'cnn']
