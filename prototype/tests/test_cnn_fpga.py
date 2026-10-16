"""Independent integer-kernel, export, DMA contract and RTL comparison checks."""

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]


def module(name):
    specification = importlib.util.spec_from_file_location(name, ROOT / "fpga" / f"{name}.py")
    result = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(result)
    return result


cnn = module("cnn")
host = module("cnn_host")


def test_integer_rounding_and_probability():
    assert cnn.round_away([-.5, .5, -1.5, 1.5]).tolist() == [-1, 1, -2, 2]
    assert cnn.shift_round([-128, -127, 127, 128], 8).tolist() == [-1, 0, 0, 1]
    with pytest.raises(ValueError):
        cnn.round_away([1e100])
    with pytest.raises(ValueError):
        cnn.input_codes(np.zeros((40, 32)), 10)
    values = np.zeros((40, 24))
    values[0, :2] = [.5, -.5]
    assert host.encode_features(values, 0)[:2].view(np.int32).tolist() == [1, -1]
    for fraction in (0, 6, 10, 15):
        for difference in (-16, -8, -1, 0, 1, 8, 16):
            codes = [0, int(difference * (1 << fraction))]
            actual = cnn.probability_code(codes, fraction) / 65536
            assert abs(actual - 1 / (1 + np.exp(-difference))) < .0021
    assert cnn.probability_code([-32768, 32767], 15) > 32768


def test_integer_padding_depthwise_pointwise_and_saturation():
    values = np.zeros((2, 4, 5), dtype=np.int16)
    values[0, 0, 0], values[1, -1, -1] = 7, -3
    kernels = np.arange(36).reshape(2, 2, 3, 3) - 17
    expected = np.zeros((2, 4, 5), dtype=np.int64)
    for outgoing in range(2):
        for row in range(4):
            for column in range(5):
                for incoming in range(2):
                    for vertical in range(3):
                        for horizontal in range(3):
                            y, x = row + vertical - 1, column + horizontal - 1
                            if 0 <= y < 4 and 0 <= x < 5:
                                expected[outgoing, row, column] += kernels[outgoing, incoming, vertical, horizontal] * values[incoming, y, x]
    actual, clips = cnn.integer_layer(values, kernels, np.zeros(2, dtype=int), "conv", 0, False)
    assert np.array_equal(actual, expected) and clips == 0
    depth = np.zeros((2, 1, 3, 3), dtype=np.int8)
    depth[0, 0, 1, 1], depth[1, 0, 1, 1] = 2, -2
    actual, _ = cnn.integer_layer(values, depth, np.zeros(2, dtype=int), "depth", 0, True)
    assert actual[0, 0, 0] == 14 and actual[1, -1, -1] == 6
    point = np.eye(2, dtype=np.int8)[:, :, None, None]
    actual, _ = cnn.integer_layer(values, point, np.zeros(2, dtype=int), "point", 0, False)
    assert np.array_equal(actual, values)
    actual, clips = cnn.integer_layer(np.array([32767, -32768]), np.eye(2, dtype=int)*127,
                                      np.zeros(2, dtype=int), "linear", 0, False)
    assert actual.tolist() == [32767, -32768] and clips == 2


def exported(directory, vectors=2):
    torch = pytest.importorskip("torch")
    torch.manual_seed(31)
    model = cnn.tiny_model()
    features = np.random.default_rng(13).normal(-6, 3, (4, 40, 24)).astype(np.float32)
    metadata = cnn.export_model(model, features, directory, vectors, {"evidence": "test-only untrained model"})
    return model, features, metadata


def test_export_roundtrip_and_file_integrity(tmp_path):
    model, features, metadata = exported(tmp_path)
    calibrated = cnn.calibrate(model, features)
    restored = cnn.load_fixed(tmp_path)
    for first, second in zip(calibrated["layers"], restored["layers"]):
        assert np.array_equal(first["weights"], second["weights"])
        assert np.array_equal(first["bias"], second["bias"])
        assert first["shift"] == second["shift"]
    codes, clipped = cnn.input_codes(features[0], restored["input_fraction"])
    result = cnn.fixed_forward(restored, codes)
    expected = cnn.read_memory(tmp_path / "expected.mem", 32)
    assert int(expected[5]) == result["probability"] | result["flags"] << 16
    assert metadata["model_tag"] == restored["model_tag"] and clipped == 0
    path = tmp_path / "weights0.mem"
    path.write_text("01\n" + "00\n"*255, encoding="ascii")
    with pytest.raises(ValueError, match="checksum"):
        cnn.check_export(tmp_path)


@pytest.mark.parametrize("failure", [None, "tag", "sequence", "protocol", "dma", "timeout"])
def test_host_dma_order_errors_and_cleanup(failure):
    allocated = []
    transfers = []
    class Buffer(np.ndarray):
        def freebuffer(self):
            self.released = True
    def allocate(shape, dtype):
        value = np.zeros(shape, dtype=dtype).view(Buffer)
        value.released = False
        allocated.append(value)
        return value
    metadata = {"format": 1, "architecture": "TinyCNN-1834", "input_shape": [40, 24],
                "input_fraction": 10, "logit_fraction": 15, "model_tag": 731}
    class Channel:
        def __init__(self, direction):
            self.direction = direction
        def transfer(self, buffer):
            transfers.append(self.direction)
            if self.direction == "send":
                assert transfers[-2:] == ["receive", "send"]
                sequence = transfers.count("send") - 1
                allocated[1][:] = [0x314e4e43, 0x00010008, sequence + (failure == "sequence"),
                                   0xffffffff, 2, 0x00020000 if failure == "protocol" else 32768,
                                   445512, 732 if failure == "tag" else 731]
        def wait(self):
            pass
    class Registers:
        def read(self, address):
            if address == 0:
                return 0
            return 0x10 if failure == "dma" else 0 if failure == "timeout" else 2
        def write(self, address, value):
            assert address == 0 and value == 4
    dma = SimpleNamespace(mmio=Registers(), recvchannel=Channel("receive"), sendchannel=Channel("send"))
    device = host.CnnDevice(SimpleNamespace(cnn_dma=dma), metadata, allocate, timeout=.001)
    try:
        if failure:
            with pytest.raises((RuntimeError, TimeoutError)):
                device.infer(np.zeros((40, 24)), frame=17)
            assert device.faulted
        else:
            result = device.infer(np.zeros((40, 24)), frame=17)
            assert result["frame"] == 17 and result["horn_probability"] == .5
            assert result["logit_codes"] == [-1, 2]
            assert device.probability(np.zeros((2, 1, 40, 24))).tolist() == [.5, .5]
            assert device.sequence == 3
            with pytest.raises(ValueError, match="shape"):
                device.probability(np.zeros((1, 40, 24)))
    finally:
        device.close()
    assert all(buffer.released for buffer in allocated)
    with pytest.raises(RuntimeError):
        device.infer(np.zeros((40, 24)))


def test_rtl_integer_stress(tmp_path):
    if not shutil.which("iverilog") or not shutil.which("vvp"):
        pytest.skip("Icarus Verilog is not available")
    _, _, metadata = exported(tmp_path)
    fixed = cnn.load_fixed(tmp_path)
    codes = np.random.default_rng(23).integers(-32768, 32768, (2, 40, 24), dtype=np.int16)
    codes[0] = 0
    codes[0, 0, 0], codes[0, -1, -1] = 32767, -32768
    results = []
    for index, values in enumerate(codes):
        output = cnn.fixed_forward(fixed, values)
        logits = [int(value) for value in output["logits"]]
        results.extend([0x314e4e43, 0x00010008, index, *logits,
                        output["probability"] | output["flags"] << 16, 0, metadata["model_tag"]])
    cnn.write_memory(tmp_path / "inputs.mem", codes.astype(np.int32), 32)
    cnn.write_memory(tmp_path / "expected.mem", results, 32)
    for name in ("inputs.mem", "expected.mem"):
        metadata["sha256"][name] = hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
    (tmp_path / "manifest.json").write_text(json.dumps(metadata), encoding="utf-8")
    assert "PASS CNN" in cnn.verify_rtl(tmp_path)
