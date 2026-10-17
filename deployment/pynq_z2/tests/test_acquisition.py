from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ps.acquisition import CaptureDevice, PacketDecoder, WindowBuffer
from ps.configuration import load_config


def packet(first, dropped=0):
    words = np.zeros(8 + 512 * 14, dtype=np.uint32)
    words[:8] = [0x314d4450, 0x0001080e, first & 0xffffffff, first >> 32, dropped, 512, 48000, 0]
    words[8:] = np.arange(512 * 14, dtype=np.int32).view(np.uint32)
    return words


def test_signed_packets_and_discontinuities():
    decoder = PacketDecoder()
    words = packet(0)
    words[8] = 0xffffffff
    assert decoder.decode(words)['audio'][0, 0] == -1
    assert not decoder.decode(packet(512))['gap']
    assert decoder.decode(packet(1536, 1))['gap']
    assert not decoder.decode(packet(2048, 1))['gap']
    assert decoder.decode(packet(2560, 2))['gap']


def test_payload_must_be_sign_extended():
    words = packet(0)
    words[8] = 65535
    with pytest.raises(ValueError, match='sign-extended'):
        PacketDecoder().decode(words)


def test_window_has_exact_packet_order_and_gap_resets_history():
    settings = load_config()['audio']
    history, decoder = WindowBuffer(settings), PacketDecoder()
    first = None
    for index in range(94):
        words = packet(index * 512)
        words[8:] = index
        first = history.feed(decoder.decode(words))
    assert first['first'] == 0 and first['end'] == 48128
    np.testing.assert_array_equal(first['audio'][:, 0], np.repeat(np.arange(94), 512))
    for index in range(94, 141):
        next_window = history.feed(decoder.decode(packet(index * 512)))
    assert next_window['first'] == 47 * 512
    assert history.feed(decoder.decode(packet(150 * 512, 1))) is None
    assert history.epoch == 1 and history.filled == 512


def test_capture_resets_before_releasing_dma_memory():
    actions = []
    class Buffer(np.ndarray):
        def freebuffer(self):
            actions.append('free')
    def allocate(shape, dtype):
        return np.zeros(shape, dtype=dtype).view(Buffer)
    class Channel:
        def transfer(self, buffer):
            buffer[:] = packet(0)
            actions.append('transfer')
        def wait(self):
            pass
    class Registers:
        def read(self, address):
            return 2 if address in (0x34, 0x08) else 0
        def write(self, address, value):
            actions.append((address, value))
    registers = Registers()
    overlay = SimpleNamespace(audio_dma=SimpleNamespace(recvchannel=Channel(), mmio=registers), capture_control=registers)
    device = CaptureDevice(overlay, allocate, load_config()['audio'])
    assert device.receive()['first'] == 0
    device.close()
    assert actions.index('transfer') < actions.index((0, 1))
    assert actions.index((0x30, 4)) < actions.index('free')
