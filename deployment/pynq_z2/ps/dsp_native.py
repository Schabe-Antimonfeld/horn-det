"""Checked double-precision beam, sparse Mel and temporal statistics kernels."""

import ctypes
from pathlib import Path
import sys

import numpy as np


class NativeDSP:
    def __init__(self, path=None):
        path = Path(path) if path else Path(__file__).with_name('dsp_v3.dll' if sys.platform == 'win32' else 'libdsp_v3.so')
        if not path.is_file():
            raise ValueError('DSP library missing; run python scripts/build.py on the target')
        self.library = ctypes.CDLL(str(path.resolve()))
        self.library.dsp_abi.restype = ctypes.c_int
        if self.library.dsp_abi() != 3:
            raise ValueError('DSP ABI mismatch')
        pointer = np.ctypeslib.ndpointer(dtype=np.float64, flags=('C_CONTIGUOUS', 'ALIGNED'))
        integers = np.ctypeslib.ndpointer(dtype=np.int32, flags=('C_CONTIGUOUS', 'ALIGNED'))
        self.library.dsp_beam.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t] + [ctypes.c_ssize_t] * 5 + [pointer]
        self.library.dsp_mel.argtypes = [pointer, ctypes.c_size_t, pointer, integers, integers, pointer]
        self.library.dsp_temporal.argtypes = [pointer, pointer]
        self.library.dsp_prepare.argtypes = [pointer, ctypes.c_int, ctypes.c_int, pointer,
                                             ctypes.c_int, ctypes.c_int, pointer]
        for name in ('dsp_beam', 'dsp_mel', 'dsp_temporal', 'dsp_prepare'):
            getattr(self.library, name).restype = ctypes.c_int
        self.banks = {}
        self.workspace = None

    def prepare(self, audio, window, fold, first=0):
        """Preserve the original fold order while reading adjacent PCM channels together."""
        values = np.require(audio, dtype=np.float64, requirements=['C', 'A'])
        window = np.require(window, dtype=np.float64, requirements=['C', 'A'])
        if (values.ndim != 2 or not 1024 <= len(values) <= 2147483647 or not 1 <= values.shape[1] <= 64
                or window.shape != (1024,) or type(fold) is not int or fold not in (1, 2, 4, 8)):
            raise ValueError('Invalid native folded FFT input')
        count = 1 + (len(values) - 1024) // 512
        if type(first) is not int or not 0 <= first < count:
            raise ValueError('Invalid native FFT frame offset')
        shape = (count - first, values.shape[1], 1024 // fold)
        if self.workspace is None or self.workspace.shape != shape:
            self.workspace = np.empty(shape)
        if self.library.dsp_prepare(values, len(values), values.shape[1], window, fold, first, self.workspace):
            raise ValueError('Native FFT input must be finite')
        return self.workspace

    def beam(self, spectrum, weights):
        values = np.require(spectrum, dtype=np.complex128, requirements=['A'])
        weights = np.require(weights, dtype=np.complex128, requirements=['A'])
        if values.ndim != 3 or weights.shape != values.shape[1:] or not all(values.shape):
            raise ValueError('Native beam dimensions disagree')
        output = np.empty((len(values), values.shape[2]), dtype=np.complex128)
        if self.library.dsp_beam(values.ctypes.data, weights.ctypes.data, *values.shape,
                                 *(stride // 8 for stride in values.strides),
                                 *(stride // 8 for stride in weights.strides), output.view(np.float64)):
            raise ValueError('Native beam failed')
        if not np.isfinite(output).all():
            raise ValueError('Native beam output is nonfinite')
        return output

    def mel(self, spectrum, bank):
        values = np.require(spectrum, dtype=np.complex128, requirements=['C', 'A'])
        if values.ndim != 2 or values.shape[1] != 513 or not len(values):
            raise ValueError('Native Mel needs nonempty 513-bin spectra')
        identity = id(bank)
        cached = self.banks.get(identity)
        immutable = isinstance(bank, np.ndarray) and not bank.flags.writeable
        if cached is None or cached[0] is not bank or not immutable:
            weights = np.require(bank, dtype=np.float64, requirements=['C', 'A'])
            if weights.shape != (40, 513) or not np.isfinite(weights).all() or np.any(weights < 0):
                raise ValueError('Native Mel bank must be finite nonnegative 40x513')
            bounds = [np.flatnonzero(row) for row in weights]
            starts = np.asarray([int(row[0]) if len(row) else 0 for row in bounds], dtype=np.int32)
            ends = np.asarray([int(row[-1]) + 1 if len(row) else 0 for row in bounds], dtype=np.int32)
            cached = bank, weights, starts, ends
            if len(self.banks) >= 4:
                self.banks.clear()
            self.banks[identity] = cached
        output = np.empty((len(values), 40))
        if self.library.dsp_mel(values.view(np.float64), len(values), cached[1], cached[2], cached[3], output):
            raise ValueError('Native Mel failed')
        if not np.isfinite(output).all():
            raise ValueError('Native Mel output is nonfinite')
        return output

    def temporal(self, contexts):
        values = np.require(contexts, dtype=np.float64, requirements=['C', 'A'])
        if values.shape != (7, 40, 24) or not np.isfinite(values).all():
            raise ValueError('Native temporal input must be finite 7x40x24')
        output = np.empty(640)
        if self.library.dsp_temporal(values, output) or not np.isfinite(output).all():
            raise ValueError('Native temporal statistics failed')
        return output
