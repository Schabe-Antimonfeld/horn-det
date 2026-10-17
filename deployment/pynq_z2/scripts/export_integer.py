"""Extract the verified integer reference into a NumPy-only board module."""

import ast
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parents[1] / 'fpga/cnn.py'
NAMES = {'round_away', 'shift_round', 'input_codes', 'softmax_table', 'probability_code',
         'integer_layer', 'fixed_forward', 'check_export', 'read_memory', 'load_fixed'}
CONSTANTS = {'LAYERS', 'WEIGHT_BASE', 'BIAS_BASE'}


class Windows(ast.NodeTransformer):
    def visit_Attribute(self, node):
        if ast.unparse(node) == 'np.lib.stride_tricks.sliding_window_view':
            return ast.copy_location(ast.Name(id='sliding_windows', ctx=ast.Load()), node)
        return self.generic_visit(node)


def generate():
    source = SOURCE.read_bytes()
    selected = []
    for node in ast.parse(source.decode('utf-8')).body:
        if isinstance(node, ast.FunctionDef) and node.name in NAMES:
            selected.append(node)
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id in CONSTANTS for target in node.targets):
            selected.append(node)
    tree = Windows().visit(ast.Module(body=selected, type_ignores=[]))
    ast.fix_missing_locations(tree)
    header = '''"""Generated integer reference; regenerate with scripts/export_integer.py."""
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

'''
    result = header + 'SOURCE_SHA256 = ' + repr(hashlib.sha256(source).hexdigest()) + '\n\n' + ast.unparse(tree) + '\n'
    ast.parse(result, feature_version=(3, 6))
    return result


if __name__ == '__main__':
    (ROOT / 'ps/integer_kernel.py').write_bytes(generate().encode('utf-8'))
