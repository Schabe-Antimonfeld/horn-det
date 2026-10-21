"""Generate Python 3.6-compatible geometry from the tested PC source."""

import ast
import copy
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parents[1] / 'prototype/src/horn/surround.py'
NAMES = {'wrap_angle', 'Lens', 'CameraModel', 'Rig', 'route_cameras', 'Detection', 'Association',
         'associate', 'direction_vectors', 'Bearing', 'AcousticEvidence', 'acoustic_support'}


class Compatibility(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        node.returns = None
        for value in node.args.args + node.args.kwonlyargs:
            value.annotation = None
        return self.generic_visit(node)

    def visit_ClassDef(self, node):
        fields = [value for value in node.body if isinstance(value, ast.AnnAssign)]
        node.decorator_list = []
        if fields:
            arguments = [ast.arg(arg='self')] + [ast.arg(arg=value.target.id) for value in fields]
            defaults = [copy.deepcopy(value.value) for value in fields if value.value is not None]
            statements = [ast.Assign(targets=[ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()),
                                                           attr=value.target.id, ctx=ast.Store())],
                                      value=ast.Name(id=value.target.id, ctx=ast.Load())) for value in fields]
            if any(isinstance(value, ast.FunctionDef) and value.name == '__post_init__' for value in node.body):
                statements.append(ast.Expr(value=ast.Call(func=ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()),
                                                                            attr='__post_init__', ctx=ast.Load()), args=[], keywords=[])))
            constructor = ast.FunctionDef(name='__init__',
                                          args=ast.arguments(posonlyargs=[], args=arguments, kwonlyargs=[],
                                                             kw_defaults=[], defaults=defaults),
                                          body=statements, decorator_list=[])
            node.body = [value for value in node.body if not isinstance(value, ast.AnnAssign)] + [constructor]
        return self.generic_visit(node)


def generate():
    source = SOURCE.read_bytes()
    tree = ast.parse(source.decode('utf-8'))
    selected = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in NAMES]
    module = ast.Module(body=selected, type_ignores=[])
    module = Compatibility().visit(module)
    ast.fix_missing_locations(module)
    result = '"""Generated geometry; regenerate with scripts/export_geometry.py."""\n\nimport numpy as np\n\n'
    result += 'SOURCE_SHA256 = ' + repr(hashlib.sha256(source).hexdigest()) + '\n\n'
    result += ast.unparse(module) + '\n'
    ast.parse(result, feature_version=(3, 6))
    return result


if __name__ == '__main__':
    (ROOT / 'ps/geometry.py').write_bytes(generate().encode('utf-8'))
