import ast
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import pytest

from scripts.package import package, verify
from ps.evaluation import compare, outcomes, verify_assets

ROOT = Path(__file__).resolve().parents[1]


def test_bundle_is_deterministic_complete_and_independent_of_checkout(tmp_path):
    first, second = tmp_path/'first.zip', tmp_path/'second.zip'
    manifest = package(first)
    package(second)
    assert first.read_bytes() == second.read_bytes()
    bundle = tmp_path/'bundle'
    with zipfile.ZipFile(first) as archive:
        names = archive.namelist()
        assert not any(name.endswith(('.dll','.so','.bit','.hwh')) or 'prototype/' in name for name in names)
        assert [name for name in names if name.startswith('config/')] == ['config/default.json']
        assert 'model/weights.npz' in names and 'evaluation/test-original.npz' in names
        assert {'frontend/index.html', 'frontend/app.js', 'frontend/README.md', 'web.py', 'ps/dashboard.py'} <= set(names)
        archive.extractall(bundle)
    assert verify(bundle) == manifest
    completed = subprocess.run([sys.executable, '-I', '-B', str(bundle/'reproduce.py'), '--output', str(tmp_path/'reproduced'),
                                '--libraries', str(ROOT/'ps')], cwd=tmp_path, capture_output=True, text=True, timeout=120)
    assert completed.returncode == 0, completed.stderr
    report = json.loads((tmp_path/'reproduced/report.json').read_text())
    assert report['evaluation']['scenes'] == 672 and report['result'] == 'passed'
    with (bundle/'model/weights.npz').open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='integrity'):
        verify(bundle)


def test_entry_points_are_python36_compatible_and_exclude_training_imports():
    paths = list((ROOT/'ps').glob('*.py')) + [ROOT/'run.py', ROOT/'reproduce.py'] + list((ROOT/'scripts').glob('*.py'))
    for path in paths:
        tree = ast.parse(path.read_text(encoding='utf-8'), feature_version=(3,6))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(value.name.split('.')[0] not in {'torch','scipy','horn'} for value in node.names)
            if isinstance(node, ast.ImportFrom):
                assert (node.module or '').split('.')[0] not in {'torch','scipy','horn'}


def test_metrics_count_wrong_direction_and_ambiguous_negative_without_hiding_them():
    cases = [dict(label=1, angle=0, bearings=[dict(azimuth=0),dict(azimuth=90)]),
             dict(label=1, angle=0, bearings=[dict(azimuth=90),dict(azimuth=0)]),
             dict(label=0, angle=0, bearings=[dict(azimuth=0),dict(azimuth=90)])]
    value = outcomes(cases, [[0.99,0.1],[0.99,0.1],[0.96,0.96]],0.95,0)
    assert value['tp'] == 1 and value['wrong_direction'] == 1
    assert value['fp'] == 2 and value['fn'] == 1 and value['ambiguous_negative'] == 1
    assert value['detection_recall'] == 1
    with pytest.raises(ValueError):
        compare([dict(state='CONFIRMED')],[dict(state='IDLE')])


def test_asset_manifest_cannot_escape_package(tmp_path):
    value = dict(format=1,model_tag=405187002,files={'../outside':'ignored'})
    (tmp_path/'assets.json').write_text(json.dumps(value))
    with pytest.raises(ValueError,match='escapes'):
        verify_assets(tmp_path)
