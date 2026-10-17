from pathlib import Path
import shutil
import subprocess

import pytest


def test_optional_srp_kernel(tmp_path):
    if not shutil.which('iverilog') or not shutil.which('vvp'):
        pytest.skip('Icarus is not installed')
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / 'srp.vvp'
    subprocess.run(['iverilog', '-g2012', '-Wall', '-s', 'tb_srp', '-o', str(output),
                    str(root / 'rtl/srp_accumulator.v'), str(root / 'tests/tb_srp.sv')], check=True)
    result = subprocess.run(['vvp', str(output)], check=True, capture_output=True, text=True)
    assert 'PASS SRP' in result.stdout
