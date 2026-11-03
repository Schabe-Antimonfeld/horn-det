"""Check current entry points, documentation links and the single-version asset contract."""

import ast
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ps.configuration import load_config
from ps.evaluation import verify_assets


def audit():
    documents = list(ROOT.glob('*.md'))
    repository = ROOT.parents[1]
    if (repository / 'AGENTS.md').is_file():
        documents += list(repository.glob('*.md')) + list((repository / 'docs').glob('*.md'))
        documents += [repository / 'prototype/README.md', repository / 'fpga/README.md', repository / 'references/README.md']
    links = 0
    for path in documents:
        text = path.read_text(encoding='utf-8')
        if sum(line.startswith('```') for line in text.splitlines()) % 2:
            raise ValueError('Unclosed fence: ' + str(path))
        for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)', text):
            if target.startswith(('http:', 'https:', '#', 'mailto:')):
                continue
            if not (path.parent / target.split('#', 1)[0]).exists():
                raise ValueError('Broken local link: ' + str(path) + ' -> ' + target)
            links += 1
    for path in list((ROOT / 'ps').glob('*.py')) + list((ROOT / 'scripts').glob('*.py')) + [ROOT / 'run.py', ROOT / 'reproduce.py']:
        ast.parse(path.read_text(encoding='utf-8'), feature_version=(3, 6))
    if [path.name for path in (ROOT / 'config').glob('*.json')] != ['default.json']:
        raise ValueError('Only one current configuration may be shipped')
    config = load_config()
    if (config['detector']['threshold'], config['detector']['capture_threshold'], config['detector']['model_tag']) != (0.9, 0.9, 405187002):
        raise ValueError('Frozen operating point changed')
    if config['event']['confirm'] != 1:
        raise ValueError('Current operating point requires one-window confirmation')
    verify_assets(ROOT)
    return dict(result='passed', documents=len(documents), local_links=links, model_tag=405187002,
                scope='Local links, Python syntax and current configuration/assets; no external URL or hardware acceptance')


if __name__ == '__main__':
    print(json.dumps(audit()))
