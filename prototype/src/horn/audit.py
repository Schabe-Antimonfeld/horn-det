"""Read-only document links, source mapping, whitespace and parameter consistency checks."""

import ast
from pathlib import Path
import re
import tokenize
from urllib.parse import unquote

from .config import Config


def audit_repository(root=None):
    root = Path(__file__).resolve().parents[3] if root is None else Path(root)
    documents = [root / name for name in ("README.md", "AGENTS.md", "PROJECT_TREE.md")]
    documents += list((root / "docs").glob("*.md")) + list((root / "references").glob("*.md"))
    documents += [root / "prototype" / "README.md"]
    failures = []
    mapping = (root / "docs" / "13_文献来源映射与复现边界.md").read_text(encoding="utf-8")
    references = set(re.findall(r"## \d+\. (R\d+)\b", mapping))
    for path in documents:
        content = path.read_text(encoding="utf-8")
        if sum(line.lstrip().startswith("```") for line in content.splitlines()) % 2:
            failures.append(f"Unclosed code fence: {path}")
        if len(re.findall(r"^\s*\$\$\s*$", content, flags=re.MULTILINE)) % 2:
            failures.append(f"Unclosed display math: {path}")
        for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", content):
            if target.startswith(("http:", "https:", "#")):
                continue
            target = unquote(target.split("#", 1)[0])
            if not (path.parent / target).exists():
                failures.append(f"Broken link in {path}: {target}")
        for citation in re.findall(r"\[(R\d+)\]", content):
            if citation not in references:
                failures.append(f"Unmapped source in {path}: {citation}")
        for index, line in enumerate(content.splitlines(), 1):
            whitespace = line[len(line.rstrip()):]
            if whitespace and whitespace != "  ":
                failures.append(f"Accidental trailing whitespace: {path}:{index}")
    bibliography = (root / "references" / "references.bib").read_text(encoding="utf-8")
    if bibliography.count("{") != bibliography.count("}"):
        failures.append("Unbalanced BibTeX braces")
    config = Config()
    parameters = {"03_STFT与频域前端.md": [f"N={config.frame}", f"H={config.hop}", f"f_s={config.sample_rate}"],
                  "02_阵列几何与远场声学模型.md": [f"r={config.diameter / 2:.2f}", f"M={config.microphones}"],
                  "07_鸣笛检测_LogMel_TinyCNN.md": [f"B = {config.mel_bands}", f"context = {config.context}"]}
    for name, snippets in parameters.items():
        content = (root / "docs" / name).read_text(encoding="utf-8")
        for snippet in snippets:
            if snippet not in content:
                failures.append(f"Document/default mismatch in {name}: {snippet}")
    sources = list((root / "prototype" / "src").rglob("*.py")) + list((root / "prototype" / "tests").glob("*.py"))
    sources += [root / "prototype" / "main.py"]
    for path in sources:
        content = path.read_text(encoding="utf-8")
        tree = ast.parse(content, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)):
                docstring = ast.get_docstring(node)
                if docstring and any("\u4e00" <= letter <= "\u9fff" for letter in docstring):
                    failures.append(f"Non-English docstring: {path}:{getattr(node, 'lineno', 1)}")
        with path.open("rb") as stream:
            for token in tokenize.tokenize(stream.readline):
                if token.type == tokenize.COMMENT and any("\u4e00" <= letter <= "\u9fff" for letter in token.string):
                    failures.append(f"Non-English comment: {path}:{token.start[0]}")
        for index, line in enumerate(content.splitlines(), 1):
            if line != line.rstrip():
                failures.append(f"Code trailing whitespace: {path}:{index}")
        if any(0x1F300 <= ord(letter) <= 0x1FAFF for letter in content):
            failures.append(f"Emoji in source: {path}")
    if failures:
        raise ValueError("\n".join(failures))
    return {"markdown_files": len(documents), "python_files": len(sources), "mapped_sources": len(references),
            "result": "passed", "scope": "local links, fences, source IDs, documented defaults and source style; no external URL claim"}
