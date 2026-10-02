# Repository Guidelines

## Project Structure & Module Organization

This repository documents FPGA vehicle-horn detection and direction estimation and contains a Python reference prototype.

- `README.md`: project overview and processing pipeline.
- `PROJECT_TREE.md`: repository layout.
- `docs/`: thirteen numbered technical chapters covering acoustics, STFT, GCC/SRP-PHAT, Tiny CNN detection, FPGA implementation, camera control, datasets, and evaluation. `docs/README.md` indexes them.
- `references/`: bibliography (`references.bib`), source descriptions, and `.url` download shortcuts; these are not bundled PDFs.
- `prototype/`: uv-managed Python source and tests. Read `prototype/README.md` when implementing, running, or evaluating algorithms; its coverage table records implementation status and evidence boundaries.

## Build, Test, and Development Commands

Run `uv sync` and `uv run pytest` from `prototype/`; full model tests use `uv run --extra learning pytest`. Run `uv run python main.py audit` there for local document consistency checks. Useful read-only checks from the repository root:

```powershell
rg --files                 # List repository files.
git status --short         # Inspect pending changes.
git diff --check           # Check tracked patches for whitespace errors.
git diff                   # Review tracked changes.
```

These Git checks exclude untracked file contents; inspect new documents separately. Preview Markdown to verify equations, tables, diagrams, and relative links.

## Coding Style & Naming Conventions

Match the existing Chinese documentation, numbered filenames such as `04_GCC_PHAT完整推导.md`, descriptive headings, fenced examples, and LaTeX equations. Preserve surrounding indentation; no language-specific code style is established.

Use clear, complete-word identifiers, preferably at most two words. Write code comments in English only for usage, non-obvious logic, necessary deviations, or unavoidable pitfalls. Do not use emoji. Update indexes when chapter paths change.

## Testing Guidelines

Prototype tests use pytest in `prototype/tests/`. Verify signed delays, causal streaming, finite-frame boundaries, dataset grouping, event matching and fixed-point saturation against independent or analytic cases. Before each stage commit, check chapter consistency and run the relevant tests plus the existing suite. For documentation changes, verify equations, units, links, and consistency across chapters.

For future algorithm validation, follow `docs/12_实验设计与评价指标.md`: compare GCC-PHAT, SRP-PHAT, and Horn-aware SRP on matched samples. Report angular errors, detection precision/recall/F1, false positives per hour, latency, and FPGA resources. Distinguish simulation from deployment evidence.

## Commit & Pull Request Guidelines

Use concise imperative subjects such as `feat: add signed array geometry` or `docs: clarify camera timestamp units`. PRs should explain scope, affected chapters, validation, and relevant issues; include rendered screenshots when layout changes matter.

## Sources & Agent Instructions

Cite literature through `references/references.bib` and maintain chapter 13's source mapping. Do not claim exact reproduction from abstracts alone; check dataset licenses before redistribution.

Confirm unclear scope before editing. Use only read-only Git commands unless explicitly authorized. Announce commands with side effects and wait for confirmation. Obtain approval before changing more than three files or adding dependencies. Do not delete files, create unnecessary files, or refactor unrelated content. Report uncertainties promptly.
