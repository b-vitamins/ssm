# Repository Guidelines

## Project Structure & Module Organization
- Source: `src/ssm/`
  - `modules/` (Mamba1, Mamba2, MHA, Block, GatedMLP stubs)
  - `models/` (LM wrapper and config stubs)
  - `ops/` (`python/` reference signatures; `cpu/` and `cuda/` kernel entry stubs)
  - `utils/` (dispatch, generation, weights stubs)
- Tests: `tests/` (API/contract tests, `goldens/` opt‑in tests)
- Docs: `docs/` (API, DESIGN, ROADMAP)
- Scripts: `scripts/` (e.g., `regenerate_goldens.py`)

## Build, Test, and Development Commands
- Install dependencies immediately after starting a new container or shell session: `python -m pip install -e .[dev]`
- Run tests: `pytest -q`
- Run golden tests: `pytest -q --run-goldens`
- Regenerate placeholder goldens: `python scripts/regenerate_goldens.py`
- Before sending a PR, you **must** run the full quality gate locally—no exceptions.
  Run `ruff check .`, `pyright`, and `pytest -q` before opening or updating any pull request, and include the command outputs in the PR discussion as evidence.

### Guix Environment (optional)
- Enter dev shell: `guix shell -m manifest.scm --`
- Then install and test inside the shell:
  - `python -m pip install -e .`
  - `pytest -q`

## Coding Style & Naming Conventions
- Python: PEP 8/PEP 257 with Google‑style docstrings (include “Args:” and “Returns:”).
- Docstrings must follow idiomatic Google style with explicit sections (for example, "Args:", "Returns:", "Raises:") and type hints where practical.
- Naming: packages/modules `snake_case`, classes `CamelCase`, functions/args `snake_case`.
- Stubs: unimplemented public functions/classes must raise `NotImplementedError` and document contracts.
- Linters/formatters: none enforced yet; recommend `ruff` + `black` locally.

## Testing Guidelines
- Framework: `pytest`.
- Test layout: filename `tests/test_*.py`; golden tests in `tests/goldens/` (skipped by default).
- Add tests for: API shape/contract, error paths, and later correctness/perf once implementations land.
- Target: keep high coverage for ops and modules as they are implemented.

## Commit & Pull Request Guidelines
- Commits: follow Conventional Commits (e.g., `feat:`, `fix:`, `docs:`, `test:`). Use imperative mood and a concise subject.
  - Examples: `feat(mamba2): add varlen seq_idx plumbing`, `fix(ops): handle empty chunk edge case`.
- CHANGELOG: maintain `CHANGELOG.md` in the repository root. Update it for user‑visible changes per PR or during release.
  - Format: Keep a Changelog style with versions, dates, and categorized entries (Added/Changed/Fixed/Removed).
  - Tooling (optional): `git-cliff` or `conventional-changelog` can help generate entries from commit history.
- PRs: clear description, limited scope, link issues, include tests/docs, and call out API changes or migration notes. Add example commands if relevant.

## Architecture Overview (for contributors)
- Core abstraction: ops API (Python signatures) with optional backends (`python` reference, `cpu` C++, `cuda` CUDA). Runtime dispatch selects best available.
- Modules build on ops; models compose modules; utils provide decoding and backend detection.
- Keep APIs stable; prefer adding backends behind existing surfaces over changing signatures.
