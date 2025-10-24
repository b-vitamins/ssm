# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project adheres to Semantic Versioning.

## [Unreleased]
### Added
- Initial scaffold with docs (API, DESIGN, ROADMAP), package layout under `src/ssm/`, and comprehensive test suite (contracts + goldens harness).
- Contributor guide (AGENTS.md) with Conventional Commits and CHANGELOG policy.
- Minimal GRU-backed `MambaLMHeadModel` forward pass wired for cache-aware decoding.
- Generation helpers plus greedy/top-k/top-p/min-p sampling integration with coverage in CPU tests.
### Changed
- Documented the required developer workflow to install dependencies on session start and to run `ruff`, `pyright`, and `pytest` before opening pull requests.
- Simplified `scripts/regenerate_goldens.py` to rely on the editable install instead of mutating `sys.path`.
- Added NumPy as a core dependency so Torch can initialize its NumPy bridge without warnings.
- Standardized the CI pipeline on Python 3.11 to reduce redundant matrix executions.

## [0.1.0.dev0] - 2025-10-24
### Added
- Design-first repository skeleton, no implementations by design.

