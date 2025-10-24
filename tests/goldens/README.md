Golden Tests
============

This directory contains a harness and static data for golden tests. The goal is to
compare operation outputs against known-good results without requiring the reference
library at test runtime.

Guidelines:
- Goldens are small tensors serialized to JSON for portability. They should be
  generated with a deterministic seed and the authoritative implementation.
- Use `scripts/regenerate_goldens.py` to rebuild goldens when updating the reference.
- Running golden tests is opt-in via `pytest --run-goldens`.

Note: Placeholders are included for initial scaffolding. Replace with real values
after implementing the Python reference or when you have an authoritative source.

