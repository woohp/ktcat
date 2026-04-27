# AGENTS

- Trust `pyproject.toml`, `Makefile`, and `src/ktcat/__init__.py` as the main executable sources of truth.
- Python target is `3.12` (`.python-version`, `pyproject.toml`, mypy, and Ruff).
- This is a `src`-layout package. The code lives in `src/ktcat`; the library API (`ktcat(...)`) and shared CLI entrypoint (`main()`) are in `src/ktcat/__init__.py`, and `src/ktcat/__main__.py` enables `python -m ktcat`.
- The installed CLI is the console script `ktcat = "ktcat:main"`. In an installed environment, `ktcat ...` and `python -m ktcat ...` should behave the same.

## Commands

- Use `uv` for environment management, locking, and command execution. Do not rely on `workon register` or direct global tool invocations.
- If another virtualenv is active in the terminal, deactivate it before working in this repo. Repo commands are written to ignore leaked `VIRTUAL_ENV`, but the intended environment is the project `.venv`.
- Sync the project environment with `make sync` (wraps `uv sync --dev`).
- Format edited Python files with `make format`. This runs `uv run ruff check --fix-only --ignore F401 src tests` and then `uv run ruff format src tests`.
- `make fformat` is the stricter formatter pass; unlike `make format`, it does not ignore `F401` during the fix step.
- Run lint/typecheck with `make lint`. It runs `uv run ruff check src tests` and then `uv run mypy src tests`.
- Use `make test` for the default test run.
- For focused checks, prefer `uv run pytest tests/test_ktcat.py -q` or `uv run pytest tests/test_ktcat.py -k <name> -q`.
- For CLI checks, use `uv run ktcat --help` and `uv run python -m ktcat --help`.

## Testing Quirks

- `uv sync --dev` is required for a self-contained test environment because tests exercise optional NumPy, Pillow, and Torch support.
- Tests are smoke tests in `tests/test_ktcat.py`; they call `ktcat(...)` with PIL, NumPy, and Torch inputs but do not assert on terminal output.

## Code Shape

- The core behavior is simple and centralized: `ktcat(image)` converts supported inputs to PNG bytes and writes Kitty graphics escape sequences to stdout in 4096-byte chunks.
- Runtime deps are intentionally optional. `ktcat(...)` can encode NumPy-backed inputs without Pillow, while the file-based CLI still requires Pillow to open image files.
- `_to_png_bytes(...)` and `_to_array(...)` are the main places to change supported input types or shape handling. Keep test updates in `tests/test_ktcat.py` aligned with any changes there.
