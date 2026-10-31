# AGENTS

- Use `uv` through the Makefile targets (`make sync`, `make format`, `make lint`, `make test`). Recipes unset `VIRTUAL_ENV`, so a leaked active virtualenv won't hijack them, but deactivate it anyway.
- Run `make sync` before testing: the tests need NumPy, Pillow, and Torch from the `dev` dependency group.
- Runtime dependencies are intentionally optional; `pyproject.toml` declares none. Never import NumPy, Pillow, or Torch at module level. Import them lazily, and detect their types with `_isinstance`, which checks `sys.modules` without importing.
- Without Pillow, `ktcat(...)` still encodes NumPy-backed inputs by writing PNG bytes by hand. Only the file-based CLI requires Pillow.
