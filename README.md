# ktcat

Display images in a Kitty-compatible terminal.

## What It Does

- Library API: `ktcat(image)`
- CLI: `ktcat path/to/image.png [more files...]`
- Supported inputs in the library API include PIL images, NumPy arrays, and Torch tensors.

## Development

- Activate the project environment first: `workon register`
- Format: `make format`
- Lint/typecheck: `make lint`
- Run tests: `pytest tests/test_ktcat.py -q`

## Notes

- The package uses a `src` layout; `src/ktcat/__main__.py` forwards `python -m ktcat` to the same `main()` entrypoint used by the installed `ktcat` console script.
