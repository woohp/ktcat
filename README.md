# ktcat

Display images in a Kitty-compatible terminal by writing Kitty graphics escape sequences to stdout.

## What It Does

- Library API: `ktcat(image)`
- CLI: `ktcat path/to/image.png [more files...]`
- Supported inputs in the library API include PIL images, NumPy arrays, and Torch tensors.

## Development

- Create or update the project environment with `make sync`.
- Format: `make format`
- Lint/typecheck: `make lint`
- Run tests: `make test`

## Notes

- The package uses a `src` layout; `src/ktcat/__main__.py` forwards `python -m ktcat` to the same `main()` entrypoint used by the installed `ktcat` console script.
- The published package has no required runtime dependencies. NumPy, Pillow, and Torch support are optional; this repo keeps them in the `dev` dependency group for tests.
- The file-based CLI requires Pillow to open image files. NumPy-backed library inputs can be encoded without Pillow.
