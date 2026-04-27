import base64
from io import BytesIO

import numpy as np
import pytest
import torch
from PIL import Image

from ktcat import _normalize_array, ktcat


def _png_from_kitty_output(output: str) -> bytes:
    chunks = []
    for sequence in output.strip().split("\033\\"):
        if not sequence:
            continue
        header, data = sequence.split(";", 1)
        assert header.startswith("\033_G")
        chunks.append(data)

    return base64.b64decode("".join(chunks))


def _image_from_output(output: str) -> Image.Image:
    return Image.open(BytesIO(_png_from_kitty_output(output)))


def test_ktcat_numpy_hwc_outputs_png(capsys: pytest.CaptureFixture[str]) -> None:
    img = np.zeros((10, 12, 3), dtype=np.uint8)

    ktcat(img)

    with _image_from_output(capsys.readouterr().out) as out:
        assert out.format == "PNG"
        assert out.mode == "RGB"
        assert out.size == (12, 10)


def test_ktcat_numpy_chw_outputs_png(capsys: pytest.CaptureFixture[str]) -> None:
    img = np.zeros((3, 10, 12), dtype=np.uint8)

    ktcat(img)

    with _image_from_output(capsys.readouterr().out) as out:
        assert out.mode == "RGB"
        assert out.size == (12, 10)


def test_ktcat_torch_hwc_outputs_png(capsys: pytest.CaptureFixture[str]) -> None:
    img = torch.zeros((10, 12, 3))

    ktcat(img)

    with _image_from_output(capsys.readouterr().out) as out:
        assert out.mode == "RGB"
        assert out.size == (12, 10)


def test_ktcat_pil_outputs_png(capsys: pytest.CaptureFixture[str]) -> None:
    img = Image.new("RGB", (12, 10), color="red")

    ktcat(img)

    with _image_from_output(capsys.readouterr().out) as out:
        assert out.mode == "RGB"
        assert out.size == (12, 10)


def test_float_values_are_clipped_and_scaled() -> None:
    img = np.array([[-1.0, 0.5, 2.0]], dtype=np.float32)

    normalized = _normalize_array(img)

    assert normalized.dtype == np.uint8
    assert normalized.tolist() == [[0, 127, 255]]


def test_integer_values_are_clipped_not_wrapped() -> None:
    img = np.array([[-1, 0, 255, 256]], dtype=np.int16)

    normalized = _normalize_array(img)

    assert normalized.dtype == np.uint8
    assert normalized.tolist() == [[0, 0, 255, 255]]


def test_large_output_is_split_into_kitty_chunks(capsys: pytest.CaptureFixture[str]) -> None:
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(80, 80, 3), dtype=np.uint8)

    ktcat(img)

    output = capsys.readouterr().out
    assert "m=1" in output
    assert "m=0" in output
    assert _png_from_kitty_output(output).startswith(b"\x89PNG\r\n\x1a\n")
