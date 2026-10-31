import base64
from io import BytesIO
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image

from ktcat import ktcat


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


@pytest.mark.parametrize(
    ("img", "mode"),
    [
        (np.zeros((10, 12), dtype=np.uint8), "L"),
        (np.zeros((10, 12, 3), dtype=np.uint8), "RGB"),
        (np.zeros((3, 10, 12), dtype=np.uint8), "RGB"),
        (np.zeros((10, 12, 4), dtype=np.uint8), "RGBA"),
        (torch.zeros((10, 12, 3), requires_grad=True), "RGB"),
        (Image.new("RGB", (12, 10), color="red"), "RGB"),
    ],
    ids=["numpy-gray", "numpy-hwc", "numpy-chw", "numpy-rgba", "torch-hwc", "pil"],
)
def test_ktcat_outputs_png(img: Any, mode: str, capsys: pytest.CaptureFixture[str]) -> None:
    ktcat(img)

    with _image_from_output(capsys.readouterr().out) as out:
        assert out.format == "PNG"
        assert out.mode == mode
        assert out.size == (12, 10)


@pytest.mark.parametrize(
    ("img", "expected"),
    [
        (np.array([[-1.0, 0.5, 2.0]], dtype=np.float32), [[0, 127, 255]]),
        (np.array([[-1, 0, 255, 256]], dtype=np.int16), [[0, 0, 255, 255]]),
    ],
    ids=["float-scaled", "int-not-wrapped"],
)
def test_values_are_clipped(img: Any, expected: list[list[int]], capsys: pytest.CaptureFixture[str]) -> None:
    ktcat(img)

    with _image_from_output(capsys.readouterr().out) as out:
        assert np.array(out).tolist() == expected


def test_large_output_is_split_into_kitty_chunks(capsys: pytest.CaptureFixture[str]) -> None:
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(80, 80, 3), dtype=np.uint8)

    ktcat(img)

    output = capsys.readouterr().out
    assert "m=1" in output
    assert "m=0" in output
    assert _png_from_kitty_output(output).startswith(b"\x89PNG\r\n\x1a\n")
