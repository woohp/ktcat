import numpy as np
import torch
from PIL import Image

from ktcat import ktcat


def test_ktcat_pil() -> None:
    img = Image.new("RGB", (10, 10), color="red")
    ktcat(img)


def test_ktcat_numpy_hwc() -> None:
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    ktcat(img)


def test_ktcat_numpy_chw() -> None:
    img = np.zeros((3, 10, 10), dtype=np.uint8)
    ktcat(img)


def test_ktcat_torch_chw() -> None:
    img = torch.zeros((3, 10, 10))
    ktcat(img)


def test_ktcat_torch_hcw() -> None:
    img = torch.zeros((10, 10, 3))
    ktcat(img)


def test_ktcat_float() -> None:
    img = np.random.rand(10, 10, 3).astype(np.float32)
    ktcat(img)
