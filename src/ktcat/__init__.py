import base64
import sys
from io import BytesIO
from typing import Any

from PIL import Image


def ktcat(image: Any) -> None:
    """
    Displays an image in the Kitty terminal.

    Args:
        image: A PIL Image, a Numpy array, or a Torch tensor.
               Supports shapes [H, W], [H, W, C], and [C, H, W].
               If values are floats, they are assumed to be in [0, 1] and are scaled to [0, 255].
    """
    # Convert to PIL Image
    pil_img = _to_pil(image)

    # Save to PNG buffer
    with BytesIO() as buf:
        pil_img.save(buf, format="PNG")
        b64_data = base64.b64encode(buf.getvalue()).decode("ascii")

    # Kitty escape sequence chunking (4096 bytes per chunk)
    chunk_size = 4096
    chunks = [b64_data[i : i + chunk_size] for i in range(0, len(b64_data), chunk_size)]

    for i, chunk in enumerate(chunks):
        is_last = i == len(chunks) - 1
        m = "0" if is_last else "1"

        if i == 0:
            # a=T: Action=Transmit and display
            # f=100: Format=PNG
            header = f"\033_Ga=T,f=100,m={m};"
        else:
            header = f"\033_Gm={m};"

        sys.stdout.write(header + chunk + "\033\\")

    sys.stdout.write("\n")
    sys.stdout.flush()


def _to_pil(data: Any) -> Image.Image:
    """Converts various image formats to a PIL Image."""
    from PIL import Image

    # 1. Already a PIL Image
    if _isinstance(data, "PIL.Image", "Image"):
        return data

    # 2. Handle Torch Tensors
    if _isinstance(data, "torch", "Tensor"):
        data = data.detach().cpu().numpy()

    # 3. Handle Numpy Arrays (or things that can be converted to them)
    import numpy as np

    if not isinstance(data, np.ndarray) and hasattr(data, "__array__"):
        data = np.array(data)

    if isinstance(data, np.ndarray):
        img = data
        if img.ndim == 2:
            # Grayscale [H, W]
            pass
        elif img.ndim == 3:
            # Decide between [C, H, W] and [H, W, C]
            # If the first dimension is 1, 3, or 4 and the last is not, it's likely CHW
            if img.shape[0] in (1, 3, 4) and img.shape[2] not in (1, 3, 4):
                img = np.transpose(img, (1, 2, 0))

            # Now it should be [H, W, C]
            # Handle [H, W, 1] -> [H, W] for PIL
            if img.shape[2] == 1:
                img = img.squeeze(2)
        else:
            raise ValueError(f"Unsupported array shape: {img.shape}. Expected 2D or 3D.")

        # Normalize float to uint8 [0, 255]
        if img.dtype.kind == "f":
            img = (img * 255).astype(np.uint8)

        return Image.fromarray(img)

    # 4. Fallback or error
    raise TypeError(f"Unsupported image type: {type(data)}")


def _isinstance(obj: Any, module: str, clsname: str) -> bool:
    """A helper that works like isinstance(obj, module:clsname), even if module is not imported."""
    if module not in sys.modules:
        return False
    try:
        clstype = getattr(sys.modules[module], clsname)
        return isinstance(obj, clstype)
    except AttributeError:
        return False


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Display images in Kitty terminal")
    parser.add_argument("files", nargs="+", help="Image files to display")
    args = parser.parse_args()

    for file in args.files:
        with Image.open(file) as img:
            ktcat(img)
