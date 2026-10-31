import base64
import struct
import sys
import zlib
from io import BytesIO
from typing import Any


def ktcat(image: Any) -> None:
    """
    Displays an image in the Kitty terminal.

    Args:
        image: A PIL Image, a Numpy array, or a Torch tensor.
               Supports shapes [H, W], [H, W, C], and [C, H, W].
               If values are floats, they are assumed to be in [0, 1] and are scaled to [0, 255].
    """
    _write_kitty_png(_to_png_bytes(image))


def _write_kitty_png(png_bytes: bytes) -> None:
    raw_chunk_size = 3072  # Encodes to exactly 4096 base64 bytes.

    for offset in range(0, len(png_bytes), raw_chunk_size):
        encoded = base64.b64encode(png_bytes[offset : offset + raw_chunk_size]).decode("ascii")
        is_last = offset + raw_chunk_size >= len(png_bytes)
        m = "0" if is_last else "1"

        if offset == 0:
            # a=T: Action=Transmit and display
            # f=100: Format=PNG
            header = f"\033_Ga=T,f=100,m={m};"
        else:
            header = f"\033_Gm={m};"

        sys.stdout.write(header + encoded + "\033\\")

    sys.stdout.write("\n")
    sys.stdout.flush()


def _to_png_bytes(data: Any) -> bytes:
    """Converts supported image inputs into PNG bytes."""

    # If the input is already a PIL image, let Pillow encode it.
    if _isinstance(data, "PIL.Image", "Image"):
        with BytesIO() as buf:
            data.save(buf, format="PNG")
            return buf.getvalue()

    import numpy as np

    img = _to_array(data)
    height, width = img.shape[:2]
    channels = 1 if img.ndim == 2 else img.shape[2]
    color_types = {1: 0, 3: 2, 4: 6}  # grayscale, RGB, RGBA
    if channels not in color_types:
        raise ValueError(f"Unsupported channel count: {channels}. Expected 1, 3, or 4.")

    # PNG scanlines are prefixed with a filter byte. We always use filter type 0.
    raw = np.pad(img.reshape(height, -1), ((0, 0), (1, 0))).tobytes()

    return b"".join([
        b"\x89PNG\r\n\x1a\n",
        _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, color_types[channels], 0, 0, 0)),
        _png_chunk(b"IDAT", zlib.compress(raw)),
        _png_chunk(b"IEND", b""),
    ])


def _to_array(data: Any) -> Any:
    """Converts supported inputs to a uint8 numpy array shaped [H, W] or [H, W, C]."""
    import numpy as np

    if _isinstance(data, "torch", "Tensor"):
        data = data.detach().cpu().numpy()

    img = np.asarray(data)
    if img.ndim not in (2, 3):
        raise ValueError(f"Unsupported array shape: {img.shape}. Expected 2D or 3D.")

    if img.ndim == 3:
        if img.shape[0] in (1, 3, 4) and img.shape[2] not in (1, 3, 4):
            img = np.transpose(img, (1, 2, 0))
        if img.shape[2] == 1:
            img = img.squeeze(2)

    if img.dtype.kind == "f":
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    return img


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    return b"".join([
        struct.pack(">I", len(data)),
        chunk_type,
        data,
        struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF),
    ])


def _isinstance(obj: Any, module: str, clsname: str) -> bool:
    """A helper that works like isinstance(obj, module:clsname), even if module is not imported."""
    cls = getattr(sys.modules.get(module), clsname, None)
    return cls is not None and isinstance(obj, cls)


def main() -> None:
    import argparse

    from PIL import Image

    parser = argparse.ArgumentParser(prog="ktcat", description="Display images in Kitty terminal")
    parser.add_argument("files", nargs="+", help="Image files to display")
    args = parser.parse_args()

    for file in args.files:
        with Image.open(file) as img:
            ktcat(img)
