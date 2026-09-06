"""Turn sticker files into a plain PNG the vision model can read.

static  (.webp)  -> first frame, alpha flattened onto white
video   (.webm)  -> frame grabbed with ffmpeg
animated(.tgs)   -> Telegram's own thumbnail (a .webp) is used instead
"""
from __future__ import annotations

import asyncio
import io
import subprocess
import tempfile

from PIL import Image

MAX_SIDE = 512


def _flatten(img: Image.Image) -> bytes:
    img = img.convert("RGBA")
    bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
    flat = Image.alpha_composite(bg, img).convert("RGB")
    if max(flat.size) > MAX_SIDE:
        flat.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
    buf = io.BytesIO()
    flat.save(buf, format="PNG")
    return buf.getvalue()


def _from_image_bytes(data: bytes) -> bytes:
    img = Image.open(io.BytesIO(data))
    try:
        img.seek(0)
    except EOFError:
        pass
    return _flatten(img)


def _from_video_bytes(data: bytes) -> bytes:
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = f"{tmp}/in.webm", f"{tmp}/out.png"
        with open(src, "wb") as fh:
            fh.write(data)
        for args in (["-ss", "0.4", "-i", src], ["-i", src]):
            proc = subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", *args, "-frames:v", "1", dst],
                capture_output=True,
            )
            if proc.returncode == 0:
                try:
                    with open(dst, "rb") as fh:
                        return _from_image_bytes(fh.read())
                except FileNotFoundError:
                    continue
        raise RuntimeError("ffmpeg could not extract a frame")


async def to_png(data: bytes, kind: str) -> bytes:
    if kind == "video":
        return await asyncio.to_thread(_from_video_bytes, data)
    return await asyncio.to_thread(_from_image_bytes, data)
