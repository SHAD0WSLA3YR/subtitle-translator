#!/usr/bin/env python
"""Generate tray/app icons into assets/.

Creates icon.png (64x64) and icon.ico (multi-size) with a blue CC badge.
Requires Pillow:  pip install pillow
"""

from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError as exc:
    raise SystemExit("Install Pillow first: pip install pillow") from exc


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
ASSETS.mkdir(exist_ok=True)


def make_cc_image(size: int) -> Image.Image:
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    margin = max(2, size // 16)
    draw.rounded_rectangle(
        [margin, margin, size - margin - 1, size - margin - 1],
        radius=size // 5,
        fill=(37, 99, 235, 255),
    )
    font_size = max(10, size * 11 // 32)
    try:
        font = ImageFont.truetype("segoeui.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()
    text = "CC"
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((size - tw) / 2, (size - th) / 2 - size * 0.03), text, fill="white", font=font)
    return img


def main() -> None:
    png = make_cc_image(256)
    png_path = ASSETS / "icon.png"
    png.save(png_path)
    ico_path = ASSETS / "icon.ico"
    icons = [make_cc_image(s) for s in (16, 32, 48, 64, 128, 256)]
    icons[-1].save(ico_path, format="ICO", sizes=[(im.width, im.height) for im in icons])
    print(f"Wrote {png_path}")
    print(f"Wrote {ico_path}")


if __name__ == "__main__":
    main()
