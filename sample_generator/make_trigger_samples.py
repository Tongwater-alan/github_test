import os
import numpy as np
from PIL import Image, ImageDraw

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def main():
    out_dir = os.path.join(".", "assets", "triggers")
    ensure_dir(out_dir)

    # trigger size: 32x32 RGB
    w = h = 32
    trig = Image.new("RGB", (w, h), (0, 0, 0))
    d = ImageDraw.Draw(trig)

    # A simple colored square + cross (high contrast)
    d.rectangle([2, 2, w - 3, h - 3], outline=(255, 0, 0), width=2)
    d.line([0, 0, w - 1, h - 1], fill=(0, 255, 0), width=2)
    d.line([0, h - 1, w - 1, 0], fill=(0, 255, 0), width=2)
    d.ellipse([10, 10, 21, 21], fill=(0, 0, 255))

    # mask: white=apply, black=ignore
    mask = Image.new("L", (w, h), 0)
    md = ImageDraw.Draw(mask)
    md.rectangle([2, 2, w - 3, h - 3], fill=255)

    trig_path = os.path.join(out_dir, "trigger.png")
    mask_path = os.path.join(out_dir, "mask.png")
    trig.save(trig_path)
    mask.save(mask_path)

    print("Wrote:", trig_path)
    print("Wrote:", mask_path)
    print("Upload these two files into the UI.")

if __name__ == "__main__":
    main()