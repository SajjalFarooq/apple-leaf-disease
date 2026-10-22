"""Augment train split to exactly 3000 images per class.

Technique (cycle method, seed 42):
- For each class, needed = 3000 - original train count.
- Cycle through original images in order; each original gets one new
  augmented copy per cycle, with a different random combination each time.
- Stop exactly when the class reaches 3000 (last cycle may be partial).
- Augmentation per copy (random combination):
  horizontal and/or vertical flip, rotation -25 to +25 degrees,
  zoom 0.8-1.2 (centre crop / resize), brightness 0.8-1.2,
  contrast 0.8-1.2, slight colour jitter, slight Gaussian noise.
- Validation and test are NEVER augmented.
- Augmented files are named <original_stem>_aug_<n>.JPG
"""
import random, pathlib
from PIL import Image, ImageEnhance, ImageOps
import numpy as np

random.seed(42)
np.random.seed(42)
BASE = pathlib.Path('/home/hatch/workspace/goals/github-professional-setup/files/apple-leaf-disease/data/train')
TARGET = 3000

def augment_one(img):
    # random flip
    if random.random() < 0.5:
        img = ImageOps.mirror(img)
    if random.random() < 0.5:
        img = ImageOps.flip(img)
    # rotation -25 to +25 degrees
    angle = random.uniform(-25, 25)
    img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=(0, 0, 0))
    # zoom 0.8-1.2 via centre crop / resize
    w, h = img.size
    zoom = random.uniform(0.8, 1.2)
    if zoom > 1.0:
        cw, ch = int(w / zoom), int(h / zoom)
        left, top = (w - cw)//2, (h - ch)//2
        img = img.crop((left, top, left+cw, top+ch)).resize((w, h), Image.BILINEAR)
    # brightness / contrast / colour, each 0.8-1.2
    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.8, 1.2))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.8, 1.2))
    img = ImageEnhance.Color(img).enhance(random.uniform(0.85, 1.15))
    # slight Gaussian noise
    arr = np.asarray(img).astype(np.float32)
    arr = arr + np.random.normal(0, 5, arr.shape)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

for class_dir in sorted(BASE.iterdir()):
    if not class_dir.is_dir():
        continue
    originals = sorted([p for p in class_dir.glob('*.JPG') if '_aug_' not in p.name])
    orig_count = len(originals)
    needed = TARGET - orig_count
    print(f"{class_dir.name}: original={orig_count} needed={needed}", flush=True)
    made = 0
    cycle = 0
    while made < needed:
        cycle += 1
        for idx, src in enumerate(originals):
            if made >= needed:
                break
            img = Image.open(src).convert('RGB')
            out = augment_one(img)
            out_path = class_dir / f"{src.stem}_aug_{cycle}.JPG"
            out.save(out_path, quality=90)
            made += 1
        print(f"  cycle {cycle}: made so far {made}", flush=True)
    final = len(list(class_dir.glob('*.JPG')))
    print(f"{class_dir.name}: FINAL train count = {final} (original {orig_count} + augmented {made})", flush=True)
    assert final == TARGET, f"{class_dir.name} final {final} != {TARGET}"
print("DONE: all train classes = 3000")
