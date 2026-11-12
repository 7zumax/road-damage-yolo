"""Step 1 - Dataset check: classes, images and boxes per split, missing/empty labels."""
import ast
from collections import Counter

from config import RAW_DATA

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

names = []
for line in (RAW_DATA / "data.yaml").read_text().splitlines():
    if line.strip().startswith("names:"):
        names = list(ast.literal_eval(line.split("names:", 1)[1].strip()))
print(f"Classes ({len(names)}): {names}\n")

total = Counter()
for split in ["train", "valid", "test"]:
    imgs = [p for p in (RAW_DATA / split / "images").iterdir() if p.suffix.lower() in IMG_EXT]
    lbl_dir = RAW_DATA / split / "labels"
    boxes, missing, empty = Counter(), 0, 0
    for img in imgs:
        lf = lbl_dir / (img.stem + ".txt")
        if not lf.exists():
            missing += 1
            continue
        rows = [r for r in lf.read_text().splitlines() if r.strip()]
        empty += not rows
        for r in rows:
            boxes[int(r.split()[0])] += 1          # first number on each line = class id
    total.update(boxes)
    print(f"[{split}] images: {len(imgs)} | missing labels: {missing} | empty labels: {empty} | boxes: {sum(boxes.values())}")
    for i, n in enumerate(names):
        print(f"    {i} {n:<14} {boxes[i]:>6}")

grand = sum(total.values())
print("\nALL SPLITS - boxes per class:")
for i, n in enumerate(names):
    print(f"    {i} {n:<14} {total[i]:>6}  ({100 * total[i] / grand:5.1f}%)")
