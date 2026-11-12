"""Step 2 - Build a 6-class copy of the dataset: drop 'edge' and renumber the remaining classes.
Images are hard-linked (no extra disk space); labels are rewritten."""
import os
import shutil

from config import CLASSES, DATA_6CLS, ORIGINAL_CLASSES, RAW_DATA

REMAP = {ORIGINAL_CLASSES.index(n): CLASSES.index(n) for n in CLASSES}   # old id -> new id

for split in ["train", "valid", "test"]:
    src_img, src_lbl = RAW_DATA / split / "images", RAW_DATA / split / "labels"
    dst_img, dst_lbl = DATA_6CLS / split / "images", DATA_6CLS / split / "labels"
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)

    for f in src_img.iterdir():
        d = dst_img / f.name
        if not d.exists():
            try:
                os.link(f, d)
            except OSError:
                shutil.copy2(f, d)

    removed = 0
    for f in src_lbl.glob("*.txt"):
        kept = []
        for r in f.read_text().splitlines():
            if not r.strip():
                continue
            parts = r.split()
            old = int(parts[0])
            if old not in REMAP:
                removed += 1
                continue
            kept.append(" ".join([str(REMAP[old])] + parts[1:]))
        (dst_lbl / f.name).write_text("\n".join(kept) + ("\n" if kept else ""))
    print(f"[{split}] edge boxes removed: {removed}")

(DATA_6CLS / "data.yaml").write_text(
    f"path: {DATA_6CLS.as_posix()}\ntrain: train/images\nval: valid/images\ntest: test/images\n"
    f"nc: {len(CLASSES)}\nnames: {CLASSES}\n")
print("data.yaml written to", DATA_6CLS / "data.yaml")
