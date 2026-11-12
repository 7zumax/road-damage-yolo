"""Step 5 - Final test-set comparison: accuracy, IoU, speed, size and ONNX deployment efficiency.
Evaluates every trained model in runs/ and copies the best (by test mAP50-95) into app/best.pt."""
import csv
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from ultralytics import YOLO

from config import APP_DIR, CLASSES, DATA_6CLS, DATA_YAML, RESULTS, RUNS

TEST_IMG = DATA_6CLS / "test" / "images"
TEST_LBL = DATA_6CLS / "test" / "labels"


def box_iou(a, b):
    """IoU between every box in a (N,4) and every box in b (M,4); boxes are x1,y1,x2,y2."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def load_gt(img_path, h, w):
    """YOLO label file (class xc yc bw bh, normalised) -> classes and pixel xyxy boxes."""
    lf = TEST_LBL / (img_path.stem + ".txt")
    rows = [r.split() for r in lf.read_text().splitlines() if r.strip()] if lf.exists() else []
    if not rows:
        return np.zeros(0, int), np.zeros((0, 4))
    a = np.array(rows, float)
    xc, yc, bw, bh = a[:, 1] * w, a[:, 2] * h, a[:, 3] * w, a[:, 4] * h
    return a[:, 0].astype(int), np.stack([xc - bw / 2, yc - bh / 2, xc + bw / 2, yc + bh / 2], 1)


def mean_iou(model, images, chunk=16):
    """Match predictions to ground truth (same class, IoU >= 0.5, highest confidence first).
    Returns (mean IoU of matched boxes, mean best IoU per ground-truth box with misses counted as 0)."""
    tp_ious, gt_best = [], []
    for k in range(0, len(images), chunk):                       # small chunks so it fits in GPU memory
        for r in model.predict([str(p) for p in images[k:k + chunk]], conf=0.25, imgsz=640, device=0, verbose=False):
            h, w = r.orig_shape
            g_cls, g_box = load_gt(Path(r.path), h, w)
            p_box = r.boxes.xyxy.cpu().numpy(); p_cls = r.boxes.cls.cpu().numpy().astype(int)
            ious = box_iou(p_box, g_box)
            if len(p_box) and len(g_box):
                ious = np.where(p_cls[:, None] == g_cls[None, :], ious, 0.0)
            used = set()
            for i in np.argsort(-r.boxes.conf.cpu().numpy()):
                if ious.shape[1] == 0:
                    break
                cand = ious[i].copy(); cand[list(used)] = 0
                j = int(np.argmax(cand))
                if cand[j] >= 0.5:
                    used.add(j); tp_ious.append(cand[j])
            gt_best += [ious[:, j].max() if len(p_box) else 0.0 for j in range(len(g_box))]
    return float(np.mean(tp_ious)), float(np.mean(gt_best))


def fps(model, images, device):
    """End-to-end images per second (preprocess + model + postprocess), batch size 1."""
    for im in images[:5]:
        model.predict(str(im), imgsz=640, device=device, verbose=False)        # warm-up
    if device != "cpu":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for im in images:
        model.predict(str(im), imgsz=640, device=device, verbose=False)
    if device != "cpu":
        torch.cuda.synchronize()
    return len(images) / (time.perf_counter() - t0)


if __name__ == "__main__":
    images = sorted(p for p in TEST_IMG.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    models = {p.parents[1].name: p for p in sorted(RUNS.glob("*/weights/best.pt"))
              if p.parents[1].name.startswith(("baseline_", "tuned_"))}
    rows, cls_rows = [], []

    for name, w in models.items():
        print(f"\n===== {name} =====", flush=True)
        model = YOLO(w)
        m = model.val(data=str(DATA_YAML), split="test", imgsz=640, batch=16, device=0, plots=False,
                      project=str(RUNS), name=f"test_{name}", exist_ok=True, verbose=False)
        iou_tp, iou_gt = mean_iou(model, images)
        fps_gpu, fps_cpu = fps(model, images[:100], 0), fps(model, images[:30], "cpu")
        onnx_path = YOLO(w).export(format="onnx", imgsz=640, simplify=False, verbose=False)
        fps_onnx = fps(YOLO(onnx_path, task="detect"), images[:30], "cpu")

        rows.append(dict(model=name, precision=round(float(m.box.mp), 4), recall=round(float(m.box.mr), 4),
                         mAP50=round(float(m.box.map50), 4), mAP50_95=round(float(m.box.map), 4),
                         mean_IoU_matched=round(iou_tp, 4), mean_IoU_per_gt=round(iou_gt, 4),
                         gpu_infer_ms=round(m.speed["inference"], 2), fps_gpu=round(fps_gpu, 1),
                         fps_cpu_pytorch=round(fps_cpu, 2), fps_cpu_onnx=round(fps_onnx, 2),
                         size_pt_mb=round(w.stat().st_size / 1e6, 1),
                         size_onnx_mb=round(Path(onnx_path).stat().st_size / 1e6, 1),
                         params_millions=round(sum(p.numel() for p in model.model.parameters()) / 1e6, 2)))
        cls_rows.append({"model": name, **{CLASSES[i]: round(float(m.box.ap50[k]), 3)
                                           for k, i in enumerate(m.box.ap_class_index)}})
        print("RESULT:", rows[-1], flush=True)

    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "test_results.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=rows[0].keys()); wr.writeheader(); wr.writerows(rows)
    with open(RESULTS / "test_per_class_ap50.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["model"] + CLASSES); wr.writeheader(); wr.writerows(cls_rows)

    best = max(rows, key=lambda r: r["mAP50_95"])
    APP_DIR.mkdir(exist_ok=True)
    shutil.copy(models[best["model"]], APP_DIR / "best.pt")
    (APP_DIR / "best_model.txt").write_text(best["model"])
    print(f"\nBEST MODEL: {best['model']} -> app/best.pt")
