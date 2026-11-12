"""Step 3 - Baseline training: YOLOv8s vs YOLOv10s vs YOLO11s with identical settings."""
import csv
import time

from ultralytics import YOLO

from config import DATA_YAML, MODELS, RESULTS, RUNS, TRAIN_SETTINGS

if __name__ == "__main__":                      # required on Windows for data-loader workers
    rows = []
    for weights in MODELS:
        name = "baseline_" + weights.replace(".pt", "")
        t0 = time.time()
        YOLO(weights).train(data=str(DATA_YAML), project=str(RUNS), name=name, exist_ok=True, **TRAIN_SETTINGS)
        minutes = (time.time() - t0) / 60

        m = YOLO(RUNS / name / "weights" / "best.pt").val(data=str(DATA_YAML), split="val", device=0,
                                                         project=str(RUNS), name=name + "_val", exist_ok=True)
        rows.append(dict(model=name, train_minutes=round(minutes, 1), precision=round(m.box.mp, 4),
                         recall=round(m.box.mr, 4), mAP50=round(m.box.map50, 4), **{"mAP50-95": round(m.box.map, 4)}))
        print("RESULT:", rows[-1])

    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "val_baseline.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
