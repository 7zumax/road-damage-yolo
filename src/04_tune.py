"""Step 4 - Hyperparameter tuning (Ultralytics genetic tuner), then a full retrain with the winners.
If the default settings win the search, the baseline model is reused instead of retraining."""
import argparse
import csv
import json
import time

import yaml
from ultralytics import YOLO

from config import DATA_YAML, MODELS, RESULTS, RUNS, TRAIN_SETTINGS

COMMON = dict(data=str(DATA_YAML), imgsz=640, batch=16, workers=4, device=0,
              optimizer="SGD")   # fixed optimizer: with 'auto', short runs switch to AdamW and ignore lr0


def trial_fitness(tune_dir):
    nd = tune_dir / "tune_results.ndjson"                   # newer Ultralytics
    if nd.exists():
        return [float(json.loads(l)["fitness"]) for l in nd.read_text().splitlines() if l.strip()]
    return [float(r["fitness"]) for r in csv.DictReader(open(tune_dir / "tune_results.csv"))]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=6, help="epochs per tuning trial")
    ap.add_argument("--iterations", type=int, default=4, help="number of tuning trials")
    args = ap.parse_args()

    baseline = {r["model"]: r for r in csv.DictReader(open(RESULTS / "val_baseline.csv"))}
    rows = []
    for weights in MODELS:
        base = weights.replace(".pt", "")
        tune_dir = RUNS / ("tune_" + base)
        YOLO(weights).tune(epochs=args.epochs, iterations=args.iterations, plots=False, save=False,
                           project=str(RUNS), name="tune_" + base, exist_ok=True, **COMMON)
        fit = trial_fitness(tune_dir)
        best_i = fit.index(max(fit))
        print(f"{base}: trial fitness {fit} -> best trial {best_i + 1}")

        if best_i == 0:                                     # trial 1 = default settings
            b = baseline["baseline_" + base]
            rows.append(dict(model="tuned_" + base, train_minutes=0.0, precision=b["precision"], recall=b["recall"],
                             mAP50=b["mAP50"], **{"mAP50-95": b["mAP50-95"]}, note="defaults won - baseline reused"))
            continue

        hyp = yaml.safe_load(open(tune_dir / "best_hyperparameters.yaml"))
        settings = {**TRAIN_SETTINGS, **COMMON, **hyp}
        t0 = time.time()
        YOLO(weights).train(project=str(RUNS), name="tuned_" + base, exist_ok=True, **settings)
        minutes = (time.time() - t0) / 60
        m = YOLO(RUNS / ("tuned_" + base) / "weights" / "best.pt").val(data=str(DATA_YAML), split="val", device=0)
        rows.append(dict(model="tuned_" + base, train_minutes=round(minutes, 1), precision=round(float(m.box.mp), 4),
                         recall=round(float(m.box.mr), 4), mAP50=round(float(m.box.map50), 4),
                         **{"mAP50-95": round(float(m.box.map), 4)}, note=f"trial {best_i + 1} won"))
        print("RESULT:", rows[-1])

    with open(RESULTS / "val_tuned.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
