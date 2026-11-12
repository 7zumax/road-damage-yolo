"""Shared paths and settings used by every script. All paths are relative to the repo root."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DATA = ROOT / "data" / "road_damage"            # Roboflow YOLOv8 export (unzipped here)
DATA_6CLS = ROOT / "data" / "road_damage_6cls"      # created by 02_prepare_6classes.py
DATA_YAML = DATA_6CLS / "data.yaml"
RUNS = ROOT / "runs"                                # all training / evaluation outputs
RESULTS = ROOT / "results"
APP_DIR = ROOT / "app"

ORIGINAL_CLASSES = ['alligator', 'block', 'crack', 'edge', 'longitudinal', 'pothole', 'transverse']
DROPPED_CLASS = 'edge'                              # 22 boxes total, 0 in test -> removed
CLASSES = [c for c in ORIGINAL_CLASSES if c != DROPPED_CLASS]

MODELS = ["yolov8s.pt", "yolov10s.pt", "yolo11s.pt"]
TRAIN_SETTINGS = dict(epochs=50, patience=15, imgsz=640, batch=16, seed=0, workers=4, device=0)
