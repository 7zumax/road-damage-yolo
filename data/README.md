# Data

Dataset: **road-damage** by TrafficSignsSafeway on Roboflow Universe (CC BY 4.0)
https://universe.roboflow.com/trafficsignssafeway/road-damage-l1ju7

1. Open the link, click **Fork Dataset** (the original has no published version), then in your copy: **Versions → Create New Version** (split 70/20/10, default preprocessing, no augmentation).
2. **Download Dataset → YOLOv8 → Download zip to computer**.
3. Unzip it here so you have `data/road_damage/data.yaml` and `data/road_damage/{train,valid,test}/`.
4. Run `python src/02_prepare_6classes.py` to create `data/road_damage_6cls/` (the `edge` class is removed).
