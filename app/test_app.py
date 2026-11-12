"""Automated app test: starts the app in-process and calls every endpoint once."""
import base64
from pathlib import Path

from fastapi.testclient import TestClient

from app import app

HERE = Path(__file__).resolve().parent
TEST_IMG = sorted((HERE.parent / "data" / "road_damage_6cls" / "test" / "images").glob("*.jpg"))[0]
client = TestClient(app)

print("GET /health ->", client.get("/health").json())
print("GET /       ->", client.get("/").status_code)
with open(TEST_IMG, "rb") as f:
    d = client.post("/detect", files={"file": ("road.jpg", f, "image/jpeg")}, data={"conf": "0.25"}).json()
print("POST /detect ->", d["count"], "detections", d["counts"], f'{d["inference_ms"]} ms')
(HERE / "test_output.jpg").write_bytes(base64.b64decode(d["image"].split(",", 1)[1]))
bad = client.post("/detect", files={"file": ("x.txt", b"not an image", "text/plain")})
print("POST /detect (bad file) ->", bad.status_code)
