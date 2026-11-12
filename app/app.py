# Step 6 - Road Damage Detection App (FastAPI)
# Run (from the app/ folder):  uvicorn app:app --host 127.0.0.1 --port 8000
import base64
import os
import time

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from ultralytics import YOLO

APP_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(APP_DIR, "best.pt")
MODEL_NAME = open(os.path.join(APP_DIR, "best_model.txt")).read().strip() \
    if os.path.exists(os.path.join(APP_DIR, "best_model.txt")) else "best.pt"
DEVICE = 0 if torch.cuda.is_available() else "cpu"

model = YOLO(WEIGHTS)                       # loaded once when the server starts
model.predict(np.zeros((640, 640, 3), np.uint8), device=DEVICE, verbose=False)   # warm-up

app = FastAPI(title="Road Damage Detection", version="1.0")


def run_detection(img_bgr, conf):
    """Run YOLO on one BGR image -> (annotated BGR image, list of detections, inference ms)."""
    t0 = time.perf_counter()
    r = model.predict(img_bgr, conf=conf, imgsz=640, device=DEVICE, verbose=False)[0]
    ms = (time.perf_counter() - t0) * 1000
    dets = [{"class": r.names[int(c)], "confidence": round(float(p), 3),
             "box": [round(float(v), 1) for v in b]}
            for b, c, p in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist())]
    return r.plot(), dets, ms


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME, "device": str(DEVICE)}


@app.post("/detect")
async def detect(file: UploadFile = File(...), conf: float = Form(0.25)):
    data = await file.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not read the image file.")
    annotated, dets, ms = run_detection(img, conf)
    ok, jpg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
    counts = {}
    for d in dets:
        counts[d["class"]] = counts.get(d["class"], 0) + 1
    return {"model": MODEL_NAME, "inference_ms": round(ms, 1), "count": len(dets),
            "counts": counts, "detections": dets,
            "image": "data:image/jpeg;base64," + base64.b64encode(jpg.tobytes()).decode()}


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE.replace("{{MODEL}}", MODEL_NAME).replace("{{DEVICE}}", "GPU" if DEVICE == 0 else "CPU")


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Road Damage Detector</title>
<style>
:root{--bg:#0f1115;--panel:#181b22;--line:#2a2f3a;--text:#e8eaf0;--muted:#9aa3b2;--accent:#f5a524}
*{box-sizing:border-box}body{margin:0;font-family:system-ui,Segoe UI,Arial,sans-serif;background:var(--bg);color:var(--text)}
header{padding:20px 24px;border-bottom:1px solid var(--line);display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px}
h1{font-size:20px;margin:0}.tag{color:var(--muted);font-size:13px}
main{max-width:1100px;margin:0 auto;padding:20px;display:grid;grid-template-columns:320px 1fr;gap:20px}
@media(max-width:800px){main{grid-template-columns:1fr}}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:16px}
label{display:block;font-size:13px;color:var(--muted);margin:12px 0 6px}
input[type=file]{width:100%;color:var(--muted)}
input[type=range]{width:100%}
button{width:100%;margin-top:14px;padding:10px;border:0;border-radius:8px;background:var(--accent);color:#111;font-weight:600;cursor:pointer}
button.secondary{background:transparent;color:var(--text);border:1px solid var(--line)}
button:disabled{opacity:.5;cursor:default}
#view{width:100%;border-radius:8px;background:#000;min-height:280px;display:block}
video{display:none}
table{width:100%;border-collapse:collapse;font-size:13px;margin-top:12px}
th,td{padding:6px 8px;border-bottom:1px solid var(--line);text-align:left}th{color:var(--muted);font-weight:500}
.stats{display:flex;gap:16px;flex-wrap:wrap;font-size:13px;color:var(--muted);margin-top:10px}
.stats b{color:var(--text)}
</style></head><body>
<header><h1>Road Damage Detector</h1>
<span class="tag">Model: {{MODEL}} &middot; Running on {{DEVICE}}</span></header>
<main>
<section class="panel">
  <label>1. Upload a road image</label>
  <input type="file" id="file" accept="image/*">
  <label>Confidence threshold: <span id="cv">0.25</span></label>
  <input type="range" id="conf" min="0.05" max="0.9" step="0.05" value="0.25">
  <button id="go">Detect damage</button>
  <label style="margin-top:22px">2. Or use your webcam (real-time)</label>
  <button id="cam" class="secondary">Start webcam</button>
</section>
<section class="panel">
  <img id="view" alt="">
  <video id="video" autoplay playsinline muted></video>
  <div class="stats"><span>Detections: <b id="n">-</b></span><span>Inference: <b id="ms">-</b></span><span id="summary"></span></div>
  <table><thead><tr><th>#</th><th>Damage type</th><th>Confidence</th><th>Box (x1, y1, x2, y2)</th></tr></thead><tbody id="rows"></tbody></table>
</section>
</main>
<script>
const $=id=>document.getElementById(id);
const conf=$("conf"); conf.oninput=()=>$("cv").textContent=conf.value;
function show(d){
  $("view").src=d.image; $("n").textContent=d.count; $("ms").textContent=d.inference_ms+" ms";
  $("summary").textContent=Object.entries(d.counts).map(([k,v])=>k+": "+v).join("  |  ");
  $("rows").innerHTML=d.detections.map((x,i)=>`<tr><td>${i+1}</td><td>${x.class}</td><td>${(x.confidence*100).toFixed(1)}%</td><td>${x.box.join(", ")}</td></tr>`).join("");
}
async function send(blob){
  const fd=new FormData(); fd.append("file",blob,"frame.jpg"); fd.append("conf",conf.value);
  const r=await fetch("/detect",{method:"POST",body:fd});
  if(!r.ok){alert((await r.json()).detail||"Error");return null}
  return r.json();
}
$("go").onclick=async()=>{
  const f=$("file").files[0]; if(!f){alert("Choose an image first");return}
  $("go").disabled=true; const d=await send(f); $("go").disabled=false; if(d)show(d);
};
let stream=null, running=false;
$("cam").onclick=async()=>{
  if(running){running=false;stream.getTracks().forEach(t=>t.stop());$("cam").textContent="Start webcam";return}
  try{stream=await navigator.mediaDevices.getUserMedia({video:true})}catch(e){alert("Webcam not available");return}
  const v=$("video"); v.srcObject=stream; running=true; $("cam").textContent="Stop webcam";
  const c=document.createElement("canvas");
  (async function loop(){
    while(running){
      if(v.videoWidth){c.width=v.videoWidth;c.height=v.videoHeight;c.getContext("2d").drawImage(v,0,0);
        const b=await new Promise(res=>c.toBlob(res,"image/jpeg",0.8)); const d=await send(b); if(d)show(d);}
      else await new Promise(r=>setTimeout(r,100));
    }
  })();
};
</script></body></html>"""
