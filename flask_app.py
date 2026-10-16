import base64
import io
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from flask import Flask, jsonify, render_template_string, request
from PIL import Image
from ultralytics import YOLO

# ----------------------------- Config -----------------------------
CNN_WEIGHTS = "best_resnet_gtsrb.pth"
YOLO_WEIGHTS = "best.pt"

IMG_SIZE = 128
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = [
    "Speed limit 20", "Speed limit 30", "Speed limit 50", "Speed limit 60",
    "Speed limit 70", "Speed limit 80", "End of speed limit 80", "Speed limit 100",
    "Speed limit 120", "No passing", "No passing for vehicles over 3.5 tonnes",
    "Right-of-way at next intersection", "Priority road", "Yield", "Stop",
    "No vehicles", "Vehicles over 3.5 tonnes prohibited", "No entry",
    "General caution", "Dangerous curve left", "Dangerous curve right",
    "Double curve", "Bumpy road", "Slippery road", "Road narrows on the right",
    "Road work", "Traffic signals", "Pedestrians", "Children crossing",
    "Bicycles crossing", "Beware of ice/snow", "Wild animals crossing",
    "End of all speed and passing limits", "Turn right ahead", "Turn left ahead",
    "Ahead only", "Go straight or right", "Go straight or left", "Keep right",
    "Keep left", "Roundabout mandatory", "End of no passing",
    "End of no passing for vehicles over 3.5 tonnes",
]

# ----------------------------- Models -----------------------------
def load_cnn():
    m = models.resnet34(weights=None)
    m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.fc.in_features, 43))
    m.load_state_dict(torch.load(CNN_WEIGHTS, map_location=DEVICE, weights_only=True))
    return m.to(DEVICE).eval()

cnn_model = load_cnn()
yolo_model = YOLO(YOLO_WEIGHTS)

cnn_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

def classify(img: Image.Image):
    x = cnn_transforms(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(cnn_model(x), dim=1)[0]
    top_p, top_i = torch.topk(probs, 5)
    return [
        {"class_id": int(i), "name": CLASS_NAMES[int(i)], "prob": round(float(p) * 100, 2)}
        for p, i in zip(top_p.cpu(), top_i.cpu())
    ]

def detect(img: Image.Image, conf: float):
    img_np = np.array(img)
    res = yolo_model(img_np, conf=conf, device=DEVICE, verbose=False)[0]
    annotated = cv2.cvtColor(res.plot(), cv2.COLOR_BGR2RGB)
    buf = io.BytesIO()
    Image.fromarray(annotated).save(buf, format="PNG")
    boxes = [
        {
            "class_id": int(b.cls),
            "name": CLASS_NAMES[int(b.cls)],
            "conf": round(float(b.conf) * 100, 2),
            "xyxy": [round(float(v), 1) for v in b.xyxy[0].tolist()],
        }
        for b in res.boxes
    ]
    return base64.b64encode(buf.getvalue()).decode(), boxes

# ----------------------------- Flask -----------------------------
app = Flask(__name__)

PAGE = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ML CSC Final Project</title>
<style>
  :root { --bg:#0e1117; --panel:#262730; --text:#fafafa; --muted:#a3a8b8; --accent:#ff4b4b; }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--text); font-family:"Source Sans Pro",system-ui,sans-serif; }
  main { max-width: 1200px; margin: 0 auto; padding: 48px 24px 80px; }
  h1 { font-size: 2.6rem; margin: 0 0 24px; }
  h2 { font-size: 2rem; margin: 40px 0 16px; }
  .drop { background:var(--panel); border-radius:8px; padding:16px; display:flex; align-items:center; gap:16px; }
  .drop button, .drop label { background:var(--bg); color:var(--text); border:1px solid #444; padding:10px 18px;
    border-radius:8px; cursor:pointer; font-size:1rem; }
  .muted { color:var(--muted); font-size:.9rem; }
  select { width:100%; background:var(--panel); color:var(--text); border:none; padding:14px; border-radius:8px; font-size:1rem; }
  input[type=range] { width:100%; accent-color:var(--accent); }
  .row { display:flex; gap:24px; flex-wrap:wrap; }
  .row > * { flex:1 1 400px; }
  
  /* Increased max-width to make images much larger and clear */
  img { width: 100%; max-width: 700px; height: auto; border-radius: 8px; display: block; }

  .metric .label { color:var(--muted); font-size:.95rem; }
  .metric .value { font-size:2.4rem; }
  .metrics { display:flex; gap:48px; flex-wrap:wrap; }
  table { width:100%; border-collapse:collapse; border:1px solid #333; border-radius:8px; overflow:hidden; }
  th, td { padding:12px; text-align:left; border-bottom:1px solid #333; }
  th { color:var(--muted); font-weight:400; }
  td.num { text-align:right; }
  .info { background:#10283f; color:#3d9df3; padding:16px; border-radius:8px; }
  .ok { color:#21c354; } .bad { color:#ff4b4b; }
  hr { border:0; border-top:1px solid #333; margin:32px 0; }
  .hidden { display:none; }
</style>
</head>
<body>
<main>
  <h1>ML CSC Final Project</h1>
  <p>Upload a traffic sign image. ResNet34 CNN classifies the image, and YOLOv8 detects traffic signs with bounding boxes.</p>

  <p>Choose a traffic sign image</p>
  <div class="drop">
    <label for="file">&#8593; Upload</label>
    <input id="file" type="file" accept="image/png,image/jpeg,image/webp" class="hidden">
    <span class="muted" id="fname">200MB per file &bull; JPG, PNG, WEBP</span>
  </div>

  <div id="results" class="hidden">
    <h2>Uploaded Image</h2>
    <img id="orig">

    <p style="margin-top:24px" class="muted">Real answer (optional, used only for comparison)</p>
    <select id="real">
      <option value="">Not specified</option>
      {% for i, n in classes %}<option value="{{ i }}">{{ i }} - {{ n }}</option>{% endfor %}
    </select>
    <p><b>Real answer:</b> <span id="real-text">Not specified</span></p>
    <p id="verdict"></p>

    <hr>
    <h2>CNN Classification &mdash; ResNet34</h2>
    <div class="metrics">
      <div class="metric"><div class="label">Predicted class</div><div class="value" id="pred-name"></div></div>
      <div class="metric"><div class="label">Class ID</div><div class="value" id="pred-id"></div></div>
      <div class="metric"><div class="label">Confidence</div><div class="value" id="pred-conf"></div></div>
    </div>
    <p><b>Top 5 CNN predictions</b></p>
    <table>
      <thead><tr><th>Traffic sign</th><th>Class ID</th><th>Probability (%)</th></tr></thead>
      <tbody id="top5"></tbody>
    </table>

    <hr>
    <h2>Object Detection &mdash; YOLOv8</h2>
    <p>Detection confidence threshold: <span id="thr-val" style="color:var(--accent)">0.25</span></p>
    <input id="thr" type="range" min="0.10" max="1.00" step="0.05" value="0.25">
    <div class="row" style="margin-top:16px">
      <div><img id="yolo-img"></div>
      <div>
        <div class="metric"><div class="label">Objects detected</div><div class="value" id="n-det">0</div></div>
        <div id="det-info" style="margin-top: 16px;"></div>
      </div>
    </div>
  </div>
</main>

<script>
const $ = id => document.getElementById(id);
let currentFile = null, timer = null;

$("file").addEventListener("change", e => {
  currentFile = e.target.files[0];
  if (!currentFile) return;
  $("fname").textContent = currentFile.name;
  $("orig").src = URL.createObjectURL(currentFile);
  run();
});
$("thr").addEventListener("input", () => {
  $("thr-val").textContent = (+$("thr").value).toFixed(2);
  clearTimeout(timer);
  timer = setTimeout(run, 250);
});
$("real").addEventListener("change", updateVerdict);

let lastPred = null;
function updateVerdict() {
  const sel = $("real");
  $("real-text").textContent = sel.value === "" ? "Not specified" : sel.options[sel.selectedIndex].text;
  const v = $("verdict");
  if (sel.value === "" || lastPred === null) { v.textContent = ""; return; }
  const ok = +sel.value === lastPred;
  v.className = ok ? "ok" : "bad";
  v.textContent = ok ? "CNN prediction matches the real answer." : "CNN prediction does not match the real answer.";
}

async function run() {
  if (!currentFile) return;
  const fd = new FormData();
  fd.append("image", currentFile);
  fd.append("conf", $("thr").value);
  const r = await fetch("/predict", { method: "POST", body: fd });
  const d = await r.json();
  if (d.error) { alert(d.error); return; }

  $("results").classList.remove("hidden");
  const top = d.cnn[0];
  lastPred = top.class_id;
  $("pred-name").textContent = top.name;
  $("pred-id").textContent = top.class_id;
  $("pred-conf").textContent = top.prob.toFixed(2) + "%";
  $("top5").innerHTML = d.cnn.map(c =>
    `<tr><td>${c.name}</td><td class="num">${c.class_id}</td><td class="num">${c.prob.toFixed(2)}</td></tr>`).join("");
  updateVerdict();

  $("yolo-img").src = "data:image/png;base64," + d.yolo_image;
  $("n-det").textContent = d.boxes.length;
  $("det-info").innerHTML = d.boxes.length === 0
    ? '<div class="info">No objects detected at this confidence threshold.</div>'
    : `<table><thead><tr><th>Sign</th><th>Conf (%)</th></tr></thead><tbody>` +
      d.boxes.map(b => `<tr><td>${b.name}</td><td class="num">${b.conf.toFixed(2)}</td></tr>`).join("") +
      `</tbody></table>`;
}
</script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(PAGE, classes=list(enumerate(CLASS_NAMES)))

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("image")
    if file is None:
        return jsonify(error="No image uploaded."), 400
    try:
        img = Image.open(file.stream).convert("RGB")
    except Exception:
        return jsonify(error="Could not read the uploaded image."), 400

    conf = min(max(float(request.form.get("conf", 0.25)), 0.01), 1.0)
    yolo_b64, boxes = detect(img, conf)
    return jsonify(cnn=classify(img), yolo_image=yolo_b64, boxes=boxes)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)