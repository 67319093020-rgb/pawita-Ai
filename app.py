from flask import Flask, request, jsonify
from flask_cors import CORS
import os, cv2, pickle, numpy as np
from utils import generate_filename, count_images
from train import train_model

app = Flask(__name__)
CORS(app)

# ---------------- Path Config ----------------
DATA_CAT = "dataset/DATA_CAT"
DATA_DOG = "dataset/DATA_DOG"
MODEL_PATH = "models/animal_classifier.pkl"

os.makedirs(DATA_CAT, exist_ok=True)
os.makedirs(DATA_DOG, exist_ok=True)
os.makedirs("models", exist_ok=True)

MIN_IMAGES = 20
# --------------------------------------------

# ---------------- Upload --------------------
@app.route("/upload/<label>", methods=["POST"])
def upload(label):
    if label not in ["cat", "dog"]:
        return jsonify({"error": "label ต้องเป็น cat หรือ dog"}), 400

    name = request.form.get("name")
    if not name:
        return jsonify({"error": "กรุณาระบุชื่อสัตว์ (name)"}), 400

    files = request.files.getlist("images")
    if not files:
        return jsonify({"error": "ไม่พบไฟล์"}), 400

    # สร้างโฟลเดอร์ย่อยตามชื่อ
    folder_base = DATA_CAT if label == "cat" else DATA_DOG
    folder = os.path.join(folder_base, name)
    os.makedirs(folder, exist_ok=True)

    start_idx = count_images(folder) + 1
    saved = []

    for i, f in enumerate(files, start=start_idx):
        fname = f"{name}_{i}.jpg"
        path = os.path.join(folder, fname)
        f.save(path)
        saved.append(fname)

    return jsonify({
        "ok": True,
        "saved": saved,
        "count": count_images(folder)
    })
# --------------------------------------------

# ---------------- Train ---------------------
@app.route("/train", methods=["POST"])
def train():
    n_cat = count_images(DATA_CAT)
    n_dog = count_images(DATA_DOG)
    if n_cat < MIN_IMAGES or n_dog < MIN_IMAGES:
        return jsonify({
            "error": f"ต้องมีรูปแมวและหมาอย่างน้อย {MIN_IMAGES} รูป",
            "count_cat": n_cat,
            "count_dog": n_dog
        }), 400

    try:
        metrics = train_model(DATA_CAT, DATA_DOG, MODEL_PATH)
        return jsonify({"ok": True, "metrics": metrics})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400
# --------------------------------------------

# ---------------- Predict -------------------
@app.route("/predict", methods=["POST"])
def predict():
    if not os.path.exists(MODEL_PATH):
        return jsonify({"error": "ยังไม่มีโมเดล"}), 400

    file = request.files.get("image")
    if not file:
        return jsonify({"error": "ไม่พบไฟล์"}), 400

    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return jsonify({"error": "โหลดภาพไม่สำเร็จ"}), 400

    img = cv2.resize(img, (200, 200)).flatten().reshape(1, -1)

    with open(MODEL_PATH, "rb") as f:
        data = pickle.load(f)
        model = data["model"]
        classes = data["classes"]

    probs = model.predict_proba(img)[0]
    pred_idx = int(np.argmax(probs))
    pred = classes[pred_idx]   # ✅ จะคืน "hum" หรือ "mee"
    confidence = float(probs[pred_idx])

    return jsonify({
        "ok": True,
        "prediction": pred,
        "confidence": confidence
    })


# ---------------- Status --------------------
@app.route("/status", methods=["GET"])
def status():
    return jsonify({
        "counts": {
            "cat": count_images(DATA_CAT),
            "dog": count_images(DATA_DOG)
        },
        "min_required": MIN_IMAGES,
        "model_exists": os.path.exists(MODEL_PATH)
    })
# --------------------------------------------

if __name__ == "__main__":
    app.run(host="10.15.2.16", port=5000, debug=True)
