from flask import Flask, request, jsonify
import os, cv2, pickle, numpy as np

app = Flask(__name__)

# ---------------- Path Config ----------------
MODEL_CAT = "models/cat_classifier.pkl"
MODEL_DOG = "models/dog_classifier.pkl"

face_cat_path = './Library/haarcascade_frontalcatface.xml'
face_dog_path = './Library/dog_face.xml'

if not os.path.exists(face_cat_path) or not os.path.exists(face_dog_path):
    raise FileNotFoundError("Haar Cascade files not found!")

face_cat = cv2.CascadeClassifier(face_cat_path)
face_dog = cv2.CascadeClassifier(face_dog_path)
# --------------------------------------------

def load_model(label):
    model_path = MODEL_CAT if label == "cat" else MODEL_DOG
    if not os.path.exists(model_path):
        return None, f"ยังไม่มีโมเดล {label}"
    with open(model_path, "rb") as f:
        return pickle.load(f), None

def predict_image(file, label):
    # โหลดโมเดล
    model, err = load_model(label)
    if err:
        return None, None, err

    # อ่านภาพ
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None, None, "โหลดภาพไม่สำเร็จ"
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # เลือก cascade
    faces = []
    if label == "cat":
        faces = face_cat.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
    else:
        faces = face_dog.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))

    if len(faces) == 0:
        return img, None, "ไม่พบใบหน้า"

    # predict แค่ใบหน้าแรก
    (x, y, w, h) = faces[0]
    cropped_face = gray[y:y+h, x:x+w]
    resized = cv2.resize(cropped_face, (200, 200)).flatten().reshape(1, -1)

    probs = model.predict_proba(resized)[0]
    pred = int(model.predict(resized)[0])
    confidence = float(np.max(probs))

    return img, {"prediction": pred, "confidence": confidence}, None

# ---------------- API ----------------
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
        model, classes = data["model"], data["classes"]

    probs = model.predict_proba(img)[0]
    pred_idx = np.argmax(probs)
    pred = classes[pred_idx]
    confidence = float(probs[pred_idx])

    return jsonify({
        "ok": True,
        "prediction": pred,
        "confidence": confidence
    })
