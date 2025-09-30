import os, cv2, pickle, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

IMG_SIZE = (200, 200)

def _load_images_from_folder(base_folder):
    X, y = [], []
    if not os.path.exists(base_folder):
        return X, y

    # วนทุกโฟลเดอร์ย่อย เช่น hum, boom, mee
    for name in os.listdir(base_folder):
        subfolder = os.path.join(base_folder, name)
        if not os.path.isdir(subfolder):
            continue

        for fname in os.listdir(subfolder):
            path = os.path.join(subfolder, fname)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            try:
                img = cv2.resize(img, IMG_SIZE)
                X.append(img.flatten())
                y.append(name)  # ✅ ใช้ชื่อโฟลเดอร์เป็น label
            except Exception:
                continue
    return X, y

def train_model(cat_dir, dog_dir, model_path):
    X_cat, y_cat = _load_images_from_folder(cat_dir)
    X_dog, y_dog = _load_images_from_folder(dog_dir)

    X = np.array(X_cat + X_dog, dtype=np.float32)
    y = np.array(y_cat + y_dog)

    if len(set(y)) < 2:
        raise ValueError("ต้องมีอย่างน้อย 2 class")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = SVC(kernel="linear", probability=True, class_weight="balanced")
    model.fit(X_train, y_train)

    acc = model.score(X_test, y_test)

    with open(model_path, "wb") as f:
        pickle.dump({
            "model": model,
            "classes": model.classes_.tolist()  # เช่น ["hum", "mee", "boom"]
        }, f)

    return {
        "accuracy": float(acc),
        "n_classes": len(set(y)),
        "classes": list(set(y))
    }
