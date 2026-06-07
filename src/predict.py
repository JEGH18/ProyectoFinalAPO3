"""
predict.py
==========
Clasifica una imagen de manzana: predice la CALIDAD (good/regular/bad) con todos
los modelos entrenados y estima el TAMAÑO (diámetro normalizado + pequeño/
mediano/grande).

Uso:
    python src/predict.py ruta/a/imagen.jpg
"""
from __future__ import annotations
import os
import sys
import json
import argparse
import warnings
import numpy as np
from PIL import Image

warnings.filterwarnings("ignore")
import joblib

from features import extract_features
from segment import fg_mask, largest_component, measure_size

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(HERE, "models")


def _load_classical():
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))
    classes = json.load(open(os.path.join(MODELS_DIR, "labels.json")))["classes"]
    models = {}
    for key, fn in [("LogReg", "logreg.joblib"), ("SVM", "svm.joblib"),
                    ("RandomForest", "randomforest.joblib")]:
        path = os.path.join(MODELS_DIR, fn)
        if os.path.exists(path):
            models[key] = joblib.load(path)
    return scaler, models, classes


def _load_cnn(classes):
    path = os.path.join(MODELS_DIR, "cnn.pt")
    if not os.path.exists(path):
        return None
    import torch
    from train_cnn import SmallCNN
    model = SmallCNN(len(classes))
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


def _features_for(im: Image.Image):
    a = np.asarray(im.resize((128, 128)), dtype=np.float32) / 255.0
    mask = largest_component(fg_mask(a))
    return extract_features(a, mask)[None, :]


def _cnn_proba(im: Image.Image, model, classes):
    import torch
    from train_cnn import _load_masked
    # replicar el preprocesamiento de entrenamiento sobre la imagen en memoria
    a = np.asarray(im.resize((128, 128)), dtype=np.float32) / 255.0
    m = largest_component(fg_mask(a))
    out = a.copy(); out[~m] = 0.5
    x = np.asarray(Image.fromarray((out * 255).astype(np.uint8)).resize((64, 64)),
                   dtype=np.float32) / 255.0
    t = torch.tensor(x.transpose(2, 0, 1)[None, ...])
    with torch.no_grad():
        p = torch.softmax(model(t), dim=1)[0].numpy()
    return {classes[i]: float(p[i]) for i in range(len(classes))}


def predict_image(path: str) -> dict:
    scaler, models, classes = _load_classical()
    im = Image.open(path).convert("RGB")
    x = scaler.transform(_features_for(im))
    out = {"quality": {}, "size": {}}
    for name, model in models.items():
        proba = model.predict_proba(x)[0]
        order = list(model.classes_)
        probs = {c: float(proba[order.index(c)]) for c in classes}
        out["quality"][name] = {"pred": model.predict(x)[0], "proba": probs}
    cnn = _load_cnn(classes)
    if cnn is not None:
        probs = _cnn_proba(im, cnn, classes)
        out["quality"]["CNN"] = {"pred": max(probs, key=probs.get), "proba": probs}
    d, scls = measure_size(path)
    out["size"] = {"diam_norm": round(d, 3), "class": scls}
    return out


def main():
    ap = argparse.ArgumentParser(description="Clasifica calidad y tamaño de una manzana.")
    ap.add_argument("image")
    args = ap.parse_args()
    if not os.path.exists(args.image):
        sys.exit(f"No existe la imagen: {args.image}")
    r = predict_image(args.image)
    print(f"\nImagen: {args.image}")
    print("CALIDAD:")
    for name, q in r["quality"].items():
        bar = "  ".join(f"{c}={q['proba'][c]*100:5.1f}%" for c in q["proba"])
        print(f"  [{name:12s}] -> {q['pred'].upper():8s} | {bar}")
    print(f"TAMAÑO: {r['size']['class']}  (diámetro normalizado={r['size']['diam_norm']})")


if __name__ == "__main__":
    main()
