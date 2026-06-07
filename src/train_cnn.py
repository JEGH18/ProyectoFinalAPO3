"""
train_cnn.py
============
Modelo de DEEP LEARNING exigido por el enunciado: una red neuronal convolucional
(CNN) pequeña entrenada desde cero para clasificar la calidad de la manzana en
{good, regular, bad}.

Para que la comparación con los modelos clásicos sea justa, la CNN recibe las
mismas imágenes con el FONDO enmascarado (solo la fruta), redimensionadas a
64x64. Usa la MISMA partición estratificada (semilla 42) que train.py.

Arquitectura: 3 bloques convolucionales (Conv-ReLU-MaxPool) + 2 capas densas,
con Dropout. Pérdida: entropía cruzada (con pesos por clase para el desbalance).
Optimizador: Adam.

Salidas: models/cnn.pt (pesos), entrada en results/metrics.json -> "CNN".
"""
from __future__ import annotations
import os
import json
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                             precision_recall_fscore_support, classification_report)

from dataset import _sources, _list_images, CLASSES
from segment import fg_mask, largest_component

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(HERE, "models")
RESULTS_DIR = os.path.join(HERE, "results")
SEED = 42
IMG = 64
torch.manual_seed(SEED)
np.random.seed(SEED)


def _load_masked(path: str) -> np.ndarray:
    """Imagen 64x64 con el fondo en gris neutro (solo fruta)."""
    im = Image.open(path).convert("RGB").resize((128, 128))
    a = np.asarray(im, dtype=np.float32) / 255.0
    m = largest_component(fg_mask(a))
    out = a.copy()
    out[~m] = 0.5                      # fondo neutro
    out = np.asarray(Image.fromarray((out * 255).astype(np.uint8)).resize((IMG, IMG)),
                     dtype=np.float32) / 255.0
    return out


def build_arrays():
    paths, labels = [], []
    for label, _origen, folder in _sources():
        if os.path.isdir(folder):
            for p in _list_images(folder):
                paths.append(p); labels.append(label)
    y = np.array([CLASSES.index(l) for l in labels])
    print(f"Cargando y enmascarando {len(paths)} imágenes para la CNN...")
    X = np.stack([_load_masked(p) for p in paths]).transpose(0, 3, 1, 2)  # N,C,H,W
    return X.astype(np.float32), y


class SmallCNN(nn.Module):
    def __init__(self, n_classes=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 32
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 16
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.4),
            nn.Linear(64 * 8 * 8, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def _augment(xb):
    """Aumento simple en batch: flip horizontal aleatorio."""
    if torch.rand(1).item() < 0.5:
        xb = torch.flip(xb, dims=[3])
    return xb


def main(epochs: int = 25, batch: int = 64):
    os.makedirs(MODELS_DIR, exist_ok=True)
    X, y = build_arrays()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.30, stratify=y,
                                          random_state=SEED)
    dev = torch.device("cpu")
    Xtr_t = torch.tensor(Xtr); ytr_t = torch.tensor(ytr)
    Xte_t = torch.tensor(Xte).to(dev); yte_t = torch.tensor(yte).to(dev)
    dl = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=batch, shuffle=True)

    # pesos por clase (desbalance de 'regular')
    counts = np.bincount(ytr, minlength=len(CLASSES))
    w = torch.tensor((counts.sum() / (len(CLASSES) * counts)), dtype=torch.float32)
    model = SmallCNN(len(CLASSES)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss(weight=w.to(dev))

    print("Entrenando CNN...")
    for ep in range(1, epochs + 1):
        model.train(); tot = 0.0
        for xb, yb in dl:
            xb = _augment(xb).to(dev); yb = yb.to(dev)
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward(); opt.step()
            tot += loss.item() * len(xb)
        if ep % 5 == 0 or ep == 1:
            model.eval()
            with torch.no_grad():
                acc = (model(Xte_t).argmax(1) == yte_t).float().mean().item()
            print(f"  época {ep:2d}  loss={tot/len(Xtr):.4f}  test_acc={acc:.4f}")

    # evaluación final
    model.eval()
    with torch.no_grad():
        pred = model(Xte_t).argmax(1).cpu().numpy()
    acc = accuracy_score(yte, pred); f1m = f1_score(yte, pred, average="macro")
    p, r, f, _ = precision_recall_fscore_support(yte, pred,
                                                 labels=range(len(CLASSES)), zero_division=0)
    cm = confusion_matrix(yte, pred, labels=range(len(CLASSES)))
    print(f"\nCNN  test acc={acc:.4f}  f1-macro={f1m:.4f}")
    print(classification_report(yte, pred, target_names=CLASSES, zero_division=0))

    torch.save(model.state_dict(), os.path.join(MODELS_DIR, "cnn.pt"))
    metrics = json.load(open(os.path.join(RESULTS_DIR, "metrics.json")))
    metrics["models"]["CNN"] = {
        "architecture": "3xConv(16,32,64)+2FC, 64x64, Adam, CE pesada",
        "epochs": epochs,
        "test_accuracy": float(acc), "test_f1_macro": float(f1m),
        "per_class": {CLASSES[i]: {"precision": float(p[i]), "recall": float(r[i]),
                                   "f1": float(f[i])} for i in range(len(CLASSES))},
        "confusion_matrix": cm.tolist(),
    }
    # La CNN no se evalúa por validación cruzada, por lo que NO se usa para
    # reelegir el "mejor modelo" (que se elige por CV entre los clásicos en
    # train.py). Solo se registran sus métricas para la comparación.
    json.dump(metrics, open(os.path.join(RESULTS_DIR, "metrics.json"), "w"), indent=2)
    print(f"Guardado models/cnn.pt (best_model se mantiene: {metrics.get('best_model')}).")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=25)
    main(epochs=ap.parse_args().epochs)
