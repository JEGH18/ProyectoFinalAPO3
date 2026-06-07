"""
dataset.py
==========
Construye la matriz de características X y las etiquetas a partir de las carpetas
de imágenes. Combina el dataset oficial (data/Apple/{good,regular,bad}) con las
fotos propias del estudiante (data/own/{good,regular,bad}).

Para cada imagen se guarda:
  * 49 características de calidad (color/textura/defectos) -> features.py
  * el diámetro normalizado y la clase de tamaño          -> segment.py
  * el origen (kaggle / propio)

Resultado cacheado en results/features.csv.
"""
from __future__ import annotations
import os
import glob
import numpy as np
import pandas as pd
from PIL import Image

from features import extract_features, feature_names
from segment import normalized_diameter, size_class, fg_mask, largest_component

CLASSES = ["good", "regular", "bad"]
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APPLE_DIR = os.path.join(HERE, "data", "Apple")
OWN_DIR = os.path.join(HERE, "data", "own")
RESULTS_DIR = os.path.join(HERE, "results")
CACHE_CSV = os.path.join(RESULTS_DIR, "features.csv")

_IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
_IMG_SIZE = 128


def _list_images(folder: str) -> list[str]:
    files: list[str] = []
    for ext in _IMG_EXT:
        files += glob.glob(os.path.join(folder, f"*{ext}"))
        files += glob.glob(os.path.join(folder, f"*{ext.upper()}"))
    return sorted(set(files))


def _sources():
    """(label, origen, carpeta) para cada clase y origen disponible."""
    out = []
    for label in CLASSES:
        out.append((label, "kaggle", os.path.join(APPLE_DIR, label)))
        out.append((label, "propio", os.path.join(OWN_DIR, label)))
    return out


def build_features(verbose: bool = True) -> pd.DataFrame:
    rows, meta = [], []
    for label, origen, folder in _sources():
        if not os.path.isdir(folder):
            continue
        imgs = _list_images(folder)
        if verbose and imgs:
            print(f"  [{label}/{origen}] {len(imgs)} imágenes")
        for p in imgs:
            try:
                im = Image.open(p).convert("RGB")
                arr128 = np.asarray(im.resize((_IMG_SIZE, _IMG_SIZE)),
                                    dtype=np.float32) / 255.0
                mask = largest_component(fg_mask(arr128))   # fruta vs fondo
                feats = extract_features(arr128, mask)       # features solo fruta
                arr256 = np.asarray(im.resize((256, 256)),
                                    dtype=np.float32) / 255.0
                d = normalized_diameter(arr256)
                rows.append(feats)
                meta.append((label, origen, os.path.relpath(p, HERE),
                             d, size_class(d)))
            except Exception as e:
                if verbose:
                    print(f"    aviso: {p}: {e}")
    X = np.vstack(rows)
    df = pd.DataFrame(X, columns=feature_names())
    df.insert(0, "label", [m[0] for m in meta])
    df.insert(1, "origen", [m[1] for m in meta])
    df.insert(2, "path", [m[2] for m in meta])
    df["diam_norm"] = [m[3] for m in meta]
    df["size_class"] = [m[4] for m in meta]
    return df


def load_dataset(rebuild: bool = False, verbose: bool = True):
    """Devuelve (X, y, df). Usa cache CSV salvo rebuild=True."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if (not rebuild) and os.path.exists(CACHE_CSV):
        df = pd.read_csv(CACHE_CSV)
        if verbose:
            print(f"Cargado cache: {CACHE_CSV} ({len(df)} muestras)")
    else:
        if verbose:
            print("Extrayendo características e info de tamaño...")
        df = build_features(verbose=verbose)
        df.to_csv(CACHE_CSV, index=False)
        if verbose:
            print(f"Guardado: {CACHE_CSV} ({len(df)} muestras)")
    X = df[feature_names()].to_numpy(dtype=np.float64)
    y = df["label"].to_numpy()
    return X, y, df


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()
    X, y, df = load_dataset(rebuild=a.rebuild)
    print("\nPor clase:\n", df["label"].value_counts())
    print("\nPor origen:\n", df["origen"].value_counts())
    print("\nTamaño:\n", df["size_class"].value_counts())
    print("X:", X.shape)
