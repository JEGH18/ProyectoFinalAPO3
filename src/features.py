"""
features.py
===========
Extracción de características (feature engineering) para clasificar la CALIDAD de
una manzana. Para evitar que el modelo aprenda el FONDO de la foto en lugar de la
fruta, las características se calculan SOLO sobre los píxeles de la fruta
(primer plano), usando la máscara de segmentación de segment.py.

Idea central
------------
1. Todas las imágenes se redimensionan a un tamaño fijo (IMG_SIZE) -> elimina el
   sesgo de resolución entre clases.
2. Se segmenta la fruta y las estadísticas de color/textura/defectos se calculan
   únicamente dentro de la máscara -> el descriptor describe la fruta, no el
   fondo (césped, mano, mesa, etc.).

Vector de características (49 dimensiones)
-----------------------------------------
  * Histograma de Matiz (H)            -> 16 bins
  * Histograma de Saturación (S)       ->  8 bins
  * Histograma de Valor/brillo (V)     ->  8 bins
  * Media y desviación de R,G,B,H,S,V  -> 12 valores
  * Media y desviación del gradiente   ->  2 valores (textura/bordes)
  * Proporción de píxeles oscuros      ->  1 valor   (podredumbre/sombra)
  * Proporción de píxeles "marrones"   ->  1 valor   (manchas de pudrición)
  * Entropía del histograma de matiz   ->  1 valor
                                          ----
                                           49 características
"""
from __future__ import annotations
import numpy as np
from PIL import Image
from matplotlib.colors import rgb_to_hsv

IMG_SIZE = 128


def load_image(path: str, size: int = IMG_SIZE) -> np.ndarray:
    im = Image.open(path).convert("RGB").resize((size, size))
    return np.asarray(im, dtype=np.float32) / 255.0


def _hist(x: np.ndarray, bins: int, rng=(0.0, 1.0)) -> np.ndarray:
    """Histograma NORMALIZADO (suma 1) -> invariante al número de píxeles."""
    h, _ = np.histogram(x, bins=bins, range=rng)
    h = h.astype(np.float64)
    total = h.sum()
    return h / total if total > 0 else h


def feature_names() -> list[str]:
    names = [f"hist_H{i}" for i in range(16)]
    names += [f"hist_S{i}" for i in range(8)]
    names += [f"hist_V{i}" for i in range(8)]
    for c in ["R", "G", "B", "H", "S", "V"]:
        names += [f"{c}_mean", f"{c}_std"]
    names += ["grad_mean", "grad_std", "dark_ratio", "brown_ratio", "hue_entropy"]
    return names


def extract_features(img: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Vector de 49 características de una imagen RGB en [0,1].
    Si se entrega una máscara binaria (fruta=True) del mismo tamaño, las
    estadísticas se calculan solo sobre la fruta; si no, sobre toda la imagen.
    """
    hsv = rgb_to_hsv(img)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    R, G, B = img[..., 0], img[..., 1], img[..., 2]

    # Máscara de primer plano: si no hay o es muy pequeña, se usa toda la imagen.
    if mask is None or mask.sum() < 0.02 * mask.size:
        sel = np.ones(H.shape, dtype=bool)
    else:
        sel = mask.astype(bool)

    Hs, Ss, Vs = H[sel], S[sel], V[sel]
    Rs, Gs, Bs = R[sel], G[sel], B[sel]

    feats: list[float] = []
    # --- Histogramas de color (solo fruta) ---
    hist_H = _hist(Hs, 16)
    feats += list(hist_H)
    feats += list(_hist(Ss, 8))
    feats += list(_hist(Vs, 8))
    # --- Estadísticos por canal (solo fruta) ---
    for arr in (Rs, Gs, Bs, Hs, Ss, Vs):
        feats += [float(arr.mean()), float(arr.std())]
    # --- Textura: gradiente (calculado en toda la imagen, leído en la fruta) ---
    gray = 0.299 * R + 0.587 * G + 0.114 * B
    gy, gx = np.gradient(gray)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    mag_sel = mag[sel]
    feats += [float(mag_sel.mean()), float(mag_sel.std())]
    # --- Indicadores de defecto (solo fruta) ---
    dark_ratio = float((Vs < 0.35).mean())
    brown = ((Hs >= 0.04) & (Hs <= 0.13) & (Ss > 0.2) & (Vs < 0.6))
    brown_ratio = float(brown.mean())
    feats += [dark_ratio, brown_ratio]
    # --- Entropía del histograma de matiz ---
    p = hist_H[hist_H > 0]
    feats += [float(-(p * np.log2(p)).sum())]
    return np.asarray(feats, dtype=np.float64)
