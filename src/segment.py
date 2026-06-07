"""
segment.py
==========
Segmentación de la fruta respecto a un fondo simple y estimación de su TAMAÑO.

El enunciado pide como segunda salida una "estimación de tamaño (pequeño,
mediano, grande) o diámetro en píxeles normalizados". Aquí se implementa:

  * fg_mask(img)            -> máscara binaria fruta/fondo (Otsu sobre la
                              distancia al color de fondo estimado en los bordes).
  * largest_component(mask) -> componente conexa más grande (una fruta).
  * normalized_diameter(img)-> diámetro equivalente de la fruta dividido por el
                              lado de la imagen  (invariante a la resolución).
  * size_class(d)           -> clasifica el diámetro normalizado en
                              {pequeño, mediano, grande} según umbrales.
  * extract_individual_crops(img) -> recortes individuales de cada fruta de una
                              imagen con varias (se usa para la carpeta Mixed).

No requiere OpenCV: usa numpy + scipy.ndimage.
"""
from __future__ import annotations
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

# Umbrales de tamaño sobre el diámetro normalizado (fracción del lado de la
# imagen). Se calibran con la distribución del dataset (ver size_thresholds()).
SIZE_SMALL_MAX = 0.45
SIZE_MEDIUM_MAX = 0.62
SIZE_LABELS = ["pequeño", "mediano", "grande"]


def _otsu(values01: np.ndarray) -> float:
    """Umbral de Otsu sobre un arreglo en [0,1]."""
    d = (np.clip(values01, 0, 1) * 255).astype(np.uint8)
    hist = np.bincount(d.ravel(), minlength=256).astype(float)
    p = hist / hist.sum()
    omega = np.cumsum(p)
    mu = np.cumsum(p * np.arange(256))
    sigma_b = (mu[-1] * omega - mu) ** 2 / (omega * (1 - omega) + 1e-12)
    return float(np.argmax(sigma_b)) / 255.0


def fg_mask(img: np.ndarray) -> np.ndarray:
    """Máscara binaria de la fruta. img: HxWx3 float [0,1]."""
    h, w, _ = img.shape
    b = max(4, h // 40)
    border = np.concatenate([
        img[:b].reshape(-1, 3), img[-b:].reshape(-1, 3),
        img[:, :b].reshape(-1, 3), img[:, -b:].reshape(-1, 3)])
    bg = np.median(border, axis=0)               # color de fondo estimado
    dist = np.sqrt(((img - bg) ** 2).sum(axis=2))  # distancia al fondo
    t = max(_otsu(dist), 0.12)
    m = dist > t
    m = ndi.binary_opening(m, iterations=1)
    m = ndi.binary_closing(m, iterations=2)
    m = ndi.binary_fill_holes(m)
    return m


def largest_component(mask: np.ndarray) -> np.ndarray:
    """Devuelve la máscara con solo la componente conexa más grande."""
    lbl, n = ndi.label(mask)
    if n == 0:
        return mask
    sizes = ndi.sum(np.ones_like(lbl), lbl, index=range(1, n + 1))
    return lbl == (int(np.argmax(sizes)) + 1)


def normalized_diameter(img: np.ndarray) -> float:
    """Diámetro equivalente de la fruta / lado de la imagen (en [0,1]).
    Se usa el diámetro de un círculo con la misma área que la fruta:
        d_eq = 2*sqrt(area/pi),  normalizado por el lado de la imagen.
    """
    m = largest_component(fg_mask(img))
    area = float(m.sum())
    if area <= 0:
        return 0.0
    d_eq = 2.0 * np.sqrt(area / np.pi)
    side = (img.shape[0] * img.shape[1]) ** 0.5
    return float(d_eq / side)


def size_class(d_norm: float) -> str:
    """Clasifica el diámetro normalizado en pequeño/mediano/grande."""
    if d_norm < SIZE_SMALL_MAX:
        return SIZE_LABELS[0]
    if d_norm < SIZE_MEDIUM_MAX:
        return SIZE_LABELS[1]
    return SIZE_LABELS[2]


def measure_size(path_or_img, size: int = 256):
    """Devuelve (diámetro_normalizado, clase_tamaño) para una imagen."""
    if isinstance(path_or_img, str):
        img = np.asarray(Image.open(path_or_img).convert("RGB").resize((size, size)),
                         dtype=np.float32) / 255.0
    else:
        img = path_or_img
    d = normalized_diameter(img)
    return d, size_class(d)


def extract_individual_crops(path: str, out_size: int = 256):
    """Recorta cada fruta individual de una imagen con varias (carpeta Mixed).
    Devuelve una lista de imágenes PIL. Filtra componentes por área, relación de
    aspecto y solidez para quedarse con frutas individuales razonablemente
    redondas."""
    im = Image.open(path).convert("RGB")
    W, H = im.size
    a = np.asarray(im.resize((out_size, out_size)), dtype=np.float32) / 255.0
    m = fg_mask(a)
    lbl, n = ndi.label(m)
    crops = []
    sx, sy = W / out_size, H / out_size
    for i in range(1, n + 1):
        ys, xs = np.where(lbl == i)
        area = len(ys)
        frac = area / (out_size * out_size)
        if frac < 0.01 or frac > 0.6:
            continue
        y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
        hh, ww = y1 - y0, x1 - x0
        if hh < 12 or ww < 12:
            continue
        ar = ww / max(hh, 1)
        if ar < 0.5 or ar > 2.0:          # aproximadamente redonda
            continue
        if area / (hh * ww) < 0.55:       # bastante sólida (no ramas/hojas)
            continue
        crops.append(im.crop((int(x0 * sx), int(y0 * sy),
                              int(x1 * sx), int(y1 * sy))))
    return crops
