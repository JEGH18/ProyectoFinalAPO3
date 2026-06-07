"""
prepare_data.py
===============
Prepara el conjunto de entrenamiento de calidad en TRES clases (good / regular /
bad) a partir del dataset oficial de Kaggle ya extraído en data/Apple/:

  * good  <- data/Apple/good   (Apple_Good de Kaggle, manzana en buen estado)
  * bad   <- data/Apple/bad    (Apple_Bad de Kaggle, manzana podrida)
  * regular <- se GENERA segmentando individualmente cada fruta de las imágenes
              de data/Apple/medium (carpeta "Mixed Quality" de Kaggle, que trae
              varias frutas por foto), tal como indica el enunciado. Como la
              clase queda pequeña, se AUMENTA (flips, rotaciones, brillo) hasta
              un objetivo para reducir el desbalance.

También crea las carpetas data/own/{good,regular,bad} para que el estudiante
deposite sus 30–50 fotos propias (se integran automáticamente al entrenar).
"""
from __future__ import annotations
import os
import glob
import random
import shutil
import numpy as np
from PIL import Image, ImageEnhance

from segment import extract_individual_crops

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APPLE = os.path.join(HERE, "data", "Apple")
OWN = os.path.join(HERE, "data", "own")
REGULAR_TARGET = 600     # tamaño objetivo de la clase 'regular' tras aumento
SEED = 42


def _augment(im: Image.Image, k: int) -> Image.Image:
    """Genera una variante aumentada determinista según k."""
    rnd = random.Random(k)
    if rnd.random() < 0.5:
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
    if rnd.random() < 0.3:
        im = im.transpose(Image.FLIP_TOP_BOTTOM)
    im = im.rotate(rnd.choice([0, 90, 180, 270, 15, -15, 30, -30]), expand=False)
    im = ImageEnhance.Brightness(im).enhance(rnd.uniform(0.8, 1.2))
    im = ImageEnhance.Color(im).enhance(rnd.uniform(0.85, 1.15))
    return im


def build_regular():
    """Crea data/Apple/regular con recortes individuales de la carpeta Mixed
    + aumento de datos hasta REGULAR_TARGET."""
    src = os.path.join(APPLE, "medium")
    dst = os.path.join(APPLE, "regular")
    if not os.path.isdir(src):
        print(f"  (no existe {src}; ¿ya se preparó?)")
        return
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    os.makedirs(dst, exist_ok=True)

    # 1) recortes individuales reales
    base = []
    for f in sorted(glob.glob(os.path.join(src, "*"))):
        try:
            for j, crop in enumerate(extract_individual_crops(f)):
                name = f"reg_{os.path.splitext(os.path.basename(f))[0]}_{j}.jpg"
                crop.convert("RGB").save(os.path.join(dst, name), quality=92)
                base.append(name)
        except Exception as e:
            print(f"    aviso: {f}: {e}")
    print(f"  recortes individuales reales: {len(base)}")

    # 2) aumento de datos hasta el objetivo
    k = 0
    while len(os.listdir(dst)) < REGULAR_TARGET and base:
        src_name = base[k % len(base)]
        im = Image.open(os.path.join(dst, src_name)).convert("RGB")
        aug = _augment(im, k)
        aug.save(os.path.join(dst, f"aug_{k}.jpg"), quality=92)
        k += 1
    print(f"  total clase regular (con aumento): {len(os.listdir(dst))}")

    # quitar la carpeta 'medium' original para dejar solo good/regular/bad
    shutil.rmtree(src)


def make_own_dirs():
    for c in ["good", "regular", "bad"]:
        d = os.path.join(OWN, c)
        os.makedirs(d, exist_ok=True)
        keep = os.path.join(d, ".gitkeep")
        if not os.path.exists(keep):
            open(keep, "w").close()
    readme = os.path.join(OWN, "LEEME.txt")
    with open(readme, "w") as f:
        f.write(
            "Coloca aquí tus 30-50 fotos propias de manzana (enunciado 2.2).\n"
            "Fondo simple y uniforme, una sola manzana centrada, buena luz.\n\n"
            "  data/own/good/      -> manzanas sanas\n"
            "  data/own/regular/   -> manzanas con golpes/manchas leves\n"
            "  data/own/bad/       -> manzanas dañadas (opcional)\n\n"
            "Se integran automáticamente al ejecutar src/train.py.\n")


def main():
    random.seed(SEED)
    print("Preparando clase 'regular' desde la carpeta Mixed...")
    build_regular()
    make_own_dirs()
    print("\nConteo final por clase:")
    for c in ["good", "regular", "bad"]:
        d = os.path.join(APPLE, c)
        n = len(glob.glob(os.path.join(d, "*"))) if os.path.isdir(d) else 0
        print(f"  {c:8s}: {n}")
    print("Carpetas para fotos propias creadas en data/own/")


if __name__ == "__main__":
    main()
