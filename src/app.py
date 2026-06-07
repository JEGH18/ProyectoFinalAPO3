"""
app.py - Interfaz gráfica de despliegue (Streamlit)
===================================================
Cumple el formato de despliegue del enunciado: permite CARGAR una imagen o
CAPTURARLA con la cámara en tiempo real, y muestra la predicción de CALIDAD
(con los modelos entrenados) y la estimación de TAMAÑO, además de la
segmentación de la fruta.

Ejecutar:
    streamlit run src/app.py
"""
from __future__ import annotations
import os
import sys
import warnings
import numpy as np
import streamlit as st
from PIL import Image

warnings.filterwarnings("ignore")
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from segment import fg_mask, largest_component, measure_size  # noqa: E402
from predict import (_load_classical, _load_cnn, _features_for,  # noqa: E402
                     _cnn_proba)

st.set_page_config(page_title="Calidad y tamaño de manzanas", page_icon="🍎")
NICE = {"good": "Buena", "regular": "Regular", "bad": "Mala"}
EMOJI = {"good": "🟢", "regular": "🟠", "bad": "🔴"}


@st.cache_resource
def load_all():
    scaler, models, classes = _load_classical()
    cnn = _load_cnn(classes)
    best = open(os.path.join(os.path.dirname(__file__), "..", "models",
                             "best_model.txt")).read().strip()
    return scaler, models, classes, cnn, best


def predict(im, scaler, models, classes, cnn):
    x = scaler.transform(_features_for(im))
    out = {}
    for name, model in models.items():
        proba = model.predict_proba(x)[0]
        order = list(model.classes_)
        out[name] = {c: float(proba[order.index(c)]) for c in classes}
        out[name]["__pred__"] = model.predict(x)[0]
    if cnn is not None:
        p = _cnn_proba(im, cnn, classes)
        p["__pred__"] = max((c for c in classes), key=lambda c: p[c])
        out["CNN"] = p
    return out


def overlay(im):
    a = np.asarray(im.resize((256, 256)), dtype=np.float32) / 255.0
    m = largest_component(fg_mask(a))
    o = a.copy(); o[~m] *= 0.3
    return Image.fromarray((o * 255).astype(np.uint8))


st.title("🍎 Calidad y tamaño de manzanas")
st.caption("Proyecto Final – Algoritmos y Programación III · Visión por computador")

scaler, models, classes, cnn, best = load_all()

src = st.radio("Fuente de la imagen:", ["Subir archivo", "Cámara"], horizontal=True)
img = None
if src == "Subir archivo":
    f = st.file_uploader("Sube una foto de una manzana (fondo simple)",
                         type=["jpg", "jpeg", "png", "webp"])
    if f:
        img = Image.open(f).convert("RGB")
else:
    cam = st.camera_input("Toma una foto de la manzana")
    if cam:
        img = Image.open(cam).convert("RGB")

if img is not None:
    c1, c2 = st.columns(2)
    with c1:
        st.image(img, caption="Entrada", use_container_width=True)
    with c2:
        st.image(overlay(img), caption="Segmentación de la fruta",
                 use_container_width=True)

    preds = predict(img, scaler, models, classes, cnn)
    d, scls = measure_size(np.asarray(img.resize((256, 256)),
                                      dtype=np.float32) / 255.0)

    bp = preds[best]["__pred__"]
    st.subheader(f"Calidad (mejor modelo · {best}): "
                 f"{EMOJI[bp]} {NICE[bp]}")
    st.metric("Tamaño estimado", scls.capitalize(),
              help=f"Diámetro normalizado = {d:.3f}")

    st.write("**Probabilidades por modelo:**")
    for name, p in preds.items():
        cols = st.columns(len(classes) + 1)
        cols[0].markdown(f"**{name}**")
        for i, c in enumerate(classes):
            cols[i + 1].progress(p[c], text=f"{NICE[c]} {p[c]*100:.0f}%")

    st.info("El tamaño se estima segmentando la fruta y midiendo su diámetro "
            "equivalente normalizado. Para mayor precisión, usa fondo simple y "
            "uniforme.")
else:
    st.write("👆 Sube o captura una imagen para analizarla.")
