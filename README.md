# Clasificación de calidad y tamaño de manzanas

Sistema de visión por computador que, a partir de la foto de una manzana, predice
su **calidad** (buena / regular / mala) y estima su **tamaño** (diámetro
normalizado y clase pequeño/mediano/grande).

Proyecto final de *Algoritmos y Programación III* (Universidad ICESI, 2026-1).

## Resultados (conjunto de prueba, n = 867)

| Modelo               | Exactitud | F1-macro | F1-macro CV |
|----------------------|:---------:|:--------:|:-----------:|
| Regresión logística  | 0.848     | 0.857    | 0.899       |
| **SVM (RBF)**        | **0.957** | **0.958**| **0.970**   |
| Random Forest        | 0.949     | 0.950    | 0.954       |
| CNN (3 conv.)        | 0.946     | 0.947    | —           |

El mejor modelo se selecciona automáticamente por F1-macro (SVM).

## Datos

Se usa el dataset público **Fruit Quality Classification** de Kaggle
(ryandpark), subconjunto de manzanas:
<https://www.kaggle.com/datasets/ryandpark/fruit-quality-classification>

- `good` (1149, *Apple_Good*) y `bad` (1141, *Apple_Bad*).
- `regular` (600): se obtiene **segmentando individualmente** cada fruta de la
  carpeta *Mixed Quality* (177 recortes) y aumentando los datos.
- `data/own/{good,regular,bad}/`: aquí van las **fotos propias** (30–50, sobre
  fondo simple). Se integran automáticamente al entrenar.

> Las características de calidad se calculan **solo sobre la fruta segmentada**,
> de modo que el modelo aprende la calidad y no el fondo de la foto.

## Estructura

```
proyectoAPo3FInal/
├── data/Apple/{good,regular,bad}/   imágenes del dataset
├── data/own/{good,regular,bad}/     fotos propias del estudiante
├── src/
│   ├── features.py      49 características de color/textura/defectos (solo fruta)
│   ├── segment.py       segmentación de la fruta y estimación de tamaño
│   ├── prepare_data.py  genera la clase 'regular' desde la carpeta Mixed
│   ├── dataset.py       arma X,y y cachea results/features.csv
│   ├── train.py         entrena LogReg, SVM y Random Forest (rejilla + CV)
│   ├── train_cnn.py     entrena la CNN (PyTorch)
│   ├── predict.py       clasifica calidad y estima tamaño de una imagen
│   └── app.py           interfaz web (Streamlit) con cámara
├── models/              modelos entrenados (.joblib, .pt) + scaler
├── results/             features.csv, metrics.json, figures/ (PDF vectorial)
├── docs/                informe_IEEE.tex/.pdf, figures/, video_guion.md
├── requirements.txt
└── README.md
```

## Reproducir

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python src/prepare_data.py     # genera la clase 'regular' (una sola vez)
python src/train.py --rebuild  # extrae características y entrena los 3 clásicos
python src/train_cnn.py        # entrena la CNN
python src/evaluate.py         # genera las figuras del informe

python src/predict.py data/Apple/bad/<archivo>.jpg   # predicción por consola
streamlit run src/app.py                              # interfaz gráfica
```

Los modelos ya entrenados están en `models/`, por lo que `predict.py` y la app
funcionan sin reentrenar.

## Metodología (CRISP-DM)

Comprensión del negocio → comprensión de los datos → preparación (segmentación +
características) → modelado (4 modelos, ajuste de hiperparámetros) → evaluación
(F1-macro, matrices de confusión, análisis de errores) → despliegue (app).

## Créditos

Dataset: *Fruit Quality Classification*, R. Park, Kaggle. Bibliotecas:
scikit-learn, PyTorch, NumPy, Matplotlib, Pillow, SciPy, Streamlit.
