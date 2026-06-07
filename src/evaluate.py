"""
evaluate.py
===========
Genera TODAS las figuras del informe en formato VECTORIAL (PDF), como exige el
enunciado ("gráficos de calidad vectorial"). Guarda en results/figures/ y copia
a docs/figures/ para el documento LaTeX.

Figuras:
  fig_class_distribution.pdf  distribución de clases de calidad
  fig_size_distribution.pdf   distribución del tamaño (diámetro normalizado)
  fig_segmentation.pdf        ejemplos de segmentación de la fruta
  fig_pca.pdf                 proyección PCA del espacio de características
  fig_confusions.pdf          matrices de confusión de los 4 modelos
  fig_model_comparison.pdf    accuracy y F1-macro de los 4 modelos
  fig_feature_importance.pdf  importancia de características (Random Forest)
  fig_examples.pdf            ejemplos de predicción (calidad + tamaño)
  fig_crispdm.pdf             diagrama de flujo CRISP-DM personalizado
"""
from __future__ import annotations
import os
import json
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from PIL import Image

warnings.filterwarnings("ignore")
from sklearn.decomposition import PCA
import joblib

from dataset import load_dataset, CLASSES, APPLE_DIR
from features import feature_names
from segment import fg_mask, largest_component, measure_size

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIRS = [os.path.join(HERE, "results", "figures"), os.path.join(HERE, "docs", "figures")]
MODELS_DIR = os.path.join(HERE, "models")
SEED = 42
COL = {"good": "#2ca02c", "regular": "#ff7f0e", "bad": "#d62728"}
NICE = {"good": "Buena", "regular": "Regular", "bad": "Mala"}


def _save(fig, name):
    for d in FIG_DIRS:
        os.makedirs(d, exist_ok=True)
        fig.savefig(os.path.join(d, name), bbox_inches="tight")  # PDF vectorial
    plt.close(fig)
    print("  ->", name)


def fig_class_distribution(df):
    counts = df["label"].value_counts().reindex(CLASSES)
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.bar([NICE[c] for c in CLASSES], counts.values, color=[COL[c] for c in CLASSES])
    for i, v in enumerate(counts.values):
        ax.text(i, v + 8, str(int(v)), ha="center")
    ax.set_ylabel("N° de imágenes"); ax.set_title("Distribución de clases de calidad")
    _save(fig, "fig_class_distribution.pdf")


def fig_size_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    axes[0].hist(df["diam_norm"], bins=30, color="#1f77b4")
    axes[0].set_xlabel("Diámetro normalizado"); axes[0].set_ylabel("Frecuencia")
    axes[0].set_title("Tamaño medido (segmentación)")
    sc = df["size_class"].value_counts().reindex(["pequeño", "mediano", "grande"])
    axes[1].bar(sc.index, sc.values, color=["#9ecae1", "#4292c6", "#08519c"])
    for i, v in enumerate(sc.values):
        axes[1].text(i, v + 8, str(int(v)), ha="center")
    axes[1].set_title("Clases de tamaño")
    _save(fig, "fig_size_distribution.pdf")


def fig_segmentation():
    import glob, random
    random.seed(4)
    fig, axes = plt.subplots(2, 4, figsize=(8, 4.2))
    cols = []
    for c in CLASSES:
        cols += random.sample(glob.glob(os.path.join(APPLE_DIR, c, "*")), 2)
    sel = random.sample(cols, 4)
    for k, f in enumerate(sel):
        a = np.asarray(Image.open(f).convert("RGB").resize((128, 128)),
                       dtype=np.float32) / 255.0
        m = largest_component(fg_mask(a))
        o = a.copy(); o[~m] *= 0.25
        d, scls = measure_size(np.asarray(Image.open(f).convert("RGB").resize((256, 256)),
                                          dtype=np.float32) / 255.0)
        axes[0, k].imshow(a); axes[0, k].set_title("original", fontsize=8)
        axes[1, k].imshow(o); axes[1, k].set_title(f"{scls}\nd={d:.2f}", fontsize=8)
        for r in (0, 1):
            axes[r, k].set_xticks([]); axes[r, k].set_yticks([])
    fig.suptitle("Segmentación de la fruta y estimación de tamaño")
    fig.tight_layout()
    _save(fig, "fig_segmentation.pdf")


def fig_pca(X, y):
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-9)
    p = PCA(n_components=2, random_state=SEED).fit(Xs)
    Z = p.transform(Xs); ev = p.explained_variance_ratio_ * 100
    fig, ax = plt.subplots(figsize=(5.2, 4))
    for c in CLASSES:
        m = y == c
        ax.scatter(Z[m, 0], Z[m, 1], s=8, alpha=0.4, color=COL[c], label=NICE[c])
    ax.set_xlabel(f"PC1 ({ev[0]:.1f}%)"); ax.set_ylabel(f"PC2 ({ev[1]:.1f}%)")
    ax.set_title("Proyección PCA del espacio de características"); ax.legend()
    _save(fig, "fig_pca.pdf")


def fig_confusions(metrics):
    models = [m for m in ["LogReg", "SVM", "RandomForest", "CNN"] if m in metrics["models"]]
    fig, axes = plt.subplots(2, 2, figsize=(8, 7.5))
    for ax, name in zip(axes.ravel(), models):
        cm = np.array(metrics["models"][name]["confusion_matrix"], dtype=float)
        cmn = cm / cm.sum(1, keepdims=True)
        im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(3)); ax.set_xticklabels([NICE[c] for c in CLASSES], fontsize=8)
        ax.set_yticks(range(3)); ax.set_yticklabels([NICE[c] for c in CLASSES], fontsize=8)
        ax.set_xlabel("Predicción"); ax.set_ylabel("Real")
        ax.set_title(f"{name}  (acc={metrics['models'][name]['test_accuracy']*100:.1f}%)")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{int(cm[i,j])}\n{cmn[i,j]*100:.0f}%", ha="center",
                        va="center", color="white" if cmn[i, j] > 0.5 else "black",
                        fontsize=8)
    fig.suptitle("Matrices de confusión (conjunto de prueba)")
    fig.tight_layout()
    _save(fig, "fig_confusions.pdf")


def fig_model_comparison(metrics):
    names = [m for m in ["LogReg", "SVM", "RandomForest", "CNN"] if m in metrics["models"]]
    acc = [metrics["models"][n]["test_accuracy"] for n in names]
    f1 = [metrics["models"][n]["test_f1_macro"] for n in names]
    x = np.arange(len(names)); w = 0.38
    fig, ax = plt.subplots(figsize=(6, 3.6))
    b1 = ax.bar(x - w/2, acc, w, label="Exactitud")
    b2 = ax.bar(x + w/2, f1, w, label="F1-macro")
    ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylim(0.7, 1.0)
    ax.set_title("Comparación de modelos (prueba)"); ax.legend(fontsize=8)
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + .005,
                f"{b.get_height():.2f}", ha="center", fontsize=7)
    _save(fig, "fig_model_comparison.pdf")


def fig_feature_importance():
    rf = joblib.load(os.path.join(MODELS_DIR, "randomforest.joblib"))
    names = np.array(feature_names())
    imp = rf.feature_importances_
    idx = np.argsort(imp)[-12:]
    fig, ax = plt.subplots(figsize=(5.6, 4))
    ax.barh(names[idx], imp[idx], color="#1f77b4")
    ax.set_xlabel("Importancia (Random Forest)")
    ax.set_title("Top-12 características más importantes")
    _save(fig, "fig_feature_importance.pdf")


def fig_examples():
    import glob, random
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))
    svm = joblib.load(os.path.join(MODELS_DIR, "svm.joblib"))
    from features import extract_features
    rng = random.Random(7)
    fig, axes = plt.subplots(3, 4, figsize=(8, 6.2))
    for r, c in enumerate(CLASSES):
        for k, f in enumerate(rng.sample(glob.glob(os.path.join(APPLE_DIR, c, "*")), 4)):
            a = np.asarray(Image.open(f).convert("RGB").resize((128, 128)),
                           dtype=np.float32) / 255.0
            m = largest_component(fg_mask(a))
            pred = svm.predict(scaler.transform(extract_features(a, m)[None, :]))[0]
            d, scls = measure_size(np.asarray(Image.open(f).convert("RGB").resize((256, 256)),
                                              dtype=np.float32) / 255.0)
            ax = axes[r, k]; ax.imshow(a); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"real:{NICE[c]} | {NICE.get(pred,pred)}\ntam:{scls}",
                         fontsize=7, color="green" if pred == c else "red")
    fig.suptitle("Ejemplos de predicción (SVM): calidad y tamaño")
    fig.tight_layout()
    _save(fig, "fig_examples.pdf")


def fig_crispdm():
    fig, ax = plt.subplots(figsize=(8, 2.4)); ax.axis("off")
    steps = ["Comprensión\ndel negocio", "Comprensión\nde los datos",
             "Preparación\n(segmentación,\ncaracterísticas)", "Modelado\n(LogReg, SVM,\nRF, CNN)",
             "Evaluación\n(F1, matrices,\nerrores)", "Despliegue\n(app Streamlit)"]
    n = len(steps); x = np.linspace(0.06, 0.94, n)
    for i, (xi, s) in enumerate(zip(x, steps)):
        ax.add_patch(FancyBboxPatch((xi - 0.07, 0.35), 0.14, 0.32,
                     boxstyle="round,pad=0.01", fc="#d6e6f5", ec="#1f77b4"))
        ax.text(xi, 0.51, s, ha="center", va="center", fontsize=7.5)
        if i < n - 1:
            ax.add_patch(FancyArrowPatch((xi + 0.07, 0.51), (x[i+1] - 0.07, 0.51),
                         arrowstyle="-|>", mutation_scale=12, color="#555"))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("Aplicación de CRISP-DM al proyecto", fontsize=10)
    _save(fig, "fig_crispdm.pdf")


def main():
    X, y, df = load_dataset()
    metrics = json.load(open(os.path.join(HERE, "results", "metrics.json")))
    print("Generando figuras vectoriales...")
    fig_class_distribution(df)
    fig_size_distribution(df)
    fig_segmentation()
    fig_pca(X, y)
    fig_confusions(metrics)
    fig_model_comparison(metrics)
    fig_feature_importance()
    fig_examples()
    fig_crispdm()
    print("Listo.")


if __name__ == "__main__":
    main()
