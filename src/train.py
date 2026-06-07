"""
train.py
========
Entrena y compara modelos CLÁSICOS de machine learning para la clasificación de
calidad de manzanas en {good, regular, bad}, cumpliendo el enunciado que pide
"al menos dos modelos distintos de machine learning" con ajuste de
hiperparámetros por validación cruzada / búsqueda en rejilla:

  1) Regresión Logística  (línea base)
  2) SVM con kernel RBF
  3) Random Forest

El modelo de deep learning (CNN) se entrena aparte en train_cnn.py.

Todos comparten el descriptor de 49 características (features.py), la
estandarización z-score y la misma partición estratificada train/test (70/30).
Se usa class_weight='balanced' por el desbalance de la clase 'regular'.
Se selecciona automáticamente el mejor modelo por F1-macro y se guardan todos.

Salidas: models/*.joblib, models/scaler.joblib, models/labels.json,
         models/best_model.txt, results/metrics.json
"""
from __future__ import annotations
import os
import json
import warnings
import numpy as np

warnings.filterwarnings("ignore")  # silencia avisos de deprecación de sklearn

from sklearn.model_selection import (train_test_split, GridSearchCV,
                                     cross_val_score, StratifiedKFold)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                             precision_recall_fscore_support, classification_report)
import joblib

from dataset import load_dataset, CLASSES

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(HERE, "models")
RESULTS_DIR = os.path.join(HERE, "results")
SEED = 42


def _grids():
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    return cv, {
        "LogReg": (LogisticRegression(max_iter=2000, class_weight="balanced"),
                    {"C": [0.1, 1, 10]}),
        "SVM": (SVC(kernel="rbf", class_weight="balanced", probability=True,
                    random_state=SEED),
                 {"C": [1, 5, 10, 50], "gamma": ["scale", 0.01, 0.1]}),
        "RandomForest": (RandomForestClassifier(class_weight="balanced",
                                                 random_state=SEED, n_jobs=-1),
                          {"n_estimators": [100, 200], "max_depth": [None, 20, 40]}),
    }


def main(rebuild: bool = False):
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    X, y, df = load_dataset(rebuild=rebuild)
    print(f"\nDataset X={X.shape}  "
          f"clases={dict(zip(*np.unique(y, return_counts=True)))}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=SEED)
    scaler = StandardScaler().fit(X_tr)
    X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)

    cv, grids = _grids()
    metrics = {"n_train": int(len(y_tr)), "n_test": int(len(y_te)),
               "classes": CLASSES, "seed": SEED, "models": {}}
    best_name, best_f1 = None, -1.0

    for name, (est, grid) in grids.items():
        print(f"\n=== {name}: búsqueda en rejilla ===")
        gs = GridSearchCV(est, grid, scoring="f1_macro", cv=cv, n_jobs=-1)
        gs.fit(X_tr_s, y_tr)
        model = gs.best_estimator_
        y_pred = model.predict(X_te_s)
        acc = accuracy_score(y_te, y_pred)
        f1m = f1_score(y_te, y_pred, average="macro")
        cvsc = cross_val_score(model, X_tr_s, y_tr, cv=cv, scoring="f1_macro")
        p, r, f, _ = precision_recall_fscore_support(
            y_te, y_pred, labels=CLASSES, zero_division=0)
        cm = confusion_matrix(y_te, y_pred, labels=CLASSES)
        print(f"best={gs.best_params_}")
        print(f"test acc={acc:.4f} f1m={f1m:.4f} | cv f1m={cvsc.mean():.4f}±{cvsc.std():.4f}")
        print(classification_report(y_te, y_pred, labels=CLASSES, zero_division=0))
        metrics["models"][name] = {
            "best_params": gs.best_params_,
            "test_accuracy": float(acc), "test_f1_macro": float(f1m),
            "cv_f1_macro_mean": float(cvsc.mean()), "cv_f1_macro_std": float(cvsc.std()),
            "per_class": {CLASSES[i]: {"precision": float(p[i]), "recall": float(r[i]),
                                       "f1": float(f[i])} for i in range(len(CLASSES))},
            "confusion_matrix": cm.tolist(),
        }
        joblib.dump(model, os.path.join(MODELS_DIR, f"{name.lower()}.joblib"))
        # El mejor modelo se elige por VALIDACIÓN CRUZADA (no por el conjunto de
        # prueba), como indica el enunciado y para no usar el test en la selección.
        if cvsc.mean() > best_f1:
            best_f1, best_name = cvsc.mean(), name

    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.joblib"))
    json.dump({"classes": CLASSES}, open(os.path.join(MODELS_DIR, "labels.json"), "w"))
    metrics["best_model"] = best_name
    open(os.path.join(MODELS_DIR, "best_model.txt"), "w").write(best_name)
    json.dump(metrics, open(os.path.join(RESULTS_DIR, "metrics.json"), "w"), indent=2)
    print(f"\n>>> Mejor modelo por F1-macro en validación cruzada: {best_name} ({best_f1:.4f})")
    print(f"Modelos en {MODELS_DIR}, métricas en results/metrics.json")
    return metrics


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    main(rebuild=ap.parse_args().rebuild)
