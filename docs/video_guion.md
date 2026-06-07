# Guion del video (máximo 10 minutos)

El enunciado pide un video corto presentando el proyecto, el contexto, las
técnicas, los resultados y los principales logros. Estructura sugerida con
minutos aproximados.

## 0:00 – 0:45 · Presentación e introducción
- Integrantes, curso y nombre del proyecto.
- Problema: la clasificación manual de calidad de fruta es lenta, subjetiva y
  genera pérdidas y desperdicio. Objetivo: predecir calidad (buena/regular/mala)
  y estimar el tamaño de una manzana a partir de una foto.

## 0:45 – 2:00 · Datos y metodología (CRISP-DM)
- Dataset oficial de Kaggle (Fruit Quality Classification, ryandpark): manzanas
  buena (1149) y mala (1141).
- Explicar cómo se construyó la clase "regular": segmentando individualmente
  cada fruta de la carpeta Mixed + aumento de datos.
- Mencionar las fotos propias añadidas (carpeta data/own).
- Mostrar el diagrama de flujo CRISP-DM (Fig. 1 del informe).

## 2:00 – 3:30 · Segmentación, características y tamaño
- Mostrar `src/segment.py`: cómo se separa la fruta del fondo (umbral de Otsu) y
  cómo se mide el tamaño (diámetro equivalente normalizado).
- Punto clave: las 49 características de color/textura/defectos se calculan SOLO
  sobre la fruta, para que el modelo aprenda la calidad y no el fondo.
- Mostrar la figura de segmentación (Fig. 3).

## 3:30 – 5:00 · Modelos y fundamentos
- Los cuatro modelos: regresión logística (línea base), SVM-RBF, Random Forest y
  CNN. Explicar brevemente cada uno y por qué se eligieron.
- Fundamentos: estandarización z-score, kernel RBF de la SVM, entropía cruzada y
  Adam en la CNN, y la métrica F1-macro con validación cruzada.

## 5:00 – 6:30 · Resultados
- Tabla de resultados: en prueba los tres mejores quedan casi empatados (SVM
  94.9 %, CNN 95.3 %, Random Forest 94.4 %), muy por encima de la línea base
  (83.8 %). El mejor por validación cruzada es la SVM (96.7 %), y por eso es el
  modelo seleccionado.
- Matrices de confusión (Fig. 5): los errores se concentran entre buena y mala.
- Importancia de características (Fig. 6): el color (saturación/matiz) es lo más
  decisivo.

## 6:30 – 7:30 · Análisis crítico
- Hallazgo: los modelos clásicos (sobre características de color) y la CNN (sobre
  píxeles) dan resultados equivalentes a esta escala de datos; el color es muy
  informativo para la calidad.
- Selección por validación cruzada (no por el test) para no "hacer trampa".
- Honestidad: al principio la clase "regular" daba 100 % irreal porque el modelo
  reconocía el FONDO de los recortes; al enmascarar el fondo bajó a un realista
  96 %. Mencionar las limitaciones (un solo dataset, tamaño relativo, etc.).

## 7:30 – 9:00 · Demostración en vivo
- Ejecutar `streamlit run src/app.py`.
- Subir o capturar con la cámara una manzana buena, una regular y una mala;
  mostrar la predicción de calidad de cada modelo, el tamaño estimado y la
  segmentación.

## 9:00 – 10:00 · Ética, impactos y conclusiones
- Ética: sesgo/equidad, privacidad (manos en algunas fotos), supervisión humana,
  y referenciar los datos/código de terceros (honestidad académica).
- Impactos: económico (menos desperdicio), ambiental, social, ODS 2 y 12.
- Conclusiones y trabajo futuro. Mencionar el repositorio.

## Lista de verificación (debe aparecer en el video)
- [ ] Contexto y problema.
- [ ] Datos y metodología CRISP-DM.
- [ ] Segmentación + estimación de tamaño.
- [ ] Los cuatro modelos y sus fundamentos.
- [ ] Resultados con gráficas y matrices de confusión.
- [ ] Análisis crítico (sobreajuste, sesgo del dato, qué falla).
- [ ] Demostración funcionando (app con cámara).
- [ ] Ética e impactos.
- [ ] Conclusiones.

## Consejos
- Subir el video a YouTube (no listado) o Google Drive (enlace público) y pegar
  el enlace en el informe (sección Enlaces).
- Duración máxima 10 minutos: ensayar una vez para ajustar el ritmo.
- Compartir pantalla mostrando el código y las figuras reales.
