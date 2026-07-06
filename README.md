# Mejoramiento de Video en Sistemas de Videovigilancia

Proyecto de tesis: **"Desarrollo de un modelo de inteligencia artificial para el escalamiento visual en videovigilancia mediante técnicas de superresolución y transformadores visuales"**

- **Autor:** Jose Rodriguez Botello (UFPS, Cúcuta)
- **Director:** Sergio Castro Casadiego (UFPS)
- **Co-director:** Sebastian Rojas-Ortega (University of Delaware)
- **Artículo asociado:** ColCom 2026 (ya enviado, `ColCom Paper/`)

Este documento explica **cómo el proyecto cumple cada uno de los cuatro objetivos específicos del anteproyecto aprobado**, qué scripts implementan cada uno, y qué falta por ejecutar. Los jurados verificarán el cumplimiento de cada objetivo, así que cada sección indica la evidencia concreta que la tesis debe presentar.

---

## Objetivo General

> Desarrollar un modelo de inteligencia artificial basado en técnicas de superresolución y transformadores visuales para escalar la calidad y fluidez de videos de videovigilancia.

**Cómo se cumple:** el pipeline completo combina superresolución con GAN (Real-ESRGAN), superresolución con transformadores visuales (SwinIR), interpolación de fotogramas para fluidez (RIFE) y un fine-tuning propio sobre datos de CCTV. Todo está automatizado en la carpeta `pipeline/` (11 scripts).

---

## Objetivo Específico 1: Preprocesamiento del dataset

> Preprocesar una base de datos de videos de cámaras de seguridad con distintos niveles de calidad, resolución y condiciones de iluminación, garantizando su adecuación para el entrenamiento y validación del modelo.

**Cómo se cumple:**

| Requisito del objetivo | Implementación |
|---|---|
| Videos de cámaras de seguridad | `step1_get_dataset.py` valida los videos en `dataset/raw_videos/` |
| Distintos niveles de calidad y resolución | `step2_prepare_frames.py` aplica degradación sintética con el protocolo estándar de Real-ESRGAN: blur gaussiano (σ=1.5), submuestreo bicúbico ×4, compresión JPEG (q=50) y ruido gaussiano (σ=5). Esto genera pares LR/HR controlados y reproducibles |
| Distintas condiciones de iluminación | Convención de nombres de video: `day_*.mp4`, `night_*.mp4`, `indoor_*.mp4`. El step2 procesa todos los videos de `raw_videos/` y los fotogramas heredan el prefijo, de modo que la condición se propaga por todo el pipeline. La función `config.condition_of()` la extrae del nombre del archivo |
| Adecuación para entrenamiento y validación | Los pares LR/HR de step2 alimentan tanto el fine-tuning (step9) como las métricas (step5/step10) |

**Acción pendiente de Jose:** conseguir 3 o 4 videos HD de vigilancia (cámara fija, exterior/interior) y nombrarlos con los prefijos de condición. Fuente sugerida: Pexels (buscar "surveillance camera outdoor", "security camera night", descarga gratuita). Colocarlos en `dataset/raw_videos/` y correr steps 1-2.

**Evidencia para la tesis:** descripción del protocolo de degradación (ya redactado en el paper de ColCom), tabla con número de fotogramas por condición, figura de ejemplo LR vs HR.

---

## Objetivo Específico 2: Diseño del modelo (superresolución + transformadores visuales)

> Diseñar un modelo de inteligencia artificial basado en transformadores visuales y técnicas de superresolución, con el propósito de mejorar la calidad de los videos mediante la optimización de la resolución y la fluidez.

**Cómo se cumple:** el diseño es un **pipeline modular** que integra las tres familias exigidas por el objetivo:

1. **Superresolución con GAN:** Real-ESRGAN ×4 (`step3_esrgan_inference.py`), arquitectura RRDBNet.
2. **Transformadores visuales:** SwinIR-M ×4 (`step8_swinir_inference.py`), modelo oficial basado en Swin Transformer entrenado para superresolución de mundo real. Este es el componente que cumple explícitamente la parte de "transformadores visuales" del título de la tesis.
3. **Fluidez temporal:** RIFE ×2 (`step4_rife_inference.py`), interpolación de fotogramas que duplica la tasa de cuadros.

**Evidencia para la tesis:** diagrama del pipeline (ya existe: `ColCom Paper/figures/pipeline_cctv.png`), descripción de las arquitecturas RRDBNet, Swin Transformer y RIFE (figuras 2 y 3 del paper; falta agregar una figura de la arquitectura SwinIR al documento de tesis).

**Nota importante:** la sección 4.1.2 del anteproyecto afirma que ya se implementó una arquitectura transformer, lo cual no era cierto en ese momento. Con SwinIR integrado (step8), esa afirmación queda respaldada, pero la redacción del capítulo debe corregirse para describir lo que realmente se hizo.

---

## Objetivo Específico 3: Entrenamiento del modelo

> Entrenar el modelo de inteligencia artificial utilizando la base de datos seleccionada, aplicando estrategias de optimización y ajuste de hiperparámetros.

**Cómo se cumple:** `step9_finetune_esrgan.py` hace **fine-tuning del generador RealESRGAN_x4plus** sobre los pares LR/HR del dataset propio de CCTV (los de step2), usando el framework oficial basicsr:

- **Pérdidas:** L1 + perceptual (VGG19) + adversarial (discriminador U-Net con normalización espectral).
- **Hiperparámetros ajustables** (definidos en `config.py` y registrados con cada corrida): learning rate (1e-4), batch size (4), iteraciones (5000 por defecto, ajustable con `--iters`), scheduler MultiStepLR con reducción al 80% del entrenamiento.
- El modelo afinado se exporta a `weights/RealESRGAN_x4plus_finetuned.pth` y se evalúa como un método más en step10.

**Requisito:** GPU (Google Colab T4). Tiempo estimado: 2 a 3 horas con 5000 iteraciones.

**Evidencia para la tesis:** tabla de hiperparámetros, curva o registro de pérdidas del entrenamiento (basicsr genera logs en `Real-ESRGAN/experiments/`), y comparación cuantitativa preentrenado vs fine-tuned en la Tabla de resultados. El ajuste de hiperparámetros se documenta corriendo al menos dos configuraciones (por ejemplo `--iters 2000` y `--iters 5000`) y reportando cuál dio mejor LPIPS.

---

## Objetivo Específico 4: Validación con métricas y comparación

> Validar el modelo desarrollado mediante métricas cuantitativas como PSNR y SSIM, junto con evaluaciones cualitativas basadas en percepción visual, y comparar los resultados con enfoques tradicionales de mejora de video.

**Cómo se cumple:**

| Requisito del objetivo | Implementación |
|---|---|
| PSNR y SSIM | `step10_thesis_metrics.py` calcula PSNR, SSIM y además LPIPS (métrica perceptual, va más allá de lo prometido) para todos los métodos |
| Evaluación cualitativa | `step11_thesis_figures.py` genera tiras de comparación visual por condición de iluminación (entrada LR, cada método, referencia HR) a 300 DPI |
| Comparación con enfoques tradicionales | La interpolación **bicúbica ×4** es la línea base tradicional en todas las tablas y figuras |
| Análisis por condición | step10 agrega resultados por condición (day/night/indoor) además del global, y genera la tabla LaTeX lista para pegar en la tesis |

Los métodos comparados son: bicúbico (tradicional), ESRGAN preentrenado (GAN), SwinIR (transformer) y ESRGAN fine-tuned (modelo propio entrenado).

**Nota sobre resultados esperados:** es normal que ESRGAN tenga PSNR menor que el bicúbico (brecha percepción-distorsión, Blau & Michaeli 2018). La superioridad se demuestra con LPIPS y con las comparaciones visuales. Esto ya está explicado en el paper de ColCom y debe explicarse igual en la tesis.

**Evidencia para la tesis:** `output/metrics/thesis_metrics.csv`, `output/metrics/thesis_summary.json`, tabla LaTeX impresa por step10, figuras de `output/thesis_figures/`.

---

## Scripts del pipeline

| Script | Qué hace | Objetivo de la tesis |
|---|---|---|
| `config.py` | Rutas, hiperparámetros y parámetros compartidos. Detecta Colab automáticamente | Todos |
| `step1_get_dataset.py` | Valida los videos fuente en `dataset/raw_videos/` | Obj. 1 |
| `step2_prepare_frames.py` | Extrae fotogramas HR y genera los LR con degradación sintética (blur, bicúbico ×4, JPEG, ruido). También crea la línea base bicúbica | Obj. 1 |
| `step3_esrgan_inference.py` | Superresolución con Real-ESRGAN ×4 preentrenado (GAN) | Obj. 2 |
| `step4_rife_inference.py` | Interpolación de fotogramas con RIFE ×2 (fluidez temporal) | Obj. 2 |
| `step5_compute_metrics.py` | PSNR/SSIM/LPIPS, solo bicúbico vs ESRGAN | Paper ColCom |
| `step6_generate_figures.py` | Figuras 4 a 7 del artículo a 300 DPI | Paper ColCom |
| `step7_compile_results.py` | Tabla LaTeX y texto sugerido para el artículo | Paper ColCom |
| `step8_swinir_inference.py` | Superresolución con SwinIR ×4 (transformador visual) | Obj. 2 |
| `step9_finetune_esrgan.py` | Fine-tuning de Real-ESRGAN sobre el dataset propio de CCTV, con hiperparámetros documentados. Exporta los pesos afinados y corre inferencia | Obj. 3 |
| `step10_thesis_metrics.py` | PSNR/SSIM/LPIPS de los cuatro métodos, global y por condición de iluminación. Imprime la tabla LaTeX de la tesis | Obj. 4 |
| `step11_thesis_figures.py` | Tiras de comparación visual por condición y gráfica de métricas, a 300 DPI | Obj. 4 |

Los steps 5, 6 y 7 pertenecen al artículo de ColCom y ya cumplieron su función; para la tesis los reemplazan los steps 10 y 11, que cubren los cuatro métodos y el análisis por condición.

## Orden de ejecución (Google Colab, GPU T4)

```
pip install -r pipeline/requirements.txt

python pipeline/step1_get_dataset.py       # valida los videos
python pipeline/step2_prepare_frames.py    # extrae y degrada fotogramas (Obj. 1)
python pipeline/step3_esrgan_inference.py  # ESRGAN preentrenado (Obj. 2)
python pipeline/step8_swinir_inference.py  # SwinIR transformer (Obj. 2)
python pipeline/step4_rife_inference.py    # interpolación RIFE (Obj. 2, fluidez)
python pipeline/step9_finetune_esrgan.py   # entrenamiento/fine-tuning (Obj. 3)
python pipeline/step10_thesis_metrics.py   # métricas por método y condición (Obj. 4)
python pipeline/step11_thesis_figures.py   # figuras de la tesis (Obj. 4)
```

Una copia congelada de los scripts y métricas exactos del artículo está en `ColCom Paper/reproducibility/`; esa carpeta no se toca, todo el trabajo de tesis se hace en `pipeline/`.

Consejos para Colab:
- Subir las carpetas `pipeline/`, `dataset/raw_videos/` y `weights/` a `/content/jose/` (o montar Drive). `config.py` detecta Colab automáticamente.
- Si la GPU se queda sin memoria en step8: `--tile 256`. En step9: `--tile 400` para la inferencia.
- Descargar al final `output/metrics/` y `output/thesis_figures/` para el documento.

---

## Trabajo pendiente en el documento de tesis

El anteproyecto (`Anteproyecto.pdf`) tiene los capítulos 1 a 4 casi completos. Para convertirlo en tesis falta:

1. **Sección 4.1.2:** corregir la redacción; describe una implementación transformer que en ese momento no existía. Ahora debe describir SwinIR tal como se integró en step8.
2. **Sección 4.4.2:** reemplazar el marcador "AQUÍ IRIA LAS LIBRERIAS UTILIZADAS" con las librerías reales: PyTorch, basicsr, realesrgan, timm, OpenCV, scikit-image, lpips, NumPy, pandas, matplotlib.
3. **Capítulo 5 (Resultados):** escribirlo con la tabla y figuras de steps 10 y 11. El marcador "QUEDAMOS AQUÍ PARA EMPEZAR EL CAPITULO 5" señala dónde.
4. **Capítulo 6 (Conclusiones y trabajo futuro):** redactar a partir de los resultados, incluyendo la discusión percepción-distorsión.
5. **Tabla de contenido:** reparar los "¡Error! Marcador no definido" (actualizar campos en Word).
6. **Cronograma y presupuesto:** llenar las tablas vacías.
7. **Regla del proyecto:** no usar guiones largos en ningún documento y no afirmar resultados que no se hayan obtenido experimentalmente.

---

## Estructura del repositorio

```
Jose/
├── pipeline/                  Scripts del experimento (steps 1-11)
├── dataset/
│   ├── raw_videos/            Videos fuente (nombrar day_*, night_*, indoor_*)
│   ├── frames_hr/             Fotogramas de referencia (ground truth)
│   └── frames_lr/             Fotogramas degradados (entrada)
├── output/
│   ├── bicubic/               Línea base tradicional
│   ├── sr/                    ESRGAN preentrenado
│   ├── swinir/                SwinIR (transformer)
│   ├── sr_finetuned/          ESRGAN afinado (modelo propio)
│   ├── rife/                  Fotogramas interpolados
│   ├── metrics/               CSV y JSON de métricas
│   └── thesis_figures/        Figuras para la tesis (300 DPI)
├── weights/                   Pesos preentrenados y afinados
├── ColCom Paper/              Artículo IEEE (ya enviado)
│   └── reproducibility/       Copia congelada de scripts y métricas del artículo (no editar)
├── Archive/                   Código viejo fuera de uso (no está en el repo)
└── Anteproyecto.pdf           Anteproyecto aprobado
```
