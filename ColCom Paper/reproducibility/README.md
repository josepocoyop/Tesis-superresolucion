# Reproducibilidad del artículo ColCom 2026

Copia congelada de los scripts y resultados usados para generar el artículo sometido a ColCom 2026 (`../main.pdf`). **No editar estos archivos**: son el registro de lo que produjo la Tabla I y las Figuras 4 a 7 del artículo. El desarrollo de la tesis continúa en `Jose/pipeline/`.

## Contenido

```
reproducibility/
├── pipeline/      Scripts tal como se usaron para el artículo
│   ├── config.py              Rutas y parámetros compartidos
│   ├── requirements.txt       Dependencias
│   ├── step1_get_dataset.py   Validación del video fuente
│   ├── step2_prepare_frames.py    Extracción y degradación sintética
│   ├── step3_esrgan_inference.py  Superresolución ESRGAN x4
│   ├── step4_rife_inference.py    Interpolación RIFE x2
│   ├── step5_compute_metrics.py   PSNR, SSIM, LPIPS
│   ├── step6_generate_figures.py  Figuras 4 a 7 (300 DPI)
│   └── step7_compile_results.py   Tabla LaTeX y texto sugerido
└── results/       Métricas que respaldan la Tabla I del artículo
    ├── metrics.csv        Métricas por fotograma
    └── summary.json       Promedios (bicúbica vs ESRGAN)
```

## Cómo reproducir

1. Instalar dependencias: `pip install -r pipeline/requirements.txt`
2. Colocar el video fuente en `dataset/raw_videos/`
3. Ejecutar en orden step1 a step7 (Google Colab con GPU T4 recomendado; `config.py` detecta Colab automáticamente)

Las figuras finales del artículo están en `../figures/` y el documento fuente es `../main.tex`.

## Parámetros del experimento

- Degradación sintética: blur gaussiano σ=1.5, submuestreo bicúbico ×4, JPEG q=50, ruido gaussiano σ=5 (protocolo Real-ESRGAN)
- 60 fotogramas, muestreo cada 5 fotogramas del video fuente
- ESRGAN: RealESRGAN_x4plus (pesos oficiales), escala ×4
- RIFE: v4.6 (ECCV2022-RIFE oficial), interpolación ×2
- Métricas: PSNR, SSIM (scikit-image), LPIPS (AlexNet)

Nota: `config.py` incluye también rutas añadidas después para la tesis (SwinIR, fine-tuning); los steps 1 a 7 no las usan.
