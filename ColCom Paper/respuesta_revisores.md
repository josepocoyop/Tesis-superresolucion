# Respuesta a los comentarios de los revisores

**Artículo:** Superresolución Espacial e Interpolación Temporal de Video CCTV con ESRGAN y RIFE
**Autores:** Jose Rodriguez Botello, Sergio Castro Casadiego, Sebastian Rojas-Ortega
**Conferencia:** 19th IEEE Colombian Conference on Communications and Computing (ColCom 2026)

Agradecemos a los dos revisores sus observaciones. Todas fueron atendidas en la versión final. A continuación indicamos, para cada una, qué se cambió y en qué parte del documento quedó. El artículo se mantiene dentro del límite de 6 páginas.

---

## Revisor 1

### 1. Aclarar que la contribución principal es la integración y la evaluación, no un algoritmo nuevo, y decirlo explícitamente en la Introducción y en la Conclusión

Se añadió un párrafo al final de la **Sección I** dedicado exclusivamente a delimitar el alcance:

> "El alcance de la contribución es acotado. Ambos modelos son preexistentes y se emplean con sus pesos preentrenados, sin modificar sus arquitecturas ni reentrenarlos. Lo que aporta este trabajo es la integración de las dos etapas en un solo flujo, su evaluación sobre grabaciones de una cámara de vigilancia comercial bajo un protocolo de degradación reproducible, y la medición del costo computacional que esa combinación impone en una GPU de escritorio. No se propone un algoritmo nuevo de superresolución ni de interpolación."

El primer párrafo de la **Sección VI-A** lo repite en los mismos términos. También se ajustó el resumen, que ahora dice "The contribution is the integration of both stages and its evaluation on CCTV footage under a reproducible degradation protocol, rather than a new architecture", y se cambió el verbo del artículo de "propone" a "evalúa" en la Introducción y en las Conclusiones.

### 2. Añadir métricas computacionales: velocidad de inferencia y uso de VRAM en la RTX 4070 Ti

La **Tabla I** incorpora dos columnas nuevas, fps de salida y milisegundos por fotograma, y la **Sección V** dedica un párrafo completo al costo:

- ESRGAN: 45.5 ms por fotograma, pico de 360 MB de memoria de video
- RIFE: 14.0 ms por fotograma sintetizado, pico de 274 MB
- Flujo completo: 52.5 ms por fotograma de entrada, es decir 19.0 fotogramas de entrada por segundo

Las mediciones se tomaron sobre la RTX 4070 Ti con el modelo ya cargado en memoria, promediando 20 fotogramas tras un calentamiento de 3, a la resolución del experimento (212x120 a 848x480). El pico de memoria se midió con `torch.cuda.max_memory_reserved()`. Como las dos etapas se ejecutan una después de la otra, el pico del flujo es el mayor de los dos y no la suma, cosa que se indica en el texto.

Estos números permiten además concluir que el sistema sostiene en tiempo real una cámara de 15 fps sobre una GPU de escritorio, y se señala que el margen restante es estrecho para atender varias cámaras en paralelo.

### 3. Analizar las limitaciones, dado que el conjunto de evaluación es pequeño, y cómo otros tipos de degradación mejorarían la robustez

Se añadió un párrafo completo al final de la **Sección V** que reconoce cuatro limitaciones:

1. La evaluación se apoya en 60 fotogramas de una sola escena exterior diurna, suficiente para observar la tendencia entre métodos pero no para sostener conclusiones sobre otras condiciones. Falta material nocturno e interior.
2. La degradación se generó con un único protocolo sintético de parámetros fijos, mientras que las cámaras reales combinan desenfoque de movimiento, artefactos de compresión variables según el códec y pérdidas de transmisión.
3. ESRGAN procesa cada fotograma de forma independiente, lo que puede producir variaciones de textura entre fotogramas contiguos.
4. RIFE es sensible a los cambios de iluminación abruptos y a los movimientos muy rápidos.

La **Sección VI-A** cierra recordando que los resultados provienen de una única escena diurna y constituyen una caracterización inicial, no una validación en condiciones variadas. La **Sección VI-B** sitúa como prioridad inmediata ampliar el conjunto a escenas nocturnas e interiores y a degradaciones más diversas.

### 4. Explicar cómo se podrían evaluar los fotogramas sintéticos añadidos por RIFE

En lugar de describir solamente el procedimiento, se realizó la medición. La **Sección IV-B** presenta una prueba de exclusión: de tripletes consecutivos de la grabación original se ocultó el fotograma central, se interpoló a partir de los dos extremos y el resultado se comparó con el fotograma real.

La prueba se restringió a los tripletes con movimiento apreciable, definido como una diferencia absoluta media superior a 3 DN entre los dos extremos, porque en los tramos estáticos cualquier método reproduce el fotograma oculto y la comparación no distingue nada. Sobre 23 tripletes así seleccionados:

| Método | PSNR | SSIM | LPIPS |
|---|---|---|---|
| Promedio de los dos vecinos | 33.95 | 0.9759 | 0.0161 |
| RIFE | **37.12** | **0.9836** | **0.0111** |

Se advierte además que la prueba es más exigente que el uso real, porque los dos extremos del triplete están separados por dos intervalos de captura mientras que en el flujo propuesto la interpolación ocurre entre fotogramas separados por uno solo.

### 5. Las figuras 2 y 3 no se mencionan en el texto

Corregido. Ambas se citan ahora en dos lugares cada una: en la descripción de las etapas del flujo (**Sección III-B**, puntos 2 y 3) y en la subsección que describe cada modelo (**Secciones III-D1** y **III-D2**). Todas las figuras y la tabla del artículo están referenciadas en el cuerpo del texto.

---

## Revisor 2

### 1. Establecer claramente la contribución, para que el lector no espere un modelo nuevo

Atendido junto con el punto 1 del Revisor 1. El párrafo añadido a la Introducción dice de forma literal que no se propone un algoritmo nuevo.

### 2. Presentar métricas computacionales que permitan evaluar el rendimiento

Atendido junto con el punto 2 del Revisor 1: dos columnas nuevas en la Tabla I y un párrafo de costo computacional en la Sección V, con tiempos y memoria de video.

### 3. Reconocer las limitaciones en el texto

Atendido junto con el punto 3 del Revisor 1: párrafo de limitaciones en la Sección V y una frase de alcance al cierre de las Conclusiones.

### 4. Corregir problemas de redacción, gramática y formato; hay figuras que no se referencian

Se hizo una revisión completa del texto. Además de las referencias a las figuras, ya comentadas:

- Se eliminaron expresiones que sobrevendían el resultado: "capacidad superior", "texturas fotorrealistas superiores", "balance superior", "trabajo seminal", "transformó el paradigma".
- Se corrigieron muletillas repetidas y se unificó la terminología, que alternaba entre "pipeline" y "flujo" en un texto en español.
- Se unificó el separador decimal, que aparecía como punto en el texto y como coma en las expresiones matemáticas.
- Se corrigió una ambigüedad de la Tabla I, donde la columna "fps" podía confundirse con la velocidad de procesamiento; ahora dice "fps de salida" y el texto distingue ambas magnitudes.
- El resumen aparecía rotulado "Resumen" estando escrito en inglés; ahora dice "Abstract".

---

## Cambios adicionales realizados por iniciativa propia

Durante la preparación de la versión final revisamos el documento completo y corregimos algunos puntos que los revisores no señalaron pero que afectaban la exactitud del artículo. Los declaramos por transparencia.

**Figura de interpolación temporal (ahora Fig. 5).** En la versión enviada, esa figura se había construido con dos de los 60 fotogramas que el procedimiento reparte de manera uniforme por toda la grabación, que están separados unos 65 fotogramas entre sí. No eran contiguos y por tanto no ilustraban correctamente la interpolación. Se regeneró con dos fotogramas consecutivos reales, separados 66.7 ms.

**Descripción del material.** El texto decía que se emplearon secuencias de dominio público. En realidad se trata de una grabación propia, capturada con una cámara de vigilancia comercial, cuya marca es visible en las propias figuras del artículo. La Sección III-A ahora lo describe de forma exacta.

**Contradicción sobre el entorno de ejecución.** La Discusión afirmaba que el procesamiento requiere entornos de nube, mientras que la Sección III-C describía una ejecución local. Todo el trabajo se realizó localmente y ahora se reportan los tiempos medidos, que respaldan esa afirmación.

**Figuras 1 a 3.** Se rehicieron como gráficos vectoriales. Las versiones anteriores contenían imprecisiones: el pie de la Fig. 2 mencionaba componentes que el dibujo no mostraba, y la Fig. 3 describía una etapa de refinamiento que no está activa en la implementación de RIFE utilizada. Los diagramas actuales se verificaron contra el código de los modelos e incorporan información adicional, como el número de escalas de IFNet, la máscara de fusión, la conexión residual del generador y el costo por etapa.

**Figuras de comparación.** Las antiguas Figs. 4 y 5 compartían casi por completo su fila superior, de modo que se fundieron en una sola figura. Se regeneraron por encima de 300 DPI efectivos, que es la resolución que IEEE recomienda para contenido fotográfico.

**Análisis de la región ampliada.** La Sección IV-A señala ahora de forma explícita que, en la región que se amplía, la textura que reconstruye ESRGAN es nítida pero no coincide con la de la referencia, y precisa el alcance forense de ese comportamiento. Nos pareció más honesto exponerlo que presentar la ampliación sin comentario.

**Bibliografía.** Se verificaron las 24 entradas contra las actas originales. Se corrigieron tres listas de autores incorrectas, se sustituyeron dos citas que apuntaban a repositorios de código por las publicaciones correspondientes, y se unificó la paginación, que mezclaba dos numeraciones distintas.

**Cita retirada.** Se eliminó una referencia que se empleaba para respaldar una afirmación que esa fuente no sostiene.

---

## Sobre el límite de páginas

La versión final ocupa 6 páginas, el máximo permitido, incluyendo figuras y referencias. Para acomodar el contenido nuevo se fusionaron dos figuras y se ajustó la redacción de varias secciones, sin suprimir ningún resultado.
