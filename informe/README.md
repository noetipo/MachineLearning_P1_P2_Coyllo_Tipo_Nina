# Informes

| Archivo | Contenido |
|---|---|
| `Producto1.md` / `Producto1.pdf` | Informe técnico: problema, datos y hallazgos, métrica, preprocesamiento sin fuga, 5 modelos, resultados y conclusiones. |
| `Producto2.md` / `Producto2.pdf` | Artículo: resumen, introducción, datos, método, experimentos (Optuna + Wilcoxon, k-means, red neuronal), interpretación, ética y limitaciones, conclusiones y referencias APA 7. |

Las figuras se enlazan desde `../resultados/figuras/`, así que los `.md` se ven bien en GitHub.

## Regenerar los PDF

No se dispone de `pandoc`; los PDF se generan convirtiendo el Markdown a HTML con la librería `markdown` e imprimiéndolo con Google Chrome en modo *headless*:

```bash
.venv/bin/pip install markdown==3.10
.venv/bin/python informe/generar_pdf.py
```

Alternativas: con `pandoc` instalado, `pandoc Producto1.md -o Producto1.pdf --pdf-engine=xelatex` (desde esta carpeta, para que las rutas de las figuras funcionen); o abrir el `.md` en VS Code / Typora y exportar a PDF.
