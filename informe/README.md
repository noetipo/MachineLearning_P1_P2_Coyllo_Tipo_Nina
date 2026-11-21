# Informes de las prácticas

| Práctica | Archivo | Contenido |
|---|---|---|
| P1 | `P1_Informe_Tecnico.md` / `P1_Informe_Tecnico.pdf` | Informe técnico: problema, datos y hallazgos, métrica, preprocesamiento sin fuga, 5 modelos, resultados y conclusiones. |
| P2 | `P2_Articulo.md` / `P2_Articulo.pdf` | Artículo: resumen, introducción, datos, método, experimentos (Optuna + Wilcoxon, k-means, red neuronal), interpretación, ética y limitaciones, conclusiones y referencias APA 7. |

Las figuras se enlazan desde `../resultados/figuras/`, por lo que los archivos `.md` se visualizan correctamente en GitHub.

## Regenerar los PDF

No se dispone de `pandoc`; los PDF se generan convirtiendo el Markdown a HTML con la librería `markdown` e imprimiéndolo con Google Chrome en modo *headless*:

```bash
.venv/bin/pip install markdown==3.10
.venv/bin/python informe/generar_pdf.py
```

Alternativas: con `pandoc` instalado, `pandoc P1_Informe_Tecnico.md -o P1_Informe_Tecnico.pdf --pdf-engine=xelatex` (desde esta carpeta, para que las rutas de las figuras funcionen); o abrir el `.md` en VS Code / Typora y exportar a PDF.
