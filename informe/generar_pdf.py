"""Convierte informe/P1_Informe_Tecnico.md y P2_Articulo.md a PDF sin pandoc.

Markdown → HTML (librería `markdown`) → PDF (Google Chrome en modo headless).
Uso:  python informe/generar_pdf.py
"""
import re
import shutil
import subprocess
import sys
from pathlib import Path

import markdown

AQUI = Path(__file__).resolve().parent
CHROME_CANDIDATOS = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    shutil.which("google-chrome"), shutil.which("chromium"), shutil.which("chrome"),
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
]
CSS = """
@page { size: A4; margin: 2.2cm 2cm; }
body { font-family: 'Times New Roman', Georgia, serif; font-size: 11pt; line-height: 1.35; color: #111; }
h1 { font-size: 18pt; margin: 0 0 6pt; } h2 { font-size: 14pt; margin-top: 18pt; border-bottom: 1px solid #bbb; }
h3 { font-size: 12pt; margin-top: 12pt; }
table { border-collapse: collapse; margin: 8pt 0; font-size: 9.5pt; width: 100%; }
th, td { border: 1px solid #999; padding: 3pt 5pt; text-align: left; } th { background: #eee; }
img { max-width: 78%; max-height: 7.2cm; display: block; margin: 6pt auto; page-break-inside: avoid; }
code { font-family: Menlo, Consolas, monospace; font-size: 9.5pt; background: #f3f3f3; padding: 0 2px; }
blockquote { border-left: 3px solid #999; margin-left: 0; padding-left: 10pt; color: #333; }
p, li { text-align: justify; }
"""


def chrome():
    for c in CHROME_CANDIDATOS:
        if c and Path(c).exists():
            return c
    sys.exit("No se encontró Google Chrome/Chromium para imprimir el PDF.")


def convertir(nombre):
    md = (AQUI / f"{nombre}.md").read_text(encoding="utf-8")
    md = re.sub(r"\A---\n.*?\n---\n", "", md, flags=re.S)  # quita el bloque YAML
    cuerpo = markdown.markdown(md, extensions=["tables", "sane_lists", "nl2br"])
    html = f"<!DOCTYPE html><html lang='es'><head><meta charset='utf-8'><title>{nombre}</title>" \
           f"<style>{CSS}</style></head><body>{cuerpo}</body></html>"
    ruta_html = AQUI / f".{nombre}.html"
    ruta_html.write_text(html, encoding="utf-8")
    pdf = AQUI / f"{nombre}.pdf"
    subprocess.run([chrome(), "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={pdf}", ruta_html.as_uri()], check=True, capture_output=True, timeout=120)
    ruta_html.unlink()
    print("PDF generado:", pdf)


if __name__ == "__main__":
    for n in ["P1_Informe_Tecnico", "P2_Articulo"]:
        convertir(n)
