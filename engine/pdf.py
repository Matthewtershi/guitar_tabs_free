"""MusicXML -> PDF pages laid out like the example tab.

Verovio engraves the notation + tab (as SVG); svglib + reportlab put the pages
into an A4 PDF with a header: title, level, "Capo N", tempo mark.
The tab staff is cleaned to bare fret numbers (no stems/flags/rests), like the example.
"""
from __future__ import annotations

import io
from pathlib import Path

from lxml import etree

A4_W, A4_H = 595.0, 842.0           # points
MARGIN = 36.0
HEADER_FIRST = 95.0                  # points reserved for the big header on page 1
HEADER_OTHER = 28.0
# Verovio page size in its units (1/10 mm). Height leaves room for our header.
VRV_W = 2100
VRV_H = 2550

_SYMBOL_FONT = None


def _symbol_font():
    """A font with the quarter-note sign for the tempo mark (Windows: Segoe UI Symbol)."""
    global _SYMBOL_FONT
    if _SYMBOL_FONT is None:
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        _SYMBOL_FONT = ""
        for p in (Path("C:/Windows/Fonts/seguisym.ttf"), Path("C:/Windows/Fonts/seguiemj.ttf")):
            if p.exists():
                try:
                    pdfmetrics.registerFont(TTFont("Sym", str(p)))
                    _SYMBOL_FONT = "Sym"
                    break
                except Exception:
                    pass
    return _SYMBOL_FONT


def _clean_svg(svg: str) -> bytes:
    """Remove rhythm stems/flags and rests from TAB staves, and the part label."""
    root = etree.fromstring(svg.encode("utf-8"))

    def has_class(el, name):
        return name in (el.get("class") or "").split()

    for el in list(root.iter("{*}g")):
        if has_class(el, "tabDurSym") or has_class(el, "label") or has_class(el, "labelAttr"):
            el.getparent().remove(el)
    for staff in [g for g in root.iter("{*}g") if has_class(g, "staff")]:
        if not any(has_class(d, "tabGrp") for d in staff.iter("{*}g")):
            continue
        for el in [d for d in staff.iter("{*}g") if has_class(d, "rest") or has_class(d, "meterSig")]:
            el.getparent().remove(el)
    return etree.tostring(root)


def render_pdf(musicxml: Path, pdf_path: Path, title: str, subtitle: str, left: str,
               tempo: float) -> int:
    """Returns the number of pages."""
    import verovio
    from reportlab.graphics import renderPDF
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from svglib.svglib import svg2rlg

    tk = verovio.toolkit()
    tk.setOptions({
        "pageWidth": VRV_W, "pageHeight": VRV_H,
        "pageMarginTop": 40, "pageMarginBottom": 40, "pageMarginLeft": 60, "pageMarginRight": 60,
        "scale": 42, "header": "none", "footer": "none", "breaks": "auto",
        "spacingSystem": 14, "spacingStaff": 8, "adjustPageHeight": False,
    })
    if not tk.loadFile(str(musicxml)):
        raise RuntimeError(f"Verovio could not read {musicxml}")
    n_pages = tk.getPageCount()
    c = canvas.Canvas(str(pdf_path), pagesize=A4)
    c.setTitle(f"{title} - {subtitle}")
    sym = _symbol_font()
    for i in range(1, n_pages + 1):
        top = A4_H - MARGIN
        if i == 1:
            from reportlab.pdfbase.pdfmetrics import stringWidth
            size = 22
            while size > 11 and stringWidth(title, "Times-Roman", size) > A4_W - 2 * MARGIN:
                size -= 1                      # shrink long titles to fit the page
            c.setFont("Times-Roman", size)
            c.drawCentredString(A4_W / 2, top - 18, title)
            c.setFont("Times-Roman", 12)
            c.drawCentredString(A4_W / 2, top - 36, subtitle)
            c.setFont("Times-Roman", 9)
            c.drawString(MARGIN, top - 58, left)
            if sym:
                c.setFont(sym, 13)
                c.drawString(MARGIN + 18, top - 78, "\u2669")
                c.setFont("Times-Bold", 10)
                c.drawString(MARGIN + 28, top - 77, f"= {round(tempo)}")
            else:
                c.setFont("Times-Bold", 10)
                c.drawString(MARGIN + 18, top - 77, f"Tempo {round(tempo)}")
            avail_top = top - HEADER_FIRST
        else:
            c.setFont("Times-Italic", 8)
            c.drawString(MARGIN, top - 8, f"{title[:60]} - {subtitle}")
            avail_top = top - HEADER_OTHER
        c.setFont("Times-Roman", 8)
        c.drawRightString(A4_W - MARGIN, MARGIN - 14, f"Page {i}/{n_pages}")
        drawing = svg2rlg(io.BytesIO(_clean_svg(tk.renderToSVG(i))))
        scale = min((A4_W - 2 * MARGIN) / drawing.width, (avail_top - MARGIN) / drawing.height)
        h = drawing.height * scale
        drawing.width, drawing.height = drawing.width * scale, h
        drawing.scale(scale, scale)
        renderPDF.draw(drawing, c, MARGIN, avail_top - h)
        c.showPage()
    c.save()
    return n_pages
