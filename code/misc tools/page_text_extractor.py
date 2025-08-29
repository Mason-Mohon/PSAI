from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
import pytesseract
from PIL import Image

# tqdm is optional; we import lazily to avoid hard dep
try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None  # type: ignore


@dataclass
class ExtractConfig:
    pdf_path: Path
    out_path: Optional[Path] = None  # default: same stem + .txt in pdf dir
    dpi: int = 300
    lang: str = "eng"
    overwrite: bool = False
    show_progress: bool = True


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _render_page(page, dpi: int) -> Image.Image:
    """Render a PyMuPDF page to a PIL Image."""
    zoom = dpi / 72  # PyMuPDF pages are 72 dpi default
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    mode = "RGB" if pix.n < 4 else "RGBA"
    return Image.frombytes(mode, [pix.width, pix.height], pix.samples)


def _ocr_image(img: Image.Image, lang: str) -> str:
    """Run Tesseract OCR on the PIL image and return raw text."""
    return pytesseract.image_to_string(img, lang=lang)


# ---------------------------------------------------------------------------
# Public runner
# ---------------------------------------------------------------------------

def extract_pdf_to_txt(cfg: ExtractConfig) -> Path:
    """OCR the entire PDF and save one TXT file with blank lines between pages.

    Parameters
    ----------
    cfg : ExtractConfig
        Configuration dataclass.

    Returns
    -------
    Path
        Path to the written TXT file.
    """
    pdf_path = cfg.pdf_path.expanduser().resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)

    out_path = (
        cfg.out_path.expanduser().resolve()
        if cfg.out_path
        else pdf_path.with_suffix(".txt")
    )

    if out_path.exists() and not cfg.overwrite:
        logging.info("%s already exists; skipping (use overwrite=True)", out_path)
        return out_path

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(message)s",
        stream=sys.stdout,
    )

    logging.info("Opening %s", pdf_path)
    doc = fitz.open(pdf_path)
    total_pages = doc.page_count

    iterator = range(total_pages)
    if cfg.show_progress and tqdm is not None:
        iterator = tqdm(iterator, desc="OCR pages", unit="page")

    page_texts: list[str] = []
    for i in iterator:
        page = doc.load_page(i)
        img = _render_page(page, cfg.dpi)
        text = _ocr_image(img, cfg.lang).strip()
        page_texts.append(text)

    joined = "\n\n".join(page_texts) + "\n"  # final newline

    out_path.write_text(joined, encoding="utf-8")
    logging.info("Wrote %d pages → %s (%.1f KB)", total_pages, out_path, out_path.stat().st_size / 1024)
    return out_path

# Alias for backward compatibility
extract_pages_to_txt = extract_pdf_to_txt

__all__ = ["ExtractConfig", "extract_pdf_to_txt", "extract_pages_to_txt"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="OCR a PDF to a single TXT file.")
    p.add_argument("--pdf", required=True, type=Path, help="Path to input PDF")
    p.add_argument("--out", type=Path, help="Path to output TXT (optional)")
    p.add_argument("--dpi", type=int, default=300, help="Render DPI (default 300)")
    p.add_argument("--lang", default="eng", help="Tesseract language (default 'eng')")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing output file")
    p.add_argument("--no-progress", dest="show_progress", action="store_false", help="Disable tqdm progress bar")
    return p


def _main_from_cli(argv: list[str] | None = None):  # pragma: no cover
    args = _build_cli().parse_args(argv)
    cfg = ExtractConfig(
        pdf_path=args.pdf,
        out_path=args.out,
        dpi=args.dpi,
        lang=args.lang,
        overwrite=args.overwrite,
        show_progress=args.show_progress,
    )
    extract_pdf_to_txt(cfg)


if __name__ == "__main__":  # pragma: no cover
    _main_from_cli()
