"""page_text_extractor.py
==================================================
Extract raw OCR text **page‑by‑page** from a single PDF and save
each page’s text to an individual `.txt` file.

This utility is intentionally simple so you can confirm the OCR
output before moving on to more complex post‑processing.

Public API
----------
>>> from page_text_extractor import ExtractConfig, extract_pages
>>> cfg = ExtractConfig(pdf_path="/path/to/01.pdf", out_dir="/tmp/out")
>>> extract_pages(cfg)

It also doubles as a CLI tool:
>>> python -m page_text_extractor --pdf /path/to/01.pdf --out_dir /tmp/out
"""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

import pytesseract
from PIL import Image

# --- Optional / Lazy imports -------------------------------------------------
try:
    import fitz  # PyMuPDF
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "PyMuPDF is not installed. Run: pip install PyMuPDF"
    ) from exc

# tqdm is optional; fall back to plain range if unavailable
try:
    from tqdm import tqdm  # type: ignore
except ImportError:  # pragma: no cover
    tqdm = None  # type: ignore

__all__ = ["ExtractConfig", "extract_pages"]

# ---------------------------------------------------------------------------
# Dataclass holding configuration -------------------------------------------
# ---------------------------------------------------------------------------

@dataclass
class ExtractConfig:
    """Runtime parameters for page‑by‑page OCR extraction."""

    pdf_path: Path | str
    out_dir: Path | str
    dpi: int = 300
    lang: str = "eng"
    overwrite: bool = False
    show_progress: bool = True

    def __post_init__(self) -> None:  # normalise to Path objects
        self.pdf_path = Path(self.pdf_path).expanduser().resolve()
        self.out_dir = Path(self.out_dir).expanduser().resolve()
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {self.pdf_path}")
        self.out_dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Core function --------------------------------------------------------------
# ---------------------------------------------------------------------------

def extract_pages(cfg: ExtractConfig) -> List[Path]:
    """OCR every page of *cfg.pdf_path* and save as text files in *cfg.out_dir*.

    Returns a list of `Path`s to the text files that were written (or
    overwritten) during this run.
    """

    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    log = logging.getLogger("extract_pages")

    log.info("Opening %s", cfg.pdf_path)
    doc = fitz.open(cfg.pdf_path)

    dpi_scale = cfg.dpi / 72  # PyMuPDF renders at 72 dpi by default
    matrix = fitz.Matrix(dpi_scale, dpi_scale)

    iterator = range(doc.page_count)
    if cfg.show_progress and tqdm is not None:
        iterator = tqdm(iterator, desc="OCR pages", unit="page")  # type: ignore

    written: List[Path] = []

    for page_index in iterator:  # type: ignore
        page = doc.load_page(page_index)
        pix = page.get_pixmap(matrix=matrix)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        text = pytesseract.image_to_string(img, lang=cfg.lang)

        txt_name = f"page_{page_index + 1:03}.txt"
        txt_path = cfg.out_dir / txt_name

        if txt_path.exists() and not cfg.overwrite:
            log.debug("Skipping existing %s", txt_path.name)
            continue

        txt_path.write_text(text, encoding="utf-8", errors="replace")
        written.append(txt_path)

    log.info("Wrote %s text files to %s", len(written), cfg.out_dir)
    return written


# ---------------------------------------------------------------------------
# CLI entry‑point ------------------------------------------------------------
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> ExtractConfig:
    p = argparse.ArgumentParser(description="OCR every page of a PDF to .txt files")
    p.add_argument("--pdf", required=True, help="Path to the input PDF")
    p.add_argument("--out_dir", required=True, help="Directory to store output .txt files")
    p.add_argument("--dpi", type=int, default=300, help="Rendering DPI for OCR (default: 300)")
    p.add_argument("--lang", default="eng", help="Tesseract language code (default: eng)")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing .txt files")
    p.add_argument("--no_progress", action="store_true", help="Disable tqdm progress bar")

    ns = p.parse_args(argv)
    return ExtractConfig(
        pdf_path=ns.pdf,
        out_dir=ns.out_dir,
        dpi=ns.dpi,
        lang=ns.lang,
        overwrite=ns.overwrite,
        show_progress=not ns.no_progress,
    )


def main(argv: list[str] | None = None) -> None:  # pragma: no cover
    cfg = _parse_args(argv)
    extract_pages(cfg)


if __name__ == "__main__":  # pragma: no cover
    main()
