"""psai_extract.py
================================================
Comprehensive extraction pipeline for Phyllis Schlafly
Christian Radio commentaries.  Designed to work on a single
month (one PDF) at a time and to be imported inside a
Jupyter notebook or called via CLI.

Author: ChatGPT (o3) - generated implementation
First release: 2025-04-22
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import fitz  # PyMuPDF
import numpy as np
import pandas as pd
import pytesseract
import cv2 as cv
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_DPI = 300
_NUM_PATTERN = re.compile(r"\b(\d{2}-\d{2})\b")
_WEEK_PATTERN = re.compile(r"Week of\s+([A-Za-z]+\s+\d{1,2},?\s+\d{4})")
_TITLE_CASE_PATTERN = re.compile(r"^[A-Z][^a-z]*[A-Z][A-Za-z'].+")  # loose


def _month_int(month: int | str) -> int:
    m = int(month)
    if not 1 <= m <= 12:
        raise ValueError("month must be 1‑12")
    return m


@dataclass
class Config:
    year: int
    month: int
    tesseract_lang: str = "eng"
    dpi: int = DEFAULT_DPI
    min_reference_char_len: int = 250
    log_level: str = "INFO"
    project_root: Path = Path.cwd()

    # derived paths (filled post‑init)
    raw_pdf: Path = field(init=False)
    ocr_dir: Path = field(init=False)
    tmp_dir: Path = field(init=False)
    json_path: Path = field(init=False)

    def __post_init__(self):
        self.month = _month_int(self.month)
        yy = str(self.year)
        mm = f"{self.month:02d}"
        self.raw_pdf = (
            self.project_root / "data" / "raw" / "comm" / yy / f"{mm}.pdf"
        )
        self.ocr_dir = self.project_root / "data" / "ocr" / yy / mm
        self.tmp_dir = self.project_root / "data" / "tmp" / yy / mm
        self.json_path = self.project_root / "data" / "json" / yy / f"{mm}.jsonl"
        self.ocr_dir.mkdir(parents=True, exist_ok=True)
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.json_path.parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=getattr(logging, self.log_level.upper(), logging.INFO),
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(self.tmp_dir / "extract.log", mode="w"),
            ],
        )


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class PageInfo:
    page_id: int
    img_path: Path
    txt_path: Path
    text: str
    commentary_number: Optional[str]
    page_type: str  # TOC | Script | Reference | Continuation | Unknown
    week_header: Optional[str] = None  # only for TOC


@dataclass
class Commentary:
    number: str
    title: str
    date_string: str
    year: int
    month: int
    page_ids: List[int]
    page_type: str  # Script or Reference
    text: str

    def to_json(self) -> str:
        rec = {
            "metadata": {
                "speaker": "Phyllis Schlafly",
                "commentary_number": self.number,
                "title": self.title,
                "date_string": self.date_string,
                "year": self.year,
                "month": self.month,
                "page_ids": self.page_ids,
                "page_type": self.page_type,
            },
            "text": self.text.strip(),
        }
        return json.dumps(rec, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Utility functions – OCR & CV helpers
# ---------------------------------------------------------------------------


def render_page(page: fitz.Page, dpi: int) -> np.ndarray:
    zoom = dpi / 72  # default 72 dpi in PDF
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    return cv.cvtColor(img, cv.COLOR_RGB2BGR)


def ocr_image(img: np.ndarray, lang: str = "eng", config: str = "") -> str:
    pil = Image.fromarray(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    return pytesseract.image_to_string(pil, lang=lang, config=config)


def detect_number_cv(img: np.ndarray) -> Optional[str]:
    """Attempt to read the commentary number from the top‑right corner via CV."""
    h, w = img.shape[:2]
    # ROI – upper‑right 15% width, 15% height
    roi = img[0 : int(0.15 * h), int(0.85 * w) : w]
    gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
    thr = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C,
                               cv.THRESH_BINARY_INV, 25, 15)
    # OCR on ROI with digit whitelist
    custom_oem_psm_config = (
        "--oem 1 --psm 6 -c tessedit_char_whitelist=0123456789- "
    )
    text = pytesseract.image_to_string(
        Image.fromarray(thr), config=custom_oem_psm_config, lang="eng"
    )
    match = _NUM_PATTERN.search(text)
    return match.group(1) if match else None


# ---------------------------------------------------------------------------
# Stage 0‑1 · Load + OCR
# ---------------------------------------------------------------------------


def load_and_ocr(config: Config) -> List[PageInfo]:
    doc = fitz.open(config.raw_pdf)
    pages: List[PageInfo] = []
    logging.info("Opened %s (%d pages)", config.raw_pdf, doc.page_count)
    for i in tqdm(range(doc.page_count), desc="OCR pages"):
        p = doc.load_page(i)
        img = render_page(p, config.dpi)
        # Save image for traceability
        img_path = config.ocr_dir / f"page_{i:03d}.png"
        cv.imwrite(str(img_path), img)
        # OCR full page
        txt = ocr_image(img, lang=config.tesseract_lang, config="--oem 3 --psm 1")
        txt_path = config.ocr_dir / f"page_{i:03d}.txt"
        txt_path.write_text(txt, encoding="utf-8")
        # Detect commentary number
        num = detect_number_cv(img)
        if not num:
            # fallback: regex in text
            m = _NUM_PATTERN.search(txt)
            num = m.group(1) if m else None
        pages.append(
            PageInfo(
                page_id=i,
                img_path=img_path,
                txt_path=txt_path,
                text=txt,
                commentary_number=num,
                page_type="Unknown",
            )
        )
    return pages


# ---------------------------------------------------------------------------
# Stage 2 · Classification functions
# ---------------------------------------------------------------------------


def classify_pages(pages: List[PageInfo]):
    for idx, page in enumerate(pages):
        txt = page.text
        # TOC detection
        if "Week of" in txt and _NUM_PATTERN.search(txt):
            page.page_type = "TOC"
            wh = _WEEK_PATTERN.search(txt)
            if wh:
                page.week_header = f"Week of {wh.group(1)}"
            continue
        if page.commentary_number:
            # Could be Script or Reference (first page)
            # Check for centered title: we approximate by first 3 lines average indent
            lines = [l.strip("\n\r ") for l in txt.splitlines() if l.strip()]
            first_line = lines[0] if lines else ""
            # rudimentary title test
            if first_line.istitle() or _TITLE_CASE_PATTERN.match(first_line):
                page.page_type = "Script"
            else:
                page.page_type = "Reference"
        else:
            # possible continuation
            prev = pages[idx - 1] if idx > 0 else None
            if prev and prev.page_type == "Reference":
                page.page_type = "Continuation"
            else:
                page.page_type = "Unknown"
        logging.debug("Page %d classified as %s", page.page_id, page.page_type)


# ---------------------------------------------------------------------------
# Stage 3 · Parse TOC
# ---------------------------------------------------------------------------


def build_toc_mapping(pages: List[PageInfo]) -> Dict[str, Dict[str, str]]:
    toc: Dict[str, Dict[str, str]] = {}
    for page in pages:
        if page.page_type != "TOC":
            continue
        # Extract week header from current page (captured earlier)
        week_header = page.week_header or ""
        for line in page.text.splitlines():
            m = _NUM_PATTERN.search(line)
            if not m:
                continue
            num = m.group(1)
            # take remainder of line as title (strip numbers and extra spaces)
            title = line[m.end() :].strip()
            # remove trailing dot leaders / columns etc.
            title = re.sub(r"\s{2,}.*", "", title)
            if num not in toc:
                toc[num] = {"title": title, "week_header": week_header}
    logging.info("Built TOC mapping for %d entries", len(toc))
    return toc


# ---------------------------------------------------------------------------
# Stage 4 · Assemble commentaries
# ---------------------------------------------------------------------------


def assemble_commentaries(
    pages: List[PageInfo], config: Config, toc: Dict[str, Dict[str, str]]
) -> List[Commentary]:
    commentaries: List[Commentary] = []
    idx = 0
    while idx < len(pages):
        page = pages[idx]
        if page.page_type in {"Script", "Reference"}:
            num = page.commentary_number
            if not num:
                logging.warning(
                    "Page %d marked %s but no commentary number found" % (page.page_id, page.page_type)
                )
                idx += 1
                continue
            body_lines: List[str] = []
            page_ids: List[int] = []
            # gather pages (this and continuation!)
            while idx < len(pages):
                p = pages[idx]
                if idx != page.page_id and p.page_type in {"Script", "Reference"} and p.commentary_number != num:
                    break  # start of next commentary
                if idx != page.page_id and p.page_type == "Unknown":
                    break
                body_lines.append(p.text)
                page_ids.append(p.page_id)
                idx += 1
                # stop if next page is not continuation and not same number
                if idx < len(pages):
                    nxt = pages[idx]
                    if nxt.page_type == "Continuation":
                        continue
                    if nxt.page_type in {"Script", "Reference"} and nxt.commentary_number == num:
                        continue  # multi‑page reference
                    # else break out to outer loop logic
            # Use TOC data if available
            if num in toc:
                title = toc[num]["title"]
                date_string = toc[num]["week_header"] or f"{config.year}-{config.month:02d}"
            else:
                title = page.text.split("\n", 1)[0].strip()
                date_string = f"{config.year}-{config.month:02d}"
            commentary_text = clean_text("\n".join(body_lines))
            commentaries.append(
                Commentary(
                    number=num,
                    title=title,
                    date_string=date_string,
                    year=config.year,
                    month=config.month,
                    page_ids=page_ids,
                    page_type=page.page_type,
                    text=commentary_text,
                )
            )
        else:
            idx += 1
    logging.info("Assembled %d commentaries", len(commentaries))
    return commentaries


# ---------------------------------------------------------------------------
# Stage 5 · Text cleanup
# ---------------------------------------------------------------------------


_HYPHEN_RE = re.compile(r"(\w+)-\n(\w+)")
_MULTISPACE_RE = re.compile(r"[ \t]{2,}")


def clean_text(text: str) -> str:
    """De‑hyphenate wrapped words, normalise whitespace and smart quotes."""
    # De‑hyphenate if line ends with hyphen and next line continues word
    while True:
        new_text = _HYPHEN_RE.sub(r"\1\2", text)
        if new_text == text:
            break
        text = new_text
    # Replace smart quotes with ASCII (minimal subset)
    text = text.replace("“", '"').replace("”", '"').replace("’", "'")
    # Collapse multiple spaces
    text = _MULTISPACE_RE.sub(" ", text)
    # Collapse multiple blank lines to max 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Stage 6 · Emit JSONL
# ---------------------------------------------------------------------------


def write_jsonl(commentaries: Sequence[Commentary], path: Path):
    with path.open("w", encoding="utf-8") as f:
        for c in commentaries:
            f.write(c.to_json() + "\n")
    logging.info("Wrote %s (%d records)", path, len(commentaries))


# ---------------------------------------------------------------------------
# Runner function – notebook & CLI entry
# ---------------------------------------------------------------------------


def run_month(config: Config) -> Tuple[Path, List[Commentary]]:
    """Main orchestration function. Returns path to JSONL and the list."""
    pages = load_and_ocr(config)
    classify_pages(pages)
    toc = build_toc_mapping(pages)
    commentaries = assemble_commentaries(pages, config, toc)
    write_jsonl(commentaries, config.json_path)
    return config.json_path, commentaries

# ---------------------------------------------------------------------------
# CLI wrapper
# ---------------------------------------------------------------------------


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract Phyllis Schlafly commentaries from PDF")
    p.add_argument("year", type=int, help="4‑digit year, e.g. 2002")
    p.add_argument("month", type=int, help="Month 1‑12")
    p.add_argument("--root", type=Path, default=Path.cwd(), help="Project root path")
    p.add_argument("--debug", action="store_true", help="Set log level to DEBUG")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    cfg = Config(
        year=args.year,
        month=args.month,
        log_level="DEBUG" if args.debug else "INFO",
        project_root=args.root,
    )
    try:
        json_path, _ = run_month(cfg)
        print(f"✅ Extraction complete. JSON saved to {json_path}")
    except Exception as e:
        logging.exception("Extraction failed: %s", e)
        sys.exit(1)