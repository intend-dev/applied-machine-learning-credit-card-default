#!/usr/bin/env python3
"""Build module_summary.pdf from module_summary.md (Chrome headless) and verify it.

Run from the repo root (or anywhere; paths resolve from this file):

    .venv\\Scripts\\python.exe build\\build_report.py

Exit 0 only when every gate passes; exit 1 prints each failing gate.
The script never writes module_summary.md.
"""

from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
import sys
from pathlib import Path

import markdown
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent.parent
MD = ROOT / "module_summary.md"
PDF = ROOT / "module_summary.pdf"
OUT_DIR = ROOT / "build" / "out"
HTML = OUT_DIR / "report.html"
PREV_PDF = ROOT / "test-runs" / "module_summary.previous.pdf"
CHROME = Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")

HEADINGS = (
    "Overview",
    "Dataset Description",
    "Modeling Approach",
    "Results",
    "Interpretation for a Non-Technical Audience",
    "Limitations and Potential Bias",
    "References",
)

DATE_LINE = "3 November 2026"

# Pinned headline values from test-runs/2026-11-03-runall.txt (summary block).
PINNED = (
    ("0.7788", "majority-class baseline"),
    ("0.7266", "logistic regression CV AUC"),
    ("0.7617", "random forest CV AUC"),
    ("0.7795", "tuned forest CV AUC"),
    ("0.8220", "test accuracy"),
    ("0.6837", "test precision"),
    ("0.3632", "test recall"),
    ("0.4744", "test F1"),
    ("0.7799", "test ROC AUC"),
    ("0.174", "cost-tuned cut-off"),
    ("4,448", "cost at the 0.5 cut-off"),
    ("3,347", "cost at the tuned cut-off"),
    ("0.0694", "X6 permutation importance"),
    ("0.7796", "deduplicated ROC AUC"),
    ("0.3389", "male false-positive rate"),
    ("0.4314", "male flagged share"),
    ("0.2958", "female false-positive rate"),
    ("0.3819", "female flagged share"),
    ("0.4810", "high-school flagged share"),
    ("0.4020", "overall flagged share"),
)

IMAGE_COUNT = 2

CSS = """
@page {
  size: A4;
  margin: 20mm 18mm 20mm 18mm;
}
body {
  font-family: "Times New Roman", Georgia, serif;
  font-size: 11.5pt;
  line-height: 1.45;
  color: #111;
  margin: 0;
}
h1 {
  font-size: 19pt;
  line-height: 1.2;
  margin: 0 0 10px;
  padding-bottom: 8px;
  border-bottom: 2px solid #222;
}
h2 {
  font-size: 14.5pt;
  margin: 24px 0 8px;
  padding-bottom: 3px;
  border-bottom: 1px solid #999;
  page-break-after: avoid;
}
h3 {
  font-size: 12pt;
  margin: 18px 0 6px;
  page-break-after: avoid;
}
p { margin: 0 0 8px; }
ul, ol { margin: 4px 0 10px 20px; padding: 0; }
li { margin-bottom: 3px; }
table {
  border-collapse: collapse;
  width: 100%;
  margin: 8px 0 14px;
  font-size: 10.5pt;
  page-break-inside: avoid;
}
th, td {
  border: 1px solid #888;
  padding: 4px 7px;
  text-align: left;
}
th { background: #eeeeee; }
img {
  max-width: 100%;
  height: auto;
  display: block;
  margin: 14px auto 6px;
  page-break-inside: avoid;
}
"""

TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Machine Learning Analysis Report - Credit Card Default Prediction</title>
<style>
{css}
</style>
</head>
<body>
{body}
</body>
</html>
"""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fail(lines: list[str]) -> int:
    for line in lines:
        print(f"FAIL {line}")
    return 1


def absolutize_images(html_text: str) -> str:
    """Point every <img src> at an absolute file:/// URI; a missing figure fails."""
    def repl(match: re.Match[str]) -> str:
        src = match.group(1)
        target = (ROOT / src).resolve()
        if not target.is_file():
            raise SystemExit(f"FAIL figure not found: {src} -> {target}")
        return f'src="{target.as_uri()}"'

    return re.sub(r'src="([^"]+)"', repl, html_text)


def main() -> int:
    if not MD.is_file():
        return fail([f"source missing: {MD}"])
    if not CHROME.is_file():
        return fail([f"chrome missing: {CHROME}"])

    md_hash_before = sha256(MD)
    body = markdown.markdown(
        MD.read_text(encoding="utf-8"), extensions=["extra", "sane_lists"]
    )
    body = absolutize_images(body)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    HTML.write_text(TEMPLATE.format(css=CSS, body=body), encoding="utf-8")

    if PDF.exists():
        PREV_PDF.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PDF, PREV_PDF)
        PDF.unlink()

    cmd = [
        str(CHROME),
        "--headless",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={PDF}",
        HTML.as_uri(),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return fail([f"chrome exit {proc.returncode}", (proc.stderr or "").strip()[:500]])
    if not PDF.is_file():
        return fail(["chrome reported success but wrote no PDF"])
    if PDF.stat().st_size == 0:
        return fail(["PDF is empty"])
    if PDF.read_bytes()[:4] != b"%PDF":
        return fail(["PDF does not start with %PDF"])

    reader = PdfReader(str(PDF))
    problems: list[str] = []
    texts: list[str] = []
    image_count = 0
    for number, page in enumerate(reader.pages, 1):
        text = page.extract_text() or ""
        texts.append(text)
        page_images = len(page.images)
        image_count += page_images
        if not text.strip() and page_images == 0:
            problems.append(f"blank page {number}")

    flat = re.sub(r"\s+", " ", "\n".join(texts))

    for heading in HEADINGS:
        if heading not in flat:
            problems.append(f"heading missing: {heading}")
    if DATE_LINE not in flat:
        problems.append(f"date line missing: {DATE_LINE}")
    for value, label in PINNED:
        if value not in flat:
            problems.append(f"pinned value missing: {value} ({label})")
    if image_count != IMAGE_COUNT:
        problems.append(f"image count {image_count} != {IMAGE_COUNT}")

    text = "\n".join(texts)
    if "`" in text:
        problems.append("literal backtick in extracted text")
    if "$" in text:
        problems.append("literal dollar sign in extracted text")
    if "\ufffd" in text:
        problems.append("U+FFFD replacement character in extracted text")

    md_hash_after = sha256(MD)
    if md_hash_after != md_hash_before:
        problems.append("source Markdown changed during the build")

    if problems:
        return fail(problems)

    print(f"PASS pages={len(reader.pages)} images={image_count} size={PDF.stat().st_size} bytes")
    print(f"pdf sha256={sha256(PDF)}")
    print(f"md  sha256={md_hash_after}  source md unchanged")
    return 0


if __name__ == "__main__":
    sys.exit(main())
