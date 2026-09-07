"""Dump full extracted text from a PDF for inspection."""
from pypdf import PdfReader
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
r = PdfReader(sys.argv[1])
for p in r.pages:
    print(p.extract_text())
