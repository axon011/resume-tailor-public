#!/usr/bin/env python3
"""Produce upload-safe ATS-flat copies of resume/cover PDFs.

Strict portals (SAP, Atos, ...) reject PDFs with interactive elements (the
clickable email/LinkedIn/GitHub/portfolio hyperlinks). The WRONG workaround is
Print->Save-as-PDF, which re-renders the page and destroys the text layer ->
ATS reads a blank page. This strips the link annotations while keeping the text
layer 100% intact.

Usage: python make_ats_flat.py output/Your_Name_acme_20260101.pdf [more.pdf ...]
Writes <name>_ATSflat.pdf next to each input and verifies text survived.
"""
import sys, os
from pypdf import PdfReader, PdfWriter

def flatten(path):
    r = PdfReader(path)
    w = PdfWriter()
    annots_removed = 0
    for p in r.pages:
        if "/Annots" in p:
            annots_removed += len(p["/Annots"])
            del p["/Annots"]
        w.add_page(p)
    out = path.rsplit(".pdf", 1)[0] + "_ATSflat.pdf"
    with open(out, "wb") as f:
        w.write(f)
    # verify
    r2 = PdfReader(out)
    t = "".join(pp.extract_text() or "" for pp in r2.pages)
    a2 = sum(len(pp.get("/Annots", [])) for pp in r2.pages)
    ok = a2 == 0 and len(t) > 500
    print(f"{'OK ' if ok else 'FAIL'} {os.path.basename(out)}: "
          f"{os.path.getsize(out)//1024}KB, annots {annots_removed}->{a2}, "
          f"text {len(t)} chars, spaces {t.count(' ')}")
    return ok

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python make_ats_flat.py <pdf> [pdf ...]"); sys.exit(1)
    all_ok = all(flatten(p) for p in sys.argv[1:] if p.lower().endswith(".pdf"))
    sys.exit(0 if all_ok else 1)
