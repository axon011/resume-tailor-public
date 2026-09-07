"""Scan generated PDFs for ATS-breaking patterns.

Extracts text using pypdf (same library most ATS parsers wrap) and
counts five known failure classes. Prints a per-file table.
"""
from pypdf import PdfReader
import re, os, glob, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

REPLACEMENT = chr(0xFFFD)  # U+FFFD REPLACEMENT CHARACTER

def scan(path: str) -> dict:
    r = PdfReader(path)
    txt = "\n".join(p.extract_text() or "" for p in r.pages)
    return {
        "replacement_chars": txt.count(REPLACEMENT),
        "label_no_space": len(re.findall(r"[A-Za-z]:[A-Za-z]", txt)),
        "pipe_attached": len(re.findall(r"[A-Za-z0-9]\|", txt)),
        "hyphen_linebreak": len(re.findall(r"-\n", txt)),
        "curly_quotes": sum(txt.count(c) for c in "‘’“”"),
        "year_to_present": len(re.findall(r"\d{4}\s*\S\s*Present", txt)),
    }

def main(pattern: str):
    files = sorted(glob.glob(pattern))
    files = [f for f in files if "CoverLetter" not in os.path.basename(f)]
    if not files:
        print("No files matched", pattern); return
    print(f"{'file':<48} {'�':>4} {'lbl:no_sp':>10} {'pipe|':>6} {'hyph-LF':>8} {'curly':>6} {'YYYY?Pres':>10}")
    print("-" * 100)
    for f in files:
        try:
            s = scan(f)
        except Exception as e:
            print(os.path.basename(f), "ERROR", e); continue
        print(f"{os.path.basename(f):<48} {s['replacement_chars']:>4} {s['label_no_space']:>10} {s['pipe_attached']:>6} {s['hyphen_linebreak']:>8} {s['curly_quotes']:>6} {s['year_to_present']:>10}")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "*.pdf"))
