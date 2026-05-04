"""Compile or validate the paper scaffold."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    paper = root / "paper" / "main.tex"
    if not paper.is_file():
        raise FileNotFoundError(paper)
    latexmk = shutil.which("latexmk")
    pdflatex = shutil.which("pdflatex")
    if latexmk:
        subprocess.run([latexmk, "-pdf", "-interaction=nonstopmode", "main.tex"], cwd=paper.parent, check=True)
        return
    if pdflatex:
        subprocess.run([pdflatex, "-interaction=nonstopmode", "main.tex"], cwd=paper.parent, check=True)
        subprocess.run([pdflatex, "-interaction=nonstopmode", "main.tex"], cwd=paper.parent, check=True)
        return
    required = [
        root / "paper" / "appendix.tex",
        root / "paper" / "refs.bib",
        root / "tables" / "ablation_table.tex",
        root / "tables" / "ablation_table_dream4.tex",
        root / "tables" / "constraint_quality.tex",
        root / "tables" / "cross_dataset.tex",
    ]
    missing = [p.as_posix() for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing paper dependency: " + ", ".join(missing))
    print("No LaTeX engine found; validated paper source dependencies only.")


if __name__ == "__main__":
    main()
