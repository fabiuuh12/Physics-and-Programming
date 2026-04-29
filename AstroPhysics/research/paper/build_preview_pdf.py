#!/usr/bin/env python3
"""Build a readable PDF preview when a TeX compiler is unavailable.

This is not a replacement for pdflatex/latexmk. It creates a plain manuscript
preview from the current LaTeX source content using Matplotlib's PDF backend.
"""

from __future__ import annotations

from pathlib import Path
import re
import textwrap

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


ROOT = Path(__file__).resolve().parent
MAIN_TEX = ROOT / "main.tex"
OUTPUT = ROOT / "main.pdf"


SECTION_RE = re.compile(r"\\section\{([^}]+)\}")


def clean_latex(text: str) -> str:
    replacements = {
        r"\PhiN": "Phi_N",
        r"\Phieff": "Phi_eff",
        r"\SA": "S_A",
        r"\lambdaA": "lambda_A",
        r"\rhoStruct": "rho_struct",
        r"\vc": "v_c",
        r"\grad": "grad",
        r"\laplacian": "nabla^2",
        r"\bm": "",
        r"\mathrm": "",
        r"\left": "",
        r"\right": "",
        r"\approx": "~=",
        r"\pi": "pi",
        r"\rho": "rho",
        r"\epsilon": "epsilon",
        r"\omega": "omega",
        r"\chi": "chi",
        r"\Omega": "Omega",
        r"\mu": "mu",
        r"\times": "x",
        r"\sqrt": "sqrt",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"\\citep\{([^}]+)\}", r"[\1]", text)
    text = re.sub(r"\\ref\{([^}]+)\}", r"\1", text)
    text = re.sub(r"\\label\{[^}]+\}", "", text)
    text = re.sub(r"\\begin\{[^}]+\}|\\end\{[^}]+\}", "", text)
    text = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]+\])?", "", text)
    text = text.replace("{", "").replace("}", "")
    text = text.replace("$", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_content(source: str) -> tuple[str, list[tuple[str, str]]]:
    title_match = re.search(r"\\title\{([^}]+)\}", source)
    title = title_match.group(1) if title_match else "ASF Manuscript"
    body_match = re.search(r"\\begin\{document\}(.*)\\end\{document\}", source, re.S)
    body = body_match.group(1) if body_match else source
    body = re.sub(r"\\maketitle", "", body)
    body = re.sub(r"\\bibliographystyle\{[^}]+\}", "", body)
    body = re.sub(r"\\bibliography\{[^}]+\}", "", body)

    matches = list(SECTION_RE.finditer(body))
    sections: list[tuple[str, str]] = []
    abstract_match = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", body, re.S)
    if abstract_match:
        sections.append(("Abstract", clean_latex(abstract_match.group(1))))
    for index, match in enumerate(matches):
        heading = match.group(1)
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        sections.append((heading, clean_latex(body[start:end])))
    return title, sections


def draw_wrapped(ax, text: str, x: float, y: float, width: int, size: float, line_h: float) -> float:
    for line in textwrap.wrap(text, width=width):
        ax.text(x, y, line, ha="left", va="top", fontsize=size)
        y -= line_h
    return y


def main() -> None:
    source = MAIN_TEX.read_text(encoding="utf-8")
    title, sections = extract_content(source)
    page_w, page_h = 8.5, 11.0
    page = 1

    def new_page() -> tuple[plt.Figure, plt.Axes, float]:
        fig = plt.figure(figsize=(page_w, page_h))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")
        ax.text(0.5, 0.03, f"Page {page}", ha="center", va="bottom", fontsize=8, color="#667085")
        return fig, ax, 0.93

    with PdfPages(OUTPUT) as pdf:
        fig, ax, y = new_page()
        ax.text(0.5, y, title, ha="center", va="top", fontsize=16, weight="bold", wrap=True)
        y -= 0.07
        ax.text(0.5, y, "Fabio Facin", ha="center", va="top", fontsize=11)
        y -= 0.04
        ax.text(
            0.5,
            y,
            "PDF preview generated from LaTeX source. Full TeX compile requires pdflatex or latexmk.",
            ha="center",
            va="top",
            fontsize=8,
            color="#667085",
        )
        y -= 0.06

        for heading, text in sections:
            lines_needed = len(textwrap.wrap(text, width=92)) + 3
            if y - lines_needed * 0.018 < 0.08:
                pdf.savefig(fig)
                plt.close(fig)
                page += 1
                fig, ax, y = new_page()
            ax.text(0.09, y, heading, ha="left", va="top", fontsize=12, weight="bold")
            y -= 0.03
            for line in textwrap.wrap(text, width=92):
                if y < 0.08:
                    pdf.savefig(fig)
                    plt.close(fig)
                    page += 1
                    fig, ax, y = new_page()
                ax.text(0.09, y, line, ha="left", va="top", fontsize=9.5)
                y -= 0.018
            y -= 0.02

        pdf.savefig(fig)
        plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
