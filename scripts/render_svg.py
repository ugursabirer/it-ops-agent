"""Render a captured terminal transcript as an SVG 'screenshot'.

The assistant is a terminal application, so its screenshots are terminal
output. Rendering the real captured transcript to SVG keeps the images
reproducible and reviewable in the repository (GitHub renders SVG inside
Markdown), instead of pasting in a bitmap nobody can diff.

Used as a module by ``scripts/03_build_screenshots.py``, or directly:

    python scripts/render_svg.py <transcript.txt> <output.svg> "Baslik"
"""

from __future__ import annotations

import html
import sys
from pathlib import Path

FONT_SIZE = 13
LINE_HEIGHT = 19
CHAR_WIDTH = 7.22
PADDING = 18
TITLE_BAR = 34
MAX_COLUMNS = 104

BACKGROUND = "#11131a"
TITLE_BAR_BG = "#1b1e27"
DEFAULT = "#d6dae3"

# Trace prefixes are coloured so the agent's steps stand out from its answers.
COLOURS: list[tuple[str, str]] = [
    ("[SUPERVISOR]", "#f0a73a"),
    ("[AGENT]", "#58b6ff"),
    ("[PLAN]", "#c08bff"),
    ("[MCP]", "#4fd18b"),
    ("[TOOL]", "#4fd18b"),
    ("[MCP TOOL]", "#4fd18b"),
    ("[ONAY]", "#ff6d6d"),
    ("[GUARD]", "#ffd166"),
    ("[UYARI]", "#ffd166"),
    ("[HATA]", "#ff6d6d"),
    ("[BILGI]", "#8a93a6"),
    ("Kullanici:", "#ffffff"),
    ("Asistan:", "#9ae6b4"),
    ("SENARYO", "#f0a73a"),
]


def _colour(line: str) -> str:
    stripped = line.strip()

    for prefix, colour in COLOURS:
        if stripped.startswith(prefix):
            return colour

    if set(stripped) and set(stripped) <= {"=", "-", "!", "#"}:
        return "#3b4256"

    if stripped.startswith("ONAY GEREKIYOR"):
        return "#ff6d6d"

    return DEFAULT


def _wrap(line: str) -> list[str]:
    if len(line) <= MAX_COLUMNS:
        return [line]

    pieces: list[str] = []
    remaining = line

    while len(remaining) > MAX_COLUMNS:
        cut = remaining.rfind(" ", 0, MAX_COLUMNS)

        if cut <= 0:
            cut = MAX_COLUMNS

        pieces.append(remaining[:cut])
        # Continuation lines are indented so wrapped text stays readable.
        remaining = "  " + remaining[cut:].lstrip()

    pieces.append(remaining)

    return pieces


def render(transcript: str, title: str) -> str:
    lines: list[str] = []

    for raw in transcript.replace("\t", "    ").splitlines():
        lines.extend(_wrap(raw.rstrip()))

    width = int(MAX_COLUMNS * CHAR_WIDTH + PADDING * 2)
    height = int(TITLE_BAR + PADDING * 2 + len(lines) * LINE_HEIGHT)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}" '
        f'font-family="ui-monospace, SFMono-Regular, Menlo, Consolas, monospace" '
        f'font-size="{FONT_SIZE}">',
        f'<rect width="{width}" height="{height}" rx="10" fill="{BACKGROUND}"/>',
        f'<path d="M0 10 a10 10 0 0 1 10 -10 h{width - 20} a10 10 0 0 1 10 10 '
        f'v{TITLE_BAR - 10} h-{width} z" fill="{TITLE_BAR_BG}"/>',
        '<circle cx="20" cy="17" r="5.5" fill="#ff5f57"/>',
        '<circle cx="39" cy="17" r="5.5" fill="#febc2e"/>',
        '<circle cx="58" cy="17" r="5.5" fill="#28c840"/>',
        f'<text x="78" y="22" fill="#8a93a6">{html.escape(title)}</text>',
    ]

    y = TITLE_BAR + PADDING + FONT_SIZE

    for line in lines:
        if line.strip():
            parts.append(
                f'<text x="{PADDING}" y="{y}" fill="{_colour(line)}" '
                f'xml:space="preserve">{html.escape(line)}</text>'
            )
        y += LINE_HEIGHT

    parts.append("</svg>")

    return "\n".join(parts)


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2

    source = Path(sys.argv[1])
    target = Path(sys.argv[2])
    title = sys.argv[3] if len(sys.argv) > 3 else source.name

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        render(source.read_text(encoding="utf-8"), title), encoding="utf-8"
    )

    print(f"[OK] {target}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
