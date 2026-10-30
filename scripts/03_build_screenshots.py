"""Slice a captured --demo transcript into per-scenario SVG screenshots.

    python scripts/03_build_screenshots.py

Reads ``docs/screenshots/demo_transcript.txt`` (produced by
``python main.py --demo | tee docs/screenshots/demo_transcript.txt``), splits it
on the scenario separators, renders one SVG per scenario and writes the Markdown
block that goes into the README.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import render_svg  # noqa: E402

BASE = Path(__file__).resolve().parent.parent
TRANSCRIPT = BASE / "docs" / "screenshots" / "demo_transcript.txt"
OUTPUT_DIR = BASE / "docs" / "screenshots"

TITLES = {
    1: "01-karsilama",
    2: "02-politika-rag",
    3: "03-operasyon-mcp",
    4: "04-cok-adimli-talep-ve-onay",
    5: "05-politikaya-uygun-degil",
    6: "06-belirsiz-talep",
}

CAPTIONS = {
    1: "Greeting Agent — yetenek özeti",
    2: "Policy Agent — RAG ile politika sorgusu, kaynak göstererek",
    3: "Operations Agent — MCP tool'u ile zimmet sorgusu",
    4: "Çok adımlı talep — plan, RAG, MCP zinciri, karar, insan onayı, talep açma",
    5: "Aynı soru, politikaya uymayan çalışan için — talep reddi",
    6: "Belirsiz talep — agent tool çağırmak yerine soru soruyor",
}

HEADER_CAPTION = "Başlangıç — aktif transport ve yapılandırma"

_SCENARIO = re.compile(r"^\s*SENARYO\s+(\d+)/(\d+)\s*$")


def split_scenarios(transcript: str) -> dict[int, str]:
    """Split the transcript into {scenario_number: text}."""

    sections: dict[int, list[str]] = {}
    current: int | None = None

    for line in transcript.splitlines():
        match = _SCENARIO.match(line)

        if match:
            current = int(match.group(1))
            sections[current] = [line.strip()]
            continue

        if current is not None:
            # The separator rows around a scenario header add nothing once the
            # header itself is kept.
            if set(line.strip()) == {"="}:
                continue
            sections[current].append(line)

    return {number: "\n".join(lines).strip() for number, lines in sections.items()}


def header(transcript: str) -> str:
    """The startup banner, up to the first scenario."""

    lines: list[str] = []

    for line in transcript.splitlines():
        if _SCENARIO.match(line):
            break
        lines.append(line)

    return "\n".join(lines).strip()


def main() -> int:
    if not TRANSCRIPT.exists():
        print(f"[HATA] Transkript bulunamadi: {TRANSCRIPT}")
        print(f"       Once: python main.py --demo | tee {TRANSCRIPT}")
        return 1

    transcript = TRANSCRIPT.read_text(encoding="utf-8")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    written: list[tuple[str, str]] = []

    banner = header(transcript)

    if banner:
        target = OUTPUT_DIR / "00-baslangic.svg"
        target.write_text(
            render_svg.render(banner, "python main.py --demo"),
            encoding="utf-8",
        )
        written.append((HEADER_CAPTION, target.name))
        print(f"[OK] {target.name}")

    for number, text in sorted(split_scenarios(transcript).items()):
        name = TITLES.get(number, f"{number:02d}-senaryo")
        target = OUTPUT_DIR / f"{name}.svg"
        target.write_text(
            render_svg.render(text, f"Senaryo {number}"), encoding="utf-8"
        )
        written.append((CAPTIONS.get(number, f"Senaryo {number}"), target.name))
        print(f"[OK] {target.name}")

    block = "\n\n".join(
        f"**{caption}**\n\n![{caption}](docs/screenshots/{file_name})"
        for caption, file_name in written
    )

    markdown = OUTPUT_DIR / "README_block.md"
    markdown.write_text(block + "\n", encoding="utf-8")

    print(f"\n[OK] README blogu: {markdown}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
