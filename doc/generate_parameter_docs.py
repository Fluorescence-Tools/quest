"""Generate the parameter reference Markdown from the central JSON catalog.

One page per language: `doc/parameters.md` (English) and
`doc/parameters.<code>.md` for every locale in `quest/settings/locales/`. Run

    python doc/generate_parameter_docs.py

after editing `quest/settings/parameter_catalog.json` or any locale file. Never
hand-edit the output.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

DOCS_DIR = Path(__file__).resolve().parent
REPO_ROOT = DOCS_DIR.parent
CATALOG_PATH = REPO_ROOT / "quest" / "settings" / "parameter_catalog.json"
OUTPUT_PATH = DOCS_DIR / "parameters.md"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from quest.i18n import (  # noqa: E402
    DEFAULT_LOCALE,
    available_locales,
    translate_catalog,
    ui_strings,
)


def output_path(locale: str = DEFAULT_LOCALE) -> Path:
    """English keeps the unsuffixed filename; every other language is suffixed."""
    if locale == DEFAULT_LOCALE:
        return OUTPUT_PATH
    return DOCS_DIR / f"parameters.{locale}.md"


def generate(locale: str = DEFAULT_LOCALE) -> Path:
    catalog = translate_catalog(locale)
    parameters = catalog["parameters"]
    strings = ui_strings(locale)
    output = output_path(locale)
    # A comma-joined list, not "a and b": the conjunction would be English in
    # every language, and the one thing this file must not do is leak English.
    source_files = ["`parameter_catalog.json`"]
    if locale != DEFAULT_LOCALE:
        source_files.append(f"`quest/settings/locales/{locale}.json`")
    sources = ", ".join(source_files)
    
    grouped = defaultdict(list)
    for parameter_id, meta in sorted(parameters.items()):
        grouped[meta.get("category", "Other")].append((parameter_id, meta))

    lines = [
        f"# {strings['doc.title']}",
        "",
        strings["doc.generated"].format(sources=sources),
        "",
        f"## {strings['doc.summary']}",
        "",
        "| {key} | {category} | {type} | {unit} | {description} |".format(
            key=strings["doc.col.key"],
            category=strings["doc.col.category"],
            type=strings["doc.col.type"],
            unit=strings["doc.col.unit"],
            description=strings["doc.col.description"],
        ),
        "| --- | --- | --- | --- | --- |",
    ]

    for parameter_id, meta in sorted(parameters.items()):
        unit = meta.get("unit", "-")
        desc = meta.get("description", "")
        category = meta.get("category_label", meta.get("category", ""))
        lines.append(
            f"| `{parameter_id}` | {category} | {meta.get('type', '')} | {unit} | {desc} |"
        )

    lines.append("")

    # Detailed descriptions by category
    category_order = ["Structure", "Labeling site", "Accessible volume", "Simulation", "Quenchers", "Advanced", "FRET Acceptor"]
    # Add any categories that might not be in the list
    for cat in sorted(grouped.keys()):
        if cat not in category_order:
            category_order.append(cat)

    for category in category_order:
        if category not in grouped:
            continue
        # `category_label` is set by translate_catalog; the English key stays so
        # ordering and grouping do not depend on the language.
        heading = grouped[category][0][1].get("category_label", category)
        lines.extend([f"## {heading}", ""])
        for parameter_id, meta in grouped[category]:
            label = meta.get("label", parameter_id)
            lines.extend([
                f"### `{parameter_id}`",
                "",
                f"**{strings['doc.label']}:** {label}  ",
                f"**{strings['doc.type']}:** `{meta.get('type', 'unspecified')}`",
            ])
            if "unit" in meta:
                lines.append(f"**{strings['doc.unit']}:** `{meta['unit']}`")
            lines.extend([
                "",
                meta.get("description", ""),
                ""
            ])

    # Append static documentation sections
    lines.extend([
        "## CLI override examples",
        "",
        "- Set donor lifetime and attachment residue:",
        "  ```bash",
        "  quest simulate -p project.quest.json --set tau0=3.8 --set attachment.residue=55",
        "  ```",
        "- Sweep linker length and residue:",
        "  ```bash",
        "  quest simulate -p project.quest.json \\",
        "    --grid av_parameter.linker_length=10.0,12.5 \\",
        "    --grid attachment.residue=25,30,35",
        "  ```",
        "",
        "## Validation tips",
        "",
        "- Ensure donor dye has `D` set; otherwise the API raises a `KeyError`.",
        "- Provide a valid `pdb` path (or pass `--pdb`/`pdb_path`).",
        "- When enabling FRET, include two dyes and an `R0_matrix` entry for donor→acceptor.",
        ""
    ])

    output.write_text("\n".join(lines).rstrip() + "\n")
    return output


def generate_all() -> list[Path]:
    return [generate(code) for code in sorted(available_locales())]


if __name__ == "__main__":
    for written in generate_all():
        print(f"Generated {written}")
