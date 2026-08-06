#!/usr/bin/env python3
"""Conformance checker for OKF v0.2 bundles (SPEC section 11).

VENDORED COPY. The canonical file is `fpsimp/okf/references/tools/okf_validate.py`
in the sibling FPSIMP repository; this copy exists so that QuEst validates itself
without a sibling checkout -- CI clones one repo, not two. Sync by copying, and
`diff` the two when either changes.

    python3 okf/references/tools/okf_validate.py okf
    python3 okf/references/tools/okf_validate.py ../quest/okf --version 0.1

Exit code 0 when the bundle is conformant, 1 when it is not. Warnings never
fail the run: the spec is explicit that consumers must not reject a bundle for
missing optional fields, unknown types, unknown keys, broken links or missing
index files, so those are reported and tolerated.

# Errors (section 11 conformance)

* a non-reserved `.md` file without a parseable YAML frontmatter block
* a frontmatter block without a non-empty `type`
* `okf_version` outside the bundle-root `index.md` (section 12)
* a `log.md` date heading that is not ISO `YYYY-MM-DD` (section 9)

# Warnings (soft guidance, sections 4-8)

* missing `description`, `generated.by`, `status`, `stale_after`
* an actor that does not match the section 7 convention
* `status` outside draft|stable|deprecated
* a bundle-relative or relative markdown link with no target on disk
* `stale_after` in the past (the concept is stale, not invalid)
* `sources[]` entries with no `resource`
* an `Attested Computation` missing `runtime`, or naming a `computation`,
  `executor.resource` or `attester.resource` that does not exist

The point of running it in CI is that "this bundle is conformant" stays a
measured fact rather than a claim in a README.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover
    print("PyYAML is required: pip install pyyaml", file=sys.stderr)
    raise SystemExit(2)

RESERVED = {"index.md", "log.md"}
SOURCE_SUFFIXES = (
    ".py", ".md", ".json", ".yml", ".yaml", ".toml", ".cfg", ".ini", ".txt",
    ".js", ".cjs", ".mjs", ".ts", ".tsx", ".html", ".css", ".sh", ".ui",
    ".pdb", ".cif", ".fasta", ".tsv", ".dat", ".png", ".lock",
)
VALID_STATUS = {"draft", "stable", "deprecated"}
ACTOR_RE = re.compile(r"^(human:[\w.@-]+|process:[\w.-]+|[\w.-]+/[\w.:-]+)$")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
LOG_HEADING_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
LINK_RE = re.compile(r"(?<!!)\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
FRONTMATTER_RE = re.compile(r"\A---\r?\n(.*?)\r?\n---\s*(?:\r?\n|\Z)", re.DOTALL)


class Report:
    def __init__(self) -> None:
        self.errors: List[Tuple[str, str]] = []
        self.warnings: List[Tuple[str, str]] = []

    def error(self, where: str, msg: str) -> None:
        self.errors.append((where, msg))

    def warn(self, where: str, msg: str) -> None:
        self.warnings.append((where, msg))


def split_frontmatter(text: str) -> Tuple[Dict[str, Any] | None, str, str | None]:
    """Return (frontmatter, body, parse_error)."""
    match = FRONTMATTER_RE.match(text)
    if not match:
        return None, text, None
    try:
        data = yaml.safe_load(match.group(1))
    except yaml.YAMLError as exc:
        return None, text[match.end():], str(exc)
    if data is None:
        data = {}
    if not isinstance(data, dict):
        return None, text[match.end():], "frontmatter is not a mapping"
    return data, text[match.end():], None


def check_actor(report: Report, where: str, field: str, value: Any) -> None:
    if not isinstance(value, str) or not ACTOR_RE.match(value):
        report.warn(where, f"{field}: {value!r} does not match the section 7 actor convention")


def check_trust(report: Report, where: str, fm: Dict[str, Any], today: _dt.date) -> None:
    generated = fm.get("generated")
    if generated is None:
        report.warn(where, "no 'generated' -- provenance of the concept is unrecorded")
    elif isinstance(generated, dict):
        if "by" not in generated:
            report.warn(where, "generated has no 'by'")
        else:
            check_actor(report, where, "generated.by", generated["by"])
    else:
        report.warn(where, "'generated' should be a mapping {by, at}")

    verified = fm.get("verified")
    if verified is not None:
        events = verified if isinstance(verified, list) else [verified]
        for event in events:
            if isinstance(event, dict) and "by" in event:
                check_actor(report, where, "verified[].by", event["by"])
            else:
                report.warn(where, f"verified entry is not a {{by, at}} mapping: {event!r}")

    status = fm.get("status")
    if status is None:
        report.warn(where, "no 'status' (defaults to stable)")
    elif status not in VALID_STATUS:
        report.warn(where, f"status: {status!r} is not draft|stable|deprecated")

    stale_after = fm.get("stale_after")
    if stale_after is None:
        report.warn(where, "no 'stale_after' -- freshness cannot be judged")
    else:
        date = stale_after if isinstance(stale_after, _dt.date) else None
        if date is None:
            if not (isinstance(stale_after, str) and DATE_RE.match(stale_after)):
                report.warn(where, f"stale_after: {stale_after!r} is not YYYY-MM-DD")
                return
            date = _dt.date.fromisoformat(stale_after)
        if today >= date:
            report.warn(where, f"STALE: stale_after {date.isoformat()} has passed")

    for entry in fm.get("sources") or []:
        if not isinstance(entry, dict) or not entry.get("resource"):
            report.warn(where, f"source entry without 'resource': {entry!r}")


def check_computation(report: Report, where: str, path: Path, bundle: Path, fm: Dict[str, Any], body: str) -> None:
    if not fm.get("runtime"):
        report.warn(where, "Attested Computation without 'runtime' (section 10.2 requires it)")
    for field, value in (
        ("computation", fm.get("computation")),
        ("executor.resource", (fm.get("executor") or {}).get("resource")),
        ("attester.resource", (fm.get("attester") or {}).get("resource")),
    ):
        if not value:
            continue
        target = (bundle / str(value).lstrip("/")) if str(value).startswith("/") else (path.parent / str(value))
        if not target.exists():
            alt = bundle / str(value)
            if not alt.exists():
                report.warn(where, f"{field} -> {value} does not exist")
    if not fm.get("computation") and "# Computation" not in body:
        report.warn(where, "no 'computation' path and no '# Computation' body section (section 10.3)")


def check_repo_paths(report: Report, where: str, repo: Path, text: str, prefixes: List[str]) -> None:
    """Warn about source paths a concept names that no longer exist.

    This is what catches a bundle drifting behind a refactor: the frontmatter
    still parses, every link still resolves, and yet the concept points at files
    that were moved or deleted. Paths are matched by prefix (`quest/`, `src/`,
    ...) so prose and tables are covered, not just the `resource` field.

    Off by default: a concept may legitimately name a path that does not exist
    -- a proposal, a counter-example, a file that was deleted on purpose -- so
    this is a review aid, never an error. `log.md` is exempt entirely: history
    is supposed to mention paths that no longer exist.
    """
    pattern = re.compile(
        r"(?<![\w/.])((?:" + "|".join(re.escape(p.rstrip("/")) for p in prefixes) + r")/[\w./-]*\w)"
    )
    for candidate in sorted(set(pattern.findall(text))):
        tail = candidate.rsplit("/", 1)[-1]
        if "." in tail and not tail.endswith(SOURCE_SUFFIXES):
            continue  # `module.function`, not a path
        if not (repo / candidate).exists():
            report.warn(where, f"path does not exist: {candidate}")


def check_links(report: Report, where: str, path: Path, bundle: Path, body: str) -> None:
    for target in LINK_RE.findall(body):
        if target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        clean = target.split("#", 1)[0]
        if not clean:
            continue
        resolved = (bundle / clean.lstrip("/")) if clean.startswith("/") else (path.parent / clean)
        if not resolved.exists():
            report.warn(where, f"broken link -> {target}")


def check_log(report: Report, where: str, body: str) -> None:
    headings = LOG_HEADING_RE.findall(body)
    if not headings:
        report.warn(where, "log has no '## <date>' entries (section 9)")
    for heading in headings:
        if not DATE_RE.match(heading):
            report.error(where, f"log date heading is not ISO YYYY-MM-DD: '## {heading}'")


def check_index(report: Report, where: str, is_root: bool, fm: Dict[str, Any] | None, body: str) -> None:
    if fm is not None and not is_root:
        report.error(where, "only the bundle-root index.md may carry frontmatter (section 12)")
    if fm is not None and is_root and "okf_version" not in fm:
        report.warn(where, "root index.md carries frontmatter but declares no okf_version")
    if not LINK_RE.search(body):
        report.warn(where, "index lists no links (section 8)")


def validate(
    bundle: Path,
    declared_version: str | None,
    today: _dt.date,
    path_prefixes: List[str] | None = None,
) -> Report:
    report = Report()
    repo = bundle.parent
    files = sorted(p for p in bundle.rglob("*.md") if p.is_file())
    if not files:
        report.error(str(bundle), "no markdown files found")
        return report

    root_index = bundle / "index.md"
    if not root_index.exists():
        report.warn(str(bundle), "no bundle-root index.md")

    for path in files:
        where = str(path.relative_to(bundle.parent))
        text = path.read_text(encoding="utf-8")
        fm, body, parse_error = split_frontmatter(text)

        # `okf_check_paths: false` opts a concept out of the existence check.
        # Some documents name dead paths on purpose -- an audit of what rotted,
        # a naming-convention concept citing files that were never created --
        # and flagging those trains readers to ignore the warnings.
        opted_out = bool(fm is not None and fm.get("okf_check_paths") is False)
        if path_prefixes and path.name != "log.md" and not opted_out:
            check_repo_paths(report, where, repo, text, path_prefixes)

        if parse_error:
            report.error(where, f"unparseable YAML frontmatter: {parse_error}")
            continue

        if path.name == "index.md":
            check_index(report, where, path == root_index, fm, body)
            if fm and path == root_index:
                version = str(fm.get("okf_version", ""))
                if declared_version and version and version != declared_version:
                    report.warn(where, f"okf_version {version!r} != expected {declared_version!r}")
            continue

        if path.name == "log.md":
            if fm is not None:
                report.warn(where, "log.md carries frontmatter; the spec defines none for it")
            check_log(report, where, body)
            check_links(report, where, path, bundle, body)
            continue

        if fm is None:
            report.error(where, "no YAML frontmatter block (section 11.1)")
            continue
        if not str(fm.get("type", "")).strip():
            report.error(where, "frontmatter has no non-empty 'type' (section 11.2)")
        if not str(fm.get("description", "")).strip():
            report.warn(where, "no 'description'")

        check_trust(report, where, fm, today)
        if str(fm.get("type", "")).strip() == "Attested Computation":
            check_computation(report, where, path, bundle, fm, body)
        check_links(report, where, path, bundle, body)

    return report


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("bundle", nargs="?", default="okf", help="path to the bundle root (default: okf)")
    parser.add_argument("--version", default="0.2", help="expected okf_version (default: 0.2)")
    parser.add_argument("--strict", action="store_true", help="treat warnings as failures")
    parser.add_argument("--today", default=None, help="override today's date (YYYY-MM-DD) for staleness checks")
    parser.add_argument(
        "--check-paths",
        default=None,
        metavar="PREFIXES",
        help="comma-separated source prefixes (e.g. 'fpsim/,src/') to check for existence; "
        "catches a bundle drifting behind a refactor. Warnings only.",
    )
    args = parser.parse_args(argv)

    bundle = Path(args.bundle).resolve()
    if not bundle.is_dir():
        print(f"not a directory: {bundle}", file=sys.stderr)
        return 2

    today = _dt.date.fromisoformat(args.today) if args.today else _dt.date.today()
    prefixes = [p.strip() for p in args.check_paths.split(",") if p.strip()] if args.check_paths else None
    report = validate(bundle, args.version, today, prefixes)

    for where, msg in report.errors:
        print(f"ERROR  {where}: {msg}")
    for where, msg in report.warnings:
        print(f"warn   {where}: {msg}")

    n_files = sum(1 for _ in bundle.rglob("*.md"))
    print(
        f"\n{n_files} concept file(s); {len(report.errors)} error(s), {len(report.warnings)} warning(s) "
        f"-- {'CONFORMANT' if not report.errors else 'NOT CONFORMANT'} with OKF v{args.version}"
    )
    if report.errors:
        return 1
    return 1 if (args.strict and report.warnings) else 0


if __name__ == "__main__":
    raise SystemExit(main())
