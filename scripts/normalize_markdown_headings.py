"""Normalize heading levels in flattened Markdown files.

Some PDF/Word-to-Markdown converters mark every detected heading as ``##``.
This script restores a standard Markdown hierarchy:

    # Book or document title, optional via --title
    ## Chapter heading
    ### Section heading
    #### Subsection heading

The important bit is adaptive top-level detection. The script scans every
heading, learns the *dominant* chapter pattern (for example ``第 N 章`` /
``Chapter N`` / a shallow ``1`` / ``1.2``), and assigns levels accordingly.

Design notes
------------
* Heading levels are recomputed purely from heading *text* patterns; the
  original ``#`` count is intentionally ignored (converters get it wrong).
* By default an unrecognized but plausible heading is **kept** as a nested
  heading rather than demoted to a paragraph -- demoting loses structure for
  the many legitimately unnumbered headings (前言 / 参考文献 / Introduction ...).
  Use ``--demote-unknown`` to restore the older, more aggressive behavior.
* Only lines that look like genuine noise (TOC dotted leaders, page numbers,
  bare formulas, full sentences) are demoted to paragraphs.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path


ATX_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)(\s*)$")
FENCE_RE = re.compile(r"^(```+|~~~+)")
CHINESE_CHAPTER_RE = re.compile(
    r"^第\s*(?P<num>[0-9一二三四五六七八九十百零〇]+)\s*[章卷篇]\s*[：:]?\s*(?P<title>.*)$"
)
CHINESE_SECTION_RE = re.compile(
    r"^第\s*(?P<num>[0-9一二三四五六七八九十百零〇]+)\s*节\s*[：:]?\s*(?P<title>.*)$"
)
ENGLISH_CHAPTER_RE = re.compile(
    r"^(?P<label>chapter|chap\.?|ch\.)\s*"
    r"(?P<num>[0-9IVXLCDM]+|one|two|three|four|five|six|seven|eight|nine|ten)"
    r"\s*[.:\-]?\s*(?P<title>.*)$",
    re.IGNORECASE,
)
STRUCTURED_NUMBER_RE = re.compile(r"^(?P<num>\d+(?:\.\d+)+)\s*\.?\s*(?P<title>.*)$")
LOCAL_NUMBER_RE = re.compile(r"^(?P<num>\d{1,2})\s*(?:[.．、])\s*(?P<title>\S.*)$")
# Require the number to be 1-2 digits AND not the prefix of a longer number
# (so "2024 年回顾" is not parsed as "20" + "24 年回顾").
BARE_NUMBER_RE = re.compile(r"^(?P<num>\d{1,2})(?!\d)\s*(?P<title>\S.*)$")
PAREN_NUMBER_RE = re.compile(
    r"^[\(（]\s*(?P<num>\d{1,2}|[一二三四五六七八九十]+)\s*[\)）]\s*(?P<title>\S.*)$"
)
TOC_LINE_RE = re.compile(r"\.{5,}\s*\d+\s*$")
FORMULA_CHAR_RE = re.compile(r"[=+\-*/%<>^_\\|{}~`′⋅]")
WORD_CHAR_RE = re.compile(r"[\u4e00-\u9fffA-Za-z]")

# Strip a redundant "Section"/"Sec."/"§" prefix that sits in front of a number,
# so "§ 1.2 Foo" / "Section 1.2 Foo" are treated like "1.2 Foo".
SECTION_PREFIX_RE = re.compile(r"^(?:sections?|sect\.?|sec\.?|§)\s*(?=\d)", re.IGNORECASE)
# Part dividers that sit above chapters in many English textbooks.
ENGLISH_PART_RE = re.compile(
    r"^part\s+(?P<num>\d+|[IVXLCDM]+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
    r"\b\s*[.:\-–—]?\s*(?P<title>.*)$",
    re.IGNORECASE,
)
# Figure / table / equation captions that converters often emit as headings.
CAPTION_RE = re.compile(
    r"^(?:figures?|fig\.?|tables?|tbl\.?|tab\.?|equations?|eq\.?|exhibits?|"
    r"plates?|schemes?|listings?|algorithms?|charts?|boxes|box)\s*\.?\s*\d",
    re.IGNORECASE,
)
# Stand-alone page-number lines ("Page 12", "p. 12", "- 12 -").
PAGE_NUMBER_RE = re.compile(r"^(?:page|pg\.?|p\.?)\s*\d+$", re.IGNORECASE)
DASHED_PAGE_RE = re.compile(r"^[-–—]\s*\d+\s*[-–—]$")

CHAPTER_LEVEL = 2
SECTION_BASE_LEVEL = CHAPTER_LEVEL + 1

# Front/back matter that is conventionally chapter-level AND appears once.
# NOTE: deliberately excludes recurring per-chapter titles such as "Summary",
# "Introduction", "Overview", "Conclusion", "Exercises", "Problems" -- those are
# section-level and are handled as ordinary unnumbered headings instead.
FRONT_BACK_MATTER_ZH = {
    "前言", "序", "序言", "自序", "内容提要", "提要", "摘要", "绪论", "引言",
    "后记", "跋", "参考文献", "参考书目", "致谢", "鸣谢", "附录", "索引",
    "术语表", "词汇表", "缩略语", "缩略语表", "符号表", "目录",
}
FRONT_BACK_MATTER_EN = {
    "preface", "foreword", "acknowledgement", "acknowledgements",
    "acknowledgment", "acknowledgments", "references", "bibliography",
    "index", "glossary", "contents", "table of contents", "notes", "abstract",
    "epilogue", "prologue", "about the author", "about the authors",
    "further reading", "dedication", "colophon",
}


@dataclass
class Stats:
    total_headings: int = 0
    chapters: int = 0
    parts: int = 0
    structured: int = 0
    local_numbered: int = 0
    parenthesized: int = 0
    colon_headings: int = 0
    unstructured: int = 0
    demoted: int = 0
    dropped_leading_lines: int = 0
    inserted_title: int = 0


@dataclass(frozen=True)
class TopHeadingPattern:
    kind: str
    number_depth: int | None = None


def strip_heading_markup(text: str) -> str:
    text = text.strip()
    text = text.replace("**", "")
    text = text.replace("__", "")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:!?，。；：！？])", r"\1", text)
    return text.strip()


def normalize_chapter_text(text: str) -> str:
    match = CHINESE_CHAPTER_RE.match(text)
    if match:
        title = match.group("title").strip()
        return f"第 {match.group('num')} 章 {title}".strip()

    match = ENGLISH_CHAPTER_RE.match(text)
    if match:
        title = match.group("title").strip()
        return f"Chapter {match.group('num')} {title}".strip()

    return text


def clean_heading_text(raw_text: str) -> str:
    """Strip markup, standardize chapter wording, and drop a redundant
    Section/§ prefix in front of a number, once, for a raw heading."""
    text = normalize_chapter_text(strip_heading_markup(raw_text))
    text = SECTION_PREFIX_RE.sub("", text).strip()
    return text


def structured_number_depth(text: str) -> int | None:
    """Depth of a *dotted* number like ``1.2.3`` (3); ``None`` otherwise."""
    match = STRUCTURED_NUMBER_RE.match(text)
    if not match:
        return None
    return match.group("num").count(".") + 1


def numeric_depth(text: str) -> int | None:
    """Unified numbering depth treating bare integers and dotted numbers as one
    hierarchy: ``1`` -> 1, ``1.`` -> 1, ``1.2`` -> 2, ``1.2.3`` -> 3.

    ``None`` when the heading has no leading section number.
    """
    dotted = structured_number_depth(text)
    if dotted is not None:
        return dotted
    if LOCAL_NUMBER_RE.match(text) or BARE_NUMBER_RE.match(text):
        return 1
    return None


def is_front_back_matter(text: str) -> bool:
    """Chapter-level front/back matter that appears once (Preface, References,
    Appendix, 前言, 参考文献 ...). Excludes recurring per-chapter titles."""
    t = text.strip().rstrip("：: 　").strip()
    if not t:
        return False
    if t in FRONT_BACK_MATTER_ZH:
        return True
    if t.lower() in FRONT_BACK_MATTER_EN:
        return True
    if re.match(r"^附\s*录", t):
        return True
    if re.match(r"^appendix\b", t, re.IGNORECASE):
        return True
    return False


def classify_top_heading(clean_text: str) -> TopHeadingPattern | None:
    """Classify an already-cleaned heading into a candidate top-level pattern."""
    if CHINESE_CHAPTER_RE.match(clean_text):
        return TopHeadingPattern("chinese_chapter")
    if ENGLISH_CHAPTER_RE.match(clean_text):
        return TopHeadingPattern("english_chapter")
    depth = numeric_depth(clean_text)
    if depth is not None:
        return TopHeadingPattern("numbered", depth)
    return None


def matches_top_pattern(clean_text: str, pattern: TopHeadingPattern | None) -> bool:
    if pattern is None:
        return (
            CHINESE_CHAPTER_RE.match(clean_text) is not None
            or ENGLISH_CHAPTER_RE.match(clean_text) is not None
        )
    if pattern.kind == "chinese_chapter":
        return CHINESE_CHAPTER_RE.match(clean_text) is not None
    if pattern.kind == "english_chapter":
        return ENGLISH_CHAPTER_RE.match(clean_text) is not None
    if pattern.kind == "numbered":
        return numeric_depth(clean_text) == pattern.number_depth
    return False


def _iter_clean_headings(lines: list[str]):
    """Yield cleaned text for each heading line, skipping fenced code blocks."""
    in_fence = False
    fence_marker = ""
    for line in lines:
        fence = FENCE_RE.match(line.lstrip())
        if fence:
            marker = fence.group(1)[0]
            if not in_fence:
                in_fence, fence_marker = True, marker
            elif marker == fence_marker:
                in_fence, fence_marker = False, ""
            continue
        if in_fence:
            continue
        match = ATX_HEADING_RE.match(line)
        if match:
            yield clean_heading_text(match.group(2))


def detect_top_pattern(lines: list[str]) -> TopHeadingPattern | None:
    """Pick the dominant chapter pattern by scanning *all* headings.

    More robust than trusting the first heading: explicit chapter markers win;
    otherwise the shallowest numeric depth (treating ``1`` and ``1.1`` as one
    hierarchy) becomes the chapter level.
    """
    counts: dict[str, int] = {}
    numeric_depths: list[int] = []

    for text in _iter_clean_headings(lines):
        if looks_like_false_heading(text):
            continue
        pattern = classify_top_heading(text)
        if pattern is None:
            continue
        counts[pattern.kind] = counts.get(pattern.kind, 0) + 1
        if pattern.kind == "numbered" and pattern.number_depth is not None:
            numeric_depths.append(pattern.number_depth)

    if counts.get("chinese_chapter"):
        return TopHeadingPattern("chinese_chapter")
    if counts.get("english_chapter"):
        return TopHeadingPattern("english_chapter")
    if numeric_depths:
        return TopHeadingPattern("numbered", min(numeric_depths))
    return None


def looks_like_false_heading(text: str) -> bool:
    if not text:
        return True
    if TOC_LINE_RE.search(text):
        return True
    if re.fullmatch(r"[\d\s.]+", text):
        return True
    if re.fullmatch(r"[0-9A-Fa-f]{8,}", text):
        return True
    if PAGE_NUMBER_RE.match(text) or DASHED_PAGE_RE.match(text):
        return True
    # Figure/Table/Equation captions emitted as headings by PDF converters.
    if CAPTION_RE.match(text):
        return True
    # Bare formula / symbol soup: several math symbols and little real text.
    symbols = FORMULA_CHAR_RE.findall(text)
    word_chars = WORD_CHAR_RE.findall(text)
    if len(symbols) >= 2 and len(word_chars) <= len(symbols):
        return True
    if len(text) > 80 and not re.match(
        r"^(第\s*\d+\s*[章节卷篇]|\d+(?:\.\d+)+|(?:chapter|chap\.?|ch\.)\s*)",
        text,
        re.IGNORECASE,
    ):
        return True
    if text.endswith(("。", "；", ";", ".")) and not re.match(
        r"^(\d{1,2}|[\(（]\s*\d{1,2})\s*[.．、)）]",
        text,
    ):
        return True
    return False


def clamp_level(level: int) -> int:
    return max(1, min(6, level))


def normalize_heading(
    raw_text: str,
    current_structured_level: int,
    top_pattern: TopHeadingPattern | None,
    promote_colon_headings: bool,
    demote_unknown: bool,
) -> tuple[str, int | None, str]:
    """Return ``(kind, level, text)``.

    ``level`` is ``None`` when the heading should become a plain paragraph.
    """
    text = clean_heading_text(raw_text)

    if looks_like_false_heading(text):
        return "demoted", None, text

    if matches_top_pattern(text, top_pattern):
        return "chapter", CHAPTER_LEVEL, text

    if ENGLISH_PART_RE.match(text):
        # Part dividers sit at the top level alongside chapters.
        return "part", CHAPTER_LEVEL, text

    if is_front_back_matter(text):
        return "chapter", CHAPTER_LEVEL, text

    if CHINESE_SECTION_RE.match(text):
        return "structured", SECTION_BASE_LEVEL, text

    # Unified numeric hierarchy: 1 -> depth 1, 1.2 -> depth 2, 1.2.3 -> depth 3.
    depth = numeric_depth(text)
    if depth is not None:
        if top_pattern and top_pattern.kind == "numbered" and top_pattern.number_depth:
            relative_depth = max(0, depth - top_pattern.number_depth)
            return "structured", clamp_level(CHAPTER_LEVEL + relative_depth), text
        if structured_number_depth(text) is not None:
            # Dotted number under a non-numbered top (e.g. chapters are 第N章).
            return "structured", clamp_level(CHAPTER_LEVEL + depth - 1), text
        # Bare/local integer (depth 1) as a section relative to the current
        # chapter. Guard against 1-char titles to reduce false positives.
        is_local = LOCAL_NUMBER_RE.match(text) is not None
        bare = BARE_NUMBER_RE.match(text)
        if is_local or (bare and len(bare.group("title").strip()) >= 2):
            return "local_numbered", clamp_level(max(current_structured_level + 1, SECTION_BASE_LEVEL)), text

    if PAREN_NUMBER_RE.match(text):
        return "parenthesized", clamp_level(max(current_structured_level + 2, SECTION_BASE_LEVEL + 1)), text

    if promote_colon_headings and len(text) <= 40 and text.endswith(("：", ":")):
        return "colon_heading", clamp_level(max(current_structured_level + 1, SECTION_BASE_LEVEL)), text

    # Unrecognized but plausible heading. Keep it (nested under the current
    # structure) unless the caller explicitly opted into demotion.
    if demote_unknown:
        return "demoted", None, text
    return "unstructured", clamp_level(max(current_structured_level + 1, SECTION_BASE_LEVEL)), text



def drop_leading_toc(lines: list[str], top_pattern: TopHeadingPattern | None) -> tuple[list[str], int]:
    for index, line in enumerate(lines):
        match = ATX_HEADING_RE.match(line)
        if not match:
            continue
        text = clean_heading_text(match.group(2))
        # A real chapter heading that is not itself a TOC dotted-leader line.
        if matches_top_pattern(text, top_pattern) and not looks_like_false_heading(text):
            return lines[index:], index
    return lines, 0


def normalize_markdown(
    content: str,
    *,
    title: str | None = None,
    drop_toc: bool = False,
    promote_colon_headings: bool = True,
    demote_unknown: bool = False,
) -> tuple[str, Stats]:
    lines = content.splitlines()
    stats = Stats()
    top_pattern = detect_top_pattern(lines)

    if drop_toc:
        lines, stats.dropped_leading_lines = drop_leading_toc(lines, top_pattern)

    output: list[str] = []
    if title:
        output.extend([f"# {strip_heading_markup(title)}", ""])
        stats.inserted_title = 1

    in_fence = False
    fence_marker = ""
    current_structured_level = CHAPTER_LEVEL

    for line in lines:
        fence = FENCE_RE.match(line.lstrip())
        if fence:
            marker = fence.group(1)[0]
            if not in_fence:
                in_fence, fence_marker = True, marker
            elif marker == fence_marker:
                in_fence, fence_marker = False, ""
            output.append(line.rstrip())
            continue

        match = None if in_fence else ATX_HEADING_RE.match(line)
        if not match:
            output.append(line.rstrip())
            continue

        stats.total_headings += 1
        kind, level, text = normalize_heading(
            match.group(2),
            current_structured_level,
            top_pattern,
            promote_colon_headings,
            demote_unknown,
        )

        if level is None:
            stats.demoted += 1
            output.append(text)
            continue

        if kind == "chapter":
            stats.chapters += 1
            current_structured_level = level
        elif kind == "part":
            stats.parts += 1
            current_structured_level = level
        elif kind == "structured":
            stats.structured += 1
            current_structured_level = level
        elif kind == "local_numbered":
            stats.local_numbered += 1
        elif kind == "parenthesized":
            stats.parenthesized += 1
        elif kind == "colon_heading":
            stats.colon_headings += 1
        elif kind == "unstructured":
            stats.unstructured += 1

        output.append(f"{'#' * level} {text}")

    return "\n".join(output).rstrip() + "\n", stats


def print_stats(stats: Stats) -> None:
    print("Heading normalization summary:")
    print(f"  input headings:        {stats.total_headings}")
    print(f"  chapters:              {stats.chapters}")
    if stats.parts:
        print(f"  parts:                 {stats.parts}")
    print(f"  structured headings:   {stats.structured}")
    print(f"  local numbered:        {stats.local_numbered}")
    print(f"  parenthesized:         {stats.parenthesized}")
    print(f"  colon headings:        {stats.colon_headings}")
    print(f"  kept (unstructured):   {stats.unstructured}")
    print(f"  demoted to paragraphs: {stats.demoted}")
    if stats.inserted_title:
        print("  inserted document title: yes")
    if stats.dropped_leading_lines:
        print(f"  dropped leading lines: {stats.dropped_leading_lines}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Restore a standard Markdown heading hierarchy for flattened converted notes."
    )
    parser.add_argument("input", help="Source Markdown file")
    parser.add_argument(
        "-o",
        "--output",
        help="Output Markdown file. Defaults to INPUT.normalized.md unless --in-place is set.",
    )
    parser.add_argument(
        "--title",
        help="Optional document title to insert as the single top-level '# ...' heading.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite the input file after normalization.",
    )
    parser.add_argument(
        "--drop-leading-toc",
        action="store_true",
        help="Drop everything before the first detected chapter/top-level heading.",
    )
    parser.add_argument(
        "--no-colon-headings",
        action="store_true",
        help="Do not promote short unnumbered colon-ending headings.",
    )
    parser.add_argument(
        "--demote-unknown",
        action="store_true",
        help="Demote unrecognized headings to paragraphs (aggressive; may lose structure).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print normalization stats; do not write a file.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    input_path = Path(args.input)
    content = input_path.read_text(encoding="utf-8")

    normalized, stats = normalize_markdown(
        content,
        title=args.title,
        drop_toc=args.drop_leading_toc,
        promote_colon_headings=not args.no_colon_headings,
        demote_unknown=args.demote_unknown,
    )
    print_stats(stats)

    if args.dry_run:
        return 0

    if args.in_place:
        output_path = input_path
    elif args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_name(f"{input_path.stem}.normalized{input_path.suffix}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(normalized, encoding="utf-8", newline="\n")
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
