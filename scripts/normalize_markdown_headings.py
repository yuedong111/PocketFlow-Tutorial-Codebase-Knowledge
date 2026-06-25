"""Normalize heading levels in flattened Markdown files.

Some PDF/Word-to-Markdown converters mark every detected heading as ``##``.
This script restores a standard Markdown hierarchy:

    # Book or document title, optional via --title
    ## Chapter heading
    ### Section heading
    #### Subsection heading

The important bit is adaptive top-level detection. For files like ``4a.md``,
the first valid ``##`` is already a chapter heading, for example
``## 第 **1** 章 ...``. The script learns that pattern and treats later
``第 N 章`` / ``Chapter N`` headings as the same chapter level.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path


ATX_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)(\s*)$")
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
BARE_NUMBER_RE = re.compile(r"^(?P<num>\d{1,2})\s*(?P<title>\S.*)$")
PAREN_NUMBER_RE = re.compile(
    r"^[\(（]\s*(?P<num>\d{1,2}|[一二三四五六七八九十]+)\s*[\)）]\s*(?P<title>\S.*)$"
)
TOC_LINE_RE = re.compile(r"\.{5,}\s*\d+\s*$")

CHAPTER_LEVEL = 2
SECTION_BASE_LEVEL = CHAPTER_LEVEL + 1


@dataclass
class Stats:
    total_headings: int = 0
    chapters: int = 0
    structured: int = 0
    local_numbered: int = 0
    parenthesized: int = 0
    colon_headings: int = 0
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


def structured_number_depth(text: str) -> int | None:
    match = STRUCTURED_NUMBER_RE.match(text)
    if not match:
        return None
    return match.group("num").count(".") + 1


def classify_top_heading(text: str) -> TopHeadingPattern | None:
    text = normalize_chapter_text(strip_heading_markup(text))
    if CHINESE_CHAPTER_RE.match(text):
        return TopHeadingPattern("chinese_chapter")
    if ENGLISH_CHAPTER_RE.match(text):
        return TopHeadingPattern("english_chapter")
    depth = structured_number_depth(text)
    if depth is not None:
        return TopHeadingPattern("structured_number", depth)
    if LOCAL_NUMBER_RE.match(text) or BARE_NUMBER_RE.match(text):
        return TopHeadingPattern("plain_number")
    return None


def matches_top_pattern(text: str, pattern: TopHeadingPattern | None) -> bool:
    if pattern is None:
        return CHINESE_CHAPTER_RE.match(text) is not None or ENGLISH_CHAPTER_RE.match(text) is not None
    if pattern.kind == "chinese_chapter":
        return CHINESE_CHAPTER_RE.match(text) is not None
    if pattern.kind == "english_chapter":
        return ENGLISH_CHAPTER_RE.match(text) is not None
    if pattern.kind == "structured_number":
        return structured_number_depth(text) == pattern.number_depth
    if pattern.kind == "plain_number":
        if structured_number_depth(text) is not None:
            return False
        return LOCAL_NUMBER_RE.match(text) is not None or BARE_NUMBER_RE.match(text) is not None
    return False


def find_first_valid_heading_pattern(lines: list[str]) -> TopHeadingPattern | None:
    in_fence = False
    fence_marker = ""
    for line in lines:
        stripped = line.lstrip()
        fence = re.match(r"^(```+|~~~+)", stripped)
        if fence:
            marker = fence.group(1)[0]
            if not in_fence:
                in_fence = True
                fence_marker = marker
            elif marker == fence_marker:
                in_fence = False
                fence_marker = ""
            continue
        if in_fence:
            continue
        match = ATX_HEADING_RE.match(line)
        if not match:
            continue
        pattern = classify_top_heading(match.group(2))
        if pattern:
            return pattern
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
    if re.search(r"[=+\-*/%′⋅{}]", text) and not re.search(r"[\u4e00-\u9fffA-Za-z]{3,}", text):
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
) -> tuple[str, int | None, str]:
    """Return ``(kind, level, text)``.

    ``level`` is ``None`` when the original heading should become a paragraph.
    """
    text = normalize_chapter_text(strip_heading_markup(raw_text))

    if looks_like_false_heading(text):
        return "demoted", None, text

    if matches_top_pattern(text, top_pattern):
        return "chapter", CHAPTER_LEVEL, text

    if CHINESE_SECTION_RE.match(text):
        return "structured", SECTION_BASE_LEVEL, text

    structured = STRUCTURED_NUMBER_RE.match(text)
    if structured:
        number = structured.group("num")
        depth = number.count(".") + 1
        if top_pattern and top_pattern.kind == "structured_number" and top_pattern.number_depth:
            relative_depth = max(0, depth - top_pattern.number_depth)
            return "structured", clamp_level(CHAPTER_LEVEL + relative_depth), text
        return "structured", clamp_level(CHAPTER_LEVEL + depth - 1), text

    if LOCAL_NUMBER_RE.match(text):
        return "local_numbered", clamp_level(max(current_structured_level + 1, SECTION_BASE_LEVEL)), text

    if PAREN_NUMBER_RE.match(text):
        return "parenthesized", clamp_level(max(current_structured_level + 2, SECTION_BASE_LEVEL + 1)), text

    bare = BARE_NUMBER_RE.match(text)
    if bare and len(bare.group("title")) >= 2:
        return "local_numbered", clamp_level(max(current_structured_level + 1, SECTION_BASE_LEVEL)), text

    if promote_colon_headings and len(text) <= 40 and text.endswith(("：", ":")):
        return "colon_heading", clamp_level(max(current_structured_level + 1, SECTION_BASE_LEVEL)), text

    return "demoted", None, text


def drop_leading_toc(lines: list[str], top_pattern: TopHeadingPattern | None) -> tuple[list[str], int]:
    for index, line in enumerate(lines):
        match = ATX_HEADING_RE.match(line)
        if not match:
            continue
        text = normalize_chapter_text(strip_heading_markup(match.group(2)))
        if matches_top_pattern(text, top_pattern):
            return lines[index:], index
    return lines, 0


def normalize_markdown(
    content: str,
    *,
    title: str | None = None,
    drop_toc: bool = False,
    promote_colon_headings: bool = True,
) -> tuple[str, Stats]:
    lines = content.splitlines()
    stats = Stats()
    top_pattern = find_first_valid_heading_pattern(lines)

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
        stripped = line.lstrip()
        fence = re.match(r"^(```+|~~~+)", stripped)
        if fence:
            marker = fence.group(1)[0]
            if not in_fence:
                in_fence = True
                fence_marker = marker
            elif marker == fence_marker:
                in_fence = False
                fence_marker = ""
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
        )

        if level is None:
            stats.demoted += 1
            output.append(text)
            continue

        if kind == "chapter":
            stats.chapters += 1
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

        output.append(f"{'#' * level} {text}")

    return "\n".join(output).rstrip() + "\n", stats


def print_stats(stats: Stats) -> None:
    print("Heading normalization summary:")
    print(f"  input headings:        {stats.total_headings}")
    print(f"  chapters:              {stats.chapters}")
    print(f"  structured headings:   {stats.structured}")
    print(f"  local numbered:        {stats.local_numbered}")
    print(f"  parenthesized:         {stats.parenthesized}")
    print(f"  colon headings:        {stats.colon_headings}")
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
