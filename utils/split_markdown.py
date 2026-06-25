import re


def estimate_tokens(text):
    """Rough token estimate that works for mixed CJK / latin text.

    - Each CJK character counts as ~1 token.
    - Remaining (latin) text is counted by whitespace-separated words.
    This is only an approximation, good enough to drive chunk sizing.
    """
    if not text:
        return 0
    cjk = len(re.findall(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]", text))
    non_cjk = re.sub(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]", " ", text)
    words = len(non_cjk.split())
    return cjk + words


def _parse_sections(content):
    """Parse markdown into a hierarchical section tree based on ATX headings.

    Headings inside fenced code blocks (``` or ~~~) are ignored.
    Returns a root node: {level, title, lines, children}.
    """
    root = {"level": 0, "title": None, "lines": [], "children": []}
    stack = [root]
    in_fence = False
    fence_marker = None

    for line in content.split("\n"):
        stripped = line.lstrip()
        # Track fenced code blocks so '#' inside code isn't treated as a heading
        fence_match = re.match(r"^(```+|~~~+)", stripped)
        if fence_match:
            marker = fence_match.group(1)[0]
            if not in_fence:
                in_fence = True
                fence_marker = marker
            elif marker == fence_marker:
                in_fence = False
                fence_marker = None
            stack[-1]["lines"].append(line)
            continue

        heading = None if in_fence else re.match(r"^(#{1,6})\s+(.*\S)\s*$", line)
        if heading:
            level = len(heading.group(1))
            title = heading.group(2).strip()
            node = {"level": level, "title": title, "lines": [line], "children": []}
            while stack and stack[-1]["level"] >= level:
                stack.pop()
            stack[-1]["children"].append(node)
            stack.append(node)
        else:
            stack[-1]["lines"].append(line)

    return root


def _full_text(node):
    """Return the full text of a section subtree (own body + all descendants)."""
    parts = ["\n".join(node["lines"])] if node["lines"] else []
    for child in node["children"]:
        parts.append(_full_text(child))
    return "\n".join(p for p in parts if p.strip() != "" or p == "")


def _own_text(node):
    return "\n".join(node["lines"])


def _hard_split(text, max_tokens):
    """Split a single oversized block into pieces of at most ~max_tokens,
    preferring sentence boundaries (latin + CJK), falling back to word/char
    windows when a single sentence is still too large."""
    if estimate_tokens(text) <= max_tokens:
        return [text]

    # Split on sentence terminators while keeping them attached.
    sentences = re.split(r"(?<=[.!?。！？；;\n])", text)
    sentences = [s for s in sentences if s]

    pieces = []
    buf = ""
    for sent in sentences:
        if estimate_tokens(sent) > max_tokens:
            # Sentence itself too big: flush, then window-split by tokens.
            if buf:
                pieces.append(buf)
                buf = ""
            words = sent.split(" ")
            if len(words) > 1:
                window = []
                wt = 0
                for w in words:
                    tw = estimate_tokens(w) + 1
                    if window and wt + tw > max_tokens:
                        pieces.append(" ".join(window))
                        window, wt = [], 0
                    window.append(w)
                    wt += tw
                if window:
                    pieces.append(" ".join(window))
            else:
                # No spaces (e.g. CJK run): slice by characters.
                approx_chars = max(1, max_tokens)
                for i in range(0, len(sent), approx_chars):
                    pieces.append(sent[i : i + approx_chars])
        elif estimate_tokens(buf + sent) > max_tokens:
            pieces.append(buf)
            buf = sent
        else:
            buf += sent
    if buf:
        pieces.append(buf)
    return pieces


def _split_by_tokens(text, max_tokens, overlap_ratio=0.1):
    """Fallback splitter for heading-less text: pack paragraphs up to max_tokens
    with a small overlap between consecutive chunks to preserve continuity.
    Oversized paragraphs are hard-split first."""
    paragraphs = re.split(r"\n\s*\n", text)
    # Expand any paragraph that is individually larger than max_tokens.
    units = []
    for p in paragraphs:
        if not p.strip():
            continue
        units.extend(_hard_split(p, max_tokens))

    chunks = []
    buffer = []
    buffer_tokens = 0
    overlap_tokens = int(max_tokens * overlap_ratio)

    for para in units:
        ptokens = estimate_tokens(para)
        if buffer and buffer_tokens + ptokens > max_tokens:
            chunks.append("\n\n".join(buffer))
            # Build overlap tail from the end of the current buffer, but never
            # exceed the overlap budget (avoids carrying a whole large unit).
            tail = []
            tail_tokens = 0
            for prev in reversed(buffer):
                t = estimate_tokens(prev)
                if tail_tokens + t > overlap_tokens:
                    break
                tail.insert(0, prev)
                tail_tokens += t
            buffer = tail[:]
            buffer_tokens = tail_tokens
        buffer.append(para)
        buffer_tokens += ptokens

    if buffer:
        chunks.append("\n\n".join(buffer))
    return chunks


def _emit_section(node, breadcrumb, min_tokens, max_tokens, segments):
    """Recursively turn a section subtree into balanced segments.

    Strategy:
    - If the whole subtree fits in max_tokens -> one segment.
    - Otherwise keep the node's own intro text, merge small children together,
      and recurse into children that are themselves too large.
    """
    title = node["title"]
    crumb = breadcrumb + [title] if title else breadcrumb
    full = _full_text(node)

    if estimate_tokens(full) <= max_tokens:
        _add_segment(crumb, full, segments)
        return

    # Subtree too big. Start a buffer with this node's own body text.
    buffer_parts = [_own_text(node)] if _own_text(node).strip() else []
    buffer_tokens = sum(estimate_tokens(p) for p in buffer_parts)

    def flush():
        nonlocal buffer_parts, buffer_tokens
        joined = "\n".join(buffer_parts).strip()
        if joined:
            _add_segment(crumb, joined, segments)
        buffer_parts = []
        buffer_tokens = 0

    if not node["children"]:
        # No subheadings to recurse into: fall back to token-window splitting.
        for piece in _split_by_tokens(full, max_tokens):
            _add_segment(crumb, piece, segments)
        return

    for child in node["children"]:
        child_full = _full_text(child)
        child_tokens = estimate_tokens(child_full)

        if child_tokens > max_tokens:
            flush()
            _emit_section(child, crumb, min_tokens, max_tokens, segments)
        elif buffer_tokens + child_tokens > max_tokens:
            flush()
            buffer_parts = [child_full]
            buffer_tokens = child_tokens
        else:
            buffer_parts.append(child_full)
            buffer_tokens += child_tokens

    flush()


def _add_segment(breadcrumb, content, segments):
    content = content.strip("\n")
    if not content.strip():
        return
    # The last breadcrumb element is the section's own title (already present as a
    # heading in content). Use ancestors as the location context.
    location = " > ".join(c for c in breadcrumb if c)
    segments.append({"breadcrumb": location, "content": content})


def _first_heading_level(text):
    """Return the ATX level of the first heading in text, or None."""
    m = re.search(r"^(#{1,6})\s+.*\S\s*$", text, re.MULTILINE)
    return len(m.group(1)) if m else None


def split_markdown_by_headings(content, min_tokens=1500, max_tokens=10000):
    """Split a large markdown document into balanced, structure-aware segments.

    Returns a list of (virtual_path, content) tuples, where content is prefixed
    with a breadcrumb line so each segment carries its location in the document.
    """
    root = _parse_sections(content)
    segments = []
    _emit_section(root, [], min_tokens, max_tokens, segments)

    # Pass 1: combine adjacent segments that share the same location.
    merged = []
    for seg in segments:
        if (
            merged
            and merged[-1]["breadcrumb"] == seg["breadcrumb"]
            and estimate_tokens(merged[-1]["content"] + seg["content"]) <= max_tokens
        ):
            merged[-1]["content"] += "\n\n" + seg["content"]
        else:
            merged.append(seg)

    # Pass 2: absorb segments below min_tokens into a neighbor (prefer previous),
    # as long as the combined size stays within max_tokens.
    compacted = []
    for seg in merged:
        first_heading_level = _first_heading_level(seg["content"])
        # In normalized book/report Markdown, H2 is the chapter level. Keep
        # chapter starts visible instead of absorbing them into the previous
        # chapter just because the intro is short.
        starts_chapter = first_heading_level == 2
        if (
            compacted
            and not starts_chapter
            and estimate_tokens(seg["content"]) < min_tokens
            and estimate_tokens(compacted[-1]["content"] + seg["content"]) <= max_tokens
        ):
            compacted[-1]["content"] += "\n\n" + seg["content"]
        else:
            compacted.append(seg)
    # A leading tiny segment can't merge backwards; fold it into the next one.
    if (
        len(compacted) >= 2
        and not (
            (_first_heading_level(compacted[0]["content"]) is not None)
            and _first_heading_level(compacted[0]["content"]) == 2
        )
        and estimate_tokens(compacted[0]["content"]) < min_tokens
        and estimate_tokens(compacted[0]["content"] + compacted[1]["content"]) <= max_tokens
    ):
        compacted[1]["content"] = compacted[0]["content"] + "\n\n" + compacted[1]["content"]
        compacted.pop(0)
    merged = compacted

    files = []
    for i, seg in enumerate(merged):
        # Derive a readable virtual path from the breadcrumb + first heading.
        first_heading = ""
        m = re.search(r"^#{1,6}\s+(.*\S)\s*$", seg["content"], re.MULTILINE)
        if m:
            first_heading = m.group(1).strip()
        label = first_heading or seg["breadcrumb"] or f"section_{i}"
        safe = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "_", label).strip("_")
        path = f"{i:03d}_{safe}.md"

        prefix = f"<!-- Location: {seg['breadcrumb']} -->\n\n" if seg["breadcrumb"] else ""
        files.append((path, prefix + seg["content"]))

    return files
