import re

_BULLET_SPLIT_RE = re.compile(r"(```.*?```)", flags=re.DOTALL)
_BULLET_ITEM_RE = re.compile(r"(?m)^([ \t]{0,3})[-*+]\s+")
_NEWLINE_COLLAPSE_RE = re.compile(r"\n{3,}")
_CODE_BLOCK_RE = re.compile(r"```[a-zA-Z0-9_+-]*\n?(.*?)```", re.DOTALL)
_HEADING_RE = re.compile(r"^#{1,6}\s+", re.MULTILINE)
_INLINE_PATTERNS = [
    (re.compile(r"\*\*(.+?)\*\*", re.DOTALL), "BOLD"),
    (re.compile(r"__(.+?)__", re.DOTALL), "BOLD"),
    (re.compile(r"~~(.+?)~~", re.DOTALL), "STRIKETHROUGH"),
    (re.compile(r"`(.+?)`"), "MONOSPACE"),
    (re.compile(r"(?<!\*)\*(?!\*| )(.+?)(?<!\*)\*(?!\*)"), "ITALIC"),
    (re.compile(r"(?<!\w)_(?!_)(.+?)(?<!_)_(?!\w)"), "ITALIC"),
]

def _utf16_len(s: str) -> int:
    """Length of *s* in UTF-16 code units."""
    return len(s.encode("utf-16-le")) // 2

def _normalize_bullet_markers(source: str) -> str:
    """Replace Markdown bullet markers with plain Unicode bullets."""
    parts = _BULLET_SPLIT_RE.split(source)
    for idx, part in enumerate(parts):
        if idx % 2 == 1:
            continue
        parts[idx] = _BULLET_ITEM_RE.sub(r"\1• ", part)
    return "".join(parts)

def _adjust(pos: int, removals: list[tuple[int, int]]) -> int:
    shift = 0
    for remove_pos, remove_len in removals:
        if remove_pos < pos:
            shift += min(remove_len, pos - remove_pos)
        else:
            break
    return pos - shift

def markdown_to_signal(text: str) -> tuple[str, list[str]]:
    text = _NEWLINE_COLLAPSE_RE.sub("\n\n", text)
    text = text.strip()
    text = _normalize_bullet_markers(text)

    styles: list[tuple[int, int, str]] = []

    while match := _CODE_BLOCK_RE.search(text):
        inner = match.group(1).rstrip("\n")
        start = match.start()
        text = text[: match.start()] + inner + text[match.end() :]
        styles.append((start, len(inner), "MONOSPACE"))

    new_text = ""
    last_end = 0
    for match in _HEADING_RE.finditer(text):
        new_text += text[last_end : match.start()]
        last_end = match.end()
        eol = text.find("\n", match.end())
        if eol == -1:
            eol = len(text)
        heading_text = text[match.end() : eol]
        start = len(new_text)
        new_text += heading_text
        styles.append((start, len(heading_text), "BOLD"))
        last_end = eol
    new_text += text[last_end:]
    text = new_text

    all_matches: list[tuple[int, int, int, int, str]] = []
    occupied: list[tuple[int, int]] = []
    for pattern, style in _INLINE_PATTERNS:
        for match in pattern.finditer(text):
            ms, me = match.start(), match.end()
            if not any(ms < oe and me > os for os, oe in occupied):
                all_matches.append((ms, me, match.start(1), match.end(1), style))
                occupied.append((ms, me))
    all_matches.sort()

    removals: list[tuple[int, int]] = []
    for ms, me, g1s, g1e, _ in all_matches:
        if g1s > ms:
            removals.append((ms, g1s - ms))
        if me > g1e:
            removals.append((g1e, me - g1e))
    removals.sort()

    adjusted_prior: list[tuple[int, int, str]] = []
    for start, length, style in styles:
        new_start = _adjust(start, removals)
        new_end = _adjust(start + length, removals)
        if new_end > new_start:
            adjusted_prior.append((new_start, new_end - new_start, style))

    result = ""
    last_end = 0
    inline_styles: list[tuple[int, int, str]] = []
    for ms, me, g1s, g1e, style in all_matches:
        result += text[last_end:ms]
        pos = len(result)
        inner = text[g1s:g1e]
        result += inner
        inline_styles.append((pos, len(inner), style))
        last_end = me
    result += text[last_end:]
    text = result

    styles = adjusted_prior + inline_styles

    style_strings: list[str] = []
    for cp_start, cp_len, style_type in sorted(styles):
        if cp_start < 0 or cp_start + cp_len > len(text):
            continue
        u16_start = _utf16_len(text[:cp_start])
        u16_len = _utf16_len(text[cp_start : cp_start + cp_len])
        style_strings.append(f"{u16_start}:{u16_len}:{style_type}")

    return text, style_strings

print(markdown_to_signal("**hello** *world*"))
