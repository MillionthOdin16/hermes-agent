"""Utilities for preparing assistant text for speech synthesis.

The TTS provider should receive a spoken script, not raw chat Markdown.  This
module centralises the lightweight, deterministic cleanup used by explicit TTS
calls and gateway auto-TTS replies.

Non-ASCII characters are written as escapes on purpose so the file stays free of
invisible/look-alike glyphs.
"""

from __future__ import annotations

import html
import re

# ⚡ Bolt: Cache compiled regexes and test membership sets at module load
_DEG_C_NUM_RE = re.compile(r"(?<!\w)([-+]?\d+(?:\.\d+)?)\s*°\s*C\b", flags=re.IGNORECASE)
_DEG_F_NUM_RE = re.compile(r"(?<!\w)([-+]?\d+(?:\.\d+)?)\s*°\s*F\b", flags=re.IGNORECASE)
_DEG_C_RE = re.compile(r"°\s*C\b", flags=re.IGNORECASE)
_DEG_F_RE = re.compile(r"°\s*F\b", flags=re.IGNORECASE)
_DEG_NUM_RE = re.compile(r"(?<!\w)([-+]?\d+(?:\.\d+)?)\s*°")

_KMH_SPACED_RE = re.compile(r"(?<=\d)\s*km\s*/\s*h\b", flags=re.IGNORECASE)
_KMH_RE = re.compile(r"(?<=\d)\s*km/h\b", flags=re.IGNORECASE)
_MM_RE = re.compile(r"(?<=\d)\s*mm\b", flags=re.IGNORECASE)
_CM_RE = re.compile(r"(?<=\d)\s*cm\b", flags=re.IGNORECASE)
_M_RE = re.compile(r"(?<=\d)\s*m\b", flags=re.IGNORECASE)
_PER_RE = re.compile(r"(?<=\d)\s*/\s*(?=[A-Za-z])")

_NZD_RE = re.compile(r"NZ\$\s*([\d,]*\d(?:\.\d+)?)", flags=re.IGNORECASE)
_AUD_RE = re.compile(r"A\$\s*([\d,]*\d(?:\.\d+)?)", flags=re.IGNORECASE)
_USD_RE = re.compile(r"US\$\s*([\d,]*\d(?:\.\d+)?)", flags=re.IGNORECASE)
_EUR_RE = re.compile(r"€\s*([\d,]*\d(?:\.\d+)?)")
_GBP_RE = re.compile(r"£\s*([\d,]*\d(?:\.\d+)?)")
_BARE_USD_RE = re.compile(r"\$\s*([\d,]*\d(?:\.\d+)?)")
_PCT_RE = re.compile(r"(?<=\d)\s*%")

_NBSP_RE = re.compile("[\u00A0\u2007\u202F]")
_BULLET_RE = re.compile("[•◦▪▫]")

_MONEY_PCT_RE = re.compile(r"[$€£%]")
_OP_RE = re.compile(r"[&→⇒≈~•◦▪▫]")

_FLATTEN_NL_2_RE = re.compile(r"\n{2,}")
_FLATTEN_NL_PUNCT_RE = re.compile(r"(?<=[.!?;:,])\n")
_FLATTEN_DOT_SPACE_DOT_RE = re.compile(r"\.\s*\.")
_FLATTEN_SPACES_RE = re.compile(r"[ \t]{2,}")

_NEWLINES_3_RE = re.compile(r"\n{3,}")
_SPACES_2_RE = re.compile(r"[ \t]{2,}")
_PUNCT_SPACE_RE = re.compile(r"\s+([,.;:!?])")
_PUNCT_CHAR_RE = re.compile(r"([,.;:!?])([A-Za-z])")
_ELLIPSIS_4_RE = re.compile(r"\.{4,}")

# Sentinel appended to former heading lines so smooth_whitespace_for_tts can
# fold a heading into the sentence that follows it ("Weather, it will be sunny")
# rather than leaving a bare "Weather." label that reads abruptly aloud.
_HEAD = "\x00"

_MD_CODE_BLOCK_RE = re.compile(r"```[\s\S]*?```")
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\((?:[^()]|\([^)]*\))*\)")
_MD_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\((?:[^()]|\([^)]*\))*\)")
_MD_INLINE_CODE_RE = re.compile(r"`([^`]+)`")
_MD_BOLD_RE = re.compile(r"\*\*(.+?)\*\*", flags=re.DOTALL)
_MD_UNDERSCORE_BOLD_RE = re.compile(r"__(.+?)__", flags=re.DOTALL)
_MD_ITALIC_RE = re.compile(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", flags=re.DOTALL)
_MD_UNDERSCORE_ITALIC_RE = re.compile(r"(?<!_)_(?!_)(.+?)(?<!_)_(?!_)", flags=re.DOTALL)
_MD_STRIKE_RE = re.compile(r"~~(.+?)~~", flags=re.DOTALL)
_MD_HEADING_LINE_RE = re.compile(r"^[ \t]{0,3}#{1,6}[ \t]+(.+?)[ \t]*#*[ \t]*$", flags=re.MULTILINE)
_MD_BLOCKQUOTE_RE = re.compile(r"^\s*>\s?", flags=re.MULTILINE)
_MD_LIST_ITEM_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+", flags=re.MULTILINE)
_MD_HR_RE = re.compile(r"^\s*[-*_]{3,}\s*$", flags=re.MULTILINE)
_MD_TABLE_PIPE_RE = re.compile(r"\s*\|\s*")
_URL_RE = re.compile(r"https?://\S+")

# Broad emoji / pictograph cleanup.  Voice providers vary a lot here; most read
# emojis as awkward labels, so keep the speech script calm and literal.
_EMOJI_RE = re.compile(
    "["
    "\U0001F1E6-\U0001F1FF"
    "\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FAFF"
    "☀-➿"
    "]+",
    flags=re.UNICODE,
)
_VARIATION_SELECTOR_RE = re.compile("[︎️]")


def strip_markdown_for_tts(text: str) -> str:
    """Strip Markdown/Telegram formatting while preserving readable words."""
    if not text:
        return ""

    text = html.unescape(str(text))
    text = _MD_CODE_BLOCK_RE.sub(" ", text)
    text = _MD_IMAGE_RE.sub(lambda m: f" {m.group(1)} " if m.group(1) else " ", text)
    text = _MD_LINK_RE.sub(r"\1", text)
    text = _URL_RE.sub("", text)
    text = _MD_INLINE_CODE_RE.sub(r"\1", text)
    text = _MD_BOLD_RE.sub(r"\1", text)
    text = _MD_UNDERSCORE_BOLD_RE.sub(r"\1", text)
    text = _MD_ITALIC_RE.sub(r"\1", text)
    text = _MD_UNDERSCORE_ITALIC_RE.sub(r"\1", text)
    text = _MD_STRIKE_RE.sub(r"\1", text)
    # Mark headings (do not just delete the marker): the whitespace pass folds a
    # heading into the sentence after it so speech says "Weather, it will be
    # sunny" instead of a clipped "Weather." then a separate sentence.
    text = _MD_HEADING_LINE_RE.sub(lambda m: m.group(1).rstrip() + _HEAD, text)
    text = _MD_BLOCKQUOTE_RE.sub("", text)
    text = _MD_LIST_ITEM_RE.sub("", text)
    text = _MD_HR_RE.sub("", text)

    # Pipe tables are terrible read aloud.  Turn any leftover pipes into pauses
    # instead of letting a provider speak "vertical bar".
    text = _MD_TABLE_PIPE_RE.sub("; ", text)
    return text


def _normalize_temperature_ranges(text: str) -> str:
    # 11-17 degrees C -> "11 to 17 degrees Celsius" (en/em dash or hyphen).
    text = re.sub(
        r"(?<!\w)([-+\u2212]?\d+(?:\.\d+)?)\s*[\u2013\u2014-]\s*([-+\u2212]?\d+(?:\.\d+)?)\s*°\s*C\b",
        lambda m: f"{m.group(1).replace(chr(0x2212), '-')} to {m.group(2).replace(chr(0x2212), '-')} degrees Celsius",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"(?<!\w)([-+\u2212]?\d+(?:\.\d+)?)\s*[\u2013\u2014-]\s*([-+\u2212]?\d+(?:\.\d+)?)\s*°\s*F\b",
        lambda m: f"{m.group(1).replace(chr(0x2212), '-')} to {m.group(2).replace(chr(0x2212), '-')} degrees Fahrenheit",
        text,
        flags=re.IGNORECASE,
    )
    return text


def normalize_symbols_for_tts(text: str) -> str:
    """Expand common symbols/shorthand into words a TTS engine reads well."""
    if not text:
        return ""

    text = str(text)
    text = _NBSP_RE.sub(" ", text)  # non-breaking / thin spaces
    text = text.replace("\u2212", "-")  # minus sign
    text = text.replace("…", "...")  # ellipsis

    if "°" in text:
        text = _normalize_temperature_ranges(text)
        # Temperatures with a number.  Do this before generic degree handling.
        text = _DEG_C_NUM_RE.sub(r"\1 degrees Celsius", text)
        text = _DEG_F_NUM_RE.sub(r"\1 degrees Fahrenheit", text)
        # Bare units with no leading number ("measured in degrees C").
        text = _DEG_C_RE.sub("degrees Celsius", text)
        text = _DEG_F_RE.sub("degrees Fahrenheit", text)
        # Any remaining degree symbol (angles, stray cases).
        text = _DEG_NUM_RE.sub(r"\1 degrees", text)
        text = text.replace("°", " degrees")

    # Common weather/travel units.
    if "/" in text or "m" in text or "M" in text:
        text = _KMH_SPACED_RE.sub(" kilometres per hour", text)
        text = _KMH_RE.sub(" kilometres per hour", text)
        text = _MM_RE.sub(" millimetres", text)
        text = _CM_RE.sub(" centimetres", text)
        text = _M_RE.sub(" metres", text)

        # Numeric rates only ("5/month" -> "5 per month").  Requiring digit-then-letter
        # keeps "and/or", "N/A", "TCP/IP" and dates like "2026/06" intact.
        text = _PER_RE.sub(" per ", text)

    # Money and percentages.  The integer part must END in a digit so a trailing
    # comma ("A$50, ...") is not swallowed into the spoken amount.
    if _MONEY_PCT_RE.search(text):
        text = _NZD_RE.sub(r"\1 New Zealand dollars", text)
        text = _AUD_RE.sub(r"\1 Australian dollars", text)
        text = _USD_RE.sub(r"\1 US dollars", text)
        text = _EUR_RE.sub(r"\1 euros", text)
        text = _GBP_RE.sub(r"\1 pounds", text)
        text = _BARE_USD_RE.sub(r"\1 dollars", text)
        text = _PCT_RE.sub(" percent", text)

    # Operators and separators that commonly leak from formatted answers.
    if _OP_RE.search(text):
        text = text.replace("&", " and ")
        text = _BULLET_RE.sub(" ", text)  # bullet glyphs
        text = text.replace("→", " to ")  # ->
        text = text.replace("⇒", " to ")  # =>
        text = text.replace("≈", " about ")  # almost equal
        text = text.replace("~", " about ")

    text = _VARIATION_SELECTOR_RE.sub("", text)
    text = _EMOJI_RE.sub("", text)
    return text


def smooth_whitespace_for_tts(text: str) -> str:
    """Collapse visual formatting into calm spoken paragraphs.

    A former heading line (marked with the _HEAD sentinel) folds into the next
    content line as a spoken lead-in: "Weather" + "It will be sunny" becomes
    "Weather, It will be sunny."  A heading with no content after it becomes its
    own short sentence.
    """
    if not text:
        return ""

    raw_lines = text.splitlines()
    add_sentence_pauses = sum(1 for raw_line in raw_lines if raw_line.replace(_HEAD, "").strip()) > 1
    lines: list[str] = []
    pending_heading: str | None = None

    def flush_pending() -> None:
        nonlocal pending_heading
        if pending_heading is not None:
            lines.append(pending_heading.rstrip(".:;,") + ".")
            pending_heading = None

    for raw_line in raw_lines:
        is_heading = raw_line.rstrip().endswith(_HEAD)
        line = raw_line.replace(_HEAD, "").strip()
        if not line:
            # Hold a pending heading across blank lines so it still folds into
            # the next real content line; otherwise just collapse the blank.
            if pending_heading is None and lines and lines[-1] != "":
                lines.append("")
            continue
        if is_heading:
            flush_pending()
            pending_heading = line.rstrip(".:;,")
            continue
        if pending_heading is not None:
            line = f"{pending_heading.rstrip('.:;,')}, {line}"
            pending_heading = None
        if add_sentence_pauses and line[-1] not in ".!?;:":
            line += "."
        lines.append(line)

    flush_pending()

    text = "\n".join(lines)
    text = _NEWLINES_3_RE.sub("\n\n", text)
    text = _SPACES_2_RE.sub(" ", text)
    text = _PUNCT_SPACE_RE.sub(r"\1", text)
    text = _PUNCT_CHAR_RE.sub(r"\1 \2", text)
    text = _ELLIPSIS_4_RE.sub("...", text)
    return text.strip()


# Reasoning blocks: models with ``/reasoning show`` enabled emit
# ``<think>...</think>`` blocks in the final assistant message.  Users want to
# SEE reasoning, not hear it read aloud (#34213).
_THINK_BLOCK_RE = re.compile(r"<think[\s>].*?</think>", flags=re.DOTALL | re.IGNORECASE)
# An unterminated block (streaming cut-off) should still not be spoken.
_THINK_BLOCK_OPEN_RE = re.compile(r"<think[\s>].*\Z", flags=re.DOTALL | re.IGNORECASE)

# Turn-end file-mutation verifier footer appended by run_agent.py
# (``_format_file_mutation_failure_footer``).  It's a UI affordance — reading
# "warning file mutation verifier, 2 files were NOT modified..." aloud is
# noise (#40772).  The footer is a ``⚠️ File-mutation verifier:`` header line
# followed by indented ``•`` bullet lines; strip the whole block.
_VERIFIER_FOOTER_RE = re.compile(
    r"^\s*⚠️?\s*File-mutation verifier:.*(?:\n[ \t]+•.*)*",
    flags=re.MULTILINE,
)


def strip_nonspoken_blocks(text: str) -> str:
    """Remove blocks that must never reach a speech provider.

    Currently: ``<think>`` reasoning blocks and the end-of-turn
    file-mutation verifier footer.
    """
    if not text:
        return ""
    text = _THINK_BLOCK_RE.sub(" ", text)
    text = _THINK_BLOCK_OPEN_RE.sub(" ", text)
    text = _VERIFIER_FOOTER_RE.sub(" ", text)
    return text


def flatten_newlines_for_payload(text: str) -> str:
    """Collapse newlines into sentence breaks for single-line TTS payloads.

    Some OpenAI-compatible backends (e.g. Kokoro) truncate synthesis at the
    first newline (#9004).  The smoothing pass already terminates each line
    with punctuation, so newlines can safely become plain spaces.
    """
    if not text:
        return ""
    if "\n" in text:
        text = _FLATTEN_NL_2_RE.sub(". ", text)
        text = _FLATTEN_NL_PUNCT_RE.sub(" ", text)
        text = text.replace("\n", ". ")
    text = _FLATTEN_DOT_SPACE_DOT_RE.sub(".", text)
    text = _FLATTEN_SPACES_RE.sub(" ", text)
    return text.strip()


def prepare_spoken_text(text: str, max_chars: int | None = 4000) -> str:
    """Return a TTS-friendly script from assistant text.

    Deterministic cleanup, not a semantic rewrite: it removes ``<think>``
    reasoning blocks and the file-mutation verifier footer, removes Markdown,
    expands common symbols such as a degree-Celsius sign to "degrees Celsius",
    turns visual line formatting into speakable sentence pauses, and flattens
    the result to a single line so newline-sensitive providers (Kokoro) speak
    the whole script.
    """
    spoken = strip_nonspoken_blocks(text)
    spoken = strip_markdown_for_tts(spoken)
    spoken = normalize_symbols_for_tts(spoken)
    spoken = smooth_whitespace_for_tts(spoken)
    spoken = flatten_newlines_for_payload(spoken)
    if max_chars is not None and max_chars > 0 and len(spoken) > max_chars:
        spoken = spoken[:max_chars].rstrip()
    return spoken
