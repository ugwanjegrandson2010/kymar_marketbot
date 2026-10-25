"""Terminal art and layout for MarketBot.

Everything here is standard-library only and degrades to plain ASCII when the
terminal cannot render Unicode or colour. Nothing is ever emitted with a raw
escape sequence that would corrupt a piped or logged run.

Components: colour palette, cell-width measurement, banner, rule, notice box,
panel, sparkline, gauge, table, and a `stat` helper that prints an explained
number rather than a bare figure.
"""

from __future__ import annotations

import os
import sys
import unicodedata
from typing import Optional, Sequence

# --------------------------------------------------------------------------
# colour
# --------------------------------------------------------------------------
class Palette:
    """Semantic colour tokens.

    Raw escape literals scattered through the codebase is how themes rot; define
    the tokens once and reference them. Every token has a 16-colour fallback so
    a limited terminal still reads correctly.
    """

    def __init__(self, enabled: Optional[bool] = None):
        self.enabled = _colour_enabled() if enabled is None else enabled
        self._codes = {
            "accent": "38;5;38",
            "ok": "38;5;79",
            "bad": "38;5;203",
            "warn": "38;5;220",
            "accent2": "38;5;141",
            "muted": "38;5;244",
            "bold": "1",
            "dim": "2",
        }
        self._fallbacks = {
            "accent": "36",      # cyan
            "ok": "32",          # green
            "bad": "31",         # red
            "warn": "33",        # yellow
            "accent2": "35",     # magenta
            "muted": "90",       # bright black
            "bold": "1",
            "dim": "2",
        }
        self._use_256 = self.enabled and _supports_256()

    def paint(self, text: str, token: str) -> str:
        """Wrap visible characters in a token's colour.

        Spaces are never colourised: wrapping one desynchronises every
        right-padding calculation downstream.
        """
        if not self.enabled or not text:
            return text
        code = self._codes[token] if self._use_256 else self._fallbacks[token]
        return f"\033[{code}m{text}\033[0m"

    def good(self, text: str) -> str:
        """Green for positive, red for negative, plain for neutral.

        Colour is always paired with a symbol by the caller, so meaning never
        depends on hue alone.
        """
        return self.paint(text, "ok")

    def bad(self, text: str) -> str:
        return self.paint(text, "bad")

    def sign(self, value: float, text: str) -> str:
        """Colour a signed number and prefix it with an explicit marker."""
        token = "ok" if value > 0 else "bad" if value < 0 else "muted"
        return self.paint(text, token)


def _colour_enabled() -> bool:
    """Colour is on unless the environment or target says otherwise."""
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("TERM") == "dumb":
        return False
    return sys.stdout.isatty()


def _supports_256() -> bool:
    term = os.environ.get("TERM", "")
    return "256" in term or term in ("xterm", "screen", "tmux", "linux", "ansi")


DEFAULT = Palette()


def palette(force: Optional[bool] = None) -> Palette:
    """A palette for a specific sink. `force=True` for explicit styling."""
    return Palette(force)


# --------------------------------------------------------------------------
# width
# --------------------------------------------------------------------------
def cell_width(text: str) -> int:
    """Display width in terminal cells, not bytes or code points.

    CJK and emoji occupy two cells; combining marks occupy none. Getting this
    wrong desynchronises every box and table drawn afterwards.
    """
    width = 0
    for ch in text:
        if unicodedata.combining(ch):
            continue
        width += 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
    return width or len(text)


def pad(text: str, width: int, align: str = "left") -> str:
    """Pad to a cell width, truncating rather than wrapping."""
    text = text[: max(0, width)]
    gap = width - cell_width(text)
    if gap <= 0:
        return text
    if align == "right":
        return " " * gap + text
    if align == "center":
        left = gap // 2
        return " " * left + text + " " * (gap - left)
    return text + " " * gap


# --------------------------------------------------------------------------
# structure
# --------------------------------------------------------------------------
_BOX = {
    "light": ("┌", "┐", "└", "┘", "─", "│", "├", "┤", "┬", "┴", "┼"),
    "heavy": ("┏", "┓", "┗", "┛", "━", "┃", "┣", "┫", "┳", "┻", "╋"),
    "ascii": ("+", "+", "+", "+", "-", "|", "+", "+", "+", "+", "+"),
}


def banner(text: str, width: int = 78, slogan: str = "", pal: Palette = DEFAULT) -> str:
    """Block-letter banner, 5 rows tall.

    Uses `#` rather than Unicode blocks so it survives a terminal with no
    Unicode font. Colour is applied to the glyphs only.
    """
    font = _font()
    rows = [""] * 5
    for ch in text.upper():
        glyph = font.get(ch)
        if glyph is None:
            for i in range(5):
                rows[i] += "      "
            continue
        for i in range(5):
            rows[i] += glyph[i] + "  "
    # Fixed-width rows, not rstripped: stripping made each row's width depend on
    # the final glyph, so the banner no longer aligned with itself.
    lines = [pal.paint(row[:width], "accent") for row in rows]
    if slogan:
        lines.append("")
        lines.append(pal.paint(pad(slogan, width, "center"), "muted"))
    return "\n".join(lines)


def _font() -> dict:
    """5-row block font. Deliberately `#`-based: it renders anywhere."""
    return {
        "A": ["  ##  ", " #  # ", "######", "#    #", "#    #"],
        "B": ["##### ", "#    #", "##### ", "#    #", "##### "],
        "C": [" #####", "#     ", "#     ", "#     ", " #####"],
        "D": ["##### ", "#    #", "#    #", "#    #", "##### "],
        "E": ["######", "#     ", "##### ", "#     ", "######"],
        "F": ["######", "#     ", "##### ", "#     ", "#     "],
        "G": [" #####", "#     ", "#  ###", "#    #", " #####"],
        "H": ["#    #", "#    #", "######", "#    #", "#    #"],
        "I": ["######", "  ##  ", "  ##  ", "  ##  ", "######"],
        "J": ["     #", "     #", "     #", "#    #", " #### "],
        "K": ["#    #", "#  #  ", "###   ", "#  #  ", "#    #"],
        "L": ["#     ", "#     ", "#     ", "#     ", "######"],
        "M": ["#    #", "##  ##", "# ## #", "#    #", "#    #"],
        "N": ["#    #", "##   #", "# #  #", "#  # #", "#   ##"],
        "O": [" #### ", "#    #", "#    #", "#    #", " #### "],
        "P": ["##### ", "#    #", "##### ", "#     ", "#     "],
        "Q": [" #### ", "#    #", "#    #", "#  # #", " # ## "],
        "R": ["##### ", "#    #", "##### ", "#  #  ", "#   ##"],
        "S": [" #####", "#     ", " #### ", "     #", "##### "],
        "T": ["######", "  ##  ", "  ##  ", "  ##  ", "  ##  "],
        "U": ["#    #", "#    #", "#    #", "#    #", " #### "],
        "V": ["#    #", "#    #", "#    #", " #  # ", "  ##  "],
        "W": ["#    #", "#    #", "# ## #", "##  ##", "#    #"],
        "X": ["#    #", " #  # ", "  ##  ", " #  # ", "#    #"],
        "Y": ["#    #", " #  # ", "  ##  ", "  ##  ", "  ##  "],
        "Z": ["######", "    # ", "  ##  ", " #    ", "######"],
        "0": [" #### ", "#   ##", "#  # #", "# #  #", " #### "],
        "1": ["  ##  ", " # #  ", "  ##  ", "  ##  ", "######"],
        "2": [" #### ", "#    #", "   ## ", "  #   ", "######"],
        "3": [" #####", "     #", "  ### ", "     #", "##### "],
        "4": ["   ## ", "  # # ", " #  # ", "######", "    # "],
        "5": ["######", "#     ", "##### ", "     #", "##### "],
        "6": [" #### ", "#     ", "##### ", "#    #", " #### "],
        "7": ["######", "     #", "    # ", "   #  ", "  #   "],
        "8": [" #### ", "#    #", " #### ", "#    #", " #### "],
        "9": [" #### ", "#    #", " #####", "     #", " #### "],
    }


def rule(width: int = 78, style: str = "light", pal: Palette = DEFAULT) -> str:
    """Horizontal divider."""
    ch = _BOX[style][4]
    return pal.paint(ch * max(0, width), "muted")


def notice(
    title: str,
    body: Sequence[str],
    width: int = 78,
    tone: str = "info",
    style: str = "light",
    pal: Palette = DEFAULT,
) -> str:
    """Boxed callout. The marker carries the meaning without colour."""
    marks = {"info": "i", "ok": "+", "warn": "!", "bad": "x"}
    mark = marks.get(tone, "i")
    token = {"info": "muted", "ok": "ok", "warn": "warn", "bad": "bad"}.get(tone, "muted")
    c = _BOX[style]
    inner = width - 4
    out = [pal.paint(c[0] + c[4] * (width - 2) + c[1], token)]
    out.append(
        c[5] + " " + pal.paint(f"[{mark}] {title}"[:inner], token).ljust(inner) + " " + c[5]
    )
    out.append(pal.paint(c[6] + c[4] * (width - 2) + c[7], token))
    for line in body:
        for wrapped in wrap(line, inner):
            out.append(c[5] + " " + wrapped.ljust(inner) + " " + c[5])
    out.append(pal.paint(c[2] + c[4] * (width - 2) + c[3], token))
    return "\n".join(out)


def panel(
    title: str,
    body: Sequence[str],
    width: int = 78,
    style: str = "light",
    pal: Palette = DEFAULT,
    wrap_body: bool = True,
) -> str:
    """Titled section box.

    The box grows to fit its widest line rather than wrapping content that is
    already laid out — a table inside a panel must not be re-wrapped, or it
    breaks mid-character. Pass `wrap_body=False` for pre-formatted blocks and
    the lines are truncated instead of wrapped.
    """
    c = _BOX[style]
    # A body entry may be a pre-formatted block with embedded newlines (a table).
    # Flatten first: measuring the whole string would count every line plus its
    # newline as one width and the box would balloon.
    flat: list[str] = []
    for line in body:
        flat.extend(line.split("\n"))
    widest = max((cell_width(l) for l in flat), default=0)
    inner = width - 4 if wrap_body else max(width - 4, widest)
    box_width = inner + 4
    out = [pal.paint(c[0] + c[4] * (box_width - 2) + c[1], "accent")]
    out.append(c[5] + " " + pal.paint(pad(title[:inner], inner), "accent") + " " + c[5])
    out.append(pal.paint(c[6] + c[4] * (box_width - 2) + c[7], "muted"))
    for line in flat:
        if wrap_body:
            for wrapped in wrap(line, inner):
                out.append(c[5] + " " + wrapped.ljust(inner) + " " + c[5])
        else:
            out.append(c[5] + " " + pad(line, inner)[:inner].ljust(inner) + " " + c[5])
    out.append(pal.paint(c[2] + c[4] * (box_width - 2) + c[3], "muted"))
    return "\n".join(out)


def wrap(text: str, width: int) -> list[str]:
    """Word-wrap on cell width, never on character count."""
    out, line, w = [], "", 0
    for word in text.split(" "):
        ww = cell_width(word)
        if w + ww + (1 if line else 0) > width and line:
            out.append(line)
            line, w = word, ww
        else:
            line = f"{line} {word}" if line else word
            w = cell_width(line)
    out.append(line)
    return out or [""]


# --------------------------------------------------------------------------
# data display
# --------------------------------------------------------------------------
_RAMP = "▁▂▃▄▅▆▇█"
_RAMP_ASCII = "_.-~+=#@"


def sparkline(series: Sequence[float], width: int = 40, pal: Palette = DEFAULT) -> str:
    """Trend line. Down-samples rather than dropping points when too long."""
    if not series:
        return pal.paint("(no data)", "muted")
    if len(series) > width:
        step = len(series) / width
        series = [series[int(i * step)] for i in range(width)]
    lo, hi = min(series), max(series)
    unicode_ok = _unicode_ok()
    ramp = _RAMP if unicode_ok else _RAMP_ASCII
    if hi == lo:
        return pal.paint(ramp[len(ramp) // 2] * len(series), "accent2")
    return pal.paint(
        "".join(ramp[int((v - lo) / (hi - lo) * (len(ramp) - 1))] for v in series),
        "accent",
    )


def sparkline_labeled(
    series: Sequence[float],
    label: str,
    width: int = 40,
    pal: Palette = DEFAULT,
) -> str:
    """Sparkline with min/max/current annotated so it needs no separate axis."""
    if not series:
        return f"{label:<14}{pal.paint('(no data)', 'muted')}"
    lo, hi, cur = min(series), max(series), series[-1]
    spark = sparkline(series, width, pal)
    return (
        f"{pal.paint(pad(label, 14), 'muted')} {spark} "
        f"{pal.paint(f'lo {lo:,.2f}', 'muted')} {pal.paint(f'hi {hi:,.2f}', 'muted')} "
        f"now {pal.paint(f'{cur:,.2f}', 'accent')}"
    )


def gauge(value: float, total: float, width: int = 26, pal: Palette = DEFAULT) -> str:
    """Determinate bar. Fixed-width percentage field avoids refresh jitter."""
    if total <= 0:
        return pal.paint("[--]" + " " * width, "muted")
    pct = max(0.0, min(1.0, value / total))
    n = int(round(pct * width))
    filled, empty = ("█", "░") if _unicode_ok() else ("#", ".")
    bar = pal.paint(filled * n, "accent") + pal.paint(empty * (width - n), "muted")
    return f"[{bar}] {pct * 100:5.1f}%"


def table(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    pal: Palette = DEFAULT,
    align_right: Optional[Sequence[int]] = None,
) -> str:
    """Aligned table. Truncates rather than wraps — a wrapping table cannot be
    scanned; full values belong in a detail view.

    Headers count toward the column width. Measuring only the data truncates
    any column whose header is longer than every cell ("trades" became "tr"),
    which is the difference between a table and a puzzle.
    """
    if not rows:
        return notice("empty", ["no rows to display"], pal=pal)
    align_right = set(align_right or [])
    c = _BOX["light"]
    widths = [
        max([cell_width(str(h)) for h in headers]
            + [cell_width(str(r[i])) for r in rows]) + 2
        for i in range(len(headers))
    ]

    def fmt(cells, header=False):
        out = []
        for i, cell in enumerate(cells):
            cell = str(cell)[: max(0, widths[i] - 1)]
            # Headers are never right-aligned: a right-aligned header reads as a
            # data value and breaks the association with its column.
            style = "right" if (i in align_right and not header) else "left"
            out.append(pal.paint(pad(cell, widths[i] - 1, style),
                                 "accent" if header else ""))
        return c[5] + c[5].join(out) + c[5]

    # A table is a box: it needs its own corners. Reusing the row separator for
    # the top and bottom drew a T where a corner belongs.
    #
    # The border draws `widths[i] - 1` dashes per column, exactly matching the
    # cell interior — drawing `widths[i]` made every border six characters
    # wider than the rows it was supposed to frame.
    top = c[0] + c[8].join(c[4] * (w - 1) for w in widths) + c[1]
    mid = c[6] + c[10].join(c[4] * (w - 1) for w in widths) + c[7]
    bottom = c[2] + c[9].join(c[4] * (w - 1) for w in widths) + c[3]

    lines = [pal.paint(top, "muted"), fmt(headers, header=True),
             pal.paint(mid, "muted")]
    lines += [fmt(r) for r in rows]
    lines.append(pal.paint(bottom, "muted"))
    return "\n".join(lines)


# --------------------------------------------------------------------------
# explained statistics
# --------------------------------------------------------------------------
def stat(
    label: str,
    value: str,
    explanation: str = "",
    tone: str = "muted",
    pal: Palette = DEFAULT,
) -> str:
    """One line: label, value, and why it matters.

    A bare figure is noise. The explanation is what turns output into a report.
    """
    line = f"{pal.paint(pad(label, 20), 'muted')} {pal.paint(value, tone)}"
    if explanation:
        line += f"  {pal.paint('— ' + explanation, 'muted')}"
    return line


def pct(value: Optional[float], digits: int = 1) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.{digits}f}%"


def signed(value: float, digits: int = 1) -> str:
    return f"{value * 100:+.{digits}f}%"


def delta(value: float, digits: int = 1, pal: Palette = DEFAULT) -> str:
    """Signed percentage with an explicit marker, so meaning survives colour loss."""
    marker = "▲" if value > 0 else "▼" if value < 0 else "="
    token = "ok" if value > 0 else "bad" if value < 0 else "muted"
    return pal.paint(f"{marker} {value * 100:+.{digits}f}%", token)


def _unicode_ok() -> bool:
    if os.environ.get("MARKETBOT_ASCII", "").lower() in ("1", "true", "yes"):
        return False
    encoding = (os.environ.get("LANG") or os.environ.get("LC_ALL") or "").upper()
    return "UTF" in encoding
