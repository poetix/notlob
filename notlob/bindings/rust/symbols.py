"""notlob.bindings.rust.symbols — Symbol extractor for Rust code.

Extracts top-level item names from a Rust code block without a full
parser.  As with the Haskell extractor, it relies on the structural
invariants of well-formatted Rust:

  • Top-level items start at column 0.
  • Item bodies and continuation lines are indented.
  • A top-level ``}`` (the close of a block item) or a blank line
    terminates the current item.

These hold for Rust written in a literate .lob document and are
sufficient for symbol extraction.

Recognised forms
----------------
  fn name(…)                  free function (incl. pub/const/async/unsafe)
  struct Name …               struct declaration
  enum Name …                 enum declaration
  trait Name …                trait declaration
  type Name …                 type alias
  const NAME: …               associated/free constant
  static NAME: …              static item
  union Name …                union declaration
  macro_rules! name           declarative macro

Not extracted
-------------
  impl blocks (no single name; methods are an implementation detail —
    this mirrors the Haskell extractor skipping ``instance``)
  use statements, mod declarations, attributes (``#[…]``), comments

Visibility and item modifiers (``pub``, ``pub(crate)``, ``const``,
``async``, ``unsafe``, ``extern "C"``) are stripped before classifying.
"""

from __future__ import annotations

import re
import textwrap
from typing import Sequence

from notlob.bindings import SymbolInfo


# ── Modifier stripping ────────────────────────────────────────

# Leading item modifiers that precede the item keyword.  ``pub`` may
# carry a restriction in parentheses: ``pub(crate)``, ``pub(super)``.
_MODIFIER_RE = re.compile(
    r"^(?:pub(?:\([^)]*\))?|const|async|unsafe|extern(?:\s+\"[^\"]*\")?)\s+"
)


def _strip_modifiers(line: str) -> str:
    """Remove leading item modifiers (pub, const, async, unsafe, extern)."""
    prev = None
    while prev != line:
        prev = line
        line = _MODIFIER_RE.sub("", line, count=1)
    return line


# ── Per-keyword classifiers ───────────────────────────────────

_FN_RE     = re.compile(r"^fn\s+([A-Za-z_][A-Za-z0-9_]*)")
_STRUCT_RE = re.compile(r"^struct\s+([A-Za-z_][A-Za-z0-9_]*)")
_ENUM_RE   = re.compile(r"^enum\s+([A-Za-z_][A-Za-z0-9_]*)")
_TRAIT_RE  = re.compile(r"^trait\s+([A-Za-z_][A-Za-z0-9_]*)")
_UNION_RE  = re.compile(r"^union\s+([A-Za-z_][A-Za-z0-9_]*)")
_TYPE_RE   = re.compile(r"^type\s+([A-Za-z_][A-Za-z0-9_]*)")
_CONST_RE  = re.compile(r"^const\s+([A-Za-z_][A-Za-z0-9_]*)\s*:")
_STATIC_RE = re.compile(r"^static\s+(?:mut\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*:")
_MACRO_RE  = re.compile(r"^macro_rules!\s*([A-Za-z_][A-Za-z0-9_]*)")

_CLASSIFIERS = (
    _FN_RE, _STRUCT_RE, _ENUM_RE, _TRAIT_RE, _UNION_RE,
    _TYPE_RE, _CONST_RE, _STATIC_RE, _MACRO_RE,
)


def _top_level_name(line: str) -> str | None:
    """Return the item name defined by this column-0 Rust line.

    Returns ``None`` for lines that are not top-level item definitions
    (``use``, ``mod``, ``impl``, attributes, comments, ``}`` closers).

    ``const``/``static`` are matched before the modifier strip would
    consume their keyword, so the dedicated regexes run on the original
    line first.
    """
    stripped = line.strip()

    # const NAME: … and static NAME: … keep their leading keyword.
    m = _CONST_RE.match(stripped)
    if m:
        return m.group(1)
    m = _STATIC_RE.match(stripped)
    if m:
        return m.group(1)

    body = _strip_modifiers(stripped)
    for rx in _CLASSIFIERS:
        m = rx.match(body)
        if m:
            return m.group(1)
    return None


# ── Main extractor ────────────────────────────────────────────

def extract_symbols(lines: Sequence[str]) -> list[SymbolInfo]:
    """Extract top-level Rust item definitions from code-block lines.

    *lines* are the raw indented lines stored in a
    :class:`~notlob.model.CodeBlock` (leading whitespace preserved).
    They are dedented before processing so the column-0 invariant
    applies.

    One :class:`~notlob.bindings.SymbolInfo` is produced per distinct
    top-level item.  An item's indented body (and any blank lines within
    it) is folded into the ``source`` field; a top-level ``}`` or a blank
    line closes the current item.

    >>> [s.name for s in extract_symbols(["    fn add(a: i32) -> i32 {", "        a + 1", "    }"])]
    ['add']
    >>> [s.name for s in extract_symbols(["    pub struct Foo {", "        x: f32,", "    }"])]
    ['Foo']
    >>> extract_symbols([])
    []
    """
    source_text  = textwrap.dedent("\n".join(lines))
    source_lines = source_text.splitlines()

    result:    list[SymbolInfo] = []
    cur_name:  str | None       = None
    cur_lines: list[str]        = []

    def _flush() -> None:
        if cur_name is not None:
            result.append(SymbolInfo(
                name=cur_name,
                source="\n".join(cur_lines),
            ))

    for raw in source_lines:
        line = raw.rstrip()

        if not line:
            # Blank line — part of the current item body if one is open;
            # otherwise nothing to do.
            if cur_name is not None:
                cur_lines.append(line)
            continue

        if line[0] in (" ", "\t"):
            # Indented continuation — attach to current item.
            if cur_name is not None:
                cur_lines.append(line)
            continue

        # Non-blank, column-0 line.  A bare closer (`}`, `};`) ends the
        # current item and starts nothing new.
        if line.lstrip().startswith("}"):
            if cur_name is not None:
                cur_lines.append(line)
            _flush()
            cur_name  = None
            cur_lines = []
            continue

        name = _top_level_name(line)
        if name is None:
            # Unrecognised top-level line (use, mod, impl, attribute, …).
            _flush()
            cur_name  = None
            cur_lines = []
            continue

        _flush()
        cur_name  = name
        cur_lines = [line]

    _flush()
    return result
