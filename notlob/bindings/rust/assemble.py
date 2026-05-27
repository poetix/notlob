"""notlob.bindings.rust.assemble — Rust code assembler.

Assembles a Module into a flat Rust source string: ``use`` statements
followed by all code blocks as top-level items.  Unlike Haskell, a Rust
source file needs no module header — the file *is* the module — so the
assembler emits items directly at the crate root.

Assembly order
--------------
  1. ``#References`` lines  (Rust ``use`` statements; lob-refs dropped)
  2. Module-level code blocks  (preceded by ``// <module_address>``)
  3. Subheading code blocks in document order
     (each group preceded by ``// <subheading_address>``)

Cross-module composition is *flattening*: dependency items are emitted
at the same crate root as the importing module, so a symbol defined in a
dependency is in scope without a Rust ``use`` — the lob-ref edge is the
only declaration needed.  Authors therefore call dependency symbols by
bare name (``flush_denormal(x)``), not ``crate::flush_denormal(x)``.

All top-level chunks are separated by a single blank line.  A location
comment is glued directly to its first code block.

The internal ``_assemble_body`` helper returns
``(use_lines, code_chunks)`` for use by the runner when it injects user
code into a harness that supplies its own ``fn main``.
"""

from __future__ import annotations

import re
import textwrap

from notlob.graph import module_address, subheading_address
from notlob.model import CodeBlock, Module, ReferencesSection, Subheading


# ── Crate-name derivation (used by `notlob build`, not by run/test) ──

def _module_name(title: str) -> str:
    """Derive a Rust module identifier from a notlob title.

    Splits on non-alphanumerics, lowercases, joins with underscores —
    Rust module/crate naming convention (snake_case).

    >>> _module_name("Patches Dc Blocker")
    'patches_dc_blocker'
    >>> _module_name("Denormal")
    'denormal'
    """
    words = re.split(r"[^A-Za-z0-9]+", title)
    return "_".join(w.lower() for w in words if w)


# ── References parsing ────────────────────────────────────────

def _rust_uses(lines: list[str]) -> list[str]:
    """Return Rust ``use``/``extern crate`` lines from a ``#References``.

    Drops blank lines and lob-ref lines (those whose stripped form starts
    with ``#``).  All other lines are assumed to be Rust ``use`` (or
    ``extern crate``) statements and are passed through verbatim.

    >>> _rust_uses([
    ...     "    #Patches Denormal",
    ...     "    use std::f32::consts::TAU;",
    ...     "",
    ...     "    use std::cmp::min;",
    ... ])
    ['use std::f32::consts::TAU;', 'use std::cmp::min;']
    """
    result = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            result.append(stripped)
    return result


# ── Block collection ──────────────────────────────────────────

def _collect_blocks(body: list) -> list[str]:
    """Return dedented, stripped text for each CodeBlock in body."""
    result = []
    for item in body:
        if isinstance(item, CodeBlock):
            text = textwrap.dedent("\n".join(item.lines)).strip()
            if text:
                result.append(text)
    return result


def _section(comment: str, blocks: list[str]) -> str:
    """Join a location comment and its code blocks.

    The comment is glued to the first block (no blank line between);
    subsequent blocks within the same section are blank-line separated.
    """
    first, *rest = blocks
    head = f"{comment}\n{first}"
    if rest:
        return head + "\n\n" + "\n\n".join(rest)
    return head


# ── Body decomposition ────────────────────────────────────────

def _assemble_body(module: Module) -> tuple[list[str], list[str]]:
    """Return ``(use_lines, code_chunks)`` for the module.

    *use_lines* — Rust ``use`` statements from the module's
    ``#References`` section (lob-refs excluded).

    *code_chunks* — dedented code-block text groups, each preceded by a
    ``// address`` location comment.  One entry per module-level or
    subheading-level group.
    """
    use_lines: list[str] = []
    if module.post_text is not None:
        for section in module.post_text.sections:
            if isinstance(section, ReferencesSection):
                use_lines = _rust_uses(section.lines)
                break

    code_chunks: list[str] = []
    mod_addr = module_address(module.title)

    mod_blocks = _collect_blocks(module.body)
    if mod_blocks:
        code_chunks.append(_section(f"// {mod_addr}", mod_blocks))

    for item in module.body:
        if isinstance(item, Subheading):
            sub_addr = subheading_address(mod_addr, item.title)
            sub_blocks = _collect_blocks(item.body)
            if sub_blocks:
                code_chunks.append(_section(f"// {sub_addr}", sub_blocks))

    return use_lines, code_chunks


# ── Multi-module merge ────────────────────────────────────────

def _merge_modules(modules: list[Module]) -> tuple[list[str], list[str]]:
    """Merge use-lines and code chunks from an ordered list of modules.

    ``use`` lines are deduplicated (first occurrence wins); code chunks
    preserve declaration order.  Used to build a single-file harness that
    inlines all dependency modules before the current module so every
    symbol is in scope at the crate root.
    """
    seen:        set[str]  = set()
    use_lines:   list[str] = []
    code_chunks: list[str] = []

    for mod in modules:
        uses, chunks = _assemble_body(mod)
        for line in uses:
            if line not in seen:
                seen.add(line)
                use_lines.append(line)
        code_chunks.extend(chunks)

    return use_lines, code_chunks


# ── Public assembler ──────────────────────────────────────────

def assemble(module: Module) -> str:
    """Assemble module code blocks into a flat Rust source string.

    Returns ``use`` lines followed by all code blocks as top-level
    items.  Returns an empty string if the module has no code and no
    uses.

    >>> from notlob.model import Module
    >>> m = Module(title="Empty", body=[], post_text=None)
    >>> assemble(m)
    ''
    """
    use_lines, code_chunks = _assemble_body(module)
    if not code_chunks and not use_lines:
        return ""

    parts: list[str] = []
    if use_lines:
        parts.append("\n".join(use_lines))
    parts.extend(code_chunks)
    return "\n\n".join(parts)


def assemble_with_deps(module: Module, dep_modules: list[Module]) -> str:
    """Assemble *module* with inlined *dep_modules* into one Rust source.

    Dependencies are flattened before the current module so every symbol
    they define is in scope at the crate root.  This is the single-file
    equivalent of linking: ``dep_modules`` play the role of compiled
    library crates.

    Returns an empty string if neither the module nor any dependency
    contains code.
    """
    use_lines, code_chunks = _merge_modules(dep_modules + [module])
    if not code_chunks and not use_lines:
        return ""

    parts: list[str] = []
    if use_lines:
        parts.append("\n".join(use_lines))
    parts.extend(code_chunks)
    return "\n\n".join(parts)
