"""Tests for the Rust assembler."""

from __future__ import annotations

from notlob.bindings.rust.assemble import (
    assemble, assemble_with_deps, _module_name, _rust_uses, _merge_modules,
)
from notlob.model import (
    CodeBlock, Module, PostText, ReferencesSection, Subheading,
)


def _code(text):
    return CodeBlock(lines=["    " + ln for ln in text.splitlines()])


def _module(title, body=None, refs=None):
    body = body or []
    sections = []
    if refs is not None:
        sections.append(ReferencesSection(lines=refs))
    post = PostText(sections=sections) if sections else None
    return Module(title=title, body=body, post_text=post)


def test_module_name_snake_case():
    assert _module_name("Patches Dc Blocker") == "patches_dc_blocker"
    assert _module_name("Denormal") == "denormal"


def test_rust_uses_drops_lobrefs():
    lines = [
        "    #Patches Denormal",
        "    use std::f32::consts::TAU;",
        "",
        "    use std::cmp::min;",
    ]
    assert _rust_uses(lines) == [
        "use std::f32::consts::TAU;",
        "use std::cmp::min;",
    ]


def test_assemble_empty():
    assert assemble(_module("Empty")) == ""


def test_assemble_emits_uses_and_location_comment():
    m = _module(
        "Patches Denormal",
        body=[_code("pub fn flush_denormal(x: f32) -> f32 {\n    x\n}")],
        refs=["    use std::f32::consts::TAU;"],
    )
    out = assemble(m)
    assert "use std::f32::consts::TAU;" in out
    assert "// patches/denormal" in out
    assert "pub fn flush_denormal" in out
    # No module header — Rust files are implicitly modules.
    assert not out.startswith("mod ")


def test_assemble_with_deps_inlines_dependency_first():
    dep = _module(
        "Patches Denormal",
        body=[_code("pub fn flush_denormal(x: f32) -> f32 { x }")],
    )
    main = _module(
        "Patches Dc Blocker",
        body=[_code("pub fn process(x: f32) -> f32 { flush_denormal(x) }")],
        refs=["    #Patches Denormal"],
    )
    out = assemble_with_deps(main, [dep])
    # dependency appears before the importing module
    assert out.index("flush_denormal(x: f32)") < out.index("fn process")


def test_merge_modules_dedups_uses():
    a = _module("A", body=[_code("fn a() {}")],
                refs=["    use std::cmp::min;"])
    b = _module("B", body=[_code("fn b() {}")],
                refs=["    use std::cmp::min;"])
    uses, chunks = _merge_modules([a, b])
    assert uses == ["use std::cmp::min;"]
    assert len(chunks) == 2
