"""Tests for the Rust runner.

Two tiers:
  • Pure unit tests (no cargo): harness generation, output parsing,
    string escaping, main-hiding, property-name discovery.
  • Integration tests (cargo): exercise actual compilation and
    execution.  Skipped when cargo is unavailable.

Integration tests are slow on first run (proptest compile); subsequent
runs hit the warm target/ cache.
"""

from __future__ import annotations

import shutil

import pytest

from notlob.bindings import ClaimResult, Status
from notlob.bindings.rust.runner import (
    _rs_string_escape,
    _iter_assertions,
    _hide_user_main,
    _first_fn_name,
    _build_examples_harness,
    _build_property_harness,
    _parse_output,
    _parse_property_output,
    run_examples,
    run_tests,
    run_properties,
)
from notlob.model import (
    Claim, CodeBlock, Module, PostText, ReferencesSection,
    Subheading, TestsSection, TestGroup,
)


_HAS_CARGO = shutil.which("cargo") is not None
_CARGO_SKIP = pytest.mark.skipif(not _HAS_CARGO, reason="cargo not found")


# ── Builders ──────────────────────────────────────────────────

def _code(text):
    return CodeBlock(lines=["    " + ln for ln in text.splitlines()])


def _module(title, body=None, refs=None, tests=None):
    body = body or []
    sections = []
    if refs is not None:
        sections.append(ReferencesSection(lines=refs))
    if tests is not None:
        sections.append(tests)
    post = PostText(sections=sections) if sections else None
    return Module(title=title, body=body, post_text=post)


def _example(text):
    return Claim(sigil="~example", lines=["    " + ln for ln in text.splitlines()])


# ── Pure unit tests ───────────────────────────────────────────

def test_string_escape():
    assert _rs_string_escape('a"b\\c') == 'a\\"b\\\\c'
    assert _rs_string_escape("x\ty") == "x\\ty"


def test_iter_assertions_skips_blanks():
    assert list(_iter_assertions(["  a == 1", "", "  b == 2"])) == [
        "a == 1", "b == 2",
    ]


def test_hide_user_main():
    src = "fn main() {\n    println!(\"hi\");\n}"
    out = _hide_user_main(src)
    assert "fn _notlob_user_main()" in out
    assert "fn main()" not in out


def test_hide_user_main_leaves_other_fns():
    src = "fn mainline() {}"
    assert _hide_user_main(src) == "fn mainline() {}"


def test_first_fn_name_in_proptest_block():
    lines = [
        "    proptest! {",
        "        fn commutes(a in 0.0f32..1.0, b in 0.0f32..1.0) {",
        "            prop_assert_eq!(a + b, b + a);",
        "        }",
        "    }",
    ]
    assert _first_fn_name(lines) == "commutes"


def test_examples_harness_contains_check_and_main():
    m = _module("Demo", body=[_code("fn answer() -> i32 { 42 }")])
    harness = _build_examples_harness(
        m, [("demo#example#1", "answer() == 42")],
    )
    assert "fn _notlob_check" in harness
    assert "fn main()" in harness
    assert '_notlob_check("demo#example#1", "answer() == 42", (answer() == 42));' in harness
    assert harness.startswith("#![allow(")


def test_property_harness_imports_proptest_and_calls_prop():
    m = _module("Demo")
    lines = [
        "    proptest! {",
        "        fn p(x in 0i32..10) { prop_assert!(x < 10); }",
        "    }",
    ]
    harness = _build_property_harness(m, lines, "demo#property#1", "p")
    assert "use proptest::prelude::*;" in harness
    assert "catch_unwind" in harness
    assert "p();" in harness


def test_parse_output_pass_fail():
    out = "CLAIM\taddr1\te1\nPASS\nCLAIM\taddr2\te2\nFAIL\n"
    results = _parse_output(out, "", 0, [("addr1", "e1"), ("addr2", "e2")])
    assert [r.status for r in results] == [Status.PASS, Status.FAIL]
    assert results[0].address == "addr1"


def test_parse_output_compile_error():
    results = _parse_output("", "error[E0425]: cannot find value", 1,
                            [("addr1", "e1")])
    assert len(results) == 1
    assert results[0].status == Status.ERROR
    assert results[0].line == "<compile>"


def test_parse_property_output():
    pass_r = _parse_property_output("CLAIM\ta\t~property\nPASS\n", "", 0,
                                    "a", "~property")
    assert pass_r.status == Status.PASS
    fail_r = _parse_property_output("CLAIM\ta\t~property\nFAIL\n",
                                    "thread panicked", 0, "a", "~property")
    assert fail_r.status == Status.FAIL


def test_run_properties_skips_without_proptest_declaration():
    m = _module("Demo", body=[
        Claim(sigil="~property", lines=[
            "    proptest! { fn p(x in 0i32..1) { prop_assert!(true); } }",
        ]),
    ])
    results = run_properties(m, binding={})
    assert len(results) == 1
    assert results[0].status == Status.SKIP


# ── Integration tests (cargo) ─────────────────────────────────

@_CARGO_SKIP
def test_run_examples_integration():
    m = _module("Demo", body=[
        _code("fn answer() -> i32 { 42 }"),
        _example("answer() == 42\nanswer() != 0"),
    ])
    results = run_examples(m)
    assert len(results) == 2
    assert all(r.status == Status.PASS for r in results), [
        (r.status, r.error) for r in results
    ]


@_CARGO_SKIP
def test_run_examples_detects_failure():
    m = _module("Demo", body=[
        _code("fn answer() -> i32 { 42 }"),
        _example("answer() == 0"),
    ])
    results = run_examples(m)
    assert results[0].status == Status.FAIL


@_CARGO_SKIP
def test_run_tests_integration():
    tests = TestsSection(items=["1 + 1 == 2", "2 * 2 == 4"])
    m = _module("Demo", body=[_code("fn noop() {}")], tests=tests)
    results = run_tests(m, binding={})
    assert [r.status for r in results] == [Status.PASS, Status.PASS]
