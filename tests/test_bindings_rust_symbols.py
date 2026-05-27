"""Tests for the Rust symbol extractor."""

from __future__ import annotations

from notlob.bindings.rust.symbols import extract_symbols, _top_level_name


def _names(text):
    lines = ["    " + ln for ln in text.splitlines()]
    return [s.name for s in extract_symbols(lines)]


def test_empty():
    assert extract_symbols([]) == []


def test_function():
    assert _names("fn add(a: i32, b: i32) -> i32 {\n    a + b\n}") == ["add"]


def test_pub_function():
    assert _names("pub fn f(x: f32) -> f32 {\n    x\n}") == ["f"]


def test_modifier_stack():
    assert _names("pub(crate) const unsafe fn g() {}") == ["g"]


def test_struct_enum_trait():
    assert _names("pub struct Foo {\n    x: f32,\n}") == ["Foo"]
    assert _names("enum E {\n    A,\n    B,\n}") == ["E"]
    assert _names("pub trait T {\n    fn m(&self);\n}") == ["T"]


def test_const_and_static():
    assert _names("const N: usize = 8;") == ["N"]
    assert _names("pub static mut S: u32 = 0;") == ["S"]


def test_type_alias():
    assert _names("type Sample = f32;") == ["Sample"]


def test_impl_block_not_extracted():
    # impl has no single name; its methods are an implementation detail.
    text = (
        "pub struct DcBlocker {\n"
        "    r: f32,\n"
        "}\n"
        "\n"
        "impl DcBlocker {\n"
        "    pub fn new() -> Self { Self { r: 0.0 } }\n"
        "    pub fn process(&mut self, x: f32) -> f32 { x }\n"
        "}\n"
    )
    assert _names(text) == ["DcBlocker"]


def test_use_and_attributes_ignored():
    text = (
        "use std::f32::consts::TAU;\n"
        "#[inline]\n"
        "pub fn k() -> f32 {\n"
        "    TAU\n"
        "}\n"
    )
    assert _names(text) == ["k"]


def test_multiple_items():
    text = (
        "fn a() {}\n"
        "fn b() {}\n"
        "struct C;\n"
    )
    assert _names(text) == ["a", "b", "C"]


def test_top_level_name_classifiers():
    assert _top_level_name("fn foo() {}") == "foo"
    assert _top_level_name("    indented() {}") is None  # stripped, but no kw
    assert _top_level_name("impl Foo {") is None
    assert _top_level_name("use x;") is None
    assert _top_level_name("}") is None
