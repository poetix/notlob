"""notlob.bindings.rust — Rust language binding kit.

Assembles the Rust ``BindingKit`` from its submodules and exposes
``kit`` as the canonical Rust binding instance.

Usage::

    from notlob.bindings.rust import kit
    results = kit.run_examples(module, file_path=path)

Runner availability
-------------------
The runner requires ``cargo`` on PATH.  When it is absent,
``run_examples``, ``run_tests``, and ``run_properties`` return a single
ERROR result rather than raising.
"""

from notlob.bindings import BindingKit
from notlob.bindings.rust.assemble import assemble
from notlob.bindings.rust.runner import (
    run_examples, run_tests, run_properties,
)
from notlob.bindings.rust.symbols import extract_symbols

#: The assembled Rust binding kit.
kit = BindingKit(
    extract_symbols=extract_symbols,
    assemble=assemble,
    run_examples=run_examples,
    run_properties=run_properties,
    run_tests=run_tests,
)

__all__ = [
    "kit", "extract_symbols", "assemble",
    "run_examples", "run_tests", "run_properties",
]
