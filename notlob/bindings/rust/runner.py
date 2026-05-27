"""notlob.bindings.rust.runner — claim runner for the Rust binding.

Builds a single-binary Cargo harness from the module's code blocks,
compiles and runs it with ``cargo run``, and parses the output into
``ClaimResult`` objects.

Why Cargo rather than a script runner
-------------------------------------
Rust has no interpreter (no ``runghc`` equivalent); every claim run is a
compile.  ``rust-script`` would hide that compile but still perform it,
and it is not always installed.  We own the build directly: a scratch
Cargo project, reused across runs and keyed by its dependency set so the
``target/`` cache stays warm and repeat runs are incremental.  This is
also the natural target for a future ``notlob build``.

Runner discovery
----------------
The runner requires ``cargo`` on PATH.  When it is absent, every claim
function returns a single ERROR result rather than raising.

Output protocol
---------------
Each assertion produces two consecutive stdout lines::

    CLAIM\\t<address>\\t<expression>
    PASS            (or FAIL)

A compile error (non-zero exit, no CLAIM lines) is reported as a single
ERROR result with line ``"<compile>"``; the compiler message is carried
in the result's ``error``.

Property testing
----------------
``run_properties`` requires ``~property-testing proptest`` in
``binding.lob``.  Without it every ~property claim receives
``Status.SKIP``.  With it, each ~property block (a ``proptest! { fn … }``
invocation) is compiled in its own harness with the ``proptest`` crate,
the inner property function is called, and a panic (proptest's failure
signal) is mapped to FAIL.

Limitations (v1)
----------------
* Each non-blank ~example / #Tests line is one boolean assertion;
  multi-line expressions are not split.  Rust block expressions
  (``{ …; cond }``) let a stateful assertion fit on one logical line.
* Only ``use`` statements in ``#References`` are forwarded; lob-ref
  dependencies are inlined by flattening (see assemble.py).
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import tempfile
import textwrap
from pathlib import Path
from typing import Any

from notlob.bindings import ClaimResult, Status
from notlob.graph import (
    claim_address, module_address, property_address, subheading_address,
)
from notlob.model import Claim, Module, Subheading, TestsSection
from notlob.bindings.rust.assemble import _merge_modules


# ── Runner discovery ──────────────────────────────────────────

def _has_cargo() -> bool:
    """Return True if a ``cargo`` executable is on PATH."""
    return shutil.which("cargo") is not None


# ── Scratch project management ─────────────────────────────────

_ALLOW = "#![allow(dead_code, unused_variables, unused_imports, unused_mut, unused_parens)]"


def _project_dir(deps: dict[str, str]) -> Path:
    """Return a stable scratch-project directory for the given deps.

    Keyed by the dependency set so projects with the same deps share a
    warm ``target/`` cache; differing deps get separate directories.
    """
    key = hashlib.sha1(
        repr(sorted(deps.items())).encode("utf-8")
    ).hexdigest()[:12]
    return Path(tempfile.gettempdir()) / f"notlob-rust-{key}"


def _write_manifest(project: Path, deps: dict[str, str]) -> None:
    """Write Cargo.toml for the harness binary into *project*."""
    dep_lines = "\n".join(f'{name} = "{ver}"' for name, ver in deps.items())
    manifest = (
        "[package]\n"
        'name = "notlob_harness"\n'
        'version = "0.0.0"\n'
        'edition = "2021"\n'
        "\n"
        "[[bin]]\n"
        'name = "harness"\n'
        'path = "src/main.rs"\n'
        "\n"
        "[dependencies]\n"
        f"{dep_lines}\n"
        "\n"
        # Standalone workspace root so an ancestor Cargo.toml never
        # captures this scratch project.
        "[workspace]\n"
    )
    (project / "Cargo.toml").write_text(manifest, encoding="utf-8")


def _run_harness(
    source: str,
    deps: dict[str, str] | None = None,
    timeout: int = 300,
    keep_path: Path | None = None,
) -> tuple[str, str, int]:
    """Compile and run *source* as a Cargo binary; return (out, err, rc).

    The scratch project is reused across calls with the same *deps* so
    compilation is incremental.  On timeout the stderr is a message and
    returncode is 1.  If ``cargo`` is unavailable, returns an error triple
    immediately.

    If *keep_path* is provided the source is also written there for
    inspection; write failures are ignored.
    """
    deps = deps or {}

    if keep_path is not None:
        try:
            keep_path.parent.mkdir(parents=True, exist_ok=True)
            keep_path.write_text(source, encoding="utf-8")
        except OSError:
            pass

    if not _has_cargo():
        return "", "no Rust runner found (install cargo)", 1

    project = _project_dir(deps)
    src_dir = project / "src"
    src_dir.mkdir(parents=True, exist_ok=True)
    _write_manifest(project, deps)
    (src_dir / "main.rs").write_text(source, encoding="utf-8")

    try:
        proc = subprocess.run(
            ["cargo", "run", "--quiet", "--bin", "harness"],
            cwd=str(project),
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**_cargo_env()},
        )
        return proc.stdout, proc.stderr, proc.returncode
    except subprocess.TimeoutExpired:
        return "", f"timeout after {timeout}s", 1
    except FileNotFoundError as exc:
        return "", str(exc), 1


def _cargo_env() -> dict[str, str]:
    """Environment for cargo: inherit, but force plain (no colour) output."""
    env = dict(os.environ)
    env["CARGO_TERM_COLOR"] = "never"
    return env


# ── Assertion helpers ─────────────────────────────────────────

def _iter_assertions(lines: list[str]):
    """Yield stripped non-blank lines as individual assertion strings."""
    for raw in lines:
        stripped = raw.strip()
        if stripped:
            yield stripped


def _rs_string_escape(s: str) -> str:
    """Escape *s* for a Rust double-quoted string literal."""
    return (
        s.replace("\\", "\\\\")
         .replace('"',  '\\"')
         .replace('\n', '\\n')
         .replace('\r', '\\r')
         .replace('\t', '\\t')
    )


_MAIN_RE = re.compile(r"(?<![A-Za-z0-9_])fn\s+main\b")


def _hide_user_main(source: str) -> str:
    """Rename any ``fn main`` to ``fn _notlob_user_main``.

    Prevents a duplicate-``main`` error when an inlined module defines
    its own entry point and the harness also defines ``fn main``.
    """
    return _MAIN_RE.sub("fn _notlob_user_main", source)


_FN_NAME_RE = re.compile(r"\bfn\s+([A-Za-z_][A-Za-z0-9_]*)")


def _first_fn_name(lines: list[str]) -> str | None:
    """Return the first ``fn <name>`` found in a code block, or None.

    Used to locate the property function inside a ``proptest! { fn … }``
    block, where the ``fn`` is nested in the macro invocation and so is
    not a column-0 item the symbol extractor would report.
    """
    for raw in lines:
        m = _FN_NAME_RE.search(raw)
        if m:
            return m.group(1)
    return None


# ── Dependency loading ────────────────────────────────────────

def _load_dep_modules(module: Module, file_path: Path | None) -> list[Module]:
    """Return the lob-ref dependency modules of *module*, in order.

    Resolves each ``#Title`` lob-ref in *module*'s ``#References`` to its
    ``.lob`` file under the project root, parses it, and returns the
    Module objects.  Dependencies that cannot be found or parsed are
    skipped — the missing symbol surfaces as a rustc compile error.

    Returns an empty list when *file_path* is None or no project root is
    found.
    """
    if file_path is None:
        return []

    from notlob.project import (           # noqa: PLC0415
        find_project_root, module_lob_refs, resolve_module_path,
    )
    from notlob.parser import parse_file   # noqa: PLC0415
    from notlob.model import from_tree     # noqa: PLC0415

    root = find_project_root(file_path)
    if root is None:
        return []

    result: list[Module] = []
    for dep_addr in module_lob_refs(module):
        try:
            dep_path = resolve_module_path(dep_addr, root)
            result.append(from_tree(parse_file(dep_path)))
        except Exception:
            pass
    return result


# ── Harness builders ──────────────────────────────────────────

_CHECK_HELPER = """\
fn _notlob_check(addr: &str, expr: &str, result: bool) {
    println!("CLAIM\\t{}\\t{}", addr, expr);
    println!("{}", if result { "PASS" } else { "FAIL" });
}\
"""


def _build_examples_harness(
    module: Module,
    assertions: list[tuple[str, str]],
    dep_modules: list[Module] | None = None,
) -> str:
    """Build a Cargo harness for a list of boolean assertions.

    *assertions* is an ordered list of ``(address, expression)`` pairs.
    ``main`` calls ``_notlob_check`` for each, printing CLAIM / PASS /
    FAIL.  *dep_modules* are inlined before *module*'s own code.
    """
    use_lines, code_chunks = _merge_modules((dep_modules or []) + [module])

    parts: list[str] = [_ALLOW]
    if use_lines:
        parts.append("\n".join(use_lines))
    if code_chunks:
        parts.append("\n\n".join(_hide_user_main(c) for c in code_chunks))
    parts.append(_CHECK_HELPER)

    main_lines = ["fn main() {"]
    for addr, expr in assertions:
        ea = _rs_string_escape(addr)
        ee = _rs_string_escape(expr)
        main_lines.append(f'    _notlob_check("{ea}", "{ee}", ({expr}));')
    main_lines.append("}")
    parts.append("\n".join(main_lines))

    return "\n\n".join(parts) + "\n"


def _build_property_harness(
    module: Module,
    prop_lines: list[str],
    addr: str,
    prop_name: str,
    dep_modules: list[Module] | None = None,
) -> str:
    """Build a Cargo harness for a single proptest property.

    Inlines *dep_modules* and *module*'s own code, appends the
    ``proptest! { fn <prop_name> … }`` block, then calls
    ``<prop_name>()`` inside ``catch_unwind`` and prints PASS / FAIL.
    proptest signals failure by panicking, which the catch converts to
    FAIL.
    """
    use_lines, code_chunks = _merge_modules((dep_modules or []) + [module])

    parts: list[str] = [_ALLOW, "use proptest::prelude::*;"]
    if use_lines:
        parts.append("\n".join(use_lines))
    if code_chunks:
        parts.append("\n\n".join(_hide_user_main(c) for c in code_chunks))

    prop_code = "\n".join(prop_lines)
    parts.append(textwrap.dedent(prop_code).strip())

    ea = _rs_string_escape(addr)
    main_src = "\n".join([
        "fn main() {",
        f'    println!("CLAIM\\t{ea}\\t{_rs_string_escape(prop_name)}");',
        "    let _hook = std::panic::take_hook();",
        "    std::panic::set_hook(Box::new(|_| {}));",
        f"    let r = std::panic::catch_unwind(|| {{ {prop_name}(); }});",
        "    std::panic::set_hook(_hook);",
        "    match r {",
        '        Ok(_) => println!("PASS"),',
        '        Err(_) => println!("FAIL"),',
        "    }",
        "}",
    ])
    parts.append(main_src)

    return "\n\n".join(parts) + "\n"


# ── Output parsers ────────────────────────────────────────────

def _parse_output(
    stdout: str,
    stderr: str,
    returncode: int,
    assertions: list[tuple[str, str]],
) -> list[ClaimResult]:
    """Parse harness stdout into ClaimResult objects (CLAIM/PASS/FAIL)."""
    results: list[ClaimResult] = []
    lines = stdout.splitlines()
    i = 0

    while i < len(lines):
        raw = lines[i]
        if raw.startswith("CLAIM\t"):
            parts = raw.split("\t", 2)
            addr = parts[1] if len(parts) > 1 else "?"
            expr = parts[2] if len(parts) > 2 else "<assertion>"
            i += 1
            if i < len(lines):
                result_line = lines[i]
                if result_line == "PASS":
                    results.append(ClaimResult(
                        address=addr, line=expr, status=Status.PASS,
                    ))
                elif result_line == "FAIL":
                    results.append(ClaimResult(
                        address=addr, line=expr, status=Status.FAIL,
                    ))
                else:
                    results.append(ClaimResult(
                        address=addr, line=expr, status=Status.ERROR,
                        error=RuntimeError(
                            f"unexpected runner output: {result_line!r}"
                        ),
                    ))
            else:
                err_msg = stderr.strip() or "runtime error (no result)"
                results.append(ClaimResult(
                    address=addr, line=expr, status=Status.ERROR,
                    error=RuntimeError(err_msg),
                ))
        i += 1

    if not results and assertions:
        err_msg = stderr.strip() or "compile error"
        addr, _ = assertions[0]
        results.append(ClaimResult(
            address=addr, line="<compile>", status=Status.ERROR,
            error=RuntimeError(err_msg),
        ))

    return results


def _parse_property_output(
    stdout: str,
    stderr: str,
    returncode: int,
    addr: str,
    sigil: str,
) -> ClaimResult:
    """Parse a single-property harness output into one ClaimResult."""
    lines = stdout.splitlines()
    result_line = None
    for i, raw in enumerate(lines):
        if raw.startswith("CLAIM\t") and i + 1 < len(lines):
            result_line = lines[i + 1]
            break

    if result_line is None:
        err = stderr.strip() or "compile error"
        return ClaimResult(
            address=addr, line="<compile>",
            status=Status.ERROR, error=RuntimeError(err),
        )

    if result_line == "PASS":
        return ClaimResult(address=addr, line=sigil, status=Status.PASS)

    return ClaimResult(
        address=addr, line=sigil, status=Status.FAIL,
        error=RuntimeError(stderr.strip() or "property failed"),
    )


# ── Claim collectors ──────────────────────────────────────────

def _collect_example_assertions(
    body: list,
    containing_addr: str,
    assertions: list[tuple[str, str]],
) -> None:
    """Append (address, expression) pairs from ~example claims in body."""
    example_n = 0
    for item in body:
        if not (isinstance(item, Claim) and item.sigil == "~example"):
            continue
        example_n += 1
        addr = claim_address(containing_addr, "example", example_n)
        for expr in _iter_assertions(item.lines):
            assertions.append((addr, expr))


def _collect_tests_assertions(
    tests_section: TestsSection,
    tests_addr: str,
    assertions: list[tuple[str, str]],
) -> None:
    """Append (address, expression) pairs from a TestsSection."""
    bare: list[str] = []
    for item in tests_section.items:
        if isinstance(item, str):
            bare.append(item)
        else:
            for expr in _iter_assertions(bare):
                assertions.append((tests_addr, expr))
            bare = []
            group_addr = f"{tests_addr}#{item.title}"
            for expr in _iter_assertions(item.lines):
                assertions.append((group_addr, expr))
    for expr in _iter_assertions(bare):
        assertions.append((tests_addr, expr))


# ── Public claim runners ──────────────────────────────────────

def run_examples(
    module: Module,
    file_path: Path | None = None,
    cache: Any = None,
    keep_dir: Path | None = None,
) -> list[ClaimResult]:
    """Run all ~example claims in *module*; return one result per line."""
    mod_addr = module_address(module.title)
    assertions: list[tuple[str, str]] = []

    _collect_example_assertions(module.body, mod_addr, assertions)
    for item in module.body:
        if isinstance(item, Subheading):
            sub_addr = subheading_address(mod_addr, item.title)
            _collect_example_assertions(item.body, sub_addr, assertions)

    if not assertions:
        return []

    dep_modules = _load_dep_modules(module, file_path)
    harness     = _build_examples_harness(module, assertions, dep_modules)
    keep_path   = (keep_dir / "_examples.rs") if keep_dir else None
    stdout, stderr, rc = _run_harness(harness, keep_path=keep_path)
    return _parse_output(stdout, stderr, rc, assertions)


def run_tests(
    module: Module,
    binding: dict | None = None,
    file_path: Path | None = None,
    cache: Any = None,
    keep_dir: Path | None = None,
) -> list[ClaimResult]:
    """Run all #Tests assertions in *module*; return one result per line."""
    if module.post_text is None:
        return []

    tests_section = next(
        (s for s in module.post_text.sections
         if isinstance(s, TestsSection)),
        None,
    )
    if tests_section is None:
        return []

    mod_addr   = module_address(module.title)
    tests_addr = f"{mod_addr}#Tests"
    assertions: list[tuple[str, str]] = []
    _collect_tests_assertions(tests_section, tests_addr, assertions)

    if not assertions:
        return []

    dep_modules = _load_dep_modules(module, file_path)
    harness     = _build_examples_harness(module, assertions, dep_modules)
    keep_path   = (keep_dir / "_tests.rs") if keep_dir else None
    stdout, stderr, rc = _run_harness(harness, keep_path=keep_path)
    return _parse_output(stdout, stderr, rc, assertions)


def run_properties(
    module: Module,
    binding: dict | None = None,
    file_path: Path | None = None,
    cache: Any = None,
    keep_dir: Path | None = None,
) -> list[ClaimResult]:
    """Run all ~property claims in *module* with proptest; return results.

    Requires ``~property-testing proptest`` in ``binding.lob``; without
    it every ~property yields SKIP.  Each property block is compiled in
    its own harness with the proptest crate.
    """
    use_pt = (
        binding is not None
        and binding.get("property-testing") == "proptest"
    )

    mod_addr    = module_address(module.title)
    dep_modules = _load_dep_modules(module, file_path)
    results: list[ClaimResult] = []
    prop_n = 0

    def _run_prop_in(body: list, containing_addr: str) -> None:
        nonlocal prop_n
        for item in body:
            if not (isinstance(item, Claim)
                    and item.sigil.startswith("~property")):
                continue
            prop_n += 1

            parts = item.sigil.split(None, 1)
            if len(parts) > 1:
                addr = property_address(containing_addr, parts[1].strip())
            else:
                addr = claim_address(containing_addr, "property", prop_n)

            if not use_pt:
                results.append(ClaimResult(
                    address=addr, line=item.sigil, status=Status.SKIP,
                ))
                continue

            prop_name = _first_fn_name(item.lines)
            if prop_name is None:
                results.append(ClaimResult(
                    address=addr, line=item.sigil, status=Status.ERROR,
                    error=ValueError(
                        "no `fn` found in ~property block; wrap the "
                        "property in proptest! { fn name(...) { ... } }"
                    ),
                ))
                continue

            harness = _build_property_harness(
                module, item.lines, addr, prop_name, dep_modules,
            )
            safe = re.sub(r"[^A-Za-z0-9_]", "_", prop_name)
            keep_path = (keep_dir / f"_prop_{safe}.rs") if keep_dir else None
            stdout, stderr, rc = _run_harness(
                harness, deps={"proptest": "1"}, keep_path=keep_path,
            )
            results.append(
                _parse_property_output(stdout, stderr, rc, addr, item.sigil)
            )

    _run_prop_in(module.body, mod_addr)
    for item in module.body:
        if isinstance(item, Subheading):
            sub_addr = subheading_address(mod_addr, item.title)
            _run_prop_in(item.body, sub_addr)

    return results
