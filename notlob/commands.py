"""notlob.commands — Command implementations for the notlob CLI.

The public functions here implement the ``notlob run`` and
``notlob test`` commands.  They accept a resolved :class:`pathlib.Path`
and return an integer exit code.  All argument parsing and entry-point
wiring lives in :mod:`notlob.cli`, which delegates here.

Validation order in ``cmd_test``
---------------------------------
1. Parse the module.
2. Check that the module's title-derived address matches its file path.
3. Validate all prose cross-references.
4. Run examples, properties, and #Tests claims.

Steps 2 and 3 are document-integrity checks; a failure short-circuits
before any claim execution.
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

from notlob import (
    build, enrich, from_tree, parse_file, validate_refs,
    Edge, EdgeKind, NodeKind,
)
from notlob.bindings import ClaimResult, Status
from notlob.bindings.python import extract_symbols as _py_extract, kit as _py_kit
from notlob.bindings.python.loader import ModuleCache
from notlob.graph import module_address
from notlob.model import BindingSection, Claim, Subheading
from notlob.project import (
    address_from_path,
    build_package,
    find_project_root, module_lob_refs, resolve_module_path,
)


# ── Language dispatch ─────────────────────────────────────────

def _get_binding_kit(language: str | None):
    """Return ``(kit, extract_symbols)`` for the given language name.

    Defaults to the Python kit for ``None`` or unrecognised languages.
    Adding a new language binding requires only a new branch here.
    """
    if language == "haskell":
        from notlob.bindings.haskell import (   # lazy: avoids import if unused
            kit as _hs_kit,
            extract_symbols as _hs_extract,
        )
        return _hs_kit, _hs_extract
    if language == "rust":
        from notlob.bindings.rust import (       # lazy: avoids import if unused
            kit as _rs_kit,
            extract_symbols as _rs_extract,
        )
        return _rs_kit, _rs_extract
    # default — python
    return _py_kit, _py_extract


# ── Binding resolution ────────────────────────────────────────

def _parse_binding_declarations(lines: list[str]) -> dict[str, str]:
    """Extract ~sigil declarations from a #Binding section's lines."""
    result: dict[str, str] = {}
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("~"):
            parts = stripped[1:].split(None, 1)
            key = parts[0]
            value = parts[1].strip() if len(parts) > 1 else ""
            result[key] = value
    return result


def _find_binding(file_path: Path) -> dict[str, str]:
    """Walk up from *file_path* to find binding.lob; return its
    declarations.  Returns an empty dict if none is found.
    """
    root = find_project_root(file_path)
    if root is None:
        return {}
    try:
        bmod = from_tree(parse_file(root / "binding.lob"))
        if bmod.post_text:
            for section in bmod.post_text.sections:
                if isinstance(section, BindingSection):
                    return _parse_binding_declarations(section.lines)
    except Exception:
        pass
    return {}


# ── Address validation ────────────────────────────────────────

def _check_address(
    module,
    path: Path,
    root: Path | None,
) -> str | None:
    """Return an error string if title address ≠ path address, else None.

    A standalone file (no project root) is exempt: there is no folder
    structure to validate against.
    """
    if root is None:
        return None
    expected = address_from_path(path, root)
    actual   = module_address(module.title)
    if actual != expected:
        return (
            f"address mismatch: title gives {actual!r}, "
            f"file path gives {expected!r}"
        )
    return None


# ── Formatting ────────────────────────────────────────────────

def _print_result(r: ClaimResult) -> None:
    tag = r.status.name
    print(f"{tag:5}  {r.address}  {r.line}")
    if r.status == Status.FAIL:
        if r.left is not None or r.right is not None:
            print(f"         left:  {r.left!r}")
            print(f"         right: {r.right!r}")
        if r.error is not None:
            print(f"         error: {r.error}")
    elif r.status == Status.ERROR and r.error is not None:
        print(f"         error: {r.error}")


# ── Reference-graph builder ───────────────────────────────────

def _build_ref_graph(module, root, extract_symbols):
    """Build a NameGraph sufficient for validate_refs.

    Builds the structural graph for *module*, enriches it with
    symbols, then merges in a MODULE node for each directly imported
    module and adds the corresponding IMPORTS edge.  Only direct
    imports are added; transitive imports are not needed because
    validate_refs only resolves one hop via step 3.

    *extract_symbols* is the language-specific extractor from the
    binding kit; it is passed explicitly so the ref graph uses the
    same language as the active kit.

    Import errors (missing files) are silently skipped — they will
    surface as claim execution errors later.
    """
    graph    = build(module)
    enrich(graph, module, extract_symbols)
    mod_addr = module_address(module.title)

    if root is not None:
        for dep_addr in module_lob_refs(module):
            try:
                dep_path = resolve_module_path(dep_addr, root)
                dep_mod  = from_tree(parse_file(dep_path))
                graph.merge(build(dep_mod))
                graph.add_edge(Edge(
                    source=mod_addr,
                    target=dep_addr,
                    kind=EdgeKind.IMPORTS,
                ))
            except Exception:
                pass  # missing dep — caught later as a claim error

    return graph


# ── Commands ──────────────────────────────────────────────────

def _collect_run_claims(module) -> list[Claim]:
    """Return all ~run claims from the module body and subheadings,
    in document order.
    """
    claims = []
    for item in module.body:
        if isinstance(item, Claim) and item.sigil == "~run":
            claims.append(item)
        elif isinstance(item, Subheading):
            for sub_item in item.body:
                if (isinstance(sub_item, Claim)
                        and sub_item.sigil == "~run"):
                    claims.append(sub_item)
    return claims


def _resolve_keep_dir(
    cli_path: str | None,
    binding:  dict,
    root:     Path | None,
) -> Path | None:
    """Resolve the keep-generated-src directory from CLI flag and binding.

    The CLI flag takes precedence.  A relative path is resolved against
    the project root when one is available, otherwise against CWD.
    """
    raw = cli_path or binding.get("keep-generated-src")
    if not raw:
        return None
    p = Path(raw)
    if not p.is_absolute():
        p = (root / p) if root else (Path.cwd() / p)
    return p


def _cmd_run_haskell(
    module,
    path: Path,
    keep_dir: Path | None = None,
) -> int:
    """Assemble a Haskell module (with inlined deps) and run with runghc.

    The assembled source must define ``main :: IO ()``.  If no ``main``
    is present GHC will report a link error, which surfaces here as a
    non-zero exit code with the compiler message on stderr.

    If *keep_dir* is set the assembled source is also written there as
    ``<module-address-slugified>.hs`` before execution.
    """
    from notlob.bindings.haskell.assemble import assemble_with_deps
    from notlob.bindings.haskell.runner import (
        _load_dep_modules, _run_harness,
    )

    dep_modules = _load_dep_modules(module, path)
    source      = assemble_with_deps(module, dep_modules)
    if not source:
        print("ERROR  <assembly>  module contains no code", file=sys.stderr)
        return 1

    if keep_dir is not None:
        from notlob.graph import module_address as _mod_addr
        slug = _mod_addr(module.title).replace("/", "_")
        keep_path = keep_dir / f"{slug}.hs"
    else:
        keep_path = None

    stdout, stderr, rc = _run_harness(source, keep_path=keep_path)
    if stdout:
        print(stdout, end="")
    if rc != 0:
        if stderr:
            print(stderr, end="", file=sys.stderr)
        return 1
    return 0


def _cmd_run_rust(
    module,
    path: Path,
    keep_dir: Path | None = None,
) -> int:
    """Assemble a Rust module (with inlined deps) and run it via cargo.

    The assembled source must define ``fn main``.  A missing ``main``
    surfaces as a rustc error (non-zero exit) with the compiler message
    on stderr.

    If *keep_dir* is set the assembled source is also written there as
    ``<module-address-slugified>.rs`` before execution.
    """
    from notlob.bindings.rust.assemble import assemble_with_deps
    from notlob.bindings.rust.runner import (
        _ALLOW, _load_dep_modules, _run_harness,
    )

    dep_modules = _load_dep_modules(module, path)
    body        = assemble_with_deps(module, dep_modules)
    if not body:
        print("ERROR  <assembly>  module contains no code", file=sys.stderr)
        return 1
    source = f"{_ALLOW}\n\n{body}\n"

    if keep_dir is not None:
        from notlob.graph import module_address as _mod_addr
        slug = _mod_addr(module.title).replace("/", "_")
        keep_path = keep_dir / f"{slug}.rs"
    else:
        keep_path = None

    stdout, stderr, rc = _run_harness(source, keep_path=keep_path)
    if stdout:
        print(stdout, end="")
    if rc != 0:
        if stderr:
            print(stderr, end="", file=sys.stderr)
        return 1
    return 0


def cmd_run(path: Path, keep_generated_src: str | None = None) -> int:
    """Assemble and execute *path*; return an exit code."""
    try:
        module = from_tree(parse_file(path))
    except Exception as exc:
        print(f"ERROR  <parse>  {exc}", file=sys.stderr)
        return 1

    binding  = _find_binding(path)
    language = binding.get("language")
    root     = find_project_root(path)

    if language == "haskell":
        keep_dir = _resolve_keep_dir(keep_generated_src, binding, root)
        return _cmd_run_haskell(module, path, keep_dir=keep_dir)

    if language == "rust":
        keep_dir = _resolve_keep_dir(keep_generated_src, binding, root)
        return _cmd_run_rust(module, path, keep_dir=keep_dir)

    root = find_project_root(path)

    addr_err = _check_address(module, path, root)
    if addr_err:
        print(f"ERROR  <address>  {addr_err}", file=sys.stderr)
        return 1

    py_kit = _py_kit
    cache = ModuleCache(root) if root else None
    ns: dict = {"__file__": str(path.resolve())}
    try:
        if cache is not None:
            for dep_addr in module_lob_refs(module):
                ns.update(cache.load(dep_addr))
        exec(py_kit.assemble(module), ns)
    except Exception as exc:
        print(f"ERROR  <assembly>  {exc}", file=sys.stderr)
        return 1

    for claim in _collect_run_claims(module):
        try:
            exec(textwrap.dedent("\n".join(claim.lines)), ns)
        except Exception as exc:
            print(f"ERROR  <run>  {exc}", file=sys.stderr)
            return 1

    return 0


def cmd_test(path: Path, keep_generated_src: str | None = None) -> int:
    """Run all claims in *path* and return an exit code."""
    try:
        module = from_tree(parse_file(path))
    except Exception as exc:
        print(f"ERROR  <parse>  {exc}", file=sys.stderr)
        return 1

    binding  = _find_binding(path)
    root     = find_project_root(path)
    language = binding.get("language")
    kit, extract_symbols = _get_binding_kit(language)
    cache    = (
        ModuleCache(root)
        if (root and language not in ("haskell", "rust"))
        else None
    )
    keep_dir = _resolve_keep_dir(keep_generated_src, binding, root)

    doc_errors: list[str] = []

    addr_err = _check_address(module, path, root)
    if addr_err:
        doc_errors.append(f"ERROR  <address>  {addr_err}")

    for ref_err in validate_refs(
        _build_ref_graph(module, root, extract_symbols), module
    ):
        doc_errors.append(f"ERROR  <refs>  {ref_err}")

    if doc_errors:
        for msg in doc_errors:
            print(msg, file=sys.stderr)
        return 1

    results = (
        kit.run_examples(module, file_path=path, cache=cache,
                         keep_dir=keep_dir)
        + kit.run_properties(module, binding=binding, file_path=path,
                             cache=cache, keep_dir=keep_dir)
        + kit.run_tests(module, binding=binding, file_path=path,
                        cache=cache, keep_dir=keep_dir)
    )

    # SKIPs don't count toward the pass/fail tally
    non_skip = [r for r in results if r.status != Status.SKIP]
    for r in results:
        _print_result(r)

    n_fail = sum(1 for r in non_skip if r.status != Status.PASS)
    n_pass = len(non_skip) - n_fail
    if n_fail:
        print(f"\n{n_pass} passed, {n_fail} failed")
    else:
        print(f"\n{n_pass} passed")

    return 1 if n_fail else 0


# ── Graph export ──────────────────────────────────────────────

def cmd_graph(path: Path, include_content: bool = False) -> int:
    """Print the package name-graph as JSON to stdout.

    When *path* is inside a notlob project (a ``binding.lob`` is
    found), the full package graph is built and exported.  For a
    standalone file the single-module graph is used instead.

    The language binding (from the project's ``binding.lob``) controls
    which symbol extractor is used.

    Pass *include_content=True* to attach prose/code to every node.
    """
    root = find_project_root(path)
    binding = _find_binding(path) if root else {}
    language = binding.get("language")
    _, extract_symbols = _get_binding_kit(language)

    if root is not None:
        graph = build_package(root, extract_symbols)
    else:
        try:
            module = from_tree(parse_file(path))
        except Exception as exc:
            print(f"ERROR  <parse>  {exc}", file=sys.stderr)
            return 1
        graph = build(module)
        enrich(graph, module, extract_symbols)

    print(graph.to_json(include_content=include_content))
    return 0


# ── Query helpers ─────────────────────────────────────────────

def _node_dict(node, include_content: bool = False) -> dict:
    d: dict = {
        "address": node.address,
        "label":   node.label,
        "kind":    node.kind.name,
    }
    if include_content and node.content is not None:
        d["content"] = node.content
    return d


def _require_graph(hint: Path | None = None):
    """Build the package graph, or return None with an error printed."""
    root = find_project_root(hint or Path.cwd())
    if root is None:
        print(
            "ERROR  <project>  no binding.lob found — "
            "run from inside a notlob project",
            file=sys.stderr,
        )
        return None
    binding = _find_binding(root / "binding.lob")
    _, extract_symbols = _get_binding_kit(binding.get("language"))
    return build_package(root, extract_symbols)


# ── Query commands ────────────────────────────────────────────

def cmd_query_children(address: str, kind_str: str = "CONTAINS") -> int:
    """Print direct children of *address* as a JSON array."""
    graph = _require_graph()
    if graph is None:
        return 1
    kind    = EdgeKind[kind_str]
    results = list(graph.children(address, kind))
    print(json.dumps([_node_dict(n) for n in results], indent=2))
    return 0


def cmd_query_resolve(label: str, context: str | None = None) -> int:
    """Resolve *label* (with optional *context* module address).

    Exits 0 and prints the node JSON when the reference resolves;
    exits 1 and prints ``null`` when it does not.
    """
    graph = _require_graph()
    if graph is None:
        return 1
    node = graph.resolve(label, context)
    print(json.dumps(_node_dict(node) if node else None, indent=2))
    return 0 if node else 1


def cmd_query_search(
    pattern: str,
    kind_str: str | None = None,
) -> int:
    """Print nodes whose label matches *pattern* (fnmatch-style)."""
    graph = _require_graph()
    if graph is None:
        return 1
    kind    = NodeKind[kind_str] if kind_str else None
    results = list(graph.search(pattern, kind))
    print(json.dumps([_node_dict(n) for n in results], indent=2))
    return 0


def cmd_query_imports(address: str) -> int:
    """Print modules imported by *address* as a JSON array."""
    graph = _require_graph()
    if graph is None:
        return 1
    results = list(graph.children(address, EdgeKind.IMPORTS))
    print(json.dumps([_node_dict(n) for n in results], indent=2))
    return 0


def cmd_query_imported_by(address: str) -> int:
    """Print modules that import *address* as a JSON array."""
    graph = _require_graph()
    if graph is None:
        return 1
    results = list(graph.parents(address, EdgeKind.IMPORTS))
    print(json.dumps([_node_dict(n) for n in results], indent=2))
    return 0


def cmd_weave(path: Path, language: str | None = None) -> int:
    """Render *path* as Markdown and write to stdout.

    The language tag for fenced code blocks is resolved in order:
    1. The *language* argument if supplied (from ``--language`` flag).
    2. The ``~language`` declaration in the nearest ``binding.lob``.
    3. The default ``"python"``.
    """
    from notlob.weave import weave_markdown   # lazy — avoids circular dep
    try:
        module = from_tree(parse_file(path))
    except Exception as exc:
        print(f"ERROR  <parse>  {exc}", file=sys.stderr)
        return 1
    if language is None:
        binding  = _find_binding(path)
        language = binding.get("language", "python")
    print(weave_markdown(module, language), end="")
    return 0


def cmd_query_content(address: str) -> int:
    """Print the node at *address* with its source content as JSON.

    Returns exit code 0 when the address resolves, 1 (with ``null``)
    when it does not.  This is the primary F3-style lookup: given a
    known address, retrieve its prose and/or code.
    """
    graph = _require_graph()
    if graph is None:
        return 1
    node = graph.node(address)
    print(json.dumps(
        _node_dict(node, include_content=True) if node else None,
        indent=2,
    ))
    return 0 if node else 1
