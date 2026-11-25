"""House style for functions in src/ and scripts/: a docstring with Inputs, Outputs and Why, and type hints.

The check reads the source with ast, so it needs no imports and covers every function, method and nested
function. A failure lists each function with what it lacks.
"""
import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FILES = sorted([*(ROOT / "src").glob("*.py"), *(ROOT / "scripts").glob("*.py")])
SECTIONS = {"Inputs": r"\bInputs?\b", "Outputs": r"\bOutputs?\b", "Why": r"\bWhy\b"}


def _functions(tree: ast.AST) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    """Every function, method and nested function in a parsed file."""
    return [node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]


def _missing_hints(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    """The arguments and the return value that have no type hint (self and cls need none)."""
    arguments = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
    arguments += [argument for argument in (node.args.vararg, node.args.kwarg) if argument is not None]
    missing = [argument.arg for argument in arguments
               if argument.annotation is None and argument.arg not in ("self", "cls")]
    if node.returns is None:
        missing.append("return")
    return missing


def _label(path: Path) -> str:
    """The file as shown in a failure: relative to the repository when it is inside it."""
    return path.relative_to(ROOT).as_posix() if ROOT in path.parents else path.name


def _problems(path: Path) -> list[str]:
    """One line per function in the file that breaks the house style."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = []
    for node in _functions(tree):
        where = f"{_label(path)}:{node.lineno} {node.name}"
        doc = ast.get_docstring(node)
        if not doc:
            found.append(f"{where}: no docstring")
        else:
            absent = [name for name, pattern in SECTIONS.items() if not re.search(pattern, doc)]
            if absent:
                found.append(f"{where}: docstring lacks {', '.join(absent)}")
        hints = _missing_hints(node)
        if hints:
            found.append(f"{where}: no type hint for {', '.join(hints)}")
    return found


@pytest.mark.parametrize("path", FILES, ids=_label)
def test_unit_every_function_has_a_full_docstring_and_type_hints(path: Path) -> None:
    """Each function has a docstring with Inputs, Outputs and Why, and hints on every argument and the return."""
    assert _problems(path) == []


def test_unit_the_check_finds_a_function_that_breaks_the_style(tmp_path: Path) -> None:
    """A function with no docstring and no hints is reported, so the check cannot pass by doing nothing."""
    bad = tmp_path / "bad.py"
    bad.write_text("def f(x):\n    return x\n", encoding="utf-8")
    problems = _problems(bad)
    assert any("no docstring" in line for line in problems)
    assert any("no type hint for x, return" in line for line in problems)
