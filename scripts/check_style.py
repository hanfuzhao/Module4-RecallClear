"""Enforce the project's two structural code rules."""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


ALLOWED_TOP_LEVEL = (
    ast.Import,
    ast.ImportFrom,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.Assign,
    ast.AnnAssign,
    ast.Expr,
    ast.If,
    ast.Try,
)

DEFAULT_PATHS = ("scripts", "tests", "main.py", "setup.py")


def _is_docstring(node: ast.stmt) -> bool:
    """Whether a top-level expression node is a bare string, i.e. a docstring."""
    return isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(
        node.value.value, str
    )


def _is_main_guard(node: ast.stmt) -> bool:
    """Whether an ``if`` statement is the ``__name__ == "__main__"`` guard."""
    if not isinstance(node, ast.If):
        return False
    test = node.test
    return (
        isinstance(test, ast.Compare)
        and isinstance(test.left, ast.Name)
        and test.left.id == "__name__"
    )


def check_module(path: Path) -> list[str]:
    """Return a list of rule violations for one Python file."""
    problems: list[str] = []
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as error:
        return [f"{path}:{error.lineno}: syntax error: {error.msg}"]

    relative = path.relative_to(PROJECT_ROOT)

    if not ast.get_docstring(tree):
        problems.append(f"{relative}:1: module is missing a docstring")

    for node in tree.body:
        if isinstance(node, ast.Expr) and not _is_docstring(node):
            problems.append(f"{relative}:{node.lineno}: loose expression at module level")
        elif not isinstance(node, ALLOWED_TOP_LEVEL):
            problems.append(
                f"{relative}:{node.lineno}: {type(node).__name__} at module level; "
                "move it into a function or under __main__"
            )
        elif isinstance(node, ast.If) and not _is_main_guard(node):

            for inner in ast.walk(node):
                if isinstance(inner, ast.Call) and not _contains_import(node):
                    problems.append(
                        f"{relative}:{node.lineno}: conditional call at module level"
                    )
                    break

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name.startswith("_"):
                continue
            if node.name.startswith("test_") or node.name in {"setUp", "tearDown"}:
                continue
            if not ast.get_docstring(node):
                problems.append(f"{relative}:{node.lineno}: '{node.name}' is missing a docstring")

    return problems


def _contains_import(node: ast.stmt) -> bool:
    """Whether a node subtree contains an import statement."""
    return any(isinstance(inner, (ast.Import, ast.ImportFrom)) for inner in ast.walk(node))


def iter_python_files(paths: list[str]) -> list[Path]:
    """Expand the requested paths into a sorted list of Python files."""
    files: list[Path] = []
    for entry in paths:
        target = PROJECT_ROOT / entry
        if target.is_dir():
            files.extend(sorted(target.rglob("*.py")))
        elif target.suffix == ".py" and target.exists():
            files.append(target)
    return files


def main() -> int:
    """Check every requested file and report violations."""
    parser = argparse.ArgumentParser(description="Check project code-structure rules.")
    parser.add_argument("--paths", nargs="+", default=list(DEFAULT_PATHS))
    arguments = parser.parse_args()

    files = iter_python_files(arguments.paths)
    problems = [problem for path in files for problem in check_module(path)]

    if problems:
        print(f"Found {len(problems)} issue(s) in {len(files)} file(s):\n")
        for problem in problems:
            print(f"  {problem}")
        return 1

    print(f"{len(files)} files checked, no issues.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
