"""Checks the docstring sections ruff can't: it doesn't notice an Args or Raises section that is missing altogether."""

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SOURCE_FILES = sorted(path for folder in (ROOT, ROOT / "injection_suite", ROOT / "tests") for path in folder.glob("*.py"))
FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)


def own_nodes(function: ast.AST) -> list[ast.AST]:
    """Collects a function's nodes, leaving out those of nested functions and classes.

    Args:
        function: The function to walk.

    Returns:
        The nodes that run as part of this function.
    """
    nodes = []
    pending = list(ast.iter_child_nodes(function))
    while pending:
        node = pending.pop()
        nodes.append(node)
        if not isinstance(node, (*FUNCTIONS, ast.ClassDef, ast.Lambda)):
            pending.extend(ast.iter_child_nodes(node))
    return nodes


def section(docstring: str, title: str) -> str:
    """Finds one section of a Google style docstring.

    Args:
        docstring: The whole docstring.
        title: The section's title, such as "Args".

    Returns:
        The section's text; empty if it isn't there.
    """
    _, found, rest = f"\n{docstring}".partition(f"\n{title}:\n")
    return rest.split("\n\n")[0] if found else ""


def problems_in(function: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    """Lists what a function's docstring is missing.

    Args:
        function: The function to check.

    Returns:
        A description of each problem; empty if there are none.
    """
    docstring = ast.get_docstring(function)
    if not docstring:
        return ["no docstring"]

    signature = function.args
    named = [*signature.posonlyargs, *signature.args, *signature.kwonlyargs]
    starred = [arg for arg in (signature.vararg, signature.kwarg) if arg]
    parameters = [arg.arg for arg in [*named, *starred] if arg.arg not in ("self", "cls")]
    documented = section(docstring, "Args")
    problems = [f"Args doesn't list {name}" for name in parameters if f"{name}:" not in documented]

    nodes = own_nodes(function)
    returns_value = any(isinstance(node, ast.Return) and node.value is not None for node in nodes)
    if returns_value and not section(docstring, "Returns"):
        problems.append("no Returns section")

    raised = [node.exc for node in nodes if isinstance(node, ast.Raise) and node.exc is not None]
    named_errors = [error.func if isinstance(error, ast.Call) else error for error in raised]
    for error in named_errors:
        if isinstance(error, ast.Name) and f"{error.id}:" not in section(docstring, "Raises"):
            problems.append(f"Raises doesn't list {error.id}")
    return problems


def test_section_finds_only_the_named_section():
    """Checks that a section stops at the blank line before the next one."""
    docstring = "Summary.\n\nArgs:\n    a: First.\n\nReturns:\n    A value."
    assert section(docstring, "Args") == "    a: First."
    assert section(docstring, "Raises") == ""


def test_problems_are_found():
    """Checks that the check itself notices each kind of missing section."""
    source = 'def f(a):\n    """Summary."""\n    if a:\n        raise ValueError("bad")\n    return a\n'
    assert problems_in(ast.parse(source).body[0]) == ["Args doesn't list a", "no Returns section", "Raises doesn't list ValueError"]


def test_a_nested_function_does_not_count_for_its_parent():
    """Checks that a nested function's return doesn't require a Returns section on its parent."""
    source = 'def f():\n    """Summary."""\n    def g():\n        return 1\n'
    assert problems_in(ast.parse(source).body[0]) == []


@pytest.mark.parametrize("path", SOURCE_FILES, ids=lambda path: path.name)
def test_every_function_documents_its_interface(path):
    """Checks every function in the project, including nested functions and test helpers.

    Args:
        path: The source file to check.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    problems = [
        f"{path.name}:{node.lineno} {node.name}: {problem}"
        for node in ast.walk(tree) if isinstance(node, FUNCTIONS)
        for problem in problems_in(node)
    ]
    assert problems == []
