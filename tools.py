"""Basic file and command tools for the Automaton agent."""

from pathlib import Path
import shlex
import subprocess
import sys

from langchain_core.tools import tool


SKIPPED_DIRS = {".git", ".pytest_cache", "__pycache__", ".venv", "venv", "node_modules"}
BLOCKED_COMMANDS = {"rm", "sudo", "curl", "wget", "ssh"}


def _resolve_workspace_path(working_dir: str, path: str = ".") -> Path:
    """Resolve a path and ensure it stays inside the working directory."""
    root = Path(working_dir).resolve()
    target = (root / path).resolve()

    if target != root and root not in target.parents:
        raise ValueError(f"Path escapes working directory: {path}")

    return target


def _is_allowed_command(parts: list[str]) -> bool:
    if not parts:
        return False

    if parts[0] in BLOCKED_COMMANDS:
        return False

    if len(parts) >= 2 and parts[0] == "git" and parts[1] == "push":
        return False

    if parts[0] == "pytest":
        return True

    if parts[0] in {"python", "python3"}:
        return len(parts) > 2 and parts[1:3] == ["-m", "pytest"]

    if parts == ["npm", "test"]:
        return True

    if parts == ["make", "test"]:
        return True

    return False


def _subprocess_parts(parts: list[str]) -> list[str]:
    if parts and parts[0] == "pytest":
        # Importing readline during pytest startup segfaults in this local macOS venv.
        pytest_runner = (
            "import sys, types; "
            "sys.modules['readline'] = types.ModuleType('readline'); "
            "import pytest; "
            "raise SystemExit(pytest.main(sys.argv[1:]))"
        )
        return [sys.executable, "-c", pytest_runner, *parts[1:]]

    return parts


@tool
def read_file(path: str, working_dir: str) -> str:
    """Read a file relative to the locked working directory."""
    try:
        target = _resolve_workspace_path(working_dir, path)
        return target.read_text(encoding="utf-8")
    except (OSError, ValueError) as error:
        return f"Error reading {path}: {error}"


@tool
def write_file(path: str, content: str, working_dir: str) -> str:
    """Write a file relative to the locked working directory."""
    try:
        target = _resolve_workspace_path(working_dir, path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"Successfully wrote to {path}"
    except (OSError, ValueError) as error:
        return f"Error writing to {path}: {error}"


@tool
def run_command(command: str, working_dir: str, timeout: int = 60) -> str:
    """Run an allowlisted command inside the locked working directory."""
    try:
        root = _resolve_workspace_path(working_dir)
        parts = shlex.split(command)

        if not _is_allowed_command(parts):
            return f"Command not allowed: {command}"

        result = subprocess.run(
            _subprocess_parts(parts),
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=root,
            check=False,
        )
        return (
            f"EXIT_CODE: {result.returncode}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        return f"Command failed: {error}"


@tool
def list_dir(path: str, working_dir: str, depth: int = 2) -> str:
    """List files under a directory relative to the locked working directory."""
    try:
        root = _resolve_workspace_path(working_dir)
        start = _resolve_workspace_path(working_dir, path)
        if not start.is_dir():
            return f"Error listing {path}: not a directory"

        lines: list[str] = []

        def visit(directory: Path, current_depth: int) -> None:
            if current_depth > depth:
                return

            for entry in sorted(directory.iterdir(), key=lambda item: (item.is_file(), item.name)):
                if entry.is_dir() and entry.name in SKIPPED_DIRS:
                    continue

                relative = entry.relative_to(root)
                suffix = "/" if entry.is_dir() else ""
                lines.append(f"{relative}{suffix}")

                if entry.is_dir():
                    visit(entry, current_depth + 1)

        visit(start, 0)
        return "\n".join(lines)
    except (OSError, ValueError) as error:
        return f"Error listing {path}: {error}"
