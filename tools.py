"""Basic file and command tools for the Automaton agent."""

import shlex
import subprocess

from langchain_core.tools import tool


@tool
def read_file(path: str) -> str:
    """Read the content of a file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except OSError as error:
        return f"Error reading {path}: {error}"


@tool
def write_file(path: str, content: str) -> str:
    """Write content to a file."""
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully wrote to {path}"
    except OSError as error:
        return f"Error writing to {path}: {error}"


@tool
def run_command(command: str) -> str:
    """Run a shell command safely."""
    try:
        result = subprocess.run(
            shlex.split(command),
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        return f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        return f"Command failed: {error}"

@tool
def list_dir(path: str, depth: int = 2) -> str:
    """List the contents of a directory."""
    try:
        with os.scandir(path) as it:
            return "\n".join(f"{entry.name} ({'dir' if entry.is_dir() else 'file'})" for entry in it)
    except OSError as error:
        return f"Error listing {path}: {error}"
    