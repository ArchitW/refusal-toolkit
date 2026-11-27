# Created: 27-11-26
# agent: kilo/baidu/ernie-5.5-flash-preview:free
"""Subprocess runner for TUI commands.

Builds a CLI command line from a CommandSpec + form values and streams
stdout/stderr line-by-line. Runs each command in its own session so it can be
terminated as a group on cancel.
"""
from __future__ import annotations

import asyncio
import os
import signal
import sys
from pathlib import Path

from .specs import CommandSpec

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CLI = REPO_ROOT / "refusal_toolkit.py"


def build_argv(spec: CommandSpec, values: dict[str, object]) -> list[str]:
    """Build the argv list (after the script name) from form values."""
    argv: list[str] = [spec.command]
    for f in spec.fields:
        argv.extend(f.to_arg(values.get(f.name, f.default)))
    return argv


class CommandRunner:
    """Runs a single command as a subprocess, streaming output lines."""

    def __init__(self) -> None:
        self._proc: asyncio.subprocess.Process | None = None

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.returncode is None

    async def run(self, argv: list[str], on_line) -> int:
        """Execute argv, calling on_line(str) for each output line.

        Returns the process exit code.
        """
        cmd = [sys.executable, str(CLI), *argv]
        self._proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(REPO_ROOT),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
        )
        assert self._proc.stdout is not None
        while True:
            raw = await self._proc.stdout.readline()
            if not raw:
                break
            on_line(raw.decode("utf-8", errors="replace").rstrip("\n"))
        code = await self._proc.wait()
        self._proc = None
        return code

    def cancel(self) -> None:
        """Terminate the running process group, if any."""
        proc = self._proc
        if proc is None or proc.returncode is not None:
            return
        try:
            if os.name == "posix":
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            else:
                proc.terminate()
        except ProcessLookupError:
            pass