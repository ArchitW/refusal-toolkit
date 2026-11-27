# Created: 27-11-26
# agent: kilo/baidu/ernie-5.5-flash-preview:free
"""Textual TUI for the Refusal Toolkit.

Launch via `refusal-toolkit tui`. Each command runs as a subprocess so the
behavior matches the argparse CLI exactly. Textual is an optional dependency;
if it is missing, run() prints an install hint and exits non-zero.
"""
from __future__ import annotations

import sys

from .runner import CommandRunner, build_argv
from .specs import COMMANDS, CommandSpec, Field

MISSING_TEXTUAL = (
    "Textual is required for the TUI. Install it with:\n"
    "    pip install textual\n"
    "or: pip install 'refusal-toolkit[tui]'"
)


def run() -> int:
    try:
        from textual.app import App, ComposeResult
        from textual.containers import Horizontal, Vertical
        from textual.widgets import (
            Button,
            Checkbox,
            Footer,
            Header,
            Input,
            Label,
            ListItem,
            ListView,
            RichLog,
            Static,
        )
    except ImportError:
        print(MISSING_TEXTUAL, file=sys.stderr)
        return 1

    class CommandForm(Vertical):
        """Dynamic form area for the selected command."""

    class ToolkitApp(App):
        CSS = """
        #body { height: 1fr; }
        #sidebar { width: 32; border-right: solid $accent; }
        #form { padding: 1 2; height: 1fr; overflow-y: auto; }
        #output { height: 1fr; border-top: solid $accent; }
        #controls { height: auto; padding: 0 2; }
        .row { height: auto; margin: 0 0 1 0; }
        .field-label { text-style: bold; }
        Button { margin-right: 1; }
        """

        BINDINGS = [
            ("q", "quit", "Quit"),
            ("ctrl+c", "cancel", "Cancel run"),
        ]

        def __init__(self) -> None:
            super().__init__()
            self.runner = CommandRunner()
            self._widgets: dict[str, object] = {}
            self._current: CommandSpec | None = None

        def compose(self) -> ComposeResult:
            yield Header()
            with Horizontal(id="body"):
                yield ListView(
                    *[ListItem(Label(c.command), id=f"cmd-{c.command}") for c in COMMANDS],
                    id="sidebar",
                )
                with Vertical(id="main"):
                    with Vertical(id="form"):
                        yield Static("Select a command to begin.", id="form-title")
                        yield Vertical(id="fields")
                    with Horizontal(id="controls"):
                        yield Button("Run", id="run", variant="success")
                        yield Button("Cancel", id="cancel", variant="error")
                    yield RichLog(id="output", markup=True, wrap=True)
            yield Footer()

        def on_mount(self) -> None:
            self.query_one("#sidebar", ListView).focus()

        def on_list_view_selected(self, event: ListView.Selected) -> None:
            idx = event.list_view.index
            if idx is None:
                return
            self.run_worker(self._select(COMMANDS[idx]))

        async def _select(self, spec: CommandSpec) -> None:
            self._current = spec
            self.query_one("#form-title", Static).update(
                f"[b]{spec.command}[/b] — {spec.help}"
            )
            fields = self.query_one("#fields", Vertical)
            await fields.remove_children()
            self._widgets.clear()
            for f in spec.fields:
                label, widget = self._make_field(f)
                await fields.mount(label)
                if widget is not None:
                    await fields.mount(widget)
            log = self.query_one("#output", RichLog)
            log.clear()

        def _make_field(self, f: Field):
            from textual.widgets import Input, Label

            label = Label(f"{f.label}{' *' if f.required else ''}", classes="field-label")
            if f.help:
                label.tooltip = f.help
            if f.type == "bool":
                cb = Checkbox(f.label, value=bool(f.default), id=f"field-{f.name}")
                self._widgets[f.name] = cb
                return label, cb
            if f.type == "choice":
                hint = "/".join(f.choices)
                inp = Input(
                    value=str(f.default or ""),
                    placeholder=f"one of: {hint}",
                    id=f"field-{f.name}",
                )
            elif f.type == "multi":
                default = " ".join(f.default) if isinstance(f.default, list) else ""
                inp = Input(value=default, placeholder="space-separated", id=f"field-{f.name}")
            else:
                inp = Input(
                    value="" if f.default is None else str(f.default),
                    placeholder=f.type,
                    id=f"field-{f.name}",
                )
            self._widgets[f.name] = inp
            return label, inp

        def _collect(self) -> dict[str, object]:
            values: dict[str, object] = {}
            for f in self._current.fields:  # type: ignore[union-attr]
                w = self._widgets.get(f.name)
                if w is None:
                    values[f.name] = f.default
                elif f.type == "bool":
                    values[f.name] = bool(w.value)  # type: ignore[attr-defined]
                else:
                    raw = str(w.value).strip()  # type: ignore[attr-defined]
                    values[f.name] = self._coerce(f, raw)
            return values

        @staticmethod
        def _coerce(f: Field, raw: str):
            if raw == "":
                return f.default if f.default is not None else None
            if f.type == "int":
                return int(raw)
            if f.type == "float":
                return float(raw)
            if f.type == "multi":
                return raw.split()
            return raw

        def on_button_pressed(self, event: Button.Pressed) -> None:
            if event.button.id == "run":
                self.action_run()
            elif event.button.id == "cancel":
                self.action_cancel()

        def action_cancel(self) -> None:
            self.runner.cancel()
            self.query_one("#output", RichLog).write("[red]cancelled[/]")

        def action_run(self) -> None:
            if self._current is None:
                return
            if self.runner.running:
                self.query_one("#output", RichLog).write("[yellow]already running[/]")
                return
            spec = self._current
            values = self._collect()
            missing = [f.label for f in spec.fields if f.required and not values.get(f.name)]
            log = self.query_one("#output", RichLog)
            if missing:
                log.write(f"[red]missing required: {', '.join(missing)}[/]")
                return
            for f in spec.fields:
                if f.type == "choice":
                    val = values.get(f.name)
                    if val and val not in f.choices:
                        log.write(f"[red]{f.label} must be one of {f.choices}[/]")
                        return
            argv = build_argv(spec, values)
            log.write(f"[bold]running:[/] {' '.join(argv)}")
            self.run_worker(self._run_worker(argv, log), exclusive=False)

        async def _run_worker(self, argv: list[str], log: RichLog) -> None:
            code = await self.runner.run(argv, lambda line: log.write(line))
            if code == 0:
                log.write("[green]done (exit 0)[/]")
            else:
                log.write(f"[red]failed (exit {code})[/]")

    return ToolkitApp().run()