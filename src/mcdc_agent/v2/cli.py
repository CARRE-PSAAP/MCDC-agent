import json
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

from mcdc_agent.v2.app import V2App
from mcdc_agent.v2.config import AppConfig
from mcdc_agent.v2.services import GenerationService
from mcdc_agent.v2.types import GenerationArtifacts


def _build_config(args) -> AppConfig:
    return AppConfig(
        provider=args.provider or "openrouter",
        model=args.model or AppConfig().model,
        context_method=getattr(args, "context_method", "api_examples_plan_geom"),
        generation_mode=getattr(args, "generation_mode", "auto"),
        max_fix_attempts=getattr(args, "max_fix_attempts", 3),
        temperature=0.1,
        output_path=Path(args.output),
        trace_dir=Path(args.trace_dir) if getattr(args, "trace_dir", None) else None,
    )


def _render_artifacts(console: Console, artifacts: GenerationArtifacts) -> None:
    console.print(Panel(artifacts.mode or "unknown", title="Mode", border_style="cyan"))

    if artifacts.plan:
        console.print(
            Panel(
                json.dumps(artifacts.plan, indent=2),
                title="General Plan",
                border_style="magenta",
            )
        )

    if artifacts.geometry_plan:
        console.print(
            Panel(
                json.dumps(artifacts.geometry_plan, indent=2),
                title="Geometry Plan",
                border_style="magenta",
            )
        )

    for phase in ["setup", "geometry", "finalize", "full"]:
        code = artifacts.phase_scripts.get(phase)
        if code:
            console.print(
                Panel(
                    Syntax(code, "python", theme="monokai", line_numbers=False, word_wrap=True),
                    title=f"{phase.title()} Output",
                    border_style="green",
                )
            )

    if artifacts.script:
        console.print(
            Panel(
                Syntax(
                    artifacts.script,
                    "python",
                    theme="monokai",
                    line_numbers=True,
                    word_wrap=True,
                ),
                title="Final Script",
                border_style="blue",
            )
        )


def run_generate(args, prompt: str, console: Console) -> None:
    if args.no_validate:
        console.print("[yellow]The v2 backend currently always performs final validation; ignoring --no-validate[/yellow]")

    config = _build_config(args)
    service = GenerationService(config)
    if config.trace_dir:
        console.print(f"[dim]Saving v2 trace artifacts to: {config.trace_dir}[/dim]")
    status_text = "[cyan]Planning...[/cyan]" if (args.plan_only or args.dry_run) else "[cyan]Generating script...[/cyan]"
    with console.status(status_text, spinner="dots"):
        artifacts = service.plan(prompt) if (args.plan_only or args.dry_run) else service.generate(prompt)

    if args.show_trace or args.plan_only or args.dry_run:
        _render_artifacts(console, artifacts)

    if artifacts.script:
        output_path = service.save_script(artifacts.script)
        console.print(f"\n[green]Script saved to: {output_path}[/green]")


def run_interactive(args, console: Console) -> None:
    config = _build_config(args)
    if config.trace_dir:
        console.print(f"[dim]Saving v2 trace artifacts to: {config.trace_dir}[/dim]")
    app = V2App(config, console=console)
    if getattr(args, "file", None):
        app.load_prompt_file(args.file)
        console.print(f"[dim]Loaded prompt from: {args.file}[/dim]")
    app.run()
