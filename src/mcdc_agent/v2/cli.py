import json
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

from mcdc_agent.v2.config import AppConfig
from mcdc_agent.v2.interactive import InteractiveGenerationSession
from mcdc_agent.v2.services import GenerationService
from mcdc_agent.v2.types import GenerationArtifacts


def _build_config(args) -> AppConfig:
    return AppConfig(
        provider=args.provider or "openrouter",
        model=args.model or AppConfig().model,
        context_method=getattr(args, "context_method", "api_examples_plan_geom"),
        generation_mode=getattr(args, "generation_mode", "phased"),
        max_fix_attempts=getattr(args, "max_fix_attempts", 3),
        temperature=0.1,
        output_path=Path(args.output),
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

    service = GenerationService(_build_config(args))
    artifacts = service.plan(prompt) if (args.plan_only or args.dry_run) else service.generate(prompt)

    if args.show_trace or args.plan_only or args.dry_run:
        _render_artifacts(console, artifacts)

    if artifacts.script:
        output_path = service.save_script(artifacts.script)
        console.print(f"\n[green]Script saved to: {output_path}[/green]")


def run_interactive(args, console: Console) -> None:
    session = InteractiveGenerationSession(_build_config(args))
    prompt = ""
    selected_materials: dict[str, str] = {}

    if getattr(args, "file", None):
        prompt_path = Path(args.file)
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {args.file}")
        prompt = prompt_path.read_text().strip()
        console.print(f"[dim]Loaded prompt from: {args.file}[/dim]")

    console.print("[bold cyan]MCDC Agent v2 Interactive[/bold cyan]")
    console.print("Commands: prompt, load, material, use, clear, show, plan, generate, help, quit")

    while True:
        raw = input("v2> ").strip()
        if not raw:
            continue

        command, _, argument = raw.partition(" ")
        command = command.lower()
        argument = argument.strip()

        if command in {"quit", "exit", "q"}:
            return

        if command == "help":
            console.print("prompt <text>      Set the current simulation prompt")
            console.print("load <path>        Load the prompt from a text file")
            console.print("material <query>   Look up material or isotope data")
            console.print("use <query>        Add looked-up material data to generation context")
            console.print("clear              Clear added material context")
            console.print("show               Show the current prompt and added material context")
            console.print("plan               Generate and show plans only")
            console.print("generate           Generate the full script and save it")
            console.print("quit               Exit interactive mode")
            continue

        if command == "prompt":
            prompt = argument or input("Prompt: ").strip()
            console.print("[green]Prompt updated.[/green]")
            continue

        if command == "load":
            path_text = argument or input("Path: ").strip()
            prompt_path = Path(path_text)
            if not prompt_path.exists():
                console.print(f"[red]File not found: {path_text}[/red]")
                continue
            prompt = prompt_path.read_text().strip()
            console.print(f"[green]Loaded prompt from: {path_text}[/green]")
            continue

        if command == "material":
            query = argument or input("Material query: ").strip()
            context = session.lookup_material(query)
            if context:
                console.print(
                    Panel(
                        context,
                        title=f"Material Lookup: {query}",
                        border_style="yellow",
                    )
                )
            else:
                console.print(f"[yellow]No material data found for: {query}[/yellow]")
            continue

        if command == "use":
            query = argument or input("Material query: ").strip()
            context = session.lookup_material(query)
            if context:
                selected_materials[query] = context
                console.print(f"[green]Added material context for: {query}[/green]")
            else:
                console.print(f"[yellow]No material data found for: {query}[/yellow]")
            continue

        if command == "clear":
            selected_materials.clear()
            console.print("[green]Cleared material context.[/green]")
            continue

        if command == "show":
            prompt_text = prompt or "(no prompt set)"
            console.print(Panel(prompt_text, title="Current Prompt", border_style="cyan"))
            if selected_materials:
                combined = "\n\n".join(selected_materials.values())
                console.print(Panel(combined, title="Material Context", border_style="yellow"))
            else:
                console.print("[dim]No material context selected.[/dim]")
            continue

        if command in {"plan", "generate"}:
            if not prompt:
                console.print("[red]Set a prompt first with 'prompt' or 'load'.[/red]")
                continue

            extra_context = "\n\n".join(selected_materials.values())
            if command == "plan":
                artifacts = session.plan(prompt, extra_context=extra_context)
                _render_artifacts(console, artifacts)
                continue

            artifacts = session.generate(prompt, extra_context=extra_context)
            _render_artifacts(console, artifacts)
            if artifacts.script:
                output_path = Path(args.output)
                output_path.write_text(artifacts.script)
                console.print(f"\n[green]Script saved to: {args.output}[/green]")
            continue

        console.print(f"[yellow]Unknown command: {command}. Type 'help' for options.[/yellow]")
