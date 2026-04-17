import argparse
import sys
import os
import logging
from pathlib import Path


def setup_logging(verbose: bool = False):
    """Configure logging based on verbosity."""
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    # Suppress noisy dependencies
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)


def _load_prompt_text(args, console):
    """Load the generation prompt from args or file."""
    if args.file:
        prompt_path = Path(args.file)
        if not prompt_path.exists():
            console.print(f"[red]Error: File not found: {args.file}[/red]")
            sys.exit(1)
        prompt = prompt_path.read_text().strip()
        console.print(f"[dim]Loaded prompt from: {args.file}[/dim]")
        return prompt

    if args.prompt:
        return args.prompt

    console.print("[red]Error: Provide a prompt string or --file[/red]")
    sys.exit(1)


def _print_direct_trace(console, generator, script):
    """Print direct-generator artifacts for demo/debug use."""
    from rich.panel import Panel
    from rich.syntax import Syntax

    console.print(Panel(generator.last_mode or "unknown", title="Mode", border_style="cyan"))
    if generator.last_plan:
        console.print(Panel(
            str(generator.last_plan),
            title="General Plan",
            border_style="magenta",
        ))
    if generator.last_geometry_plan:
        console.print(Panel(
            str(generator.last_geometry_plan),
            title="Geometry Plan",
            border_style="magenta",
        ))
    for phase in ["setup", "geometry", "finalize", "full"]:
        if phase in generator.last_phase_scripts:
            console.print(
                Panel(
                    Syntax(
                        generator.last_phase_scripts[phase],
                        "python",
                        theme="monokai",
                        line_numbers=False,
                        word_wrap=True,
                    ),
                    title=f"{phase.title()} Output",
                    border_style="green",
                )
            )
    if script:
        console.print(
            Panel(
                Syntax(script, "python", theme="monokai", line_numbers=True, word_wrap=True),
                title="Final Script",
                border_style="blue",
            )
        )


def _cmd_generate_direct(args, prompt, console):
    """Run the direct small-model generator backend."""
    from mcdc_agent.mcdc.direct import build_direct_generator, save_generation_trace

    if args.no_validate:
        console.print("[yellow]Direct backend currently always performs final validation; ignoring --no-validate[/yellow]")

    generation_mode = "phased" if args.generation_mode == "auto" else args.generation_mode

    generator = build_direct_generator(
        model=args.model,
        provider=args.provider or "openrouter",
        context_method=args.context_method,
        generation_mode=generation_mode,
        max_fix_attempts=args.max_fix_attempts,
        temperature=0.1,
    )

    plan_only = args.plan_only or args.dry_run

    if plan_only:
        generator.generate(prompt, plan_only=True)
        if args.trace_dir:
            trace_path = save_generation_trace(
                generator=generator,
                prompt=prompt,
                script=None,
                output_dir=args.trace_dir,
            )
            console.print(f"[green]Trace saved to: {trace_path}[/green]")
        if args.show_trace:
            _print_direct_trace(console, generator, None)
        return

    script = generator.generate(prompt)
    output_path = Path(args.output)
    output_path.write_text(script)
    console.print(f"\n[green]Script saved to: {args.output}[/green]")

    if args.trace_dir:
        trace_path = save_generation_trace(
            generator=generator,
            prompt=prompt,
            script=script,
            output_dir=args.trace_dir,
        )
        console.print(f"[green]Trace saved to: {trace_path}[/green]")

    if args.show_trace:
        _print_direct_trace(console, generator, script)


def _cmd_generate_v2(args, prompt, console):
    """Run the new v2 generator backend."""
    from mcdc_agent.v2.cli import run_generate

    run_generate(args, prompt, console)


def _cmd_generate_legacy(args, prompt, console):
    """Run the existing decomposer + interactive agent generation backend."""
    # Import here to avoid slow startup for --help
    from mcdc_agent.utils import load_llm
    from mcdc_agent.onboarding.agent import MCDCAgent
    from mcdc_agent.onboarding.decomposer import Decomposer

    # Initialize LLM with provided or default settings
    llm = load_llm(
        temperature=0.1,
        model=args.model,
        provider=args.provider
    )

    agent = MCDCAgent(llm)
    decomposer = Decomposer(llm)

    # Decompose prompt into steps
    tasks = decomposer.decompose(prompt)
    console.print(f"[cyan]Decomposed into {len(tasks)} steps[/cyan]")

    if args.dry_run:
        for i, task in enumerate(tasks, 1):
            console.print(f"  {i}. [{task.step.upper()}] {task.instruction}")
        console.print("\n[yellow]Dry run - no execution[/yellow]")
        return

    # Execute batch
    result = agent.execute_batch(
        [t.to_dict() for t in tasks],
        enable_validation=not args.no_validate,
        verbose=args.verbose
    )

    # Save result
    output_path = Path(args.output)
    output_path.write_text(result)
    console.print(f"\n[green]Script saved to: {args.output}[/green]")


def cmd_generate(args):
    """Handle the 'generate' subcommand."""
    from rich.console import Console
    
    console = Console()
    
    prompt = _load_prompt_text(args, console)
    console.print(f"[dim]{prompt[:150]}{'...' if len(prompt) > 150 else ''}[/dim]\n")
    
    try:
        if args.backend == "direct":
            _cmd_generate_direct(args, prompt, console)
        elif args.backend == "v2":
            _cmd_generate_v2(args, prompt, console)
        else:
            _cmd_generate_legacy(args, prompt, console)
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def cmd_interactive(args):
    """Handle the 'interactive' subcommand."""
    from rich.console import Console
    
    console = Console()
    
    try:
        if args.backend == "v2":
            from mcdc_agent.v2.cli import run_interactive

            run_interactive(args, console)
            return

        from mcdc_agent.utils import load_llm
        from mcdc_agent.onboarding.agent import MCDCAgent

        llm = load_llm(
            temperature=0.1,
            model=args.model,
            provider=args.provider
        )
        agent = MCDCAgent(llm)
        agent.run_onboarding()
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Exiting.[/yellow]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="mcdc-agent",
        description="Generate and analyze MCDC Monte Carlo simulation scripts using the v2 AI agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  mcdc-agent generate "[Simulation description]"
  mcdc-agent generate --file prompt.txt -o my_script.py
  mcdc-agent generate --provider openrouter --model anthropic/claude-opus-4.6 --file prompt.txt
  mcdc-agent generate --backend legacy --provider gemini --model gemini-3-flash-preview --file prompt.txt
  mcdc-agent generate --backend direct --provider openrouter --model anthropic/claude-opus-4.6 --file prompt.txt
  mcdc-agent interactive

Environment Variables:
  OPENROUTER_API_KEY  Required for the default v2 and direct generation flows
  OPENROUTER_MODEL    Optional default model override for v2/direct
  GEMINI_API_KEY      Required only when explicitly using --backend legacy --provider gemini
  OLLAMA_MODEL        Required only when explicitly using --backend legacy --provider ollama
"""
    )
    
    parser.add_argument(
        "--version", action="version", version="mcdc-agent 0.1.0"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable verbose output"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # --- Generate subcommand ---
    gen_parser = subparsers.add_parser(
        "generate", aliases=["gen", "g"],
        help="Generate a simulation script from a prompt"
    )
    gen_parser.add_argument(
        "prompt", nargs="?", default=None,
        help="Simulation description (e.g., 'sphere in cube with fission')"
    )
    gen_parser.add_argument(
        "-f", "--file", type=str,
        help="Read prompt from a text file"
    )
    gen_parser.add_argument(
        "-o", "--output", type=str, default="mcdc_input.py",
        help="Output filename (default: mcdc_input.py)"
    )
    gen_parser.add_argument(
        "--backend", type=str, default="v2",
        choices=["legacy", "direct", "v2"],
        help="Generation backend (default: v2): v2 pipeline, direct generator, or the legacy LangChain agent flow"
    )
    gen_parser.add_argument(
        "--provider", type=str, default=None,
        help="LLM provider: v2/direct use 'openrouter'; legacy supports 'gemini'/'ollama'"
    )
    gen_parser.add_argument(
        "--model", type=str, default=None,
        help="Model name (e.g., 'anthropic/claude-opus-4.6', 'google/gemini-3-flash-preview', 'qwen3:8b')"
    )
    gen_parser.add_argument(
        "--dry-run", action="store_true",
        help="Show decomposed plan without execution"
    )
    gen_parser.add_argument(
        "--no-validate", action="store_true",
        help="Skip dry-run validation of generated script"
    )
    gen_parser.add_argument(
        "--context-method", type=str, default="api_examples_plan_geom",
        choices=["api_only", "api_examples", "api_examples_plan", "api_examples_plan_geom", "no_context"],
        help="Direct/v2 only: context to include during generation"
    )
    gen_parser.add_argument(
        "--generation-mode", type=str, default="auto",
        choices=["auto", "phased", "full", "one_shot"],
        help="Direct/v2 only: auto (simple=one-shot, complex=phased), phased, or one-shot generation"
    )
    gen_parser.add_argument(
        "--max-fix-attempts", type=int, default=3,
        help="Direct/v2 only: maximum automatic fix attempts"
    )
    gen_parser.add_argument(
        "--plan-only", action="store_true",
        help="Direct/v2 only: generate plans without creating a final script"
    )
    gen_parser.add_argument(
        "--trace-dir", type=str, default=None,
        help="Direct/v2: save prompt, plans, phase outputs, and final script to this directory; v2 writes artifacts incrementally"
    )
    gen_parser.add_argument(
        "--show-trace", action="store_true",
        help="Direct/v2 only: print plans and phase outputs to the console"
    )
    gen_parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Show tool calls and debug info"
    )
    gen_parser.set_defaults(func=cmd_generate)
    
    # --- Interactive subcommand ---
    int_parser = subparsers.add_parser(
        "interactive", aliases=["int", "i"],
        help="Start interactive onboarding mode"
    )
    int_parser.add_argument(
        "--backend", type=str, default="v2",
        choices=["legacy", "v2"],
        help="Interactive backend (default: v2): the v2 flow or the legacy LangChain onboarding flow"
    )
    int_parser.add_argument(
        "--provider", type=str, default=None,
        help="LLM provider: v2 uses 'openrouter'; legacy supports 'gemini'/'ollama'"
    )
    int_parser.add_argument(
        "--model", type=str, default=None,
        help="Model name"
    )
    int_parser.add_argument(
        "-f", "--file", type=str,
        help="Optional prompt file to preload in v2 interactive mode"
    )
    int_parser.add_argument(
        "-o", "--output", type=str, default="mcdc_input.py",
        help="Output filename for v2 interactive generation (default: mcdc_input.py)"
    )
    int_parser.add_argument(
        "--context-method", type=str, default="api_examples_plan_geom",
        choices=["api_only", "api_examples", "api_examples_plan", "api_examples_plan_geom", "no_context"],
        help="v2 only: context to include during generation"
    )
    int_parser.add_argument(
        "--generation-mode", type=str, default="auto",
        choices=["auto", "phased", "full", "one_shot"],
        help="v2 only: auto (simple=one-shot, complex=phased), phased, or one-shot generation"
    )
    int_parser.add_argument(
        "--max-fix-attempts", type=int, default=3,
        help="v2 only: maximum automatic fix attempts"
    )
    int_parser.add_argument(
        "--trace-dir", type=str, default=None,
        help="v2 only: save prompt, plans, phase outputs, and final script to this directory as they are generated"
    )
    int_parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable verbose output"
    )
    int_parser.set_defaults(func=cmd_interactive)
    
    # Parse and execute
    args = parser.parse_args()
    
    # Setup logging
    verbose = getattr(args, 'verbose', False)
    setup_logging(verbose)
    
    if args.command is None:
        parser.print_help()
        sys.exit(0)
    
    args.func(args)


if __name__ == "__main__":
    main()
