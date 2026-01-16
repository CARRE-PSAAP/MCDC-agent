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


def cmd_generate(args):
    """Handle the 'generate' subcommand."""
    # Import here to avoid slow startup for --help
    from mcdc_agent.utils import load_llm
    from mcdc_agent.onboarding.agent import MCDCAgent
    from mcdc_agent.onboarding.decomposer import Decomposer
    from rich.console import Console
    
    console = Console()
    
    # Get prompt from args or file
    if args.file:
        prompt_path = Path(args.file)
        if not prompt_path.exists():
            console.print(f"[red]Error: File not found: {args.file}[/red]")
            sys.exit(1)
        prompt = prompt_path.read_text().strip()
        console.print(f"[dim]Loaded prompt from: {args.file}[/dim]")
    elif args.prompt:
        prompt = args.prompt
    else:
        console.print("[red]Error: Provide a prompt string or --file[/red]")
        sys.exit(1)
    
    console.print(f"[dim]{prompt[:150]}{'...' if len(prompt) > 150 else ''}[/dim]\n")
    
    try:
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
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def cmd_interactive(args):
    """Handle the 'interactive' subcommand."""
    from mcdc_agent.utils import load_llm
    from mcdc_agent.onboarding.agent import MCDCAgent
    from rich.console import Console
    
    console = Console()
    
    try:
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
        description="Generate MCDC Monte Carlo simulation scripts using AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  mcdc-agent generate "[Simulation description]"
  mcdc-agent generate --file prompt.txt -o my_script.py
  mcdc-agent generate "..." --provider ollama --model qwen3:8b
  mcdc-agent interactive

Environment Variables:
  LLM_PROVIDER    Set to 'ollama' or 'gemini' (default: gemini)
  OLLAMA_MODEL    Default model for Ollama (default: qwen3:8b)
  GEMINI_API_KEY  Required for Gemini provider
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
        "--provider", type=str, default=None,
        help="LLM provider: 'gemini' or 'ollama'"
    )
    gen_parser.add_argument(
        "--model", type=str, default=None,
        help="Model name (e.g., 'qwen3:8b', 'gemini-3-flash-preview')"
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
        "--provider", type=str, default=None,
        help="LLM provider: 'gemini' or 'ollama'"
    )
    int_parser.add_argument(
        "--model", type=str, default=None,
        help="Model name"
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
