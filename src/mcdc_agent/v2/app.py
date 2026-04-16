import json
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax

from mcdc_agent.v2.config import AppConfig, DIAGNOSTICS_MENU, TOP_LEVEL_MENU
from mcdc_agent.v2.services import (
    DiagnosticsService,
    ExecutionService,
    GenerationService,
    OnboardingService,
    QAService,
    VisualizationService,
)
from mcdc_agent.v2.state import AppState
from mcdc_agent.v2.types import ExecutionResult, GenerationArtifacts


class V2App:
    """Simplified interactive shell for the v2 architecture."""

    def __init__(self, config: AppConfig, console: Console | None = None):
        self.config = config
        self.console = console or Console()
        self.state = AppState(current_script_path=config.output_path)

        self.onboarding = OnboardingService()
        self.qa = QAService(config)
        self.generation = GenerationService(config)
        self.execution = ExecutionService()
        self.diagnostics = DiagnosticsService(config)
        self.visualization = VisualizationService(config)

    def load_prompt_file(self, prompt_path: str | Path) -> None:
        path = self._normalize_user_path(prompt_path)
        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
        self.state.current_prompt = path.read_text(encoding="utf-8").strip()

    def run(self) -> None:
        self._section("MCDC Agent v2", "OpenRouter-backed generation, onboarding, Q&A, and diagnostics.")

        while True:
            self._section("Main Menu")
            for key, label in TOP_LEVEL_MENU:
                self.console.print(f"  [{key}] {label}", markup=False)
            self.console.print("  [q] Quit", markup=False)

            choice = self._prompt().lower()
            if choice == "q":
                return
            if choice == "1":
                self._learn_menu()
            elif choice == "2":
                self._ask_questions()
            elif choice == "3":
                self._generate_and_run()
            elif choice == "4":
                self._view_edit_script()
            elif choice == "5":
                self._results_and_diagnostics()
            else:
                self.console.print("[yellow]Invalid option.[/yellow]")

    def _learn_menu(self) -> None:
        while True:
            self._section("Learn MCDC")
            for key, label in self.onboarding.menu_options():
                self.console.print(f"  [{key}] {label}", markup=False)
            self.console.print("  [b] Back", markup=False)

            choice = self._prompt()
            if not choice:
                continue
            if choice.lower() == "b":
                return

            try:
                topic = self.onboarding.get_topic(choice)
                lesson_text = self.onboarding.format_lesson(choice)
                self.console.print(
                    Panel(
                        Markdown(lesson_text, code_theme="monokai"),
                        title=topic.label,
                        border_style="magenta",
                    )
                )
                self.console.print("[dim]Ask a question about this topic, or press Enter to go back.[/dim]")
                while True:
                    question = self._prompt()
                    if not question:
                        break
                    scoped_question = f"In MCDC, regarding {topic.label}: {question}"
                    try:
                        with self.console.status("[cyan]Thinking...[/cyan]", spinner="dots"):
                            answer = self.qa.answer(scoped_question, top_k=4)
                        self.console.print(Panel(answer, title=f"{topic.label} Q&A", border_style="cyan"))
                    except Exception as exc:
                        self.console.print(f"[yellow]Could not generate an answer: {exc}[/yellow]")
            except KeyError:
                self.console.print("[yellow]Unknown topic.[/yellow]")

    def _ask_questions(self) -> None:
        self._section("Ask Questions")
        self.console.print("[dim]Press Enter on an empty line to return.[/dim]")

        while True:
            question = self._prompt()
            if not question:
                return

            hits = self.qa.retrieve(question, top_k=4)
            if not hits:
                self.console.print("[yellow]No relevant local matches found.[/yellow]")
                continue

            try:
                with self.console.status("[cyan]Thinking...[/cyan]", spinner="dots"):
                    answer = self.qa.answer(question, top_k=4)
                self.console.print(Panel(answer, title="Answer", border_style="cyan"))
            except Exception as exc:
                self.console.print(f"[yellow]Could not generate an LLM answer: {exc}[/yellow]")

            sources = "\n".join(f"- {hit.source}" for hit in hits)
            self.console.print(Panel(sources, title="Sources", border_style="green"))

    def _generate_and_run(self) -> None:
        self._section("Generate and Run Simulation")

        self.console.print("[dim]Paste a new prompt, or press Enter to reuse the current prompt.[/dim]")
        prompt = self._prompt()
        if prompt:
            self.state.current_prompt = prompt

        if not self.state.current_prompt:
            self.console.print("[yellow]No prompt set yet.[/yellow]")
            return

        with self.console.status("[cyan]Generating script...[/cyan]", spinner="dots"):
            artifacts = self.generation.generate(self.state.current_prompt)
        self.state.last_generation = artifacts
        self._render_artifacts(artifacts)

        if not artifacts.script:
            self.console.print("[yellow]No script was generated.[/yellow]")
            return

        output_path = self.generation.save_script(artifacts.script, self.config.output_path)
        self.state.current_script = artifacts.script
        self.state.current_script_path = output_path
        self.state.last_output_path = output_path
        self._clear_run_state()
        self.console.print(f"\n[green]Script saved to: {output_path}[/green]")
        self._run_current_script()

    def _view_edit_script(self) -> None:
        self._section("View/Edit Current Script")

        if self.state.current_script:
            self.console.print(
                Panel(
                    Syntax(
                        self.state.current_script,
                        "python",
                        theme="monokai",
                        line_numbers=True,
                        word_wrap=True,
                    ),
                    title="Current Script",
                    border_style="blue",
                )
            )
        else:
            self.console.print("[dim]No current script loaded.[/dim]")

        self.console.print("  [l] Load from file", markup=False)
        self.console.print("  [p] Paste replacement", markup=False)
        self.console.print("  [r] Run current script", markup=False)
        self.console.print("  [b] Back", markup=False)
        choice = self._prompt().lower()
        if choice == "b" or not choice:
            return
        if choice == "l":
            self.console.print("[dim]Enter a file path.[/dim]")
            path_text = self._prompt()
            if not path_text:
                return
            path = self._normalize_user_path(path_text)
            if not path.exists():
                self.console.print(f"[yellow]File not found: {path}[/yellow]")
                return
            self.state.current_script = path.read_text(encoding="utf-8")
            self.state.current_script_path = path
            self._clear_run_state()
            self.console.print(f"[green]Loaded script from: {path}[/green]")
            self._prompt_to_run_current_script()
            return
        if choice == "p":
            self.console.print("[dim]Paste the new script below. End with a line containing only END.[/dim]")
            lines = []
            while True:
                line = input("> ")
                if line == "END":
                    break
                lines.append(line)
            text = "\n".join(lines).rstrip() + "\n"
            self.state.current_script = text
            path = self.state.current_script_path or self.config.output_path
            path.write_text(text, encoding="utf-8")
            self.state.current_script_path = path
            self._clear_run_state()
            self.console.print(f"[green]Current script updated: {path}[/green]")
            self._prompt_to_run_current_script()
            return
        if choice == "r":
            self._run_current_script()
            return

        self.console.print("[yellow]Invalid option.[/yellow]")

    def _results_and_diagnostics(self) -> None:
        self._section("Results and Diagnostics")
        output_h5 = self._find_output_h5()

        try:
            with self.console.status("[cyan]Reading output...[/cyan]", spinner="dots"):
                summary = self.diagnostics.summarize_run_context(
                    output_h5=output_h5,
                    returncode=self.state.last_run_returncode,
                    stdout=self.state.last_run_stdout,
                    stderr=self.state.last_run_stderr,
                )
            self.state.last_output_h5 = output_h5 if output_h5 and output_h5.exists() else None
            self.state.last_output_summary = summary
            self.console.print(
                Panel(
                    self.diagnostics.format_summary(summary),
                    title="Output Summary",
                    border_style="blue",
                )
            )
        except Exception as exc:
            self.console.print(f"[yellow]Failed to summarize output: {exc}[/yellow]")
            return

        while True:
            for key, label in DIAGNOSTICS_MENU:
                self.console.print(f"  [{key}] {label}", markup=False)
            self.console.print("  [b] Back", markup=False)
            choice = self._prompt().lower()
            if choice == "b" or not choice:
                return
            if choice == "1":
                if not output_h5:
                    self.console.print("[yellow]Visualization requires an output .h5 file. This run did not produce one.[/yellow]")
                    continue
                self._visualize_output(output_h5, summary)
                continue
            if choice == "2":
                self._analyze_output(summary)
                continue
            self.console.print("[yellow]Invalid option.[/yellow]")

    def _record_execution(self, result: ExecutionResult) -> None:
        self.state.last_run_returncode = result.returncode
        self.state.last_run_stdout = result.stdout
        self.state.last_run_stderr = result.stderr
        self.state.last_output_h5 = result.output_h5

    def _find_output_h5(self) -> Path | None:
        if self.state.last_output_h5 and self.state.last_output_h5.exists():
            return self.state.last_output_h5
        return None

    def _print_execution(self, result: ExecutionResult) -> None:
        self.console.print(
            Panel(
                "\n".join(
                    [
                        f"Success: {result.success}",
                        f"Return code: {result.returncode}",
                        f"Timed out: {result.timed_out}",
                        f"Duration: {result.duration_seconds:.2f}s",
                        f"Output file: {result.output_h5}",
                    ]
                ),
                title="Execution Result",
                border_style="cyan",
            )
        )

        if result.stdout_display:
            self.console.print(Panel(result.stdout_display, title="Run Output", border_style="green"))
        if result.stderr.strip():
            self.console.print(Panel(result.stderr.strip(), title="Run Errors", border_style="red"))

    def _visualize_output(self, output_h5: Path, summary: dict) -> None:
        self.console.print("[dim]Describe the plot you want, or press Enter for a sensible default.[/dim]")
        request = self._prompt()
        try:
            with self.console.status("[cyan]Building visualization...[/cyan]", spinner="dots"):
                plot_path, spec = self.visualization.create_plot(
                    output_h5,
                    summary,
                    request=request,
                    script_text=self.state.current_script,
                )
            self.state.last_visualization_path = plot_path
            spec_text = json.dumps(
                {
                    "kind": spec.kind,
                    "tally": spec.tally,
                    "score": spec.score,
                    "value": spec.value,
                    "title": spec.title,
                },
                indent=2,
            )
            self.console.print(Panel(spec_text, title="Visualization Spec", border_style="magenta"))
            self.console.print(f"[green]Saved plot to: {plot_path}[/green]")
        except Exception as exc:
            self.console.print(f"[yellow]Visualization failed: {exc}[/yellow]")

    def _analyze_output(self, summary: dict) -> None:
        self.console.print("[dim]Ask a diagnosis question, or press Enter for a general analysis.[/dim]")
        question = self._prompt()
        try:
            with self.console.status("[cyan]Analyzing results...[/cyan]", spinner="dots"):
                analysis = self.diagnostics.analyze_output(
                    summary,
                    script_text=self.state.current_script,
                    stdout=self.state.last_run_stdout,
                    stderr=self.state.last_run_stderr,
                    user_question=question,
                    original_prompt=self.state.current_prompt,
                    plan=self.state.last_generation.plan if self.state.last_generation else None,
                    geometry_plan=self.state.last_generation.geometry_plan if self.state.last_generation else None,
                )
            self.state.last_analysis = analysis
            self.console.print(Panel(analysis, title="Analysis / Diagnosis", border_style="cyan"))
            self.console.print("  [f] Apply suggested fix", markup=False)
            self.console.print("  [b] Back", markup=False)
            choice = self._prompt().lower()
            if choice == "f":
                self._apply_diagnostic_fix(summary, question, analysis)
        except Exception as exc:
            self.console.print(f"[yellow]Analysis failed: {exc}[/yellow]")

    def _clear_run_state(self) -> None:
        self.state.last_run_returncode = None
        self.state.last_run_stdout = ""
        self.state.last_run_stderr = ""
        self.state.last_output_h5 = None
        self.state.last_output_summary = {}
        self.state.last_visualization_path = None
        self.state.last_analysis = ""

    @staticmethod
    def _normalize_user_path(path_text: str | Path) -> Path:
        text = str(path_text).strip()
        if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
            text = text[1:-1]
        return Path(text).expanduser().resolve()

    def _prompt(self) -> str:
        return input("> ").strip()

    def _section(self, title: str, subtitle: str | None = None) -> None:
        self.console.print()
        self.console.rule(f"[bold cyan]{title}[/bold cyan]")
        if subtitle:
            self.console.print(f"[dim]{subtitle}[/dim]")

    def _run_current_script(self) -> None:
        if not self.state.current_script.strip():
            self.console.print("[yellow]No current script is loaded.[/yellow]")
            return
        if not self.state.current_script_path:
            self.console.print("[yellow]No current script file is available to run.[/yellow]")
            return

        with self.console.status("[cyan]Running simulation...[/cyan]", spinner="dots"):
            result = self.execution.run_script(self.state.current_script_path)
        self._record_execution(result)
        self._print_execution(result)

        if result.output_h5:
            try:
                with self.console.status("[cyan]Summarizing output...[/cyan]", spinner="dots"):
                    summary = self.diagnostics.summarize_output(result.output_h5)
                self.state.last_output_summary = summary
                self.console.print(
                    Panel(
                        self.diagnostics.format_summary(summary),
                        title="Output Summary",
                        border_style="blue",
                    )
                )
            except Exception as exc:
                self.console.print(f"[yellow]Run finished, but output summary failed: {exc}[/yellow]")

    def _prompt_to_run_current_script(self) -> None:
        self.console.print("[dim]Run the current script now? [y/N][/dim]")
        if self._prompt().lower() == "y":
            self._run_current_script()

    def _apply_diagnostic_fix(self, summary: dict, question: str, analysis: str) -> None:
        if not self.state.current_script.strip():
            self.console.print("[yellow]No current script is loaded, so there is nothing to fix.[/yellow]")
            return

        try:
            with self.console.status("[cyan]Applying suggested fix...[/cyan]", spinner="dots"):
                fixed_script = self.diagnostics.suggest_script_fix(
                    summary,
                    script_text=self.state.current_script,
                    stdout=self.state.last_run_stdout,
                    stderr=self.state.last_run_stderr,
                    user_question=question,
                    analysis_text=analysis,
                    original_prompt=self.state.current_prompt,
                    plan=self.state.last_generation.plan if self.state.last_generation else None,
                    geometry_plan=self.state.last_generation.geometry_plan if self.state.last_generation else None,
                )
        except Exception as exc:
            self.console.print(f"[yellow]Fix generation failed: {exc}[/yellow]")
            return

        path = self.state.current_script_path or self.config.output_path
        path.write_text(fixed_script, encoding="utf-8")
        self.state.current_script = fixed_script
        self.state.current_script_path = path
        self._clear_run_state()

        self.console.print(
            Panel(
                Syntax(
                    fixed_script,
                    "python",
                    theme="monokai",
                    line_numbers=True,
                    word_wrap=True,
                ),
                title="Updated Script",
                border_style="green",
            )
        )
        self.console.print(f"[green]Updated current script: {path}[/green]")
        self.console.print("[dim]Run the updated script now? [y/N][/dim]")
        if self._prompt().lower() == "y":
            self._run_current_script()

    def _render_artifacts(self, artifacts: GenerationArtifacts) -> None:
        self.console.print(Panel(artifacts.mode or "unknown", title="Mode", border_style="cyan"))

        if artifacts.plan:
            self.console.print(
                Panel(
                    json.dumps(artifacts.plan, indent=2),
                    title="General Plan",
                    border_style="magenta",
                )
            )

        if artifacts.geometry_plan:
            self.console.print(
                Panel(
                    json.dumps(artifacts.geometry_plan, indent=2),
                    title="Geometry Plan",
                    border_style="magenta",
                )
            )

        for phase in ["setup", "geometry", "finalize", "full"]:
            code = artifacts.phase_scripts.get(phase)
            if code:
                self.console.print(
                    Panel(
                        Syntax(code, "python", theme="monokai", line_numbers=False, word_wrap=True),
                        title=f"{phase.title()} Output",
                        border_style="green",
                    )
                )

        if artifacts.script:
            self.console.print(
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
