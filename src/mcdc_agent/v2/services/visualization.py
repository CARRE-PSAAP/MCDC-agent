import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import matplotlib
import numpy as np

from mcdc_agent.v2.config import AppConfig
from mcdc_agent.v2.llm import extract_code, load_llm

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True, slots=True)
class VisualizationProgram:
    code: str
    mode: str
    request: str = ""


class VisualizationService:
    """LLM-driven matplotlib program generation with controlled execution."""

    def __init__(self, config: AppConfig | None = None):
        self.config = config or AppConfig()
        self._llm = None

    def create_plot(
        self,
        output_h5: str | Path,
        summary: dict[str, Any],
        *,
        request: str = "",
        script_text: str = "",
        output_path: str | Path | None = None,
    ) -> tuple[Path, VisualizationProgram]:
        program = self.plan_plot(summary, request=request, script_text=script_text)
        plot_path = self.render_plot(output_h5, program, output_path=output_path)
        return plot_path, program

    def plan_plot(
        self,
        summary: dict[str, Any],
        *,
        request: str = "",
        script_text: str = "",
    ) -> VisualizationProgram:
        request_text = request.strip() or self._default_request(summary)
        return self._plan_with_llm(summary, request=request_text, script_text=script_text)

    def render_plot(
        self,
        output_h5: str | Path,
        program: VisualizationProgram,
        *,
        output_path: str | Path | None = None,
    ) -> Path:
        output_h5 = Path(output_h5).resolve()
        if not output_h5.exists():
            raise FileNotFoundError(f"Output file not found: {output_h5}")

        plot_path = Path(output_path).resolve() if output_path else self._default_plot_path(output_h5, program)
        plot_path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(output_h5, "r") as handle:
            fig = self._execute_plot_code(handle, output_h5, program.code)

        try:
            fig.tight_layout()
        except Exception:
            pass
        fig.savefig(plot_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        return plot_path

    def _plan_with_llm(
        self,
        summary: dict[str, Any],
        *,
        request: str,
        script_text: str,
    ) -> VisualizationProgram:
        llm = self._get_llm()
        prompt = self._build_code_prompt(summary, request=request, script_text=script_text)
        response = llm.invoke({"messages": [{"role": "user", "content": prompt}]})
        content = str(response.get("content", "") if isinstance(response, dict) else response).strip()
        code = self._sanitize_generated_code(extract_code(content))
        if not code:
            raise RuntimeError("Visualization generation failed; the model returned no plotting code.")

        try:
            compile(code, "<visualization>", "exec")
        except SyntaxError as exc:
            raise RuntimeError(f"Visualization generation failed; invalid plotting code was returned: {exc.msg}.") from exc

        return VisualizationProgram(code=code, mode="llm_direct", request=request)

    def _build_code_prompt(
        self,
        summary: dict[str, Any],
        *,
        request: str,
        script_text: str,
    ) -> str:
        script_block = ""
        if script_text.strip():
            script_block = f"\nCurrent script:\n```python\n{script_text[:6000]}\n```\n"

        return (
            "You are writing Python matplotlib code to visualize an MCDC output.h5 file.\n"
            "Return ONLY Python code. Do not include Markdown fences or prose.\n"
            "Write plotting code that satisfies the user's request using only data that is actually available.\n"
            "Assume these objects already exist: handle (an open h5py.File), np, plt, h5py, output_h5,\n"
            "read_value_array(score_group, value_kind), midpoints_if_edges(grid, target_length),\n"
            "and value_label(value_kind).\n"
            "Do not import modules.\n"
            "Do not open other files.\n"
            "Do not call savefig, show, or close.\n"
            "Create a matplotlib figure and assign it to `fig`.\n"
            "Raise ValueError with a short message if the requested data is not available.\n"
            "Prefer readable axis labels, titles, and light grids where they help.\n\n"
            "Relevant MCDC HDF5 layout:\n"
            "- k_cycle, k_mean, k_sdev for eigenvalue runs\n"
            "- runtime/<name> for timing data\n"
            "- global_tally/neutron/{mean,sdev,max} and global_tally/precursor/{mean,sdev,max} in eigenvalue runs\n"
            "- tallies/<tally>/grid/{mu,azi,energy,time[,x,y,z]}\n"
            "- tallies/<tally>/<score>/{mean,sdev}\n"
            "- mesh tally spatial grids are usually bin edges suitable for pcolormesh\n"
            "- 1D grids may need midpoints_if_edges(grid, len(values)) before plotting\n\n"
            f"Output summary:\n{json.dumps(summary, indent=2)}\n"
            f"{script_block}\n"
            f"User request:\n{request}\n"
        )

    @staticmethod
    def _sanitize_generated_code(code: str) -> str:
        cleaned_lines = []
        for line in code.splitlines():
            stripped = line.strip()
            if re.match(r"^(from\s+\S+\s+import|import\s+\S+)", stripped):
                continue
            if any(token in stripped for token in (".savefig(", "plt.savefig(", "plt.show(", "show()", "plt.close(")):
                continue
            cleaned_lines.append(line)
        return "\n".join(cleaned_lines).strip()

    def _execute_plot_code(self, handle: h5py.File, output_h5: Path, code: str):
        existing_figures = set(plt.get_fignums())
        safe_builtins = {
            "abs": abs,
            "all": all,
            "any": any,
            "bool": bool,
            "dict": dict,
            "enumerate": enumerate,
            "Exception": Exception,
            "float": float,
            "int": int,
            "isinstance": isinstance,
            "len": len,
            "list": list,
            "max": max,
            "min": min,
            "next": next,
            "range": range,
            "reversed": reversed,
            "round": round,
            "set": set,
            "slice": slice,
            "sorted": sorted,
            "str": str,
            "sum": sum,
            "tuple": tuple,
            "ValueError": ValueError,
            "RuntimeError": RuntimeError,
            "zip": zip,
        }
        globals_dict = {
            "__builtins__": safe_builtins,
            "h5py": h5py,
            "handle": handle,
            "matplotlib": matplotlib,
            "midpoints_if_edges": self._midpoints_if_edges,
            "np": np,
            "output_h5": str(output_h5),
            "plt": plt,
            "read_value_array": self._extract_value_array,
            "value_label": self._value_label,
        }
        locals_dict: dict[str, Any] = {}

        try:
            exec(compile(code, "<visualization>", "exec"), globals_dict, locals_dict)
        except Exception as exc:
            self._close_new_figures(existing_figures)
            if isinstance(exc, ValueError):
                raise
            raise RuntimeError(f"Generated visualization code failed: {exc}") from exc

        fig = locals_dict.get("fig") or globals_dict.get("fig")
        if fig is None:
            ax = locals_dict.get("ax") or globals_dict.get("ax")
            if ax is not None and hasattr(ax, "figure"):
                fig = ax.figure
        if fig is None:
            new_figure_numbers = [num for num in plt.get_fignums() if num not in existing_figures]
            if new_figure_numbers:
                fig = plt.figure(new_figure_numbers[-1])
        if fig is None or not hasattr(fig, "savefig"):
            self._close_new_figures(existing_figures)
            raise RuntimeError("Generated visualization code did not create a matplotlib figure.")
        return fig

    @staticmethod
    def _close_new_figures(existing_figures: set[int]) -> None:
        for figure_number in list(plt.get_fignums()):
            if figure_number not in existing_figures:
                plt.close(figure_number)

    @staticmethod
    def _default_request(summary: dict[str, Any]) -> str:
        if summary.get("has_eigenvalue"):
            return "Create a sensible default plot that highlights the k-effective convergence."
        if summary.get("has_tallies"):
            return "Create a sensible default plot that highlights the most informative tally result."
        if summary.get("has_runtime"):
            return "Create a sensible default runtime breakdown plot."
        return "Create a sensible default visualization for this MCDC output."

    @staticmethod
    def _default_plot_path(output_h5: Path, program: VisualizationProgram) -> Path:
        return output_h5.with_name("visualization.png")

    @staticmethod
    def _extract_value_array(score_group: h5py.Group, value_kind: str):
        mean = np.asarray(score_group["mean"][()])
        sdev = np.asarray(score_group["sdev"][()]) if "sdev" in score_group else None
        if value_kind == "mean" or sdev is None:
            return mean
        if value_kind == "sdev":
            return sdev
        if value_kind == "rel_err":
            denom = np.where(mean == 0, np.nan, mean)
            return np.nan_to_num(sdev / denom)
        return mean

    @staticmethod
    def _midpoints_if_edges(grid: np.ndarray, target_length: int) -> np.ndarray:
        if grid.ndim == 1 and len(grid) == target_length + 1:
            return 0.5 * (grid[1:] + grid[:-1])
        return grid

    @staticmethod
    def _value_label(value_kind: str) -> str:
        return {
            "mean": "mean",
            "sdev": "standard deviation",
            "rel_err": "relative error",
        }.get(value_kind, value_kind)

    def _get_llm(self):
        if self._llm is None:
            self._llm = load_llm(
                temperature=self.config.temperature,
                model=self.config.model,
                provider=self.config.provider,
            )
        return self._llm
