import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import matplotlib
import numpy as np

from mcdc_agent.v2.config import AppConfig
from mcdc_agent.v2.llm import load_llm

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True, slots=True)
class VisualizationSpec:
    kind: str
    tally: str = ""
    score: str = ""
    value: str = "mean"
    title: str = ""


class VisualizationService:
    """LLM-assisted plot planning with deterministic matplotlib rendering."""

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
    ) -> tuple[Path, VisualizationSpec]:
        spec = self.plan_plot(summary, request=request, script_text=script_text)
        plot_path = self.render_plot(output_h5, spec, output_path=output_path)
        return plot_path, spec

    def plan_plot(
        self,
        summary: dict[str, Any],
        *,
        request: str = "",
        script_text: str = "",
    ) -> VisualizationSpec:
        if request.strip():
            spec = self._plan_with_llm(summary, request=request, script_text=script_text)
            if not spec:
                raise RuntimeError("Visualization planning failed; the model did not return a valid plot specification.")
            return spec
        return self._fallback_spec(summary, request=request)

    def render_plot(
        self,
        output_h5: str | Path,
        spec: VisualizationSpec,
        *,
        output_path: str | Path | None = None,
    ) -> Path:
        output_h5 = Path(output_h5).resolve()
        if not output_h5.exists():
            raise FileNotFoundError(f"Output file not found: {output_h5}")

        plot_path = Path(output_path).resolve() if output_path else self._default_plot_path(output_h5, spec)
        plot_path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(output_h5, "r") as handle:
            if spec.kind == "eigenvalue_convergence":
                fig = self._plot_eigenvalue(handle, spec)
            elif spec.kind == "runtime_breakdown":
                fig = self._plot_runtime(handle, spec)
            elif spec.kind == "mesh_heatmap":
                fig = self._plot_mesh_heatmap(handle, spec)
            elif spec.kind == "tally_line":
                fig = self._plot_tally_line(handle, spec)
            elif spec.kind == "tally_bar":
                fig = self._plot_tally_bar(handle, spec)
            else:
                raise ValueError(f"Unsupported visualization kind: {spec.kind}")

        fig.savefig(plot_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        return plot_path

    def _plan_with_llm(
        self,
        summary: dict[str, Any],
        *,
        request: str,
        script_text: str,
    ) -> VisualizationSpec | None:
        try:
            llm = self._get_llm()
        except Exception:
            return None

        supported = {
            "supported_kinds": [
                "eigenvalue_convergence",
                "runtime_breakdown",
                "mesh_heatmap",
                "tally_line",
                "tally_bar",
            ],
            "allowed_values": ["mean", "sdev", "rel_err"],
        }
        prompt = (
            "You are planning a plot for MCDC output data.\n"
            "Return ONLY a JSON object with keys: kind, tally, score, value, title.\n"
            "Choose ONLY from the supported kinds and use only tally/score names that appear in the summary.\n"
            "If the request is about scalar tally values, choose tally_bar.\n"
            "If it is about a mesh tally map, choose mesh_heatmap.\n"
            "If it is about time/energy trends and the data is 1D, choose tally_line.\n"
            "If it is about k-effective convergence, choose eigenvalue_convergence.\n"
            "If it is about runtime, choose runtime_breakdown.\n\n"
            "Example 1:\n"
            '{"kind":"eigenvalue_convergence","tally":"","score":"","value":"mean","title":"k-effective convergence"}\n'
            "Example 2:\n"
            '{"kind":"mesh_heatmap","tally":"mesh_tally_0","score":"flux","value":"mean","title":"Mesh flux heatmap"}\n'
            "Example 3:\n"
            '{"kind":"tally_bar","tally":"surface_tally_0","score":"","value":"mean","title":"Surface tally values"}\n\n'
            f"Supported options:\n{json.dumps(supported, indent=2)}\n\n"
            f"Output summary:\n{json.dumps(summary, indent=2)}\n\n"
            f"Current script:\n```python\n{script_text[:4000]}\n```\n\n"
            f"User request:\n{request}\n"
        )
        response = llm.invoke(
            {
                "messages": [{"role": "user", "content": prompt}],
                "response_format": {"type": "json_object"},
            }
        )
        content = str(response.get("content", "") if isinstance(response, dict) else response).strip()
        try:
            spec_data = json.loads(content)
        except json.JSONDecodeError:
            return None
        return self._validate_spec(summary, spec_data)

    def _fallback_spec(self, summary: dict[str, Any], *, request: str = "") -> VisualizationSpec:
        request_lower = request.lower()
        tallies = summary.get("tallies", [])

        if summary.get("has_eigenvalue") and any(word in request_lower for word in ("k", "eigen", "convergence")):
            return VisualizationSpec(kind="eigenvalue_convergence", title="k-effective convergence")

        if any(word in request_lower for word in ("runtime", "timing", "performance")) and summary.get("has_runtime"):
            return VisualizationSpec(kind="runtime_breakdown", title="Runtime breakdown")

        tally = self._choose_tally(tallies, request_lower)
        if tally:
            score = self._choose_score(tally, request_lower)
            shape = tally.get("scores", {}).get(score, {}).get("mean_shape", []) if score else []
            if tally.get("is_mesh") and len(shape) >= 2:
                return VisualizationSpec(
                    kind="mesh_heatmap",
                    tally=tally["name"],
                    score=score,
                    value="rel_err" if "uncert" in request_lower or "error" in request_lower else "mean",
                    title=f"{score} heatmap for {tally['name']}",
                )
            if len(shape) == 1:
                return VisualizationSpec(
                    kind="tally_line",
                    tally=tally["name"],
                    score=score,
                    value="rel_err" if "uncert" in request_lower or "error" in request_lower else "mean",
                    title=f"{score} line plot for {tally['name']}",
                )
            return VisualizationSpec(
                kind="tally_bar",
                tally=tally["name"],
                value="mean",
                title=f"Scalar scores for {tally['name']}",
            )

        if summary.get("has_eigenvalue"):
            return VisualizationSpec(kind="eigenvalue_convergence", title="k-effective convergence")
        if summary.get("has_runtime"):
            return VisualizationSpec(kind="runtime_breakdown", title="Runtime breakdown")
        raise ValueError("No supported plot could be inferred from the output summary.")

    def _validate_spec(self, summary: dict[str, Any], spec_data: dict[str, Any]) -> VisualizationSpec | None:
        kind = str(spec_data.get("kind", "")).strip()
        if kind not in {"eigenvalue_convergence", "runtime_breakdown", "mesh_heatmap", "tally_line", "tally_bar"}:
            return None

        spec = VisualizationSpec(
            kind=kind,
            tally=str(spec_data.get("tally", "")).strip(),
            score=str(spec_data.get("score", "")).strip(),
            value=str(spec_data.get("value", "mean")).strip() or "mean",
            title=str(spec_data.get("title", "")).strip(),
        )

        if spec.kind in {"eigenvalue_convergence", "runtime_breakdown"}:
            return spec

        tally = next((item for item in summary.get("tallies", []) if item["name"] == spec.tally), None)
        if not tally:
            return None

        if spec.kind in {"mesh_heatmap", "tally_line"}:
            if spec.score not in tally.get("scores", {}):
                return None
        return spec

    @staticmethod
    def _choose_tally(tallies: list[dict[str, Any]], request_lower: str) -> dict[str, Any] | None:
        if not tallies:
            return None
        for tally in tallies:
            if tally["name"].lower() in request_lower:
                return tally
        if "surface" in request_lower:
            for tally in tallies:
                if tally["name"].startswith("surface_tally"):
                    return tally
        if "mesh" in request_lower:
            for tally in tallies:
                if tally.get("is_mesh"):
                    return tally
        return tallies[0]

    @staticmethod
    def _choose_score(tally: dict[str, Any], request_lower: str) -> str:
        score_names = list(tally.get("scores", {}).keys())
        for score in score_names:
            if score.lower() in request_lower:
                return score
        if score_names:
            return score_names[0]
        return ""

    @staticmethod
    def _default_plot_path(output_h5: Path, spec: VisualizationSpec) -> Path:
        stem = output_h5.stem
        suffix_parts = [spec.kind]
        if spec.tally:
            suffix_parts.append(spec.tally)
        if spec.score:
            suffix_parts.append(spec.score)
        safe = "_".join(re.sub(r"[^a-zA-Z0-9_]+", "_", part) for part in suffix_parts if part)
        return output_h5.with_name(f"{stem}_{safe}.png")

    def _plot_eigenvalue(self, handle: h5py.File, spec: VisualizationSpec):
        if "k_cycle" not in handle:
            raise ValueError("This output file does not contain k_cycle data.")
        k_cycle = np.asarray(handle["k_cycle"][:])
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(np.arange(1, len(k_cycle) + 1), k_cycle, marker="o")
        ax.set_xlabel("Cycle")
        ax.set_ylabel("k-effective")
        ax.set_title(spec.title or "k-effective convergence")
        ax.grid(alpha=0.3)
        return fig

    def _plot_runtime(self, handle: h5py.File, spec: VisualizationSpec):
        if "runtime" not in handle:
            raise ValueError("This output file does not contain runtime data.")
        names = sorted(handle["runtime"].keys())
        values = [float(np.asarray(handle["runtime"][name][()]).reshape(-1)[0]) for name in names]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(names, values)
        ax.set_ylabel("Seconds")
        ax.set_title(spec.title or "Runtime breakdown")
        ax.tick_params(axis="x", rotation=25)
        return fig

    def _plot_mesh_heatmap(self, handle: h5py.File, spec: VisualizationSpec):
        data, x_axis, y_axis, axis_labels = self._extract_mesh_plane(handle, spec)
        fig, ax = plt.subplots(figsize=(6, 5))
        mesh = ax.pcolormesh(x_axis, y_axis, data.T, shading="auto")
        fig.colorbar(mesh, ax=ax, label=self._value_label(spec))
        ax.set_xlabel(axis_labels[0])
        ax.set_ylabel(axis_labels[1])
        ax.set_title(spec.title or f"{spec.score} heatmap")
        return fig

    def _plot_tally_line(self, handle: h5py.File, spec: VisualizationSpec):
        base = handle[f"tallies/{spec.tally}"]
        score_group = base[spec.score]
        y = self._extract_value_array(score_group, spec.value)
        grid_name = self._choose_line_grid(base)
        if not grid_name:
            x = np.arange(len(y))
            xlabel = "Index"
        else:
            grid = np.asarray(base["grid"][grid_name][:])
            x = self._midpoints_if_edges(grid, len(y))
            xlabel = grid_name
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(x, y, marker="o")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(self._value_label(spec))
        ax.set_title(spec.title or f"{spec.score} line plot")
        ax.grid(alpha=0.3)
        return fig

    def _plot_tally_bar(self, handle: h5py.File, spec: VisualizationSpec):
        base = handle[f"tallies/{spec.tally}"]
        score_names = [name for name in sorted(base.keys()) if name != "grid"]
        values = []
        for score_name in score_names:
            score_group = base[score_name]
            value = self._extract_value_array(score_group, "mean")
            values.append(float(np.asarray(value).reshape(-1)[0]))
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(score_names, values)
        ax.set_ylabel("mean")
        ax.set_title(spec.title or f"Scalar scores for {spec.tally}")
        return fig

    def _extract_mesh_plane(self, handle: h5py.File, spec: VisualizationSpec):
        base = handle[f"tallies/{spec.tally}"]
        score_group = base[spec.score]
        data = np.asarray(self._extract_value_array(score_group, spec.value))
        if data.ndim < 2:
            raise ValueError("Mesh heatmap requires at least 2D data.")
        if data.ndim > 2:
            data = data[..., data.shape[-1] // 2]

        grids = base["grid"]
        axes = [(axis, np.asarray(grids[axis][:])) for axis in ("x", "y", "z") if axis in grids]
        if len(axes) < 2:
            raise ValueError("Mesh heatmap requires at least two spatial grid axes.")
        x_name, x_grid = axes[0]
        y_name, y_grid = axes[1]
        return data, x_grid, y_grid, (x_name, y_name)

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
    def _choose_line_grid(base: h5py.Group) -> str:
        if "grid" not in base:
            return ""
        for grid_name in ("time", "energy", "mu", "azi", "x", "y", "z"):
            if grid_name in base["grid"]:
                return grid_name
        return ""

    @staticmethod
    def _midpoints_if_edges(grid: np.ndarray, target_length: int) -> np.ndarray:
        if grid.ndim == 1 and len(grid) == target_length + 1:
            return 0.5 * (grid[1:] + grid[:-1])
        return grid

    @staticmethod
    def _value_label(spec: VisualizationSpec) -> str:
        return {
            "mean": "mean",
            "sdev": "standard deviation",
            "rel_err": "relative error",
        }.get(spec.value, spec.value)

    def _get_llm(self):
        if self._llm is None:
            self._llm = load_llm(
                temperature=self.config.temperature,
                model=self.config.model,
                provider=self.config.provider,
            )
        return self._llm
