from pathlib import Path
from typing import Any

import h5py


class DiagnosticsService:
    """Deterministic parser and formatter for MCDC HDF5 output files."""

    def summarize_output(self, output_h5: str | Path) -> dict[str, Any]:
        output_path = Path(output_h5).resolve()
        if not output_path.exists():
            raise FileNotFoundError(f"Output file not found: {output_h5}")

        with h5py.File(output_path, "r") as handle:
            summary = {
                "path": str(output_path),
                "version": self._read_value(handle.get("version")),
                "settings": self._summarize_settings(handle),
                "tallies": self._summarize_tallies(handle),
                "eigenvalue": self._summarize_eigenvalue(handle),
                "runtime": self._summarize_runtime(handle),
                "particles_saved": "particles" in handle,
            }

        summary["has_tallies"] = bool(summary["tallies"])
        summary["has_runtime"] = bool(summary["runtime"])
        summary["has_eigenvalue"] = summary["eigenvalue"].get("present", False)
        return summary

    def format_summary(self, summary: dict[str, Any]) -> str:
        lines = [
            f"Output file: {summary.get('path', '')}",
        ]

        version = summary.get("version")
        if version not in (None, ""):
            lines.append(f"MCDC version: {version}")

        settings = summary.get("settings", {})
        if settings:
            lines.append("Settings:")
            for key in ["output_name", "N_particle", "N_batch", "N_cycle", "eigenvalue_mode"]:
                if key in settings:
                    lines.append(f"  {key}: {settings[key]}")

        tallies = summary.get("tallies", [])
        lines.append(f"Tallies found: {len(tallies)}")
        for tally in tallies:
            grids = ", ".join(tally.get("grids", [])) or "none"
            score_text = ", ".join(
                f"{name} {tuple(info.get('mean_shape', []))}"
                for name, info in tally.get("scores", {}).items()
            ) or "none"
            kind = "mesh" if tally.get("is_mesh") else "non-mesh"
            lines.append(f"  - {tally['name']} ({kind}; grids: {grids}; scores: {score_text})")

        eigenvalue = summary.get("eigenvalue", {})
        if eigenvalue.get("present"):
            lines.append("Eigenvalue data:")
            for key in ["k_mean", "k_sdev", "k_cycle_length"]:
                if key in eigenvalue:
                    lines.append(f"  {key}: {eigenvalue[key]}")

        runtime = summary.get("runtime", {})
        if runtime:
            lines.append("Runtime:")
            for key, value in runtime.items():
                lines.append(f"  {key}: {value}")

        lines.append(f"Particles saved: {summary.get('particles_saved', False)}")
        return "\n".join(lines)

    def build_analysis_context(
        self,
        summary: dict[str, Any],
        *,
        script_text: str = "",
        stdout: str = "",
        stderr: str = "",
    ) -> str:
        parts = [self.format_summary(summary)]
        if script_text.strip():
            parts.append("Current script:\n```python\n" + script_text.strip() + "\n```")
        if stdout.strip():
            parts.append("Run stdout:\n```text\n" + stdout.strip() + "\n```")
        if stderr.strip():
            parts.append("Run stderr:\n```text\n" + stderr.strip() + "\n```")
        return "\n\n".join(parts)

    def _summarize_settings(self, handle: h5py.File) -> dict[str, Any]:
        settings = {}
        if "settings" not in handle:
            return settings

        group = handle["settings"]
        for key in ["output_name", "N_particle", "N_batch", "N_cycle", "eigenvalue_mode"]:
            if key in group:
                settings[key] = self._read_value(group[key])
        return settings

    def _summarize_tallies(self, handle: h5py.File) -> list[dict[str, Any]]:
        if "tallies" not in handle:
            return []

        tallies = []
        tally_group = handle["tallies"]
        for tally_name in sorted(tally_group.keys()):
            group = tally_group[tally_name]
            grids = []
            if "grid" in group:
                grids = sorted(group["grid"].keys())

            scores = {}
            for child_name in sorted(group.keys()):
                if child_name == "grid":
                    continue
                score_group = group[child_name]
                if not isinstance(score_group, h5py.Group):
                    continue
                score_info: dict[str, Any] = {}
                if "mean" in score_group:
                    score_info["mean_shape"] = list(score_group["mean"].shape)
                if "sdev" in score_group:
                    score_info["sdev_shape"] = list(score_group["sdev"].shape)
                if score_info:
                    scores[child_name] = score_info

            tallies.append(
                {
                    "name": tally_name,
                    "grids": grids,
                    "is_mesh": any(axis in grids for axis in ("x", "y", "z")),
                    "scores": scores,
                }
            )
        return tallies

    def _summarize_eigenvalue(self, handle: h5py.File) -> dict[str, Any]:
        summary: dict[str, Any] = {"present": False}

        if "k_cycle" in handle:
            summary["present"] = True
            summary["k_cycle_length"] = int(handle["k_cycle"].shape[0])
        if "k_mean" in handle:
            summary["present"] = True
            summary["k_mean"] = self._read_value(handle["k_mean"])
        if "k_sdev" in handle:
            summary["present"] = True
            summary["k_sdev"] = self._read_value(handle["k_sdev"])
        if "gyration_radius" in handle:
            summary["gyration_radius_length"] = int(handle["gyration_radius"].shape[0])

        return summary

    def _summarize_runtime(self, handle: h5py.File) -> dict[str, Any]:
        runtime = {}
        if "runtime" not in handle:
            return runtime

        group = handle["runtime"]
        for key in sorted(group.keys()):
            runtime[key] = self._read_value(group[key])
        return runtime

    @staticmethod
    def _read_value(dataset: h5py.Dataset | None) -> Any:
        if dataset is None:
            return None

        value = dataset[()]
        if hasattr(value, "tolist"):
            value = value.tolist()
        if isinstance(value, bytes):
            return value.decode()
        if isinstance(value, list) and len(value) == 1:
            item = value[0]
            if isinstance(item, bytes):
                return item.decode()
            return item
        return value
