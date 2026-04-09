from pathlib import Path
from typing import Any

import h5py

from mcdc_agent.mcdc.tools.api_reference import APIReference
from mcdc_agent.v2.config import AppConfig
from mcdc_agent.v2.llm import load_llm


class DiagnosticsService:
    """Deterministic parser and formatter for MCDC HDF5 output files."""

    def __init__(self, config: AppConfig | None = None):
        self.config = config or AppConfig()
        self.api_reference = APIReference()
        self._llm = None

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

    def analyze_output(
        self,
        summary: dict[str, Any],
        *,
        script_text: str = "",
        stdout: str = "",
        stderr: str = "",
        user_question: str = "",
    ) -> str:
        findings = self._deterministic_findings(summary, stdout=stdout, stderr=stderr)
        findings_text = "\n".join(f"- {finding}" for finding in findings) if findings else "- No obvious deterministic issues detected."

        relevant_api = self.api_reference.get_relevant_sections(
            "\n".join(filter(None, [user_question, stderr, stdout, script_text[:2000]]))
        )
        if not relevant_api.strip():
            relevant_api = self.api_reference.get_full_text()[:12000]

        context = self.build_analysis_context(
            summary,
            script_text=script_text,
            stdout=stdout,
            stderr=stderr,
        )

        prompt = (
            "You are helping diagnose an MCDC simulation run.\n"
            "Use the deterministic findings as primary evidence.\n"
            "Use the script and API reference to identify likely issues in the setup.\n"
            "Separate confirmed findings from suspected issues.\n"
            "Be concise and practical.\n\n"
            f"User question:\n{user_question or 'Does anything look wrong or noteworthy?'}\n\n"
            f"Deterministic findings:\n{findings_text}\n\n"
            f"Run/output context:\n{context}\n\n"
            f"Relevant API reference:\n{relevant_api}\n\n"
            "Return a short diagnosis with these headings:\n"
            "Status\nConfirmed findings\nLikely issues\nSuggested next steps\n"
        )

        llm = self._get_llm()
        response = llm.invoke({"messages": [{"role": "user", "content": prompt}]})
        if isinstance(response, dict):
            text = str(response.get("content", "")).strip()
        else:
            text = str(response).strip()
        if not text:
            raise RuntimeError("Diagnosis generation returned an empty response.")
        return text

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

    def _deterministic_findings(
        self,
        summary: dict[str, Any],
        *,
        stdout: str = "",
        stderr: str = "",
    ) -> list[str]:
        findings: list[str] = []

        if not summary.get("has_tallies"):
            findings.append("No tally datasets were found in the output file.")

        tallies = summary.get("tallies", [])
        for tally in tallies:
            if not tally.get("scores"):
                findings.append(f"Tally {tally['name']} has no score datasets.")
                continue
            for score_name, score_info in tally["scores"].items():
                shape = tuple(score_info.get("mean_shape", []))
                findings.append(f"Tally {tally['name']} includes score {score_name} with mean shape {shape}.")

        if summary.get("has_eigenvalue"):
            eigen = summary.get("eigenvalue", {})
            findings.append(
                f"Eigenvalue data is present with k_mean={eigen.get('k_mean')} and k_sdev={eigen.get('k_sdev')}."
            )
        else:
            findings.append("This output looks like a fixed-source run; no eigenvalue datasets are present.")

        runtime = summary.get("runtime", {})
        if runtime:
            total = runtime.get("total")
            simulation = runtime.get("simulation")
            if isinstance(total, (int, float)) and isinstance(simulation, (int, float)) and total:
                share = simulation / total
                findings.append(f"Simulation time is {share:.1%} of total runtime.")

        stderr_text = (stderr or "").strip()
        if stderr_text:
            findings.append("stderr is non-empty and may indicate warnings or runtime problems.")

        stdout_text = (stdout or "").lower()
        if "particle is lost" in stdout_text or "particle is lost" in stderr_text.lower():
            findings.append("The run output mentions lost particles, which usually indicates a geometry gap or overlap.")
        if "bank" in stdout_text and "full" in stdout_text:
            findings.append("The run output mentions a full particle bank, which suggests particle bank overflow.")

        return findings

    def _get_llm(self):
        if self._llm is None:
            self._llm = load_llm(
                temperature=self.config.temperature,
                model=self.config.model,
                provider=self.config.provider,
            )
        return self._llm
