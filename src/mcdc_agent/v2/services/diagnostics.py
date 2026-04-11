from pathlib import Path
from typing import Any

import h5py

from mcdc_agent.mcdc.tools.api_reference import APIReference
from mcdc_agent.mcdc.utils import extract_code
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
        summary["output_present"] = True
        return summary

    def summarize_run_context(
        self,
        *,
        output_h5: str | Path | None = None,
        returncode: int | None = None,
        stdout: str = "",
        stderr: str = "",
    ) -> dict[str, Any]:
        if output_h5:
            path = Path(output_h5).resolve()
            if path.exists():
                summary = self.summarize_output(path)
                summary["returncode"] = returncode
                return summary

        return {
            "path": "",
            "version": None,
            "settings": {},
            "tallies": [],
            "eigenvalue": {"present": False},
            "runtime": {},
            "particles_saved": False,
            "has_tallies": False,
            "has_runtime": False,
            "has_eigenvalue": False,
            "output_present": False,
            "returncode": returncode,
            "stdout_present": bool(stdout.strip()),
            "stderr_present": bool(stderr.strip()),
        }

    def format_summary(self, summary: dict[str, Any]) -> str:
        output_present = summary.get("output_present", bool(summary.get("path")))
        lines = []
        if output_present:
            lines.append(f"Output file: {summary.get('path', '')}")
        else:
            lines.append("Output file: not produced")

        if summary.get("returncode") is not None:
            lines.append(f"Return code: {summary.get('returncode')}")

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
        findings = self._deterministic_findings(
            summary,
            script_text=script_text,
            stdout=stdout,
            stderr=stderr,
        )
        findings_text = "\n".join(f"- {finding}" for finding in findings) if findings else "- No obvious deterministic issues detected."
        script_evidence = self._build_script_evidence(script_text)
        script_evidence_text = script_evidence or "No script evidence was available."

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
            "You are helping diagnose an MCDC simulation run to find possible issues.\n"
            "Use the deterministic findings as primary evidence.\n"
            "Use the script and API reference to identify likely issues in the setup.\n"
            "Separate confirmed findings from suspected issues.\n"
            "When there is a problem, refer explicitly to relevant script lines/snippets and the API context.\n"
            "Be concise and practical.\n\n"
            f"User question:\n{user_question or 'Does anything look wrong or noteworthy?'}\n\n"
            f"Deterministic findings:\n{findings_text}\n\n"
            f"Relevant script evidence:\n{script_evidence_text}\n\n"
            f"Run/output context:\n{context}\n\n"
            f"Relevant API reference:\n{relevant_api}\n\n"
            "Return a short diagnosis with these headings:\n"
            "Status\nConfirmed findings\nRelevant script/API evidence\nLikely issues\nSuggested next steps\n"
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

    def suggest_script_fix(
        self,
        summary: dict[str, Any],
        *,
        script_text: str,
        stdout: str = "",
        stderr: str = "",
        user_question: str = "",
        analysis_text: str = "",
        original_prompt: str = "",
    ) -> str:
        if not script_text.strip():
            raise ValueError("No current script is available to fix.")

        relevant_api = self.api_reference.get_relevant_sections(
            "\n".join(filter(None, [user_question, stderr, stdout, script_text[:2000], analysis_text]))
        )
        if not relevant_api.strip():
            relevant_api = self.api_reference.get_full_text()[:12000]

        findings = self._deterministic_findings(
            summary,
            script_text=script_text,
            stdout=stdout,
            stderr=stderr,
        )
        findings_text = "\n".join(f"- {finding}" for finding in findings) if findings else "- No obvious deterministic issues detected."
        script_evidence = self._build_script_evidence(script_text) or "No script evidence was available."
        context = self.build_analysis_context(
            summary,
            script_text=script_text,
            stdout=stdout,
            stderr=stderr,
        )

        prompt = (
            "You are fixing an MCDC script after inspecting its output and diagnostics.\n"
            "Make the minimum necessary changes to improve the script.\n"
            "Preserve working structure and keep the script style simple.\n"
            "When there is a clear problem, align the fix with the API reference.\n"
            "If the original prompt requested continuous-energy materials, preserve CE usage.\n\n"
            f"Original prompt:\n{original_prompt or 'Not available.'}\n\n"
            f"User request for diagnosis:\n{user_question or 'General analysis'}\n\n"
            f"Deterministic findings:\n{findings_text}\n\n"
            f"Diagnosis:\n{analysis_text or 'No prior diagnosis text provided.'}\n\n"
            f"Relevant script evidence:\n{script_evidence}\n\n"
            f"Run/output context:\n{context}\n\n"
            f"Relevant API reference:\n{relevant_api}\n\n"
            "Return the COMPLETE updated Python script in a ```python``` block."
        )

        llm = self._get_llm()
        response = llm.invoke({"messages": [{"role": "user", "content": prompt}]})
        if isinstance(response, dict):
            text = str(response.get("content", "")).strip()
        else:
            text = str(response).strip()
        if not text:
            raise RuntimeError("Fix generation returned an empty response.")

        fixed_script = extract_code(text).strip()
        if not fixed_script:
            raise RuntimeError("Fix generation did not include any script content.")
        return fixed_script + "\n"

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
        script_text: str = "",
        stdout: str = "",
        stderr: str = "",
    ) -> list[str]:
        findings: list[str] = []

        if not summary.get("output_present", bool(summary.get("path"))):
            findings.append("No output file was produced for this run.")
        if summary.get("returncode") not in (None, 0):
            findings.append(f"The run exited with return code {summary.get('returncode')}.")

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

        script_lower = script_text.lower()
        if script_lower:
            if "mcdc.material(" in script_lower and "energy_group=" in script_lower:
                findings.append("The script appears to use continuous-energy materials while also passing energy_group to a source.")
            if "set_eigenmode" in script_lower and not summary.get("has_eigenvalue"):
                findings.append("The script calls set_eigenmode, but the output does not contain eigenvalue datasets.")
            if "tally" in script_lower and not summary.get("has_tallies"):
                findings.append("The script defines tallies, but no tally datasets were found in the output.")

        return findings

    @staticmethod
    def _build_script_evidence(script_text: str) -> str:
        if not script_text.strip():
            return ""

        lines = script_text.splitlines()
        evidence = []
        interesting = (
            "mcdc.Material(",
            "mcdc.MaterialMG(",
            "mcdc.Source(",
            "mcdc.Tally",
            "mcdc.Mesh",
            "mcdc.settings.",
            "set_eigenmode",
            "set_root_universe",
            "mcdc.Cell(",
            "mcdc.Lattice(",
            "mcdc.Universe(",
        )

        for lineno, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped and any(token in stripped for token in interesting):
                evidence.append(f"L{lineno}: {stripped}")

        return "\n".join(evidence[:40])

    def _get_llm(self):
        if self._llm is None:
            self._llm = load_llm(
                temperature=self.config.temperature,
                model=self.config.model,
                provider=self.config.provider,
            )
        return self._llm
