import csv
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class MaterialMatch:
    query: str
    kind: str
    name: str
    summary: str
    code_snippet: str = ""


class MaterialCatalog:
    """Small, dependency-light material and isotope lookup service."""

    def __init__(
        self,
        material_properties_path: Path | None = None,
        nuclear_data_path: Path | None = None,
    ):
        base = Path(__file__).resolve().parents[2] / "onboarding"
        self.material_properties_path = material_properties_path or base / "material_properties.csv"
        self.nuclear_data_path = nuclear_data_path or base / "nuclear_data.csv"
        self.materials = self._load_csv(self.material_properties_path)
        self.isotopes = self._load_csv(self.nuclear_data_path)

    @staticmethod
    def _load_csv(path: Path) -> list[dict[str, str]]:
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    def lookup(self, query: str) -> MaterialMatch | None:
        return self.lookup_material(query) or self.lookup_isotope(query)

    def lookup_material(self, query: str) -> MaterialMatch | None:
        query_lower = query.lower().strip()
        for row in self.materials:
            aliases = [alias.strip() for alias in row.get("aliases", "").lower().split(";") if alias.strip()]
            if query_lower == row["name"].lower() or query_lower in aliases:
                return self._build_material_match(query, row)
        return None

    def lookup_isotope(self, query: str) -> MaterialMatch | None:
        query_lower = query.lower().strip()
        exact = [row for row in self.isotopes if row["Isotope"].lower() == query_lower]
        if exact:
            row = exact[0]
            summary = (
                f"Isotope: {row['Isotope']}\n"
                f"Element: {row['Element']}\n"
                f"Atomic Mass: {row['Mass_u']} u\n"
                f"Natural Abundance: {float(row['Abundance']):.4%}"
            )
            return MaterialMatch(query=query, kind="isotope", name=row["Isotope"], summary=summary)

        element = [row for row in self.isotopes if row["Element"].lower() == query_lower]
        if element:
            lines = [f"Element: {element[0]['Element']}"]
            for row in element:
                dominant = " (dominant)" if row.get("Dominant") == "1" else ""
                lines.append(
                    f"- {row['Isotope']}: {row['Mass_u']} u, {float(row['Abundance']):.4%}{dominant}"
                )
            return MaterialMatch(query=query, kind="element", name=element[0]["Element"], summary="\n".join(lines))
        return None

    def format_context(self, query: str) -> str:
        match = self.lookup(query)
        if not match:
            return ""
        if match.code_snippet:
            return f"{match.summary}\n\n```python\n{match.code_snippet}\n```"
        return match.summary

    def _build_material_match(self, query: str, row: dict[str, str]) -> MaterialMatch:
        composition = self._parse_formula(row["formula"])
        mean_molar_mass = 0.0
        for isotope, fraction in composition.items():
            mean_molar_mass += fraction * self._get_isotope_mass(isotope)

        total_number_density = 0.0
        if mean_molar_mass > 0.0:
            total_number_density = (float(row["density"]) * 0.602214076) / mean_molar_mass

        nuclides = {}
        for isotope, fraction in composition.items():
            nuclides[isotope] = fraction * total_number_density

        lines = [
            f"Material: {row['name']}",
            f"Formula: {row['formula']}",
            f"Density: {row['density']} g/cm^3",
        ]
        if nuclides:
            lines.append("Continuous-energy nuclide composition available.")

        code_lines = [f"{row['name']} = mcdc.Material(", "    nuclide_composition={"]
        for isotope, density in nuclides.items():
            code_lines.append(f"        '{isotope}': {density:.6e},")
        code_lines.extend(["    },", ")", ""])

        return MaterialMatch(
            query=query,
            kind="material",
            name=row["name"],
            summary="\n".join(lines),
            code_snippet="\n".join(code_lines) if nuclides else "",
        )

    def _get_isotope_mass(self, isotope: str) -> float:
        for row in self.isotopes:
            if row["Isotope"] == isotope:
                return float(row["Mass_u"])
        return 0.0

    def _parse_formula(self, formula: str) -> dict[str, float]:
        matches = re.findall(r"([A-Z][a-z]?)(\d*\.?\d*)", formula)
        if not matches:
            return {}

        composition: dict[str, float] = {}
        total = 0.0
        for element, count_text in matches:
            count = float(count_text) if count_text else 1.0
            dominant_rows = [
                row for row in self.isotopes
                if row["Element"] == element and row.get("Dominant") == "1"
            ]
            for row in dominant_rows:
                abundance = float(row["Abundance"])
                composition[row["Isotope"]] = composition.get(row["Isotope"], 0.0) + count * abundance
                total += count * abundance

        if total <= 0.0:
            return {}
        return {isotope: value / total for isotope, value in composition.items()}
