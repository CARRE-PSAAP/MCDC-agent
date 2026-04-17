from dataclasses import dataclass

from mcdc_agent.v2.config import LEARN_MENU
from mcdc_agent.v2.lessons import CONCEPT_LESSONS


@dataclass(frozen=True, slots=True)
class OnboardingTopic:
    key: str
    label: str
    lesson_id: str


class OnboardingService:
    """Simple lesson-oriented onboarding service for the v2 interactive app."""

    TOPICS = [
        OnboardingTopic("1", "Materials", "material"),
        OnboardingTopic("2", "Surfaces and Regions", "surface"),
        OnboardingTopic("3", "Cells", "cell"),
        OnboardingTopic("4", "Universes and Lattices", "hierarchy"),
        OnboardingTopic("5", "Sources", "source"),
        OnboardingTopic("6", "Tallies", "tally"),
        OnboardingTopic("7", "Settings and Run", "settings"),
        OnboardingTopic("8", "Example Walkthrough", "example_walkthrough"),
    ]

    def list_topics(self) -> list[OnboardingTopic]:
        return list(self.TOPICS)

    def get_topic(self, key_or_label: str) -> OnboardingTopic:
        query = str(key_or_label).strip().lower()
        for topic in self.TOPICS:
            if query in {topic.key.lower(), topic.label.lower(), topic.lesson_id.lower()}:
                return topic
        raise KeyError(f"Unknown onboarding topic: {key_or_label}")

    def get_lesson(self, key_or_label: str) -> dict[str, str | list[str]]:
        topic = self.get_topic(key_or_label)
        if topic.lesson_id == "example_walkthrough":
            return self._example_walkthrough_lesson()
        lesson = CONCEPT_LESSONS[topic.lesson_id]
        return {
            "title": topic.label,
            "concept": lesson.get("concept", ""),
            "syntax": lesson.get("syntax", ""),
            "parts": lesson.get("parts", ""),
            "tips": lesson.get("tips", []),
        }

    def format_lesson(self, key_or_label: str) -> str:
        lesson = self.get_lesson(key_or_label)
        lines = [f"# {lesson['title']}"]

        concept = str(lesson.get("concept", "")).strip()
        if concept:
            lines.extend(["", concept])

        syntax = str(lesson.get("syntax", "")).strip()
        if syntax:
            lines.extend(["", "## Syntax", "```python", syntax, "```"])

        parts = str(lesson.get("parts", "")).strip()
        if parts:
            lines.extend(["", "## Parameters and Notes", parts])

        tips = lesson.get("tips", [])
        if tips:
            lines.append("")
            lines.append("## Tips")
            for tip in tips:
                lines.append(f"- {tip}")

        return "\n".join(lines).strip()

    def menu_options(self) -> list[tuple[str, str]]:
        return list(LEARN_MENU)

    @staticmethod
    def _example_walkthrough_lesson() -> dict[str, str | list[str]]:
        return {
            "title": "Example Walkthrough",
            "concept": (
                "A good beginner workflow is to build an MCDC script in this order: "
                "materials, surfaces, cells, hierarchy if needed, source, tally, settings, then run. "
                "This mirrors how the geometry and physics depend on each other."
            ),
            "syntax": """import numpy as np
import mcdc

fuel = mcdc.MaterialMG(capture=np.array([0.2]), scatter=np.array([[0.7]]))
left = mcdc.Surface.PlaneX(x=0.0, boundary_condition="vacuum")
right = mcdc.Surface.PlaneX(x=5.0, boundary_condition="vacuum")
cell = mcdc.Cell(region=+left & -right, fill=fuel)

mcdc.Source(x=[0.0, 5.0], isotropic=True, energy_group=0)
mcdc.TallyCell(cell=cell, scores=["flux"])
mcdc.settings.N_particle = 1000
mcdc.settings.N_batch = 2
mcdc.run()""",
            "parts": (
                "Start with the physics definition, then make the geometry finite with surfaces and cells. "
                "After that, add a source that is actually inside the geometry, add the tally you care about, "
                "set conservative settings, and run."
            ),
            "tips": [
                "Check that every source location lies inside a material-filled cell.",
                "For fixed-source problems, prefer non-multiplying multigroup materials unless the prompt explicitly asks for fission.",
                "Use tallies to answer a specific question, not just because they are available.",
            ],
        }
