import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from mcdc_agent.v2.config import AppConfig
from mcdc_agent.v2.lessons import CONCEPT_LESSONS
from mcdc_agent.v2.llm import load_llm


@dataclass(frozen=True, slots=True)
class QAHit:
    source: str
    category: str
    content: str
    score: float


class QAService:
    """Simple non-LangChain retrieval and Q&A over local docs, examples, and code."""

    DEFAULT_SCOPES = ("docs", "examples", "code")
    STOPWORDS = {
        "a", "an", "and", "are", "be", "by", "do", "does", "for", "how", "i",
        "in", "is", "it", "of", "or", "the", "to", "what", "when", "where",
        "which", "why", "with", "work", "works",
    }
    LESSON_TITLES = {
        "material": "Materials",
        "surface": "Surfaces and Regions",
        "cell": "Cells",
        "hierarchy": "Universes and Lattices",
        "source": "Sources",
        "tally": "Tallies",
        "settings": "Settings and Run",
    }

    def __init__(self, config: AppConfig | None = None):
        self.config = config or AppConfig()
        self._llm = None
        self._chunks: list[dict] | None = None

    def retrieve(self, question: str, *, scopes: Iterable[str] | None = None, top_k: int = 4) -> list[QAHit]:
        query_tokens = self._tokenize(question)
        query_text = question.lower().strip()
        hits = []
        for chunk in self._get_chunks():
            if scopes and chunk["category"] not in scopes:
                continue
            score = self._score_chunk(query_tokens, query_text, chunk)
            if score <= 0:
                continue
            hits.append(
                QAHit(
                    source=chunk["source"],
                    category=chunk["category"],
                    content=chunk["content"],
                    score=score,
                )
            )

        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:top_k]

    def answer(self, question: str, *, scopes: Iterable[str] | None = None, top_k: int = 4) -> str:
        hits = self.retrieve(question, scopes=scopes, top_k=top_k)
        if not hits:
            return "I could not find relevant local documentation or code for that question."

        llm = self._get_llm()
        context = self.format_hits(hits)
        prompt = (
            "You are MCDC-Agent. Answer the user's question using only the retrieved local context.\n"
            "Be concise, accurate, and practical. If the context is incomplete, say so.\n\n"
            f"Question:\n{question}\n\n"
            f"Retrieved context:\n{context}\n\n"
            "Answer:"
        )
        response = llm.invoke({"messages": [{"role": "user", "content": prompt}]})
        if isinstance(response, dict):
            return str(response.get("content", "")).strip()
        return str(response).strip()

    @staticmethod
    def format_hits(hits: list[QAHit]) -> str:
        parts = []
        for i, hit in enumerate(hits, 1):
            parts.append(f"[Hit {i}: {hit.source} | {hit.category}]\n{hit.content}")
        return "\n\n---\n\n".join(parts)

    def _get_llm(self):
        if self._llm is None:
            self._llm = load_llm(
                temperature=self.config.temperature,
                model=self.config.model,
                provider=self.config.provider,
            )
        return self._llm

    def _get_chunks(self) -> list[dict]:
        if self._chunks is None:
            self._chunks = self._build_chunks()
        return self._chunks

    def _build_chunks(self) -> list[dict]:
        base = Path(__file__).resolve().parents[2]
        chunks: list[dict] = []

        for lesson_id, lesson in CONCEPT_LESSONS.items():
            content_parts = [
                self.LESSON_TITLES.get(lesson_id, lesson_id),
                lesson_id,
                str(lesson.get("concept", "")),
                str(lesson.get("parts", "")),
                str(lesson.get("syntax", "")),
                "\n".join(str(tip) for tip in lesson.get("tips", [])),
            ]
            content = "\n\n".join(part.strip() for part in content_parts if part and str(part).strip())
            chunks.append(
                {
                    "source": f"concepts:{lesson_id}",
                    "category": "docs",
                    "content": content,
                }
            )

        chunk_specs = [
            ("docs", base / "mcdc" / "tools" / "mcdc_api_reference.md"),
            ("docs", base / "v2" / "lessons.py"),
        ]
        for category, path in chunk_specs:
            if path.exists():
                chunks.extend(self._chunk_file(path, category))

        examples_dirs = [
            base / "scraped_docs" / "examples",
            base / "mcdc" / "generators" / "examples",
        ]
        for directory in examples_dirs:
            if not directory.exists():
                continue
            for path in sorted(directory.glob("*.py")):
                chunks.extend(self._chunk_file(path, "examples"))

        code_files = [
            base / "mcdc" / "generators" / "small_model_generator.py",
            base / "mcdc" / "generators" / "small_model_prompts.py",
            base / "mcdc" / "tools" / "validator.py",
            base / "v2" / "services" / "generation.py",
            base / "v2" / "services" / "execution.py",
            base / "v2" / "services" / "diagnostics.py",
            base / "v2" / "services" / "onboarding.py",
            base / "v2" / "services" / "visualization.py",
        ]
        for path in code_files:
            if path.exists():
                chunks.extend(self._chunk_file(path, "code"))

        mcdc_output = Path("/home/gunna/MCDC/mcdc/output.py")
        if mcdc_output.exists():
            chunks.extend(self._chunk_file(mcdc_output, "code"))

        return chunks

    def _chunk_file(self, path: Path, category: str) -> list[dict]:
        text = path.read_text(encoding="utf-8")
        source = str(path)

        if path.suffix == ".py":
            return self._chunk_python_file(source, category, text)
        return self._chunk_text_file(source, category, text)

    @staticmethod
    def _chunk_text_file(source: str, category: str, text: str) -> list[dict]:
        chunks = []
        sections = re.split(r"\n(?=#+\s)", text)
        if len(sections) == 1:
            sections = [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]

        for section in sections:
            section = section.strip()
            if not section:
                continue
            if len(section) <= 1400:
                chunks.append({"source": source, "category": category, "content": section})
                continue
            start = 0
            while start < len(section):
                piece = section[start:start + 1400].strip()
                if piece:
                    chunks.append({"source": source, "category": category, "content": piece})
                start += 1100
        return chunks

    @staticmethod
    def _chunk_python_file(source: str, category: str, text: str) -> list[dict]:
        lines = text.splitlines()
        chunks = []
        window = 36
        step = 28
        for start in range(0, len(lines), step):
            block = "\n".join(lines[start:start + window]).strip()
            if not block:
                continue
            chunks.append(
                {
                    "source": f"{source}:{start + 1}",
                    "category": category,
                    "content": block,
                }
            )
            if start + window >= len(lines):
                break
        return chunks

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[a-zA-Z0-9_]+", text.lower())
            if len(token) >= 2 and token not in QAService.STOPWORDS
        }

    def _score_chunk(self, query_tokens: set[str], query_text: str, chunk: dict) -> float:
        content = chunk["content"].lower()
        source = chunk["source"].lower()
        content_tokens = self._tokenize(content)

        overlap = query_tokens & content_tokens
        if not overlap:
            return 0.0

        score = float(len(overlap))
        if query_text and query_text in content:
            score += 6.0

        for token in query_tokens:
            if token in source:
                score += 1.0

        if chunk["category"] == "docs":
            score += 0.5
        if str(chunk["source"]).startswith("concepts:"):
            score += 1.5
        return score
