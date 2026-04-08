import logging
import os
import re
from typing import Any, Mapping, Sequence

from openai import OpenAI

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_OPENROUTER_MODEL = "google/gemini-3-flash-preview"


class OpenRouterLLM:
    """Tiny adapter exposing the invoke() API used by the generator stack."""

    def __init__(
        self,
        model: str,
        *,
        temperature: float = 0.3,
        base_url: str = DEFAULT_OPENROUTER_BASE_URL,
        api_key: str | None = None,
        max_tokens: int = 40000,
        max_retries: int = 2,
        site_url: str | None = None,
        app_name: str | None = None,
    ):
        if not api_key:
            raise EnvironmentError(
                "OPENROUTER_API_KEY not found. Set OPENROUTER_API_KEY before using the v2 pipeline."
            )

        headers: dict[str, str] = {}
        if site_url:
            headers["HTTP-Referer"] = site_url
        if app_name:
            headers["X-Title"] = app_name

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            max_retries=max_retries,
            default_headers=headers or None,
        )

    def invoke(self, payload: Any) -> dict[str, Any]:
        request_overrides: dict[str, Any] = {}

        if isinstance(payload, Mapping) and "messages" in payload:
            messages = _normalize_messages(payload["messages"])
            request_overrides = {
                key: value for key, value in payload.items() if key != "messages"
            }
        else:
            messages = _normalize_messages(payload)

        extra_body = dict(request_overrides.pop("extra_body", {}) or {})
        if "reasoning" in request_overrides:
            extra_body["reasoning"] = request_overrides.pop("reasoning")

        response = self.client.chat.completions.create(
            model=request_overrides.pop("model", self.model),
            messages=messages,
            temperature=request_overrides.pop("temperature", self.temperature),
            max_tokens=request_overrides.pop("max_tokens", self.max_tokens),
            extra_body=extra_body or None,
            **request_overrides,
        )
        message = response.choices[0].message
        content = _coerce_content_text(getattr(message, "content", ""))

        if not content:
            refusal = getattr(message, "refusal", None)
            if refusal:
                content = str(refusal)

        return {
            "content": content,
            "role": getattr(message, "role", "assistant"),
            "model": getattr(response, "model", self.model),
        }


def load_llm(temperature: float = 0.3, model: str | None = None, provider: str | None = None):
    provider = (provider or "openrouter").lower()
    if provider != "openrouter":
        raise ValueError(
            f"Unknown LLM provider: '{provider}'. The v2 pipeline currently supports only 'openrouter'."
        )

    api_key = os.environ.get("OPENROUTER_API_KEY", os.environ.get("OPENAI_API_KEY"))
    return OpenRouterLLM(
        model=model or os.environ.get("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL),
        temperature=temperature,
        base_url=os.environ.get("OPENROUTER_BASE_URL", DEFAULT_OPENROUTER_BASE_URL),
        api_key=api_key,
        max_tokens=int(os.environ.get("OPENROUTER_MAX_TOKENS", 40000)),
        max_retries=int(os.environ.get("OPENROUTER_MAX_RETRIES", 2)),
        site_url=os.environ.get("OPENROUTER_SITE_URL"),
        app_name=os.environ.get("OPENROUTER_APP_NAME", "mcdc_agent_v2"),
    )


def strip_thinking_blocks(text: str) -> str:
    if not text:
        return text
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def extract_message_content(message: Any) -> str:
    if isinstance(message, dict):
        content = message.get("content", str(message))
    elif hasattr(message, "content"):
        content = message.content
    else:
        return str(message)

    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text", "").strip()
                if text:
                    text_parts.append(text)
            elif isinstance(block, str) and block.strip():
                text_parts.append(block)
        return "\n".join(text_parts)

    return str(content) if content else ""


def extract_response_text(response: Any) -> str:
    if isinstance(response, str):
        return strip_thinking_blocks(response)
    if isinstance(response, dict):
        if "content" in response:
            return strip_thinking_blocks(str(response["content"]))
        if "messages" in response and response["messages"]:
            return strip_thinking_blocks(extract_message_content(response["messages"][-1]))
    if hasattr(response, "content"):
        return strip_thinking_blocks(str(response.content))
    if isinstance(response, list):
        return strip_thinking_blocks("\n".join(extract_message_content(item) for item in response))
    return strip_thinking_blocks(str(response))


def extract_code(text: str) -> str:
    text = strip_thinking_blocks(text)
    matches = re.findall(r"```(?:python)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if matches:
        return "\n\n".join(match.strip() for match in matches)
    return text.strip()


def _normalize_messages(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, str):
        return [{"role": "user", "content": payload}]
    if isinstance(payload, Mapping):
        if "role" in payload and "content" in payload:
            return [_normalize_message(payload)]
        return [{"role": "user", "content": str(payload)}]
    if isinstance(payload, Sequence):
        return [_normalize_message(message) for message in payload]
    return [{"role": "user", "content": str(payload)}]


def _normalize_message(message: Any) -> dict[str, Any]:
    if isinstance(message, Mapping):
        role = str(message.get("role", "user"))
        content = message.get("content", "")
        return {"role": role, "content": _coerce_message_content(content)}
    return {"role": "user", "content": str(message)}


def _coerce_message_content(content: Any) -> Any:
    if isinstance(content, (str, list)):
        return content
    return str(content)


def _coerce_content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                else:
                    parts.append(str(item))
            elif hasattr(item, "text"):
                parts.append(str(item.text))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return str(content)
