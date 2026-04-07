import logging
import os
import re
from typing import Any, Mapping, Sequence

from openai import OpenAI

# Suppress verbose INFO logs from dependencies
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_OPENROUTER_MODEL = "openai/gpt-4.1-mini"


class OpenRouterLLM:
    """Small adapter that exposes an invoke() API over OpenRouter chat completions."""

    def __init__(
        self,
        model: str,
        *,
        temperature: float = 0.3,
        top_p: float = 0.95,
        top_k: int = 20,
        base_url: str = DEFAULT_OPENROUTER_BASE_URL,
        api_key: str | None = None,
        max_tokens: int = 40000,
        max_retries: int = 2,
        site_url: str | None = None,
        app_name: str | None = None,
    ):
        if not api_key:
            raise EnvironmentError(
                "OPENROUTER_API_KEY not found in environment variables. "
                "Please set it with: export OPENROUTER_API_KEY='your-api-key-here'"
            )

        headers = {}
        if site_url:
            headers["HTTP-Referer"] = site_url
        if app_name:
            headers["X-Title"] = app_name

        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_tokens = max_tokens
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            max_retries=max_retries,
            default_headers=headers or None,
        )

    def invoke(self, payload: Any) -> dict[str, Any]:
        """Send a prompt or message list and return a parser-friendly response."""
        request_overrides: dict[str, Any] = {}

        if isinstance(payload, Mapping) and "messages" in payload:
            messages = _normalize_messages(payload["messages"])
            request_overrides = {
                key: value for key, value in payload.items() if key != "messages"
            }
        else:
            messages = _normalize_messages(payload)

        extra_body = dict(request_overrides.pop("extra_body", {}) or {})
        for extra_key in ("reasoning",):
            if extra_key in request_overrides:
                extra_body[extra_key] = request_overrides.pop(extra_key)

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


def load_llm(temperature=0.3, model=None, provider=None):
    """Load the shared LLM client used by the generators."""
    provider = (provider or "openrouter").lower()

    if provider != "openrouter":
        raise ValueError(
            f"Unknown LLM provider: '{provider}'. Supported providers: 'openrouter'"
        )

    api_key = os.environ.get("OPENROUTER_API_KEY", os.environ.get("OPENAI_API_KEY"))
    return OpenRouterLLM(
        model=model or os.environ.get("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL),
        temperature=temperature,
        top_p=0.95,
        top_k=20,
        base_url=os.environ.get("OPENROUTER_BASE_URL", DEFAULT_OPENROUTER_BASE_URL),
        api_key=api_key,
        max_tokens=int(os.environ.get("OPENROUTER_MAX_TOKENS", 40000)),
        max_retries=int(os.environ.get("OPENROUTER_MAX_RETRIES", 2)),
        site_url=os.environ.get("OPENROUTER_SITE_URL"),
        app_name=os.environ.get("OPENROUTER_APP_NAME", "mc_agent"),
    )


def strip_thinking_blocks(text: str) -> str:
    """Strip DeepSeek R1 <think>...</think> reasoning blocks from response.
    
    R1 reasoning models produce output in the format:
        <think>
        [internal reasoning chain]
        </think>
        
        [actual answer]
    
    This function removes the thinking blocks to extract just the answer.
    Handles multiple thinking blocks and nested content.
    """
    if not text:
        return text
    
    # Remove <think>...</think> blocks (handles multiline content)
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    
    # Clean up any extra whitespace left behind
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    
    return cleaned.strip()


def extract_message_content(message) -> str:
    """
    Extract text content from a single message object.

    Handles common message formats including multimodal content blocks.
    """
    content = None
    if isinstance(message, dict):
        content = message.get("content", str(message))
    elif hasattr(message, "content"):
        content = message.content
    else:
        return str(message)

    if isinstance(content, list):
        # Handle multimodal content blocks - filter out empty ones
        text_parts = []
        for b in content:
            if isinstance(b, dict) and b.get("type") == "text":
                text = b.get("text", "").strip()
                if text:
                    text_parts.append(text)
            elif isinstance(b, str) and b.strip():
                text_parts.append(b)
        return "\n".join(text_parts)

    return str(content) if content else ""


def parse_agent_response(response) -> str:
    """
    Extract text from common response formats.

    Handles:
    - Plain strings
    - Dict responses with 'output', 'content', or 'messages' keys
    - Objects with .content attribute
    - List of content blocks
    - DeepSeek R1 <think>...</think> reasoning blocks (strips them)
    """
    result = None
    
    if isinstance(response, str):
        result = response
    elif isinstance(response, dict):
        if "output" in response:
            result = response["output"]
        elif "content" in response:
            result = response["content"]
        elif "messages" in response and response["messages"]:
            result = extract_message_content(response["messages"][-1])
    elif hasattr(response, "content"):
        result = response.content
    elif isinstance(response, list):
        try:
            text_parts = []
            for block in response:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif hasattr(block, "content"):
                    text_parts.append(block.content)
                else:
                    text_parts.append(str(block))
            result = "\n".join(text_parts)
        except Exception:
            result = str(response)
    else:
        result = str(response)
    
    # Strip R1 thinking blocks if present
    return strip_thinking_blocks(result) if result else ""




def extract_code(text: str) -> str:
    """Extract code from markdown blocks.
    
    If multiple blocks are found, joins them with newlines.
    If no blocks are found, returns the original text.
    Handles DeepSeek R1 <think> blocks by stripping them first.
    """
    # First strip any R1 thinking blocks
    text = strip_thinking_blocks(text)
    
    # Look for ```python ... ``` or just ``` ... ``` blocks
    pattern = r"```(?:python)?\s*(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    
    if matches:
        # Strip whitespace from each match
        cleaned_matches = [m.strip() for m in matches]
        return "\n\n".join(cleaned_matches)
    
    # If no markdown blocks, return full text (fallback)
    return text.strip()

extract_response_text = parse_agent_response


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
        normalized = {
            "role": str(message.get("role", "user")),
            "content": message.get("content", ""),
        }
        if "name" in message:
            normalized["name"] = message["name"]
        return normalized

    return {"role": "user", "content": str(message)}


def _coerce_content_text(content: Any) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, Mapping):
                text = item.get("text") or item.get("content")
                if text:
                    parts.append(str(text))
            elif hasattr(item, "text") and item.text:
                parts.append(str(item.text))
        return "\n".join(part for part in parts if part)

    if content is None:
        return ""

    return str(content)
