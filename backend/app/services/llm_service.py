import json
from typing import Any

from openai import OpenAI

from ..config import get_settings


def has_llm() -> bool:
    return bool(get_settings().openai_api_key)


def complete_text(messages: list[dict[str, str]], max_tokens: int = 500) -> str:
    settings = get_settings()
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured")

    client = OpenAI(api_key=settings.openai_api_key)
    kwargs: dict[str, Any] = {"model": settings.openai_model, "messages": messages}
    if settings.openai_model.startswith("gpt-5"):
        kwargs["max_completion_tokens"] = max_tokens
    else:
        kwargs["max_tokens"] = max_tokens

    try:
        response = client.chat.completions.create(**kwargs)
    except Exception as exc:
        if "max_tokens" not in str(exc):
            raise
        kwargs.pop("max_tokens", None)
        kwargs["max_completion_tokens"] = max_tokens
        response = client.chat.completions.create(**kwargs)

    content = response.choices[0].message.content or ""
    return content.strip()


def complete_json(messages: list[dict[str, str]], max_tokens: int = 600) -> dict[str, Any]:
    text = complete_text(messages, max_tokens=max_tokens)
    if not text:
        raise RuntimeError("LLM returned an empty response")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise
