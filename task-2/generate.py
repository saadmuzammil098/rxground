"""Provider-agnostic generation step. Same pattern GridScribe used, one
thin function per provider, all taking a system prompt and a user prompt
and returning plain text, so rag.py does not need to know which provider
answered.

Ollama is the default, free, local, no rate limit, which is what the real
runs in this task's README use. Gemini is confirmed working live as the
alternate provider (a GEMINI_API_KEY was available in this environment).
Groq is implemented the same way but not exercised live here, no
GROQ_API_KEY was set in this environment, consistent with how GridScribe
documented the same gap.
"""

from __future__ import annotations

import os


class GenerationError(Exception):
    pass


def ollama_generate(system_prompt: str, user_prompt: str, model: str = "qwen2.5:7b", temperature: float = 0.0) -> str:
    import ollama

    try:
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            options={"temperature": temperature},
        )
    except Exception as exc:
        raise GenerationError(f"ollama generate failed: {exc}") from exc
    return response["message"]["content"]


def gemini_generate(system_prompt: str, user_prompt: str, model: str = "gemini-flash-latest", temperature: float = 0.0) -> str:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise GenerationError("GEMINI_API_KEY is not set, cannot construct Gemini client")
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    try:
        response = client.models.generate_content(
            model=model,
            contents=user_prompt,
            config=types.GenerateContentConfig(system_instruction=system_prompt, temperature=temperature),
        )
    except Exception as exc:
        raise GenerationError(f"gemini generate failed: {exc}") from exc
    return response.text


def groq_generate(system_prompt: str, user_prompt: str, model: str = "llama-3.3-70b-versatile", temperature: float = 0.0) -> str:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise GenerationError("GROQ_API_KEY is not set, cannot construct Groq client")
    from groq import Groq

    client = Groq(api_key=api_key)
    try:
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
    except Exception as exc:
        raise GenerationError(f"groq generate failed: {exc}") from exc
    return response.choices[0].message.content


GENERATORS = {
    "ollama": ollama_generate,
    "gemini": gemini_generate,
    "groq": groq_generate,
}
