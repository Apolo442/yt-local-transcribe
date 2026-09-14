"""Chamadas ao LLM local (Ollama) com saída em JSON Schema.

Modelos locais às vezes entram em repetição e devolvem JSON truncado: a resposta tem tamanho máximo,
repetição é penalizada e há novas tentativas com leve variação antes de desistir (LLMOutputError).
"""

import json
import urllib.request

from pydantic import BaseModel

from . import config


def chat(system: str, user: str, schema: type[BaseModel], num_ctx: int, max_tokens: int = 2048) -> BaseModel:
    """Chamada ao LLM local com saída em JSON Schema.

    Modelos locais às vezes entram em repetição e geram JSON truncado: limitamos o tamanho da resposta,
    penalizamos repetição e tentamos de novo (com leve variação) antes de desistir."""
    last_error = None
    for attempt in range(3):
        body = {
            "model": config.OLLAMA_MODEL,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "format": schema.model_json_schema(),
            "stream": False,
            "think": False,
            "keep_alive": "2m",
            "options": {"temperature": 0 if attempt == 0 else 0.3, "num_ctx": num_ctx, "num_predict": max_tokens,
                        "repeat_penalty": 1.1, "num_thread": config.OLLAMA_NUM_THREAD, "seed": attempt},
        }
        req = urllib.request.Request(
            f"{config.OLLAMA_URL}/api/chat", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=1800) as resp:
            content = json.loads(resp.read())["message"]["content"]
        try:
            return schema.model_validate_json(content)
        except ValueError as exc:  # JSON truncado/inválido
            last_error = exc
    raise LLMOutputError(f"resposta inválida do LLM após 3 tentativas: {str(last_error)[:200]}")


class LLMOutputError(RuntimeError):
    pass


# ---------------------------------------------------------------- verificação determinística


_chat = chat  # compatibilidade


def unload_ollama() -> None:
    """Libera a VRAM ocupada pelo LLM local (antes das etapas de reconhecimento de fala)."""
    try:
        with urllib.request.urlopen(f"{config.OLLAMA_URL}/api/ps", timeout=3) as r:
            models = [m["name"] for m in json.loads(r.read()).get("models", [])]
        for name in models:
            body = json.dumps({"model": name, "keep_alive": 0}).encode()
            urllib.request.urlopen(urllib.request.Request(f"{config.OLLAMA_URL}/api/generate", data=body), timeout=30)
    except OSError:
        pass
