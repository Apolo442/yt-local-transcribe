"""Perfis de análise: o que fazer com a transcrição depois do consenso.

Um perfil implementa:
  analyze(vdir, force, progress) -> Path   análise estruturada (analysis.json)
  render(vdir) -> Path                      documento legível do vídeo
  render_index() -> Path                    índice do projeto
  units(vdir) -> dict[str, str]             textos traduzíveis (chaves estáveis)
  export() -> Path                          exportação agregada do projeto (opcional)

`transcript` é o perfil mínimo (sem LLM): só gera Markdown da transcrição com timestamps.
"""

from importlib import import_module
from types import ModuleType

PROFILES = {
    "transcript": "ytlt.profiles.transcript",
    "tier_list": "ytlt.profiles.tier_list",
}


def get_profile(name: str) -> ModuleType:
    if name not in PROFILES:
        raise SystemExit(f"perfil desconhecido: {name!r} (disponíveis: {', '.join(PROFILES)})")
    return import_module(PROFILES[name])
