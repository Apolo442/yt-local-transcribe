"""Configuração do projeto ativo.

Cada projeto é uma pasta de trabalho com um `project.toml` (ver examples/). Os valores vêm, em ordem de
prioridade: variáveis de ambiente YTLT_* > project.toml > padrões abaixo. `load_project()` é chamado pela CLI
antes de qualquer etapa; os módulos leem `config.X` em tempo de execução.
"""

import os
import tomllib
from pathlib import Path

# ---- padrões (ajustados para RTX 2070 Super 8 GB + Ryzen 5 3600 + 32 GB RAM) ----
PROJECT_DIR = Path.cwd()
PROJECT_NAME = "project"
PLAYLIST_URL = ""
VIDEO_URLS: list[str] = []
PROFILE = "transcript"  # transcript | tier_list

WHISPER_MODEL = "large-v3"
WHISPER_COMPUTE_TYPE = "float16"
WHISPER_BATCH_SIZE = 4
WHISPER_LANGUAGE: str | None = None  # None = detecção automática
DEVICE = "cuda"

OLLAMA_URL = "http://127.0.0.1:11434"
OLLAMA_MODEL = "qwen3:30b-a3b-instruct-2507-q4_K_M"
OLLAMA_NUM_THREAD = 4  # deixa núcleos livres para o desktop
OLLAMA_NUM_CTX = 8192

OUTPUT_LANGUAGE = "English"

DATA_DIR = PROJECT_DIR / "data"
ENCYCLOPEDIA_DIR = PROJECT_DIR / "encyclopedia"

# chave do toml → (nome da variável, conversor)
_KEYS = {
    ("project", "name"): ("PROJECT_NAME", str),
    ("project", "playlist"): ("PLAYLIST_URL", str),
    ("project", "videos"): ("VIDEO_URLS", list),
    ("project", "profile"): ("PROFILE", str),
    ("project", "output_language"): ("OUTPUT_LANGUAGE", str),
    ("asr", "whisper_model"): ("WHISPER_MODEL", str),
    ("asr", "compute_type"): ("WHISPER_COMPUTE_TYPE", str),
    ("asr", "batch_size"): ("WHISPER_BATCH_SIZE", int),
    ("asr", "language"): ("WHISPER_LANGUAGE", str),
    ("asr", "device"): ("DEVICE", str),
    ("llm", "url"): ("OLLAMA_URL", str),
    ("llm", "model"): ("OLLAMA_MODEL", str),
    ("llm", "num_thread"): ("OLLAMA_NUM_THREAD", int),
    ("llm", "num_ctx"): ("OLLAMA_NUM_CTX", int),
}


def load_project(project_dir: str | Path | None = None) -> Path:
    """Ativa um projeto: lê project.toml e aplica variáveis de ambiente YTLT_<NOME>."""
    g = globals()
    root = Path(project_dir or os.environ.get("YTLT_PROJECT") or Path.cwd()).expanduser().resolve()
    g["PROJECT_DIR"] = root
    toml_path = root / "project.toml"
    if toml_path.exists():
        data = tomllib.loads(toml_path.read_text())
        for (section, key), (var, conv) in _KEYS.items():
            if key in data.get(section, {}):
                g[var] = conv(data[section][key])
    for _, (var, conv) in _KEYS.items():
        env = os.environ.get(f"YTLT_{var}")
        if env is not None:
            g[var] = [u for u in env.split(",") if u] if conv is list else conv(env)
    if not g["PROJECT_NAME"] or g["PROJECT_NAME"] == "project":
        g["PROJECT_NAME"] = root.name
    g["DATA_DIR"] = root / "data"
    g["ENCYCLOPEDIA_DIR"] = root / "encyclopedia"
    return root


def video_dir(index: int, video_id: str) -> Path:
    return DATA_DIR / "videos" / f"{index:02d}-{video_id}"
