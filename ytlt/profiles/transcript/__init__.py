"""Perfil transcript: nenhum LLM. Gera um Markdown por vídeo com metadados, capítulos e a transcrição
por consenso (com timestamps e trechos incertos marcados), e um índice do projeto."""

import json
from pathlib import Path

from ... import config


def _mmss(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600:d}:{s // 60 % 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60:02d}:{s % 60:02d}"


def analyze(vdir: Path, force: bool = False, progress=print) -> Path:
    return vdir / "transcript.json"  # nada a analisar


def render(vdir: Path) -> Path:
    meta = json.loads((vdir / "metadata.json").read_text())
    tr = json.loads((vdir / "transcript.json").read_text())
    cons_path = vdir / "consensus.json"
    vid = meta["id"]
    lines = [f"# {meta['title']}", "", f"- Canal: {meta.get('channel')}", f"- Link: {meta.get('webpage_url')}"]
    if cons_path.exists():
        sm = json.loads(cons_path.read_text())["summary"]
        lines.append(f"- Consenso: concordância {sm['agreement_pct']}, {sm['changed_from_whisperx']} correções, "
                     f"{sm['uncertain']} trechos incertos")
    chapters = meta.get("chapters") or [{"title": "", "start_time": 0, "end_time": 10**9}]
    for c in chapters:
        segs = [s for s in tr["segments"] if c["start_time"] <= (s["start"] + s["end"]) / 2 < c["end_time"]]
        if not segs:
            continue
        lines += ["", f"## {c['title'] or 'Transcrição'}", ""]
        for s in segs:
            flag = " ❓" if s.get("uncertain") else ""
            lines.append(f"- [{_mmss(s['start'])}](https://youtu.be/{vid}?t={int(s['start'])}) {s['text']}{flag}")
    out = config.PROJECT_DIR / "transcripts" / f"{meta['playlist_index']:02d}-{vid}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    return out


def render_index() -> Path:
    rows = []
    for vdir in sorted((config.DATA_DIR / "videos").glob("*")):
        if (vdir / "transcript.json").exists():
            meta = json.loads((vdir / "metadata.json").read_text())
            rows.append(f"- [{meta['title']}]({meta['playlist_index']:02d}-{meta['id']}.md)")
    out = config.PROJECT_DIR / "transcripts" / "index.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(f"# {config.PROJECT_NAME}\n\n" + "\n".join(rows) + "\n")
    return out


def units(vdir: Path) -> dict[str, str]:
    return {}


def export() -> Path:
    raise SystemExit("o perfil 'transcript' não tem export; use um perfil de análise (ex.: tier_list)")
