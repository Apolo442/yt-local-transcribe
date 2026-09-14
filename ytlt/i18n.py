"""Tradução da análise (en → pt-BR) com chaves estáveis.

Fluxo (metodologia adaptada de claude-translation-skill: glossário → tradução → revisão independente → revisão):
  ytlt i18n extract   → translation/en/NN.json   {chave: texto em inglês}
  (tradutores escrevem)        translation/pt/NN.json   {mesmas chaves: texto em português}
  ytlt i18n check     → valida chaves, números, tiers, siglas e campos vazios
  ytlt export         → o JSON do app ganha campos {"en": …, "pt": …}

Só é traduzido conteúdo produzido pela análise (resumos, critérios, argumentos, justificativas, dicas,
foco muscular, afirmações científicas parafraseadas). Transcrição e citações literais ficam no original.
"""

import json
import re
from pathlib import Path

from . import config

def _dirs():
    base = config.PROJECT_DIR / "translation"
    return base / "en", base / "pt", base / "glossary.json"


def _video_dirs():
    return [d for d in sorted((config.DATA_DIR / "videos").glob("*")) if (d / "analysis.json").exists()]


def units(vdir: Path) -> dict[str, str]:
    """Textos traduzíveis de um vídeo — definidos pelo perfil de análise do projeto."""
    from .profiles import get_profile

    return get_profile(config.PROFILE).units(vdir)


def extract() -> list[Path]:
    EN_DIR, _, _ = _dirs()
    EN_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for vdir in _video_dirs():
        meta = json.loads((vdir / "metadata.json").read_text())
        path = EN_DIR / f"{meta['playlist_index']:02d}.json"
        path.write_text(json.dumps({"video_id": meta["id"], "title": meta["title"], "units": units(vdir)},
                                   ensure_ascii=False, indent=2))
        paths.append(path)
    return paths


# letra do tier em maiúscula: "a tier list" (artigo) não é o tier A
TIER_TOKEN = re.compile(r"\b([SABCDF]\+?)[\s-]*[Tt]ier\b(?!\s+(?:list|system|ranking))")


def _numbers(s: str) -> list[str]:
    return sorted(re.findall(r"\d+(?:[.,]\d+)?", s))


def _acronyms(s: str) -> set[str]:
    return set(re.findall(r"\b(?:EMG|ROM|RDL|DOI|PMID|EZ|RIR|RPE|SFR)\b", s))


def check(strict: bool = True) -> list[str]:
    """Verificações determinísticas de fidelidade da tradução. Retorna a lista de problemas."""
    problems = []
    EN_DIR, PT_DIR, GLOSSARY = _dirs()
    glossary = json.loads(GLOSSARY.read_text()) if GLOSSARY.exists() else {"terms": []}
    for en_path in sorted(EN_DIR.glob("*.json")):
        en = json.loads(en_path.read_text())["units"]
        pt_path = PT_DIR / en_path.name
        if not pt_path.exists():
            problems.append(f"{en_path.name}: tradução ausente")
            continue
        pt = json.loads(pt_path.read_text())
        pt = pt.get("units", pt)
        for key in en.keys() - pt.keys():
            problems.append(f"{en_path.name} {key}: chave ausente (OMISSÃO)")
        for key in pt.keys() - en.keys():
            problems.append(f"{en_path.name} {key}: chave extra (possível FABRICAÇÃO)")
        for key in en.keys() & pt.keys():
            src, dst = en[key], pt[key]
            if not isinstance(dst, str) or not dst.strip():
                problems.append(f"{en_path.name} {key}: vazio")
                continue
            term = next((t for t in glossary["terms"] if t["en"].lower() == src.lower()), None) \
                if key.endswith(".name") else None
            if term:  # nome definido no glossário verificado: a única regra é seguir o glossário
                if dst != term["pt"]:
                    problems.append(f"{en_path.name} {key}: nome fora do glossário «{dst}» ≠ «{term['pt']}»")
                continue
            if _numbers(src) != _numbers(dst):
                problems.append(f"{en_path.name} {key}: números diferem {_numbers(src)} → {_numbers(dst)}")
            src_tiers = sorted(TIER_TOKEN.findall(src))
            dst_tiers = sorted(x for pair in re.findall(r"\b(?:[Tt]ier|nível)\s+([SABCDF]\+?)(?![\w])|\b([SABCDF]\+?)[\s-]*[Tt]ier\b", dst)
                               for x in pair if x)
            if src_tiers and src_tiers != dst_tiers:
                problems.append(f"{en_path.name} {key}: tiers diferem {src_tiers} → {dst_tiers}")
            missing = _acronyms(src) - _acronyms(dst)
            if missing:
                problems.append(f"{en_path.name} {key}: siglas perdidas {sorted(missing)}")
            ratio = len(dst) / max(1, len(src))
            if strict and len(src) > 40 and not 0.6 <= ratio <= 2.0:
                problems.append(f"{en_path.name} {key}: tamanho suspeito ({ratio:.2f}× o original)")
    return problems


def load_pt(vdir: Path) -> dict[str, str]:
    meta = json.loads((vdir / "metadata.json").read_text())
    _, PT_DIR, _ = _dirs()
    p = PT_DIR / f"{meta['playlist_index']:02d}.json"
    if not p.exists():
        return {}
    data = json.loads(p.read_text())
    return data.get("units", data)
