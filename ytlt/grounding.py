"""Fidelidade de textos gerados pelo LLM ao trecho de origem.

- strip_llm_wrappers: remove prefixos de chave JSON e rótulos que modelos locais deixam no texto;
- verify_support: mantém só afirmações com apoio no trecho (filtro de radicais + verificador com
  evidência literal que precisa existir na transcrição);
- looks_portuguese: detecta troca de idioma na resposta.
"""

import re
from typing import Literal

from pydantic import BaseModel, Field

from .llm import LLMOutputError, chat
from .textmatch import Locator


KIND_LABELS = (r"explicação biomecânica|biomecânica|opinião|experiência pessoal|científica|"
               r"biomechanical explanation|biomechanical|opinion|personal experience|scientific")
WRAP_PREFIX = re.compile(r'^\s*(?:(?:tier_)?quote"?\s*:\s*"?|(?:' + KIND_LABELS + r')\s*:\s*(?:quote\s*:\s*)?)', re.I)
WRAP_SUFFIX = re.compile(r'(?:"\s*,\s*"?(?:classification|type|kind|evidence)"?\s*:.*'
                         r'|\s*[|/]\s*(?:classification|classificação|tipo|type|kind|evidence|evidência)\s*:.*'
                         r'|\s*[-–]\s*(?:' + KIND_LABELS + r')'
                         r'|\s*\((?:' + KIND_LABELS + r')\))\s*$', re.I)
LABEL_ONLY = re.compile(r'^\W*(?:(?:classification|classificação|tipo|type|kind)"?\s*:\s*"?)?(?:' + KIND_LABELS + r')\W*$', re.I)


def strip_llm_wrappers(text: str) -> str:
    """Remove os envoltórios que o modelo local deixa em volta do texto (prefixos de chave JSON, rótulos)."""
    prev = None
    while prev != text:
        prev = text
        text = WRAP_PREFIX.sub("", text)
        text = WRAP_SUFFIX.sub("", text)
        text = text.strip().strip("`'\"“” ").strip()
    return text


class Support(BaseModel):
    index: int
    verdict: Literal["supported", "not_in_excerpt", "about_other_exercise"]
    evidence: str = Field(description="If supported: the exact words from the excerpt that support it (copied verbatim, "
                                      "5-25 words). Empty otherwise.")


class SupportCheck(BaseModel):
    verdicts: list[Support]


SUPPORT_PROMPT = """You are a strict fact-checker. You get a transcript excerpt and a numbered list of claims that a summarizer
wrote about "{name}".
For each claim decide:
- "supported": the excerpt states it or directly implies it. Paraphrases, condensed wording and faithful summaries COUNT
  as supported. Give the exact supporting words from the excerpt in `evidence`.
- "not_in_excerpt": the excerpt does not say it (e.g., generic training advice the presenter did not give here,
  exaggerations, or anything that needs outside knowledge).
- "about_other_exercise": the claim describes a different subject than "{name}".
Judge every claim; do not skip any."""

GROUND_MIN = 0.15  # fração mínima de radicais de conteúdo do item presentes no capítulo


def _content_words(text: str) -> set[str]:
    """Radicais (5 primeiras letras, sem plural) das palavras de conteúdo: "isolation"≈"isolate", "quads"≈"quad"."""
    from .references import STOPWORDS

    out = set()
    for w in re.findall(r"[a-z]+", text.lower()):
        if len(w) > 2 and w not in STOPWORDS:
            out.add(w.rstrip("s")[:5] if len(w) > 3 else w)
    return out


PT_MARKERS = re.compile(r"\b(para|com|dos|das|não|mais|maior|ativação|glúteos|exercício|execução|você|também)\b", re.I)


def _looks_portuguese(items: list[str]) -> bool:
    """Qualquer item com marcador de português + acento/cedilha típicos já basta (frases curtas escapavam)."""
    text = " ".join(items)
    if len(PT_MARKERS.findall(text)) >= max(2, len(text.split()) // 15):
        return True
    return any(PT_MARKERS.search(x) and re.search(r"[ãõçáéíóúâêô]", x) for x in items)


def verify_support(r: dict, chapter_text: str, loc: Locator, lo: float, hi: float,
                   fields: tuple[str, ...] = ("reasons", "technique_tips")) -> None:
    """Remove justificativas/dicas sem apoio no próprio capítulo (conteúdo inventado pelo LLM).

    1) filtro barato: item sem radicais em comum com o capítulo; 2) verificador do LLM com evidência literal,
    e a evidência precisa existir de fato na transcrição do capítulo."""
    cw = _content_words(chapter_text)
    candidates, removed = [], r.setdefault("removed_unsupported", [])
    for field in fields:
        for text in r[field]:
            xw = _content_words(text)
            if xw and len(xw & cw) / len(xw) < GROUND_MIN:
                removed.append({"field": field, "text": text, "why": "no words from the chapter"})
            else:
                candidates.append((field, text))
    keep: dict[str, list[str]] = {f: [] for f in fields}
    if candidates:
        listing = "\n".join(f"{i}. {t}" for i, (_, t) in enumerate(candidates, 1))
        try:
            res = chat(SUPPORT_PROMPT.format(name=r["name"]), f"Excerpt:\n{chapter_text}\n\nClaims:\n{listing}",
                        SupportCheck, num_ctx=4096)
            verdict = {v.index: v for v in res.verdicts}
        except LLMOutputError:
            verdict = {}  # sem verificação não há como confirmar: nada é mantido
        for i, (field, text) in enumerate(candidates, 1):
            v = verdict.get(i)
            if v and v.verdict == "supported" and v.evidence.strip() and loc.locate(v.evidence, lo, hi)[0] >= 0.75:
                keep[field].append(text)
                r.setdefault("support_evidence", {})[text] = v.evidence
            else:
                why = v.verdict if v and v.verdict != "supported" else "evidence not found in transcript"
                removed.append({"field": field, "text": text, "why": why})
    for f in fields:
        r[f] = keep[f]
    if len(fields) == 2:  # segundo campo (ex.: dicas) não repete o primeiro (ex.: justificativas)
        r[fields[1]] = dedupe_tips(r[fields[0]], r[fields[1]])


def dedupe_tips(reasons: list[str], tips: list[str]) -> list[str]:
    """Dicas que só repetem uma justificativa (mesmos radicais) não são dicas de execução."""
    out = []
    for t in tips:
        tw = _content_words(t)
        if tw and any(len(tw & _content_words(r)) / len(tw | _content_words(r)) >= 0.5 for r in reasons):
            continue
        out.append(t)
    return out


looks_portuguese = _looks_portuguese
content_words = _content_words
