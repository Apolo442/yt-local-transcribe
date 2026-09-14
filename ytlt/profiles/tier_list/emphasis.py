"""Ênfase muscular por exercício: qual cabeça/porção/músculo o apresentador diz que o exercício mais trabalha.

Só vale o que é dito no capítulo do exercício. Uma ênfase é aceita quando:
  1. o LLM aponta uma frase literal do trecho (localizada na transcrição, similaridade ≥ 0,8, dentro do capítulo);
  2. a região descrita compartilha termos com essa frase (evita região "inferida" de conhecimento externo).
Quando não é dito, `emphasis` fica null — nunca é completado com conhecimento geral.
"""

import json
from pathlib import Path

from pydantic import BaseModel, Field

from ...grounding import content_words
from ...llm import LLMOutputError, chat
from ...textmatch import Locator, mmss

QUOTE_MIN = 0.8
# termos que sozinhos não identificam uma parte do músculo
GENERIC_REGION_WORDS = {"muscl", "exerc", "target", "work", "activ", "growt", "stimu", "overa", "entir", "whole"}


class Emphasis(BaseModel):
    stated: bool = Field(description="true only if the excerpt explicitly says which part is emphasized")
    region: str = Field(description="Short English noun phrase for the emphasized part, e.g. 'long head of the triceps', "
                                    "'upper chest', 'glute medius and minimus', 'rear delts'. Empty if not stated.")
    quote: str = Field(description="The exact words from the excerpt that state it (8-30 words). Empty if not stated.")


PROMPT = """You read one excerpt from a video about the exercise "{name}" ({group}).
Answer ONLY from the excerpt: does the presenter say which part of the muscle this exercise emphasizes most —
a head (long/lateral/medial head), a portion (upper/mid/lower, front/side/rear) or a specific muscle
(e.g., glute medius, brachialis, rectus femoris, lats, rhomboids)?

Rules:
- stated=true only if the excerpt itself says it for THIS exercise. Do not use your own anatomy knowledge.
- A generic "this works the glutes/triceps" is NOT a specific part: stated=false.
- If several parts are named as emphasized, include them in one short region phrase.
- quote must be copied word for word from the excerpt."""


def _chapter_texts(meta: dict, transcript: dict) -> dict[str, tuple[float, float, str]]:
    out = {}
    duration = meta.get("duration") or transcript["segments"][-1]["end"]
    for c in meta.get("chapters") or []:
        end = c.get("end_time") or duration
        segs = [s for s in transcript["segments"] if c["start_time"] <= (s["start"] + s["end"]) / 2 < end]
        out[c["title"]] = (c["start_time"], end, "\n".join(f"[{mmss(s['start'])}] {s['text']}" for s in segs))
    return out


def build(vdir: Path, force: bool = False, progress=print) -> Path:
    path = vdir / "analysis.json"
    analysis = json.loads(path.read_text())
    meta = json.loads((vdir / "metadata.json").read_text())
    transcript = json.loads((vdir / "transcript.json").read_text())
    loc = Locator(transcript)
    chapters = _chapter_texts(meta, transcript)
    group = meta.get("title", "")

    todo = [e for e in analysis["exercises"] if force or "emphasis" not in e]
    for n, ex in enumerate(todo, 1):
        chapter = chapters.get(ex.get("chapter") or "")
        if not chapter or not chapter[2].strip():
            ex["emphasis"] = None
            continue
        lo, hi, text = chapter
        progress(f"    ênfase {n}/{len(todo)}: {ex['name']}")
        try:
            r = chat(PROMPT.format(name=ex["name"], group=group), f"Excerpt:\n{text}", Emphasis, num_ctx=4096, max_tokens=400)
        except LLMOutputError as exc:
            ex["emphasis"], ex["emphasis_rejected"] = None, f"LLM: {exc}"
            continue
        ex.pop("emphasis_rejected", None)
        if not r.stated or not r.region.strip():
            ex["emphasis"] = None
            continue
        match, t = loc.locate(r.quote, lo - 2, hi + 2)
        region_terms = content_words(r.region) - GENERIC_REGION_WORDS
        shared = region_terms & content_words(r.quote)
        if match < QUOTE_MIN:
            ex["emphasis"], ex["emphasis_rejected"] = None, f"citação não encontrada no capítulo ({match:.2f}): {r.region}"
        elif not shared:
            ex["emphasis"], ex["emphasis_rejected"] = None, f"região sem apoio na citação: {r.region}"
        else:
            ex["emphasis"] = {"region": r.region.strip(), "quote": r.quote.strip(), "timestamp": mmss(t), "quote_match": match}

    path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2))
    return path
