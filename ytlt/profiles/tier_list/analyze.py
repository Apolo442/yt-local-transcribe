"""Etapas 3–6: exercícios, tier list, argumentos e separação opinião × ciência via LLM local (Ollama).

Estratégia (pensada para um modelo ~8B em 8 GB de VRAM):
- Uma chamada por capítulo do vídeo (normalmente 1 capítulo = 1 exercício) → contexto pequeno, 100% GPU
  e muito menos confusão entre exercícios vizinhos. Sem capítulos, blocos de ~90 s são usados.
- Uma chamada de visão geral (resumo, critérios, argumentos, escolha S+) sobre a transcrição inteira.
- Toda resposta segue um JSON Schema (structured outputs do Ollama).
- Pós-verificação determinística: cada `quote` é localizada na transcrição (dá o timestamp real e
  `quote_match`); afirmações "científicas" sem termo de evidência por perto são sinalizadas.
"""

import json
import re
import urllib.request
from difflib import SequenceMatcher
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from ... import config
from ...grounding import (
    LABEL_ONLY,
    _content_words,
    _looks_portuguese,
    dedupe_tips,
    strip_llm_wrappers,
    verify_support,
)
from ...llm import LLMOutputError, _chat
from ...textmatch import Locator, _norm, mmss

Tier = Literal["S+", "S", "A", "B", "C", "D", "F", "sem tier"]
Kind = Literal["científica", "explicação biomecânica", "opinião", "experiência pessoal"]

EVIDENCE_TERMS = re.compile(
    r"\b(stud(y|ies)|research\w*|paper|meta.?analys\w+|emg|trial|subjects|participants|data|evidence|"
    r"measured|scientists?|literature|review)\b",
    re.I,
)
# "high B tier", "low D tier", "S tier plus". Ignora "tier list" e a legenda "S tier for super".
TIER_RE = re.compile(r"\b(?:(high|low|mid)[\s-]+)?([sabcdf])[\s-]*tier(?:\s*(\+|plus))?\b(?!\s*(?:list|for\s+(?:super|fail)|or\s+two))", re.I)
STOP = {"the", "a", "and", "of", "with", "to", "machine", "exercise", "exercises"}


class Statement(BaseModel):
    text: str = Field(description="A afirmação, parafraseada no idioma de saída")
    kind: Kind
    evidence: str = Field(description="Só para 'científica': o estudo/dado citado, como descrito no vídeo. Senão vazio")
    quote: str = Field(description="Trecho LITERAL (8–20 palavras) da transcrição, no idioma original")


class ChapterExercise(BaseModel):
    is_exercise: bool = Field(description="false se o trecho não avalia um exercício (introdução, encerramento)")
    name: str = Field(description="Nome do exercício como dito no vídeo")
    name_translated: str = Field(description="Nome usual no idioma de saída")
    muscle_focus: str = Field(description="Região/músculo enfatizado, segundo o apresentador")
    tier: Tier
    tier_modifier: Literal["alto", "baixo", ""] = Field(description="'alto' se disse high X tier, 'baixo' se low X tier")
    reasons: list[str] = Field(description="Prós e contras que justificam o tier, RESUMIDOS com suas próprias palavras "
                                           "(frases curtas, até ~15 palavras cada). NÃO copie frases da transcrição.")
    technique_tips: list[str] = Field(description="Dicas de execução ditas, RESUMIDAS com suas próprias palavras "
                                                  "(até ~15 palavras cada); vazio se nenhuma. NÃO copie frases da transcrição.")
    tier_quote: str = Field(description="Frase LITERAL em que o tier é anunciado")
    statements: list[Statement]


class Argument(BaseModel):
    thesis: str
    reasoning: str
    quote: str = Field(description="Trecho LITERAL (8–20 palavras) da transcrição, no idioma original")


class Overview(BaseModel):
    summary: str = Field(description="Resumo do vídeo em 3–5 frases")
    ranking_criteria: list[str] = Field(description="Critérios que o apresentador usa para ranquear")
    top_pick: str = Field(description="Exercício promovido a S+ / escolhido como o melhor de todos; vazio se nenhum")
    top_pick_quote: str = Field(description="Frase LITERAL em que o melhor exercício é escolhido; vazio se nenhum")
    worst_pick: str = Field(description="Exercício(s) apontado(s) como o pior de todos; vazio se nenhum")
    arguments: list[Argument] = Field(description="Principais teses defendidas no vídeo inteiro (4–8)")
    closing_statements: list[Statement] = Field(description="Afirmações da introdução e do encerramento")


RULES = """Regras:
- Use SOMENTE o que está no texto fornecido. Não invente exercícios, tiers, estudos, números ou nomes de pessoas.
- Campos de citação (`quote`, `tier_quote`, `top_pick_quote`) são cópia LITERAL da transcrição, sem traduzir.
- Classificação de afirmações:
  * "científica": o apresentador cita explicitamente estudo, pesquisa, EMG, dados ou medição como base.
  * "explicação biomecânica": justificativa por mecanismo (alongamento, alavanca, função muscular) SEM citar estudo.
  * "opinião": julgamento, preferência ou recomendação sem evidência citada ("I think", "I like", "I'm putting").
  * "experiência pessoal": o que ele próprio ou clientes sentiram/fizeram.
  Na dúvida entre científica e opinião, escolha opinião.
- Escreva os demais campos em {language}."""

CHAPTER_PROMPT = """IMPORTANT: write every non-literal field (name_translated, muscle_focus, reasons, technique_tips,
statement text/evidence) in {language}.
Você analisa UM trecho de um vídeo que ranqueia exercícios numa tier list (S, A, B, C, D, F).
Extraia o exercício avaliado neste trecho e as afirmações relevantes (2–6).
Justificativas (`reasons`) e dicas (`technique_tips`) são um RESUMO analítico: condense o argumento em frases curtas
com suas palavras (ex.: "Maior amplitude que o hip thrust, mas difícil de progredir a carga"), sem rótulos, sem aspas
e sem copiar a fala. Só os campos `quote` e `tier_quote` são literais.
FIDELIDADE: cada justificativa e cada dica precisa estar dita NESTE trecho. Não acrescente conselhos gerais de treino
(ex.: "controle a fase excêntrica", "mantenha a coluna neutra") se o apresentador não os disse aqui. Não use o que você
sabe sobre o exercício. Se o trecho for curto e só trouxer o veredito, devolva listas curtas ou vazias.
O tier é o que o apresentador anuncia para este exercício ("I'm putting it in high B tier" → tier B, modificador alto).
""" + RULES

OVERVIEW_PROMPT = """Você analisa a transcrição completa de um vídeo que ranqueia exercícios numa tier list.
Extraia resumo, critérios de ranqueamento, a escolha final de melhor exercício (S+), os argumentos centrais
do apresentador e as afirmações feitas na introdução e no encerramento.
""" + RULES


ARTIFACT_RE = re.compile(r"^\s*(?:\[\d{1,2}:\d{2}\]\s*|(?:quote|tipo|classificação|científica|opinião)\s*:\s*)+", re.I)


def _clean(value):
    """Remove artefatos que modelos pequenos às vezes deixam: 'quote: ', '[11:37] ', aspas extras."""
    if isinstance(value, str):
        value = ARTIFACT_RE.sub("", value)
        value = re.sub(r"\s*/\s*tipo:.*$", "", value).strip().strip("'\"“”").strip()
        return value
    if isinstance(value, list):
        return [_clean(v) for v in value]
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    return value


def _check_statement(st: dict, loc: Locator, lo: float, hi: float) -> dict:
    st["quote_match"], t = loc.locate(st["quote"], lo, hi)
    st["timestamp"] = mmss(t)
    if st["kind"] == "científica" and not EVIDENCE_TERMS.search(loc.context(t, radius=12) + " " + st["quote"]):
        st["kind"] = "explicação biomecânica"
        st["reclassified"] = "LLM marcou como científica, mas não há estudo/dado citado perto do trecho"
        st["evidence"] = ""
    return st


def _tier_mentions(loc: "Locator", lo: float, hi: float) -> list[dict]:
    """Procura anúncios de tier no fluxo de PALAVRAS (um segmento pode cruzar a fronteira de capítulos)."""
    idx = [i for i, (_, t) in enumerate(loc.words) if lo <= t < hi]
    if not idx:
        return []
    toks = loc.tokens[idx[0] : idx[-1] + 1]
    text, starts, pos = "", [], 0
    for tok in toks:
        starts.append(pos)
        text += tok + " "
        pos = len(text)
    out = []
    for m in TIER_RE.finditer(text):
        w = max(k for k, st in enumerate(starts) if st <= m.start())
        t = loc.words[idx[0] + w][1]
        tier = m.group(2).upper() + ("+" if m.group(3) else "")
        mod = {"high": "alto", "low": "baixo"}.get((m.group(1) or "").lower(), "")
        context = " ".join(toks[max(0, w - 10) : w + 6])
        rel = text[m.end():].split()
        before = " ".join(toks[max(0, w - 10) : w])
        out.append({"t": t, "tier": tier, "mod": mod, "text": context, "before": before, "after": " ".join(rel[:3])})
    return out


def _same_exercise(name: str, title: str) -> bool:
    stem = lambda w: w[:5]
    a = {stem(w) for w in _norm(name).split() if w not in STOP}
    b = {stem(w) for w in _norm(title).split() if w not in STOP}
    return bool(a & b)


HYPOTHETICAL = re.compile(
    r"\b(close to|almost|just about|instinct|tempted|would|could|should|probably|if you|if we|on paper|"
    r"enough to get into|may|might|from|first|instead of|criteria|variations?|between|not putting|or)\b")
DECISIVE = re.compile(
    r"\b(back to|back down to|down to|bump\w*|knock\w*|drop\w*|mov\w+ (it|them) to|going in(to)?|putting|put (it|them|these|this one)|"
    r"sending|stay in|deserve|feeling|thinking|say|leave (it|them) in)\b")


def _score_mention(before: str, after: str) -> int:
    """Pontua uma menção de tier pelo contexto: veredito decidido (+1), hipótese/ressalva (−2/−1)."""
    score = 1 if DECISIVE.search(before) else 0
    if re.search(r"\bout of\b", before):
        score += 1 if re.search(r"\b(can t|cannot|couldn t)\b", before) else -2
    elif HYPOTHETICAL.search(before):
        score -= 2
    if re.match(r"\s*(plus\s+)?(but|or)\b", after):
        score -= 1
    return score


def _resolve_tier(ex: dict, loc: "Locator", lo: float, hi: float) -> None:
    """O tier vem da transcrição. Com várias menções no capítulo, vence a de melhor pontuação de contexto
    (decisão > hipótese), e em empate a mais tardia. S+ fica de fora (é a escolha final do vídeo)."""
    mentions = [m for m in _tier_mentions(loc, lo, hi + 4) if m["tier"] != "S+"]
    ex["llm_tier"] = (ex["tier"], ex["tier_modifier"])
    ex["tier_alternative"] = None
    if not mentions:
        ex["tier_source"] = "LLM (não encontrado na transcrição)"
        return
    for m in mentions:
        m["score"] = _score_mention(m["before"], m["after"])
    chosen = max(mentions, key=lambda m: (m["score"], m["t"]))
    ex["tier_ambiguous"] = len({(m["tier"], m["mod"]) for m in mentions}) > 1
    ex["tier"], ex["tier_modifier"] = chosen["tier"], chosen["mod"]
    ex["tier_quote"], ex["tier_timestamp"], ex["tier_quote_match"] = f"… {chosen['text']} …", mmss(chosen["t"]), 1.0
    ex["tier_source"] = "transcrição"
    # veredito condicional: "high C tier, but if you do them the modified way … low A tier"
    for m in mentions:
        if (m["tier"], m["mod"]) == (chosen["tier"], chosen["mod"]) or m["score"] <= -2 or m["t"] < lo + 5:
            continue  # hipóteses ("very close to S tier") e o veredito do exercício anterior não são condicionais
        a, b = sorted([m, chosen], key=lambda x: x["t"])
        between = loc.span_text(a["t"], b["t"] + 0.01)
        if abs(m["t"] - chosen["t"]) <= 12 and re.search(r"\b(if you|if we)\b", between):
            ex["tier_alternative"] = {"tier": m["tier"], "mod": m["mod"], "text": m["text"], "timestamp": mmss(m["t"])}
            break


# ---------------------------------------------------------------- orquestração

def _chunks(meta: dict, transcript: dict) -> list[dict]:
    duration = meta.get("duration") or transcript["segments"][-1]["end"]
    chapters = meta.get("chapters") or [
        {"start_time": t, "end_time": min(t + 90, duration), "title": ""} for t in range(0, int(duration), 90)
    ]
    out = []
    for c in chapters:
        segs = [s for s in transcript["segments"] if c["start_time"] <= (s["start"] + s["end"]) / 2 < c["end_time"]]
        if segs:
            out.append({"title": c["title"], "start": c["start_time"], "end": c["end_time"],
                        "text": "\n".join(f"[{mmss(s['start'])}] {s['text']}" for s in segs)})
    return out


def clean_exercise_texts(analysis: dict, transcript: dict | None = None) -> dict:
    """Remove artefatos do LLM em justificativas/dicas e separa o que é só citação literal da fala."""
    loc = Locator(transcript) if transcript else None
    for ex in analysis["exercises"]:
        for field in ("reasons", "technique_tips"):
            kept, quotes = [], ex.setdefault(f"{field}_verbatim", [])
            for raw in ex[field]:
                if LABEL_ONLY.match(raw):
                    continue  # só um rótulo, sem conteúdo
                text = strip_llm_wrappers(raw)
                if not text or LABEL_ONLY.match(text) or (TIER_RE.search(text) and len(text.split()) <= 12):
                    continue  # vazio ou anúncio de tier (já está na tier list)
                if loc and loc.locate(text)[0] >= 0.9:
                    if text not in quotes:
                        quotes.append(text)  # é fala literal: fica só no original
                    continue
                kept.append(text)
            ex[field] = kept
    return analysis


def link_top_pick(exercises: list[dict], top_pick: str, quote: str = "") -> None:
    """Marca o exercício escolhido como melhor de todos (S+). Compara sem espaços e hífens:
    "Faceaway Bayesian Cable Curl" = "Face Away Bayesian Cable Curl"."""
    compact = lambda t: _norm(t).replace(" ", "")
    key = compact(top_pick)
    for ex in exercises:
        ex.pop("top_pick", None)
    candidates = [ex for ex in exercises if key and (key in compact(ex["name"]) or compact(ex["name"]) in key)]
    # prefere a correspondência mais específica (nome mais longo contido na escolha)
    if candidates:
        best = max(candidates, key=lambda ex: len(compact(ex["name"])) if compact(ex["name"]) in key else len(key))
        best["top_pick"] = True
        best["top_pick_quote"] = quote


def postprocess(analysis: dict) -> dict:
    """Limpeza determinística das afirmações extraídas pelo LLM."""
    kept, seen = [], []
    for st in analysis["statements"]:
        quote = st["quote"].strip()
        words = _norm(quote).split()
        # anúncios de tier puros ("Low B tier.") já estão na tier list
        if TIER_RE.search(quote) and len(words) <= 9:
            continue
        # duplicatas: mesmo trecho ou trecho contido em outro já mantido
        nq = _norm(quote)
        if any(nq in o or o in nq for o in seen):
            continue
        if st["kind"] != "científica":
            st["evidence"] = ""
        elif (_norm(st["evidence"]) in {"", "none", "study", "studies", "research"} or len(st["evidence"]) > 120
              or _norm(st["evidence"])[:40] in nq):
            st["evidence"] = ""
        # paráfrase que só repete a citação em inglês não acrescenta nada
        if SequenceMatcher(None, _norm(st["text"]), nq).ratio() > 0.8:
            st["text"] = ""
        seen.append(nq)
        kept.append(st)
    analysis["statements"] = kept
    return analysis


def analyze(vdir: Path, force: bool = False, progress=print) -> Path:
    out = vdir / "analysis.json"
    if out.exists() and not force:
        return out
    meta = json.loads((vdir / "metadata.json").read_text())
    transcript = json.loads((vdir / "transcript.json").read_text())
    loc = Locator(transcript)
    lang = config.OUTPUT_LANGUAGE

    exercises, statements = [], []
    chunks = _chunks(meta, transcript)
    for i, ch in enumerate(chunks, 1):
        progress(f"    capítulo {i}/{len(chunks)}: {ch['title'] or mmss(ch['start'])}")
        user = f"Vídeo: {meta['title']}\nCapítulo: {ch['title'] or '(sem título)'}\n\nTrecho:\n{ch['text']}"
        try:
            r = _clean(_chat(CHAPTER_PROMPT.format(language=lang), user, ChapterExercise, num_ctx=4096).model_dump())
        except LLMOutputError as exc:
            progress(f"      falha do LLM neste capítulo ({exc}); mantendo só o tier")
            r = {"is_exercise": True, "name": ch["title"], "name_translated": ch["title"], "muscle_focus": "",
                 "tier": "sem tier", "tier_modifier": "", "reasons": [], "technique_tips": [], "tier_quote": "",
                 "statements": [], "llm_error": str(exc)}
        if _looks_portuguese(r["reasons"] + r["technique_tips"]):
            r = _clean(_chat(CHAPTER_PROMPT.format(language=lang), user + f"\n\nAnswer in {lang} only.",
                             ChapterExercise, num_ctx=4096).model_dump())
        copied = [t for t in r["reasons"] + r["technique_tips"] if loc.locate(t, ch["start"], ch["end"])[0] >= 0.8]
        if copied:  # uma nova tentativa pedindo reescrita do que saiu copiado da fala
            fix = ("\n\nNa tentativa anterior estes itens copiaram a fala; reescreva-os como resumo com suas palavras:\n- "
                   + "\n- ".join(copied))
            r2 = _clean(_chat(CHAPTER_PROMPT.format(language=lang), user + fix, ChapterExercise, num_ctx=4096).model_dump())
            if sum(loc.locate(t, ch["start"], ch["end"])[0] >= 0.8 for t in r2["reasons"] + r2["technique_tips"]) < len(copied):
                r = r2
        for st in r.pop("statements"):
            statements.append(_check_statement(st, loc, ch["start"], ch["end"]))
        intro = i == 1 or re.search(r"what makes|criteria|intro|outro|conclusion|book|sponsor|summary|keys|takeaway|recap|wrap.?up|final thoughts", ch["title"], re.I)
        has_tier = bool([m for m in _tier_mentions(loc, ch["start"], ch["end"]) if m["tier"] != "S+"])
        if intro or not has_tier and (not r["is_exercise"] or not _same_exercise(r["name"], ch["title"] or r["name"])):
            continue
        if ch["title"]:  # o título do capítulo (dado pelo criador) é o nome canônico do exercício
            r["llm_name"], r["name"] = r["name"], ch["title"]
        verify_support(r, ch["text"], loc, ch["start"], ch["end"])
        r["tier_quote_match"], t = loc.locate(r["tier_quote"], ch["start"], ch["end"])
        r["timestamp"] = mmss(ch["start"])
        r["tier_timestamp"] = mmss(t)
        r["chapter"] = ch["title"]
        _resolve_tier(r, loc, ch["start"], ch["end"])
        exercises.append(r)

    progress("    visão geral")
    full = "\n".join(f"[{mmss(s['start'])}] {s['text']}" for s in transcript["segments"])
    ov = _chat(OVERVIEW_PROMPT.format(language=lang), f"Vídeo: {meta['title']}\n\nTranscrição:\n{full}",
               Overview, num_ctx=config.OLLAMA_NUM_CTX).model_dump()
    ov = _clean(ov)
    for arg in ov["arguments"]:
        arg["quote_match"], t = loc.locate(arg["quote"])
        arg["timestamp"] = mmss(t)
    for st in ov.pop("closing_statements"):
        statements.append(_check_statement(st, loc, 0, float("inf")))

    if ov["top_pick"]:
        link_top_pick(exercises, ov["top_pick"], ov.get("top_pick_quote", ""))

    statements.sort(key=lambda s: s["timestamp"])
    seen = set()  # a introdução/encerramento pode repetir afirmações já extraídas por capítulo
    statements = [s for s in statements if not (_norm(s["quote"]) in seen or seen.add(_norm(s["quote"])))]
    analysis = {
        "summary": ov["summary"],
        "ranking_criteria": ov["ranking_criteria"],
        "top_pick": ov["top_pick"],
        "worst_pick": ov["worst_pick"],
        "exercises": exercises,
        "arguments": ov["arguments"],
        "statements": statements,
        "_llm": {"backend": "ollama", "model": config.OLLAMA_MODEL},
    }
    analysis = clean_exercise_texts(postprocess(analysis), transcript)
    out.write_text(json.dumps(analysis, ensure_ascii=False, indent=2))
    return out
