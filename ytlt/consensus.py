"""Etapa 2b: transcrição final por consenso de reconhecedores independentes.

Fontes: WhisperX large-v3 (base, tem tempos por palavra), Parakeet TDT v3 e legendas automáticas do YouTube.

1. Cada fonte vira uma sequência de tokens canônicos (minúsculas, sem pontuação, números por extenso,
   "gonna"→"going to", hesitações removidas), para que diferenças só de grafia não contem como divergência.
2. Parakeet e YouTube são alinhados ao WhisperX (difflib). As regiões divergentes são unificadas.
3. Em cada região:
   - divergência só de grafia (mesmas letras juntas, ex.: "warmup"/"warm up") → ignorada;
   - WhisperX concorda com ao menos uma fonte → mantém o WhisperX;
   - senão → redecodifica o trecho de áudio isolado (Whisper large-v3, beam 10, com o contexto
     anterior como prompt) e vota com 4 hipóteses; ≥3 votos decidem;
   - variantes de grafia são agrupadas; termos do título/capítulos/descrição vencem; maioria simples vence se
     tiver apoio de uma fonte independente (Parakeet/YouTube);
   - empate → mantém o WhisperX e marca como incerto quando as opções diferem em palavras de conteúdo.
4. Trechos sem pontuação/maiúsculas no WhisperX recebem a pontuação do Parakeet.
5. O texto final é dividido em frases e realinhado por palavra (wav2vec2).

Saídas: `transcript.json`, `transcript.srt` e `consensus.json` (relatório de cada decisão).
"""

import json
import re
from difflib import SequenceMatcher
from pathlib import Path

from . import config
from .llm import unload_ollama
from .transcribe import free_gpu, realign, write_srt

FILLERS = {"uh", "um", "umm", "er", "ah", "hmm", "mm", "mhm", "oh"}
EXPAND = {
    "gonna": "going to", "wanna": "want to", "gotta": "got to", "kinda": "kind of", "sorta": "sort of",
    "alright": "all right", "okay": "ok", "thisll": "this will", "thatll": "that will", "itll": "it will",
    "lbs": "pounds", "lb": "pound", "vs": "versus", "percent": "percent", "%": "percent", "°": "degrees",
    "&": "and",
}
ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen " \
       "seventeen eighteen nineteen".split()
TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def _num_words(n: int) -> str:
    if n < 20:
        return ONES[n]
    if n < 100:
        return TENS[n // 10] + ("" if n % 10 == 0 else " " + ONES[n % 10])
    if n < 1000:
        return ONES[n // 100] + " hundred" + ("" if n % 100 == 0 else " " + _num_words(n % 100))
    if n < 10000:
        return _num_words(n // 1000) + " thousand" + ("" if n % 1000 == 0 else " " + _num_words(n % 1000))
    return str(n)


def canon(surface: str) -> list[str]:
    """Um token de superfície → zero ou mais tokens canônicos."""
    s = surface.lower().replace("’", "'")
    s = s.replace("%", " percent ").replace("°", " degrees ").replace("&", " and ")
    out = []
    for part in re.split(r"[\s\-–—/]+", s):
        m = re.fullmatch(r"\W*(\d+)(st|nd|rd|th|s)?\W*", part)
        if m and int(m.group(1)) < 100000:
            out += _num_words(int(m.group(1))).split()
            continue
        w = re.sub(r"[^a-z0-9]", "", part)
        if not w or w in FILLERS:
            continue
        out += EXPAND.get(w, w).split()
    return out


class Stream:
    """Sequência de tokens canônicos com referência à palavra de superfície de origem."""

    def __init__(self, surfaces: list[str], times: list[tuple[float, float]] | None = None):
        self.surfaces, self.times = surfaces, times
        self.tokens: list[str] = []
        self.owner: list[int] = []
        for i, s in enumerate(surfaces):
            for t in canon(s):
                self.tokens.append(t)
                self.owner.append(i)

    def surface(self, t1: int, t2: int) -> str:
        idx = sorted(set(self.owner[t1:t2]))
        return " ".join(self.surfaces[i] for i in idx)


class BMap:
    """Fronteira i em a → posição em b. `lo` é usado como início de região e `hi` como fim, para que
    tokens inseridos em b numa fronteira de a (inserção pura) fiquem dentro da região."""

    def __init__(self):
        self.lo: dict[int, int] = {}
        self.hi: dict[int, int] = {}

    def set(self, i: int, j_lo: int, j_hi: int) -> None:
        self.lo[i] = min(self.lo.get(i, j_lo), j_lo)
        self.hi[i] = max(self.hi.get(i, j_hi), j_hi)

    def __contains__(self, i: int) -> bool:
        return i in self.lo

    def span(self, i1: int, i2: int) -> tuple[int, int]:
        return self.lo[i1], self.hi[i2]


def _boundary_map(a: list[str], b: list[str]) -> tuple[BMap, list[tuple[int, int]]]:
    """Mapa de fronteiras de a para b e os blocos não iguais (em coordenadas de a)."""
    bmap, diffs = BMap(), []
    for op, i1, i2, j1, j2 in SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if op == "equal":
            for k in range(i2 - i1 + 1):
                bmap.set(i1 + k, j1 + k, j1 + k)
        else:
            bmap.set(i1, j1, j1)
            bmap.set(i2, j2, j2)
            diffs.append((i1, i2))
    return bmap, diffs


def _regions(len_a: int, diff_lists: list[list[tuple[int, int]]], maps: list[dict]) -> list[tuple[int, int]]:
    spans = sorted(d for dl in diff_lists for d in dl)
    merged: list[list[int]] = []
    for i1, i2 in spans:
        # inclui 1 token de contexto de cada lado para que inserções vizinhas se unam
        if merged and i1 <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], i2)
        else:
            merged.append([i1, i2])
    # fronteiras precisam existir em todos os mapas (pontos de concordância)
    out = []
    for i1, i2 in merged:
        while i1 > 0 and any(i1 not in m for m in maps):  # noqa: B023
            i1 -= 1
        while i2 < len_a and any(i2 not in m for m in maps):
            i2 += 1
        if out and i1 <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], i2))
        else:
            out.append((i1, i2))
    return out


INDEPENDENT = {"parakeet", "youtube"}  # WhisperX e a redecodificação são o mesmo modelo
FUNCTION_WORDS = set("""a an the and or but so now then well also just really very in on at to of for with by from
as is are was were be it its it's itd it'd id i'd i im i'm you your youre you're youll you'll they theyll they'll
them their we he she this that these those there here yeah oh ok okay like""".split())


def _glossary(meta: dict) -> dict[str, str]:
    """Termos com grafia fornecida pelo criador (título, capítulos, descrição): chave canônica → grafia."""
    text = " ".join([meta.get("title") or "", meta.get("description") or ""] +
                    [c["title"] for c in meta.get("chapters") or []])
    out = {}
    for w in re.findall(r"[A-Za-z][A-Za-z\-']{3,}", text):
        key = "".join(canon(w))
        if key and key not in FUNCTION_WORDS:
            out.setdefault(key, w)
    # siglas/jargão curtos ("lat", "pec", "RDL") só dos títulos de capítulo, que são listas de exercícios
    for c in meta.get("chapters") or []:
        for w in re.findall(r"\b[A-Za-z]{3}\b", c["title"]):
            key = "".join(canon(w))
            if key and key not in FUNCTION_WORDS:
                out.setdefault(key, w)
    return out


def _similar(a: str, b: str) -> bool:
    return a == b or (min(len(a), len(b)) >= 3 and SequenceMatcher(None, a, b).ratio() >= 0.8)


def decide_region(e: dict, keys: dict[str, str], glossary: dict[str, str]) -> None:
    """Vota entre as hipóteses de uma região disputada (sem LLM)."""
    # 1) agrupa variantes de grafia ("pec"/"peck", "delt"/"del")
    clusters: list[list[str]] = []
    for src, k in keys.items():
        for c in clusters:
            if _similar(keys[c[0]], k):
                c.append(src)
                break
        else:
            clusters.append([src])
    clusters.sort(key=lambda c: -len(c))
    e["votes"] = [[*c] for c in clusters]

    # 2) grafia do criador: se um grupo corresponde a um termo do glossário, ele vence
    for c in clusters:
        for src in c:
            words = e["hypotheses"][src].split()
            if words and all("".join(canon(w)) in glossary or "".join(canon(w)) in FUNCTION_WORDS for w in words) \
                    and any("".join(canon(w)) in glossary for w in words):
                e["decision"], e["method"] = src, f"termo do glossário do vídeo ({', '.join(c)})"
                return

    best, second = clusters[0], clusters[1] if len(clusters) > 1 else []
    if len(best) > len(second) and (len(best) >= 3 or INDEPENDENT & set(best)):
        e["decision"] = _pick_source(best)
        e["method"] = f"maioria ({', '.join(best)})"
        variants = {keys[src] for src in best}
        if len(best) == 2 and len(variants) > 1 and set(canon(e["hypotheses"][e["decision"]])) - FUNCTION_WORDS:
            # dois votos que nem concordam na grafia: provável palavra rara mal reconhecida
            e["uncertain"] = True
            e["alternatives"] = [e["hypotheses"][_pick_source(c)] or "(nada)" for c in clusters]
        return

    # 3) empate ou maioria só do Whisper: mantém o WhisperX; alerta só se o sentido pode mudar
    e["decision"] = "whisperx"
    e["method"] = "empate — mantido o WhisperX"
    content = {w for src in keys for w in canon(e["hypotheses"][src])} - FUNCTION_WORDS
    if len(clusters) > 1 and content:
        e["uncertain"] = True
        e["alternatives"] = [e["hypotheses"][_pick_source(c)] or "(nada)" for c in clusters]


def _redecode(model, audio, t0: float, t1: float, prompt: str) -> str:
    sr = 16000
    clip = audio[int(max(0, t0) * sr) : int(t1 * sr)]
    segments, _ = model.transcribe(
        clip, language="en", beam_size=10, temperature=0.0, initial_prompt=prompt[-400:],
        condition_on_previous_text=False, vad_filter=False, without_timestamps=True,
    )
    return " ".join(s.text.strip() for s in segments)


def _norm_key(tokens: list[str]) -> str:
    return "".join(tokens)


def build(vdir: Path, force: bool = False, progress=print) -> Path:
    out = vdir / "transcript.json"
    if out.exists() and not force:
        return out
    meta = json.loads((vdir / "metadata.json").read_text())
    wx_raw = json.loads((vdir / "asr_whisperx.json").read_text())

    words, seg_of = [], []
    for sid, seg in enumerate(wx_raw["segments"]):
        for w in seg["words"]:
            if "start" in w:
                words.append(w)
                seg_of.append(sid)
    wx = Stream([w["word"] for w in words], [(w["start"], w["end"]) for w in words])
    others: dict[str, Stream] = {}
    for name, fname in (("parakeet", "asr_parakeet.json"), ("youtube", "asr_youtube.json")):
        p = vdir / fname
        if p.exists():
            segs = json.loads(p.read_text())["segments"]
            others[name] = Stream([w for s in segs for w in s["text"].split()])
    if not others:
        raise RuntimeError("consenso precisa de pelo menos uma fonte além do WhisperX")

    maps, diffs = {}, []
    for name, st in others.items():
        bmap, d = _boundary_map(wx.tokens, st.tokens)
        maps[name] = bmap
        diffs.append(d)
    regions = _regions(len(wx.tokens), diffs, list(maps.values()))

    report = {"sources": ["whisperx", *others], "words_whisperx": len(wx.tokens), "regions": []}
    pending = []
    for i1, i2 in regions:
        hyp = {"whisperx": wx.tokens[i1:i2]}
        spans = {name: maps[name].span(i1, i2) for name in others}
        for name, st in others.items():
            hyp[name] = st.tokens[slice(*spans[name])]
        keys = {n: _norm_key(t) for n, t in hyp.items()}
        entry = {
            "i1": i1, "i2": i2,
            "time": round(wx.times[wx.owner[min(i1, len(wx.owner) - 1)]][0], 2),
            "hypotheses": {n: (wx.surface(i1, i2) if n == "whisperx" else others[n].surface(*spans[n])) for n in hyp},
        }
        if len(set(keys.values())) == 1:
            continue  # só grafia/espaçamento
        if any(keys[n] == keys["whisperx"] for n in others):
            entry["decision"], entry["method"] = "whisperx", "maioria (WhisperX + outra fonte)"
            report["regions"].append(entry)
            continue
        entry["keys"] = keys
        pending.append(entry)

    progress(f"    {len(report['regions'])} divergências resolvidas por maioria; {len(pending)} para arbitragem")

    # --- redecodificação dos trechos disputados (GPU) ---
    if pending:
        import whisperx
        from faster_whisper import WhisperModel

        unload_ollama()
        audio = whisperx.load_audio(str(vdir / "audio.wav"))
        model = WhisperModel(config.WHISPER_MODEL, device=config.DEVICE, compute_type=config.WHISPER_COMPUTE_TYPE)
        for e in pending:
            c1, c2 = max(0, e["i1"] - 4), min(len(wx.tokens), e["i2"] + 4)
            t0 = wx.times[wx.owner[c1]][0] - 0.6
            t1 = wx.times[wx.owner[min(c2, len(wx.owner)) - 1]][1] + 0.6
            prompt = wx.surface(max(0, c1 - 60), c1)
            text = _redecode(model, audio, t0, t1, prompt)
            rd = Stream(text.split())
            # recorta, dentro da redecodificação, o que corresponde à região (âncoras = contexto)
            ctx = wx.tokens[c1:c2]
            bmap, _ = _boundary_map(ctx, rd.tokens)
            a, b = e["i1"] - c1, e["i2"] - c1
            if a in bmap and b in bmap:
                r1, r2 = bmap.span(a, b)
                e["hypotheses"]["redecode"] = rd.surface(r1, r2)
                e["keys"]["redecode"] = _norm_key(rd.tokens[r1:r2])
            e["redecode_window"] = text
        del model
        free_gpu()

    # --- votação ---
    glossary = _glossary(meta)
    for e in pending:
        keys = e.pop("keys")
        decide_region(e, keys, glossary)
        report["regions"].append(e)

    report["regions"].sort(key=lambda r: r["i1"])
    final_words = _apply(wx, others, maps, report["regions"], seg_of)
    sentences = _sentences(final_words)
    progress(f"    realinhando {len(sentences)} frases")
    unload_ollama()
    segments = realign(vdir, sentences, wx_raw["language"])
    _mark_uncertain(segments, report["regions"])

    changed = [r for r in report["regions"] if r["decision"] != "whisperx"]
    report["summary"] = {
        "regions_total": len(report["regions"]),
        "changed_from_whisperx": len(changed),
        "uncertain": sum(bool(r.get("uncertain")) for r in report["regions"]),
        "agreement_pct": {
            n: round(100 * (1 - sum(max(i2 - i1, 0) for i1, i2 in d) / max(1, len(wx.tokens))), 2)
            for n, d in zip(others, diffs)
        },
    }
    (vdir / "consensus.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    out.write_text(json.dumps({"language": wx_raw["language"], "model": f"consenso({', '.join(report['sources'])})",
                               "segments": segments}, ensure_ascii=False, indent=1))
    write_srt(segments, vdir / "transcript.srt")
    return out


def _pick_source(srcs: list[str]) -> str:
    """Prefere fontes com pontuação/maiúsculas para o texto de superfície."""
    for pref in ("whisperx", "redecode", "parakeet", "youtube"):
        if pref in srcs:
            return pref
    return srcs[0]


def _apply(wx: Stream, others: dict[str, Stream], maps: dict, regions: list[dict], seg_of: list[int]) -> list[dict]:
    """Monta a lista final de palavras {word, start, end} aplicando as decisões sobre o WhisperX."""
    by_start = {r["i1"]: r for r in regions if r["decision"] != "whisperx"}
    punct = _punctuation_donor(wx, others.get("parakeet"), maps.get("parakeet"), seg_of)
    out, i = [], 0
    n = len(wx.tokens)
    done_owner = set()
    while i <= n:
        r = by_start.pop(i, None)
        if r:
            prev_end = out[-1]["end"] if out else 0.0
            t0 = wx.times[wx.owner[i]][0] if i < n else prev_end
            t1 = wx.times[wx.owner[r["i2"] - 1]][1] if r["i2"] > i else t0
            for w in r["hypotheses"][r["decision"]].split():
                out.append({"word": w, "start": t0, "end": t1})
            done_owner.update(wx.owner[i : r["i2"]])
            if r["i2"] > i:
                i = r["i2"]
                continue
        if i == n:
            break
        o = wx.owner[i]
        if o not in done_owner:
            done_owner.add(o)
            out.append({"word": punct.get(o, wx.surfaces[o]), "start": wx.times[o][0], "end": wx.times[o][1]})
        i += 1
    return out


def _punctuation_donor(wx: Stream, pk: Stream | None, bmap: BMap | None, seg_of: list[int]) -> dict[int, str]:
    """Em segmentos do WhisperX sem pontuação (artefato do modo em lote), usa a grafia do Parakeet
    — só para palavras idênticas e alinhadas 1:1, então nenhuma palavra muda, apenas pontuação/maiúsculas."""
    if pk is None:
        return {}
    by_seg: dict[int, list[int]] = {}
    for o, sid in enumerate(seg_of):
        by_seg.setdefault(sid, []).append(o)
    bad_words = set()
    for owners in by_seg.values():
        n_punct = sum(bool(re.search(r"[.?!,;:]$", wx.surfaces[o])) for o in owners)
        if len(owners) >= 15 and n_punct < len(owners) / 25:
            bad_words.update(owners)
    donor: dict[int, str] = {}
    for ti, o in enumerate(wx.owner):
        if o not in bad_words or ti not in bmap or ti + 1 not in bmap:
            continue
        j = bmap.lo[ti]
        if bmap.hi[ti + 1] != j + 1 or j >= len(pk.owner):
            continue
        po = pk.owner[j]
        if canon(pk.surfaces[po]) == canon(wx.surfaces[o]):
            donor[o] = pk.surfaces[po]
    return donor


def _sentences(words: list[dict], soft: int = 22, hard: int = 45) -> list[dict]:
    """Uma frase por segmento; frases longas quebram na primeira vírgula após `soft` palavras."""
    segs, cur = [], []
    for w in words:
        cur.append(w)
        end = re.search(r"[.?!][\"”')]*$", w["word"])
        comma = len(cur) >= soft and w["word"].endswith((",", ";", ":"))
        if end or comma or len(cur) >= hard:
            segs.append(cur)
            cur = []
    if cur:
        segs.append(cur)
    out = []
    for s in segs:
        text = " ".join(w["word"] for w in s)
        out.append({"start": s[0]["start"], "end": max(w["end"] for w in s), "text": text})
    return out


def _mark_uncertain(segments: list[dict], regions: list[dict]) -> None:
    for r in regions:
        if r.get("uncertain"):
            for s in segments:
                if s["start"] - 0.5 <= r["time"] <= s["end"] + 0.5:
                    s.setdefault("uncertain", []).append({"time": r["time"], "alternatives": r["alternatives"]})
                    break
