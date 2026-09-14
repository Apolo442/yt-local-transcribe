"""Etapa 7: gera o Markdown de cada vídeo e o índice da enciclopédia."""

import json
import re
import unicodedata
from pathlib import Path

from ... import config
from ...references import format_citation

TIER_ORDER = ["S+", "S", "A+", "A", "B+", "B", "C+", "C", "D+", "D", "F", "sem tier"]
QUOTE_OK = 0.8  # abaixo disso o trecho citado pelo LLM não foi encontrado na transcrição


def slugify(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"\(.*?\)", "", text)
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def ts_to_seconds(ts: str) -> int:
    parts = [int(p) for p in re.findall(r"\d+", ts)] or [0]
    secs = 0
    for p in parts[-3:]:
        secs = secs * 60 + p
    return secs


def ts_link(ts: str, video_id: str) -> str:
    return f"[{ts}](https://youtu.be/{video_id}?t={ts_to_seconds(ts)})"


MOD_ARROW = {"alto": " ↑", "baixo": " ↓", "": ""}


def flag(item: dict, key: str = "quote_match") -> str:
    return "" if item.get(key, 1) >= QUOTE_OK else " ⚠️"


def fmt_date(d: str | None) -> str:
    return f"{d[:4]}-{d[4:6]}-{d[6:]}" if d and len(d) == 8 else (d or "?")


def render_video(vdir: Path) -> Path:
    meta = json.loads((vdir / "metadata.json").read_text())
    transcript = json.loads((vdir / "transcript.json").read_text())
    a = json.loads((vdir / "analysis.json").read_text())
    cons = _load(vdir / "consensus.json")
    refs = _load(vdir / "references.json") or {"references": [], "mentions": []}
    vid = meta["id"]
    # numeração das referências por DOI (a descrição pode repetir o mesmo estudo em grupos diferentes)
    ref_num: dict[int, int] = {}
    unique: list[dict] = []
    for r in refs["references"]:
        key = r.get("doi") or r["url"]
        hit = next((u for u in unique if (u.get("doi") or u["url"]) == key), None)
        if hit is None:
            hit = {**r, "groups": []}
            unique.append(hit)
        if r["group"] and r["group"] not in hit["groups"]:
            hit["groups"].append(r["group"])
        ref_num[r["id"]] = unique.index(hit) + 1

    def cite_at(ts: str) -> str:
        t = ts_to_seconds(ts)
        nums = sorted({ref_num[i] for m in refs["mentions"] if m["time"] - 5 <= t <= m["time"] + 40
                       for i in m.get("confirmed", [])})
        return " " + "".join(f"[[{n}]](#ref-{n})" for n in nums) if nums else ""

    L: list[str] = []
    add = L.append

    add("---")
    add(f'title: "{meta["title"]}"')
    add(f"video_id: {vid}")
    add(f"url: {meta['webpage_url']}")
    add(f'channel: "{meta["channel"]}"')
    add(f"upload_date: {fmt_date(meta.get('upload_date'))}")
    add(f"duration_s: {meta['duration']}")
    add(f"playlist_index: {meta['playlist_index']}")
    add(f"transcription_model: whisperx/{transcript['model']}")
    add(f"analysis_model: {a['_llm']['backend']}/{a['_llm']['model']}")
    add("---\n")
    add(f"# {meta['title']}\n")

    add("## Metadados\n")
    add("| Campo | Valor |\n|---|---|")
    m, s = divmod(meta["duration"], 60)
    for k, v in [
        ("Canal", meta["channel"]),
        ("Publicado em", fmt_date(meta.get("upload_date"))),
        ("Duração", f"{m}:{s:02d}"),
        ("Visualizações", f"{meta.get('view_count') or 0:,}".replace(",", ".")),
        ("Curtidas", f"{meta.get('like_count') or 0:,}".replace(",", ".")),
        ("Link", meta["webpage_url"]),
    ]:
        add(f"| {k} | {v} |")
    if meta.get("chapters"):
        add("\n**Capítulos:**\n")
        for c in meta["chapters"]:
            t = int(c["start_time"])
            add(f"- {ts_link(f'{t // 60:02d}:{t % 60:02d}', vid)} {c['title']}")
    add("")

    add("## Resumo\n")
    add(a["summary"] + "\n")
    if a["ranking_criteria"]:
        add("**Critérios de ranqueamento usados pelo apresentador:**\n")
        L.extend(f"- {c}" for c in a["ranking_criteria"])
        add("")

    if a.get("top_pick") or a.get("worst_pick"):
        add(f"**Melhor de todos (S+):** {a.get('top_pick') or '—'} · **Pior de todos:** {a.get('worst_pick') or '—'}\n")

    add("## Tier list\n")
    add("| Tier | Exercícios |\n|---|---|")
    by_tier: dict[str, list[dict]] = {}
    for ex in a["exercises"]:
        by_tier.setdefault("S+" if ex.get("top_pick") else ex["tier"], []).append(ex)
    mod_rank = {"alto": 0, "": 1, "baixo": 2}
    for tier in TIER_ORDER:
        if tier in by_tier:
            items = sorted(by_tier[tier], key=lambda e: mod_rank[e["tier_modifier"]])
            add(f"| **{tier}** | " + ", ".join(e["name"] + MOD_ARROW[e["tier_modifier"]] for e in items) + " |")
    add("\n↑ = *high* (topo do tier) · ↓ = *low* (base do tier)\n")

    add("## Exercícios\n")
    for ex in a["exercises"]:
        title = ex["name"] if ex["name_translated"].lower() == ex["name"].lower() else f"{ex['name']} — {ex['name_translated']}"
        add(f"### {title}{flag(ex, 'tier_quote_match')}\n")
        tier_txt = f"{ex['tier']} ({ex['tier_modifier']})" if ex["tier_modifier"] else ex["tier"]
        if alt := ex.get("tier_alternative"):
            alt_txt = f"{alt['tier']} ({alt['mod']})" if alt["mod"] else alt["tier"]
            tier_txt += f" · condicional: {alt_txt} — “… {alt['text']} …” ({ts_link(alt['timestamp'], vid)})"
        if ex.get("top_pick"):
            tier_txt += " → promovido a **S+** (melhor exercício do vídeo)"
        add(f"- **Tier:** {tier_txt} — anunciado em {ts_link(ex['tier_timestamp'], vid)}")
        if ex.get("tier_source") != "transcrição":
            add(f"- ⚠️ **Tier não confirmado na transcrição** (fonte: {ex.get('tier_source', 'LLM')})")
        add(f"- **Foco:** {ex['muscle_focus']}")
        if ex.get("emphasis"):
            em = ex["emphasis"]
            add(f"- **Ênfase principal:** {em['region']} — “{em['quote']}” ({ts_link(em['timestamp'], vid)})")
        add(f"- **Capítulo:** {ts_link(ex['timestamp'], vid)} {ex['chapter']}")
        if ex["reasons"]:
            add("- **Justificativa:**")
            L.extend(f"  - {r}" for r in ex["reasons"])
        if ex["technique_tips"]:
            add("- **Dicas de execução:**")
            L.extend(f"  - {t}" for t in ex["technique_tips"])
        add(f"\n> “{ex['tier_quote']}”\n")

    add("## Argumentos do apresentador\n")
    for i, arg in enumerate(a["arguments"], 1):
        add(f"{i}. **{arg['thesis']}**{flag(arg)} ({ts_link(arg['timestamp'], vid)})  ")
        add(f"   {arg['reasoning']}")
    add("")

    add("## Ciência × opinião\n")
    sections = [
        ("científica", "Afirmações com base científica citada"),
        ("explicação biomecânica", "Explicações biomecânicas (sem estudo citado)"),
        ("opinião", "Opiniões"),
        ("experiência pessoal", "Experiência pessoal / relatos"),
    ]
    for kind, heading in sections:
        items = [st for st in a["statements"] if st["kind"] == kind]
        if not items:
            continue
        add(f"### {heading}\n")
        for st in items:
            ev = f" — evidência citada: {st['evidence']}" if st.get("evidence") else ""
            chk = " ❓" if st.get("reclassified") else ""
            cite = cite_at(st["timestamp"]) if kind == "científica" else ""
            para = f"<br>  ↳ {st['text']}" if st.get("text") else ""
            add(f"- “{st['quote']}”{cite} ({ts_link(st['timestamp'], vid)}){flag(st)}{chk}{ev}{para}")
        add("")
    add("> ⚠️ = o trecho citado pela análise automática não foi encontrado na transcrição; revise.  ")
    add("> ❓ = o LLM classificou como científica, mas não há estudo/dado citado perto do trecho; foi rebaixada.\n")

    add("## Estudos citados\n")
    if unique:
        for n, r in enumerate(unique, 1):
            add(f'<a id="ref-{n}"></a>**[{n}]** {format_citation(r)}  ')
            add(f"   *Tema na descrição:* {', '.join(r['groups']) or '—'}")
            spoken = [m for m in refs["mentions"] if any(ref_num[i] == n for i in m.get("confirmed", []))]
            if spoken:
                add("   *Citado em:* " + ", ".join(ts_link(f"{int(m['time']) // 60:02d}:{int(m['time']) % 60:02d}", vid)
                                             for m in spoken))
            add("")
        pend = [m for m in refs["mentions"] if not m.get("confirmed")]
        if pend:
            add("**Falas que citam estudos sem referência confirmada:** " + ", ".join(
                ts_link(f"{int(m['time']) // 60:02d}:{int(m['time']) % 60:02d}", vid) for m in pend) + "\n")
    else:
        add("Nenhuma referência listada na descrição.\n")

    add("## Transcrição\n")
    if cons:
        sm = cons["summary"]
        agree = " · ".join(f"{k}: {v}%" for k, v in sm["agreement_pct"].items())
        add(f"Consenso de {', '.join(cons['sources'])} — concordância com o WhisperX: {agree}. "
            f"{sm['regions_total']} divergências analisadas, {sm['changed_from_whisperx']} correções aplicadas, "
            f"**{sm['uncertain']} trechos incertos** (marcados com ❓).\n")
    add("<details><summary>Transcrição completa com timestamps</summary>\n")
    for seg in transcript["segments"]:
        t = int(seg["start"])
        unc = "".join(f" ❓ *(alternativas: {' / '.join(u['alternatives'])})*" for u in seg.get("uncertain", []))
        add(f"- {ts_link(f'{t // 60:02d}:{t % 60:02d}', vid)} {seg['text']}{unc}")
    add("\n</details>\n")

    out = config.ENCYCLOPEDIA_DIR / f"{meta['playlist_index']:02d}-{slugify(meta['title'])}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L))
    return out


def _load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def render_index() -> Path:
    rows = []
    for vdir in sorted((config.DATA_DIR / "videos").glob("*")):
        if not (vdir / "analysis.json").exists():
            continue
        meta = json.loads((vdir / "metadata.json").read_text())
        a = json.loads((vdir / "analysis.json").read_text())
        top = [e["name"] + (" (S+)" if e.get("top_pick") else "") for e in a["exercises"] if e["tier"] in ("S+", "S")]
        fname = f"{meta['playlist_index']:02d}-{slugify(meta['title'])}.md"
        rows.append(f"| {meta['playlist_index']} | [{meta['title']}]({fname}) | {len(a['exercises'])} | {', '.join(top) or '—'} |")
    out = config.ENCYCLOPEDIA_DIR / "index.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "# Enciclopédia de exercícios\n\n"
        "| # | Vídeo | Exercícios | Tier S/S+ |\n|---|---|---|---|\n" + "\n".join(rows) + "\n"
    )
    return out
