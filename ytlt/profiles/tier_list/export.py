"""Exporta tudo em um único JSON para o web app (`export/easy-jeff.json`).

Entidades:
- `videos`: metadados, resumo, critérios, argumentos, afirmações e transcrição por frase;
- `exercises`: lista plana (todos os vídeos) para busca/filtro por tier;
- `studies`: estudos citados, deduplicados por DOI, prontos para consulta humana
  (citação APA, DOI, PubMed, PMC, acesso aberto) e com onde/para quê cada um foi citado.

Resumos (abstracts) dos artigos não são exportados: pertencem às editoras e ficam só nos dados locais
de verificação; o link da PubMed/DOI leva a eles.
"""

import json
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

from ... import config
from ...i18n import load_pt
from ...references import enrich
from .render import slugify, ts_to_seconds

SCHEMA_VERSION = 2  # v2: textos da análise como {"en": …, "pt": …}; transcrição/citações no original


def _yt(video_id: str, seconds: float | int) -> str:
    return f"https://youtu.be/{video_id}?t={int(seconds)}"


def _apa_author(name: str) -> str:
    """"Plotkin DL" → "Plotkin, D. L."; nomes já em outro formato passam como estão."""
    m = re.fullmatch(r"(.+?)\s+([A-Z]{1,4})", name.strip())
    if not m:
        return name.strip()
    family = m.group(1).strip()
    # "ESCAMILLA" ou "Rocha-JÚnior" (caixa quebrada em acentos vinda do Crossref) → caixa de título
    if family.isupper() or re.search(r"[a-zà-ÿ][A-ZÀ-Ý]|[A-ZÀ-Ý]{2}[a-zà-ÿ]", family):
        family = "-".join(w[:1].upper() + w[1:].lower() for w in family.split("-"))
        family = " ".join(w[:1].upper() + w[1:] for w in family.split(" "))
    return f"{family}, {' '.join(c + '.' for c in m.group(2))}"


def apa_citation(ref: dict) -> str:
    authors = [_apa_author(a) for a in ref.get("authors") or []]
    if len(authors) > 20:
        who = ", ".join(authors[:19]) + ", … " + authors[-1]
    elif len(authors) > 1:
        who = ", ".join(authors[:-1]) + ", & " + authors[-1]
    else:
        who = authors[0] if authors else ""
    parts = [f"{who} ({ref.get('year') or 's.d.'})." if who else f"({ref.get('year') or 's.d.'}).",
             f"{ref['title'][:1].upper() + ref['title'][1:].rstrip('.')}.", f"{ref['journal']}." if ref.get("journal") else ""]
    if ref.get("doi"):
        parts.append(f"https://doi.org/{ref['doi']}")
    return " ".join(p for p in parts if p)


def _study_id(ref: dict) -> str:
    if ref.get("doi"):
        return "doi:" + ref["doi"].lower()
    return "url:" + ref["url"]


def build(out: Path | None = None) -> Path:
    out = out or config.PROJECT_DIR / "export" / f"{config.PROJECT_NAME}.json"
    studies: dict[str, dict] = {}
    videos, exercises = [], []

    for vdir in sorted((config.DATA_DIR / "videos").glob("*")):
        if not (vdir / "analysis.json").exists():
            continue
        meta = json.loads((vdir / "metadata.json").read_text())
        a = json.loads((vdir / "analysis.json").read_text())
        tr = json.loads((vdir / "transcript.json").read_text())
        cons = json.loads((vdir / "consensus.json").read_text())
        refs_path = vdir / "references.json"
        refs = json.loads(refs_path.read_text()) if refs_path.exists() else {"references": [], "mentions": []}
        vid = meta["id"]
        pt = load_pt(vdir)

        def tr_(key: str, en_text):
            """Campo bilíngue; pt=None enquanto não houver tradução revisada."""
            if en_text in (None, ""):
                return None
            return {"en": en_text, "pt": pt.get(key)}

        # estudos: enriquece (cacheado em references.json) e deduplica por DOI
        changed = False
        local_ids: dict[int, str] = {}
        for r in refs["references"]:
            if r.get("resolved") and not r.get("enriched"):
                enrich(r)
                changed = True
            sid = _study_id(r)
            local_ids[r["id"]] = sid
            st = studies.setdefault(sid, {
                "id": sid,
                "status": "resolved" if r.get("resolved") else "unresolved",
                "citation_apa": apa_citation(r) if r.get("resolved") else None,
                "title": r.get("title"), "authors": r.get("authors") or [], "year": r.get("year"),
                "journal": r.get("journal"), "doi": r.get("doi"), "pmid": r.get("pmid"), "pmcid": r.get("pmcid"),
                "links": {k: v for k, v in {
                    "doi": r.get("doi_url"), "pubmed": r.get("pubmed_url"), "pmc": r.get("pmc_url"),
                    "open_access": r.get("open_access_url"), "source_in_description": r["url"],
                }.items() if v},
                "is_open_access": r.get("is_open_access"),
                "publication_type": r.get("publication_type"),
                "metadata_source": r.get("source"),
                "listed_in": [], "cited_in": [],
            })
            listing = {"video_id": vid, "topic": r.get("group") or None, "description_url": r["url"]}
            if listing not in st["listed_in"]:
                st["listed_in"].append(listing)
        if changed:
            refs_path.write_text(json.dumps(refs, ensure_ascii=False, indent=2))

        mentions = []
        for m in refs["mentions"]:
            confirmed = [local_ids[i] for i in m.get("confirmed", []) if i in local_ids]
            mentions.append({"timestamp_s": round(m["time"], 1), "youtube_url": _yt(vid, m["time"]),
                             "spoken_text": m["text"], "study_ids": confirmed, "confirmed": bool(confirmed)})
            for sid in confirmed:
                studies[sid]["cited_in"].append({
                    "video_id": vid, "video_title": meta["title"], "timestamp_s": round(m["time"], 1),
                    "youtube_url": _yt(vid, m["time"]), "spoken_text": m["text"],
                })

        def studies_near(t: float) -> list[str]:
            return sorted({s for m in mentions if m["timestamp_s"] - 5 <= t <= m["timestamp_s"] + 40
                           for s in m["study_ids"]})

        video_exercises, seen_ids = [], set()
        for ei, e in enumerate(a["exercises"]):
            t = ts_to_seconds(e["timestamp"])
            tt = ts_to_seconds(e["tier_timestamp"])
            item = {
                "id": _unique_id(vid, e["chapter"] or e["name"], seen_ids),
                "video_id": vid, "video_title": meta["title"], "muscle_group": _muscle_group(meta["title"]),
                "name": tr_(f"exercises.{ei}.name", e["name"]), "chapter": e.get("chapter"),
                "tier": "S+" if e.get("top_pick") else e["tier"],
                "tier_announced": e["tier"], "tier_modifier": {"alto": "high", "baixo": "low"}.get(e["tier_modifier"]),
                "tier_alternative": e.get("tier_alternative") and {
                    "tier": e["tier_alternative"]["tier"],
                    "modifier": {"alto": "high", "baixo": "low"}.get(e["tier_alternative"]["mod"]),
                    "quote": e["tier_alternative"]["text"],
                    "timestamp_s": ts_to_seconds(e["tier_alternative"]["timestamp"]),
                },
                "is_top_pick": bool(e.get("top_pick")),
                "tier_quote": e["tier_quote"].strip("… "), "tier_timestamp_s": tt, "tier_youtube_url": _yt(vid, tt),
                "chapter_timestamp_s": t, "youtube_url": _yt(vid, t),
                "muscle_focus": tr_(f"exercises.{ei}.muscle_focus", e["muscle_focus"]),
                # parte do músculo que o apresentador diz ser a mais trabalhada (null se não foi dito)
                "emphasis": e.get("emphasis") and {
                    "region": tr_(f"exercises.{ei}.emphasis", e["emphasis"]["region"]),
                    "quote": e["emphasis"]["quote"],
                    "timestamp_s": ts_to_seconds(e["emphasis"]["timestamp"]),
                    "youtube_url": _yt(vid, ts_to_seconds(e["emphasis"]["timestamp"])),
                },
                "reasons": [tr_(f"exercises.{ei}.reasons.{j}", r) for j, r in enumerate(e["reasons"])],
                "technique_tips": [tr_(f"exercises.{ei}.technique_tips.{j}", x) for j, x in enumerate(e["technique_tips"])],
                # frases do apresentador citadas literalmente (original apenas)
                "reasons_quotes": e.get("reasons_verbatim", []),
                "technique_tips_quotes": e.get("technique_tips_verbatim", []),
                "tier_verified_in_transcript": e.get("tier_source") == "transcrição",
            }
            video_exercises.append(item)
            exercises.append(item)

        statements = []
        for si, st in enumerate(a["statements"]):
            t = ts_to_seconds(st["timestamp"])
            kind = {"científica": "scientific", "explicação biomecânica": "biomechanical_reasoning",
                    "opinião": "opinion", "experiência pessoal": "personal_experience"}[st["kind"]]
            statements.append({
                "kind": kind, "quote": st["quote"],
                "paraphrase": tr_(f"statements.{si}.paraphrase", st.get("text")),
                "evidence_mentioned": tr_(f"statements.{si}.evidence", st.get("evidence")),
                "timestamp_s": t, "youtube_url": _yt(vid, t),
                "study_ids": studies_near(t) if kind == "scientific" else [],
                "downgraded_from_scientific": bool(st.get("reclassified")),
                "quote_verified": st.get("quote_match", 0) >= 0.8,
            })

        videos.append({
            "id": vid, "playlist_index": meta["playlist_index"], "title": meta["title"],
            "muscle_group": _muscle_group(meta["title"]), "url": meta["webpage_url"], "channel": meta["channel"],
            "upload_date": meta.get("upload_date"), "duration_s": meta["duration"], "view_count": meta.get("view_count"),
            "thumbnail": meta.get("thumbnail"),
            "chapters": [{"title": c["title"], "start_s": int(c["start_time"]), "youtube_url": _yt(vid, c["start_time"])}
                         for c in meta.get("chapters") or []],
            "summary": tr_("summary", a["summary"]),
            "ranking_criteria": [tr_(f"ranking_criteria.{i}", c) for i, c in enumerate(a["ranking_criteria"])],
            "top_pick": tr_("top_pick", a.get("top_pick")), "worst_pick": tr_("worst_pick", a.get("worst_pick")),
            "exercise_ids": [x["id"] for x in video_exercises],
            "arguments": [{"thesis": tr_(f"arguments.{i}.thesis", g["thesis"]),
                           "reasoning": tr_(f"arguments.{i}.reasoning", g["reasoning"]), "quote": g["quote"],
                           "timestamp_s": ts_to_seconds(g["timestamp"]), "youtube_url": _yt(vid, ts_to_seconds(g["timestamp"]))}
                          for i, g in enumerate(a["arguments"])],
            "statements": statements,
            "study_mentions": mentions,
            "transcript_quality": {
                "method": "consensus of WhisperX large-v3, Parakeet TDT v3 and YouTube auto-captions",
                "agreement_pct": cons["summary"]["agreement_pct"],
                "corrections_applied": cons["summary"]["changed_from_whisperx"],
                "uncertain_spans": cons["summary"]["uncertain"],
            },
            "transcript": [{"start_s": s["start"], "end_s": s["end"], "text": s["text"],
                            **({"uncertain_alternatives": [u["alternatives"] for u in s["uncertain"]]}
                               if s.get("uncertain") else {})}
                           for s in tr["segments"]],
        })

    tier_rank = {t: i for i, t in enumerate(["S+", "S", "A+", "A", "B+", "B", "C+", "C", "D+", "D", "F", "sem tier"])}
    mod_rank = {"high": 0, None: 1, "low": 2}
    exercises.sort(key=lambda x: (x["muscle_group"], tier_rank.get(x["tier"], 99), mod_rank[x["tier_modifier"]]))
    study_list = sorted(studies.values(), key=lambda s: (s["status"] != "resolved", -len(s["cited_in"]),
                                                         (s["authors"] or [""])[0]))
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "languages": {"original": "en", "analysis": ["en", "pt"],
                      "note": "transcript and quotes are kept in the original language only"},
        "source": {"playlist_url": config.PLAYLIST_URL, "channel": videos[0]["channel"] if videos else None},
        "stats": {
            "videos": len(videos), "exercises": len(exercises),
            "studies_resolved": sum(s["status"] == "resolved" for s in study_list),
            "studies_unresolved": sum(s["status"] != "resolved" for s in study_list),
            "study_mentions_confirmed": sum(m["confirmed"] for v in videos for m in v["study_mentions"]),
            "study_mentions_unconfirmed": sum(not m["confirmed"] for v in videos for m in v["study_mentions"]),
        },
        "videos": videos,
        "exercises": exercises,
        "studies": study_list,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    return out


def _unique_id(video_id: str, name: str, seen: set[str]) -> str:
    """ID estável e único por vídeo; mantém o conteúdo entre parênteses ("Front Foot Elevated")."""
    base = re.sub(r"[^a-z0-9]+", "-", unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()).strip("-")
    candidate, n = f"{video_id}:{base}", 2
    while candidate in seen:
        candidate, n = f"{video_id}:{base}-{n}", n + 1
    seen.add(candidate)
    return candidate


def _muscle_group(title: str) -> str:
    t = title.lower()
    for key, name in [("glute", "glutes"), ("bicep", "biceps"), ("tricep", "triceps"), ("shoulder", "shoulders"),
                      ("quad", "quads"), ("chest", "chest"), ("back", "back"), ("hamstring", "hamstrings"),
                      ("calf", "calves"), ("calves", "calves"), ("ab", "abs")]:
        if re.search(rf"\b{key}", t):
            return name
    return "other"
