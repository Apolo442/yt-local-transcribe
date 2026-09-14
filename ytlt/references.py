"""Etapa de referências: estudos citados no vídeo.

1. Lê a seção "References" da descrição (grupos por tema + links).
2. Resolve cada link em fontes bibliográficas confiáveis, sem adivinhar:
   - PubMed (link com PMID) → NCBI E-utilities;
   - link contendo DOI → Crossref;
   - LWW/journals (título no próprio link) → busca no Crossref, aceita só se o título começar com o
     texto do link e o ano bater;
   - PMC → conversor de IDs do NCBI; MDPI → Crossref por ISSN/volume/número/artigo;
   - ResearchGate com título no link → Crossref (prefixo exato do título);
   - outras páginas de periódico/preprint → DOI em <meta name="citation_doi">;
   - o resto (ex.: ResearchGate só com número, que bloqueia acesso automatizado) → "não resolvido".
   Resumos (abstracts) vêm da PubMed ou, na falta, do OpenAlex.
3. Encontra na transcrição as falas que citam estudos e pede ao LLM local para ligar cada fala a uma
   referência. A confirmação é determinística: exige números com unidade ou termos raros no catálogo
   presentes na fala e no artigo; o resto vira "possível" ou pedido de print.

Só são enviados a serviços externos identificadores e títulos de artigos — nunca vídeo, áudio ou transcrição.
Saídas: `references.json` e, se houver pendências, `screens/PEDIDOS.md`.
"""

import html
import json
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from difflib import SequenceMatcher
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

UA = {"User-Agent": "yt-encyclopedia/0.1 (ferramenta local de pesquisa)"}
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
MENTION_RE = re.compile(
    r"\b(stud(y|ies)|research(ers)?|paper|meta.?analys[ie]s|emg|trial|review|researchers|scientists|evidence)\b", re.I
)


# ---------------------------------------------------------------- HTTP

def _get(url: str, as_json: bool = True, retries: int = 3):
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
                data = r.read()
            time.sleep(0.35)  # respeita limites de taxa (NCBI: 3 req/s sem chave)
            return json.loads(data) if as_json else data.decode("utf-8", "replace")
        except urllib.error.HTTPError as exc:
            if exc.code in (404, 403, 400):
                return None
            time.sleep(2 * (attempt + 1))
        except OSError:
            time.sleep(2 * (attempt + 1))
    return None


def _norm(s: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", html.unescape(s).lower()).split())


# ---------------------------------------------------------------- descrição

def parse_description(description: str) -> list[dict]:
    lines = description.splitlines()
    try:
        start = next(i for i, l in enumerate(lines) if re.fullmatch(r"\s*(references?|sources?|studies)\s*:?\s*", l, re.I))
    except StopIteration:
        return []
    refs, group = [], ""
    for line in lines[start + 1 :]:
        line = line.strip()
        if re.fullmatch(r"-{3,}", line):
            break
        if not line:
            continue
        urls = re.findall(r"https?://\S+", line)
        if urls:
            label = line.split(urls[0])[0].strip(" :-–")
            for u in urls:
                refs.append({"group": group, "label": label, "url": u.rstrip(").,")})
        else:
            group = line.rstrip(":")
    for i, r in enumerate(refs, 1):
        r["id"] = i
    return refs


# ---------------------------------------------------------------- resolução

def _from_pubmed(pmid: str) -> dict | None:
    summ = _get(f"{EUTILS}/esummary.fcgi?db=pubmed&retmode=json&id={pmid}")
    if not summ or pmid not in summ.get("result", {}):
        return None
    d = summ["result"][pmid]
    doi = next((a["value"] for a in d.get("articleids", []) if a["idtype"] == "doi"), None)
    return {
        "title": d["title"].rstrip("."), "authors": [a["name"] for a in d.get("authors", [])],
        "journal": d.get("fulljournalname") or d.get("source"), "year": (d.get("pubdate") or "")[:4],
        "doi": doi, "pmid": pmid, "source": "PubMed",
    }


def _from_crossref(doi: str) -> dict | None:
    r = _get(f"https://api.crossref.org/works/{urllib.parse.quote(doi)}")
    return _crossref_item(r["message"]) if r else None


def _crossref_item(it: dict) -> dict:
    year = next(iter((it.get("published") or it.get("issued") or {}).get("date-parts", [[None]])[0]), None)
    return {
        "title": html.unescape((it.get("title") or [""])[0]),
        "authors": [f"{a.get('family', '')} {''.join(p[0] for p in a.get('given', '').split() if p)}".strip()
                    for a in it.get("author", [])],
        "journal": html.unescape((it.get("container-title") or [""])[0]), "year": str(year or ""),
        "doi": it["DOI"].lower(), "pmid": None, "source": "Crossref",
    }


def _search_crossref_by_slug(slug_title: str, year: str | None) -> dict | None:
    params = {"query.bibliographic": slug_title, "rows": 5}
    if year:
        params["filter"] = f"from-pub-date:{year},until-pub-date:{year}"
    q = urllib.parse.urlencode(params)
    r = _get(f"https://api.crossref.org/works?{q}")
    target = _norm(slug_title)
    for it in (r or {}).get("message", {}).get("items", []):
        title = _norm((it.get("title") or [""])[0])
        prefix = title[: len(target)]
        if SequenceMatcher(None, target, prefix).ratio() >= 0.95:
            return _crossref_item(it)
    return None


def _pmid_for_doi(doi: str) -> str | None:
    r = _get(f"{EUTILS}/esearch.fcgi?db=pubmed&retmode=json&term={urllib.parse.quote(doi)}[doi]")
    ids = (r or {}).get("esearchresult", {}).get("idlist", [])
    return ids[0] if len(ids) == 1 else None


def _abstract(ref: dict) -> str:
    if ref.get("pmid"):
        xml = _get(f"{EUTILS}/efetch.fcgi?db=pubmed&retmode=xml&id={ref['pmid']}", as_json=False)
        if xml:
            root = ET.fromstring(xml)
            parts = ["".join(n.itertext()) for n in root.iter("AbstractText")]
            if parts:
                return " ".join(parts)
    if ref.get("doi"):
        r = _get(f"https://api.openalex.org/works/doi:{urllib.parse.quote(ref['doi'])}?select=abstract_inverted_index")
        inv = (r or {}).get("abstract_inverted_index")
        if inv:
            pos = sorted((p, w) for w, ps in inv.items() for p in ps)
            return " ".join(w for _, w in pos)
    return ""


def _from_pmc(pmcid: str) -> dict | None:
    r = _get(f"https://pmc.ncbi.nlm.nih.gov/tools/idconv/api/v1/articles/?ids={pmcid}&format=json"
             "&tool=yt-encyclopedia")
    rec = ((r or {}).get("records") or [{}])[0]
    if rec.get("pmid"):
        return _from_pubmed(str(rec["pmid"]))
    return _from_crossref(rec["doi"]) if rec.get("doi") else None


def _from_mdpi(issn: str, volume: str, issue: str, article: str) -> dict | None:
    q = urllib.parse.urlencode({"filter": f"issn:{issn}", "query.bibliographic": article, "rows": 10})
    for it in (_get(f"https://api.crossref.org/works?{q}") or {}).get("message", {}).get("items", []):
        if it.get("volume") == volume and it.get("issue") == issue and it["DOI"].endswith(article):
            return _crossref_item(it)
    return None


def _from_page_meta(url: str) -> dict | None:
    """Páginas de periódicos/preprints (OJS, Highwire) expõem o DOI em <meta name="citation_doi">."""
    page = _get(url, as_json=False, retries=1)
    if not page:
        return None
    meta = dict(re.findall(r'<meta\s+name="(citation_doi|DC\.Identifier\.DOI|citation_title|citation_date|'
                           r'citation_publication_date|citation_journal_title)"\s+content="([^"]+)"', page))
    doi = meta.get("citation_doi") or meta.get("DC.Identifier.DOI")
    if doi and (found := _from_crossref(doi)):
        return found
    if doi and meta.get("citation_title"):
        authors = re.findall(r'<meta\s+name="citation_author"\s+content="([^"]+)"', page)
        date = meta.get("citation_publication_date") or meta.get("citation_date") or ""
        return {"title": html.unescape(" ".join(meta["citation_title"].split())), "authors": authors,
                "journal": meta.get("citation_journal_title", ""), "year": date[:4], "doi": doi.lower(),
                "pmid": None, "source": "metadados da página do editor"}
    return None


def _overrides() -> dict:
    """Identificações feitas manualmente (ex.: DOI lido no navegador quando o site bloqueia robôs)."""
    from . import config

    path = config.DATA_DIR / "reference_overrides.json"
    return json.loads(path.read_text()) if path.exists() else {}


def enrich(ref: dict) -> dict:
    """Links para consulta humana: acesso aberto (OpenAlex), PMCID e URLs de DOI/PubMed."""
    if not ref.get("resolved") or ref.get("enriched"):
        return ref
    if ref.get("doi"):
        r = _get(f"https://api.openalex.org/works/doi:{urllib.parse.quote(ref['doi'])}"
                 "?select=ids,open_access,best_oa_location,type") or {}
        ids = r.get("ids") or {}
        oa = r.get("open_access") or {}
        best = r.get("best_oa_location") or {}
        ref["openalex_id"] = ids.get("openalex")
        ref["pmcid"] = (ids.get("pmcid") or "").rstrip("/").rsplit("/", 1)[-1] or None
        if not ref.get("pmid") and ids.get("pmid"):
            ref["pmid"] = ids["pmid"].rstrip("/").rsplit("/", 1)[-1]
        ref["is_open_access"] = bool(oa.get("is_oa"))
        ref["open_access_url"] = best.get("pdf_url") or best.get("landing_page_url") or oa.get("oa_url")
        ref["publication_type"] = r.get("type")
    ref["doi_url"] = f"https://doi.org/{ref['doi']}" if ref.get("doi") else None
    ref["pubmed_url"] = f"https://pubmed.ncbi.nlm.nih.gov/{ref['pmid']}/" if ref.get("pmid") else None
    ref["pmc_url"] = f"https://pmc.ncbi.nlm.nih.gov/articles/{ref['pmcid']}/" if ref.get("pmcid") else None
    ref["enriched"] = True
    return ref


def resolve(ref: dict) -> dict:
    url = ref["url"]
    meta = None
    if override := _overrides().get(url):
        meta = _from_crossref(override["doi"]) or _from_page_meta(f"https://doi.org/{override['doi']}")
        if meta:
            meta["source"] += f" (identificação manual: {override.get('note', '')})"
    elif m := re.search(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)", url):
        meta = _from_pubmed(m.group(1))
    elif m := re.search(r"/(PMC\d+)", url):
        meta = _from_pmc(m.group(1))
    elif m := re.search(r"mdpi\.com/(\d{4}-\d{3}[\dX])/(\d+)/(\d+)/(\d+)", url):
        meta = _from_mdpi(*m.groups())
    elif m := re.search(r"researchgate\.net/publication/\d+_([^/?#]+)", url):
        meta = _search_crossref_by_slug(m.group(1).replace("_", " "), None)
    elif m := re.search(r"(10\.\d{4,9}/[^\s?#]+?)(?:/full|/abstract|/pdf)?(?:[?#]|$)", url):
        meta = _from_crossref(m.group(1))
    elif m := re.search(r"journals\.lww\.com/[^/]+/(?:fulltext|abstract)/(\d{4})/\d+/([^/]+?)(?:\.\d+)?\.aspx", url):
        meta = _search_crossref_by_slug(m.group(2).replace("_", " "), m.group(1))
    elif "researchgate.net" not in url:
        meta = _from_page_meta(url)
    if not meta:
        return {**ref, "resolved": False, "reason": "link sem identificador bibliográfico acessível"}
    if meta.get("doi") and not meta.get("pmid"):
        meta["pmid"] = _pmid_for_doi(meta["doi"])
    meta["abstract"] = _abstract(meta)
    return enrich({**ref, **meta, "resolved": True})


def format_citation(r: dict) -> str:
    if not r.get("resolved"):
        return f"{r.get('label') or 'Referência não identificada'} — {r['url']}"
    authors = r["authors"]
    who = ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else "")
    ids = []
    if r.get("doi"):
        ids.append(f"[doi:{r['doi']}](https://doi.org/{r['doi']})")
    if r.get("pmid"):
        ids.append(f"[PMID {r['pmid']}](https://pubmed.ncbi.nlm.nih.gov/{r['pmid']}/)")
    return f"{who} ({r['year']}). *{r['title']}*. {r['journal']}. {' · '.join(ids)}"


# ---------------------------------------------------------------- menções na fala

class Link(BaseModel):
    reference_id: int = Field(description="id da referência; 0 se nenhuma corresponde")
    confidence: Literal["alta", "média", "baixa"]
    matching_details: list[str] = Field(description="Detalhes concretos presentes TANTO na fala quanto no título/resumo "
                                                    "(desenho, duração, população, músculos, medida). Vazio se nenhum")
    reason: str


class MentionLinks(BaseModel):
    links: list[Link]


LINK_PROMPT = """Você liga falas de um vídeo sobre musculação aos estudos listados na descrição do vídeo.
Para a fala dada, indique quais referências ela descreve. Regras:
- Confiança "alta" só se houver detalhes concretos coincidentes (ex.: duração do estudo, tipo de participante,
  exercícios comparados, medida usada) entre a fala e o título/resumo. Tema parecido NÃO basta.
- Uma fala genérica ("research shows…") sem detalhes pode apontar para um grupo temático, mas com confiança média/baixa.
- Nunca invente detalhes. Se nenhuma referência corresponder, devolva reference_id 0.
- Escreva `reason` em português, em uma frase."""


def find_mentions(transcript: dict) -> list[dict]:
    segs = transcript["segments"]
    hits = [i for i, s in enumerate(segs) if MENTION_RE.search(s["text"])]
    groups: list[list[int]] = []
    for i in hits:
        if groups and segs[i]["start"] - segs[groups[-1][-1]]["start"] <= 20:
            groups[-1].append(i)
        else:
            groups.append([i])
    out = []
    for g in groups:
        lo, hi = max(0, g[0] - 1), min(len(segs), g[-1] + 3)
        out.append({"time": segs[g[0]]["start"], "text": " ".join(s["text"] for s in segs[lo:hi])})
    return out


STOPWORDS = set("""a an the and or but if so of to in on at for with by from as is are was were be been being it its this
that these those they them their there here i you your we our he she his her not no just also very really more most
less than then about into over under up down out only even still much many some any all both each such can could
will would should may might must do does did done have has had get gets got one ones which who whom what when where
why how while because like well just going pretty lot kind""".split())
# Vocabulário comum a quase todo estudo do tema: coincidir nele não prova que é o mesmo estudo.
GENERIC = set("""glute glutes gluteal gluteus maximus medius minimus muscle muscles muscular activity activation active
exercise exercises training trained train study studies research researcher researchers paper evidence effect effects
effective hip hips thrust thrusts squat squats squatting deadlift deadlifts growth grow growing hypertrophy barbell leg
legs lower upper back body strength found show shows showed compared comparison group groups participant participants
result results significant data extension flexion isolation isolate isolated target suggest high
external internal position range motion""".split())
SYNONYMS = {"emg": "electromyograph", "electromyographic": "electromyograph", "electromyography": "electromyograph",
            "deeper": "deep", "depth": "deep", "depths": "deep", "full": "deep", "stretched": "stretch",
            "lengthened": "stretch", "length": "stretch", "untrained": "untrain", "novice": "untrain",
            "beginners": "untrain", "weeks": "week", "wk": "week"}


def _stem(w: str) -> str:
    w = SYNONYMS.get(w, w)
    if w in SYNONYMS.values() or len(w) <= 5:
        return w
    return re.sub(r"(ing|ies|es|ed|s)$", "", w)


def _terms(text: str) -> set[str]:
    words = re.findall(r"[a-z]+", html.unescape(text).lower())
    return {_stem(w) for w in words if len(w) >= 3 and w not in STOPWORDS and w not in GENERIC}


def _unit_numbers(text: str) -> set[str]:
    """Números acompanhados de unidade ("9 weeks", "nine-week", "30 participants")."""
    text = re.sub(r"-", " ", text)
    words = {"two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8",
             "nine": "9", "ten": "10", "twelve": "12", "sixteen": "16", "twenty": "20"}
    text = re.sub(r"\b(" + "|".join(words) + r")\b", lambda m: words[m.group(1).lower()], text, flags=re.I)
    return {f"{n} {_stem(u.lower()).rstrip('s')}" for n, u in
            re.findall(r"\b(\d+)\s+(weeks?|wk|months?|days?|sets?|reps?|participants?|subjects?|men|women)\b", text, re.I)}


def _ref_text(ref: dict) -> str:
    return f"{ref.get('title', '')} {ref.get('abstract', '')} {ref.get('group', '')}"


def evidence_overlap(mention: str, ref: dict, catalog: list[dict]) -> dict:
    """Coincidências verificáveis entre a fala e o artigo.

    Um termo só conta se for raro no catálogo do vídeo (aparece em no máximo 1 estudo distinto), então
    vocabulário compartilhado por todos os estudos do tema ("maximus", "between") não prova nada."""
    distinct = {(r.get("doi") or r["url"]): r for r in catalog if r.get("resolved")}
    doc_freq: dict[str, int] = {}
    for r in distinct.values():
        for t in _terms(_ref_text(r)):
            doc_freq[t] = doc_freq.get(t, 0) + 1
    rare = sorted(t for t in _terms(mention) & _terms(_ref_text(ref)) if doc_freq.get(t, 0) <= 1)
    in_title = [t for t in rare if t in _terms(f"{ref.get('title', '')} {ref.get('group', '')}")]
    numbers = sorted(_unit_numbers(mention) & _unit_numbers(_ref_text(ref)))
    return {"rare_terms": rare, "title_terms": in_title, "numbers": numbers,
            "verified": bool(numbers) or (len(rare) >= 2 and len(in_title) >= 1)}


def _numbers(s: str) -> set[str]:
    words = {"two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
             "ten": "10", "twelve": "12", "fifteen": "15", "twenty": "20", "thirty": "30"}
    s = re.sub(r"\b(" + "|".join(words) + r")\b", lambda m: words[m.group(1).lower()], s, flags=re.I)
    return set(re.findall(r"\d+", s))


def link_mentions(mentions: list[dict], refs: list[dict], title: str) -> list[dict]:
    from .llm import chat as _chat

    catalog = "\n".join(
        f"[{r['id']}] grupo: {r['group'] or '-'} | "
        + (f"{r['title']} ({r['year']}, {r['journal']}) | resumo: {r['abstract'][:900]}" if r.get("resolved")
           else f"(não identificado; rótulo: {r.get('label') or '-'})")
        for r in refs
    )
    by_id = {r["id"]: r for r in refs}
    for m in mentions:
        user = f"Vídeo: {title}\n\nReferências:\n{catalog}\n\nFala [{int(m['time'])//60:02d}:{int(m['time'])%60:02d}]:\n{m['text']}"
        res = _chat(LINK_PROMPT, user, MentionLinks, num_ctx=8192)
        m["links"] = []
        for link in res.links:
            ref = by_id.get(link.reference_id)
            if not ref:
                continue
            entry = link.model_dump()
            # números citados como detalhe coincidente precisam existir na fala e no título/resumo
            haystack = f"{ref.get('title', '')} {ref.get('abstract', '')}"
            for det in link.matching_details:
                nums = _numbers(det)
                if nums and not (nums & _numbers(haystack) and nums & _numbers(m["text"])):
                    entry["confidence"] = "baixa"
                    entry["reason"] += " [rebaixado: número do detalhe não confere com o resumo/fala]"
            if entry["confidence"] == "alta" and not link.matching_details:
                entry["confidence"] = "média"
            if entry["confidence"] == "alta" and not ref.get("resolved"):
                entry["confidence"] = "média"  # sem título/resumo não há como confirmar
            m["links"].append(entry)
        # Confirmação = LLM aponta (alta/média) E há coincidência verificável de detalhes.
        for l in m["links"]:
            l["overlap"] = evidence_overlap(m["text"], by_id[l["reference_id"]], refs)
        # Candidatos também vêm da verificação determinística (o LLM local às vezes subestima o óbvio).
        proposed = {l["reference_id"] for l in m["links"]}
        for r in refs:
            if r["id"] not in proposed and r.get("resolved"):
                ov = evidence_overlap(m["text"], r, refs)
                if ov["verified"] and (ov["numbers"] or len(ov["title_terms"]) >= 2):
                    m["links"].append({"reference_id": r["id"], "confidence": "média", "matching_details": [],
                                       "reason": "coincidência determinística de termos raros/números", "overlap": ov})
        seen_doi: set[str] = set()
        m["confirmed"] = []
        for l in m["links"]:
            ref = by_id[l["reference_id"]]
            key = ref.get("doi") or ref["url"]
            if ref.get("resolved") and l["overlap"]["verified"] and key not in seen_doi:
                seen_doi.add(key)
                m["confirmed"].append(l["reference_id"])
    return mentions


# ---------------------------------------------------------------- orquestração

def build(vdir: Path, force: bool = False, progress=print) -> Path:
    out = vdir / "references.json"
    if out.exists() and not force:
        return out
    meta = json.loads((vdir / "metadata.json").read_text())
    transcript = json.loads((vdir / "transcript.json").read_text())

    refs = [resolve(r) for r in parse_description(meta.get("description") or "")]
    progress(f"    {sum(r['resolved'] for r in refs)}/{len(refs)} referências da descrição resolvidas")
    mentions = find_mentions(transcript)
    if refs and mentions:
        mentions = link_mentions(mentions, refs, meta["title"])
    confirmed = sum(bool(m.get("confirmed")) for m in mentions)
    progress(f"    {len(mentions)} menções a estudos na fala; {confirmed} confirmadas")

    pending = [m for m in mentions if not m.get("confirmed")]
    unresolved = [r for r in refs if not r["resolved"]]
    screens = vdir / "screens"
    if pending or unresolved:
        screens.mkdir(exist_ok=True)
        vid = meta["id"]
        lines = ["# Prints pedidos", "",
                 "Salve nesta pasta um print da tela em cada timestamp abaixo, com o nome `mm-ss.png`",
                 "(ex.: `03-19.png`). Só é preciso quando o estudo aparece na tela.", ""]
        if pending:
            lines += ["## Falas sobre estudos sem referência confirmada", ""]
            for m in pending:
                t = int(m["time"])
                lines.append(f"- [{t//60:02d}:{t%60:02d}](https://youtu.be/{vid}?t={t}) → `{t//60:02d}-{t%60:02d}.png`")
            lines.append("")
        if unresolved:
            lines += ["## Links da descrição que não puderam ser identificados", ""]
            lines += [f"- [{r['id']}] {r['group']}: {r['url']}" for r in unresolved]
        (screens / "PEDIDOS.md").write_text("\n".join(lines) + "\n")

    out.write_text(json.dumps({"references": refs, "mentions": mentions}, ensure_ascii=False, indent=2))
    return out
