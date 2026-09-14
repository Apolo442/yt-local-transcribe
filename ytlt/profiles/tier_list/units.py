"""Unidades traduzíveis do perfil tier_list (chaves estáveis por campo da análise).

Só conteúdo produzido pela análise; transcrição e citações literais ficam no idioma original.
"""

import json
from pathlib import Path


def units(vdir: Path) -> dict[str, str]:
    """Textos traduzíveis de um vídeo, com chaves estáveis."""
    a = json.loads((vdir / "analysis.json").read_text())
    out: dict[str, str] = {"summary": a["summary"]}
    for i, c in enumerate(a["ranking_criteria"]):
        out[f"ranking_criteria.{i}"] = c
    for k in ("top_pick", "worst_pick"):
        if a.get(k):
            out[k] = a[k]
    for i, g in enumerate(a["arguments"]):
        out[f"arguments.{i}.thesis"] = g["thesis"]
        out[f"arguments.{i}.reasoning"] = g["reasoning"]
    for i, e in enumerate(a["exercises"]):
        out[f"exercises.{i}.name"] = e["name"]
        out[f"exercises.{i}.muscle_focus"] = e["muscle_focus"]
        if e.get("emphasis"):
            out[f"exercises.{i}.emphasis"] = e["emphasis"]["region"]
        for j, r in enumerate(e["reasons"]):
            out[f"exercises.{i}.reasons.{j}"] = r
        for j, t in enumerate(e["technique_tips"]):
            out[f"exercises.{i}.technique_tips.{j}"] = t
    for i, st in enumerate(a["statements"]):
        if st["kind"] == "científica":
            if st.get("text"):
                out[f"statements.{i}.paraphrase"] = st["text"]
            if st.get("evidence"):
                out[f"statements.{i}.evidence"] = st["evidence"]
    return {k: v for k, v in out.items() if isinstance(v, str) and v.strip()}
