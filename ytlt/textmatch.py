"""Normalização de texto e localização de trechos na transcrição (tempos por palavra do WhisperX)."""

from difflib import SequenceMatcher
import re


def norm(s: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", s.lower()).split())


def mmss(seconds: float) -> str:
    return f"{int(seconds) // 60:02d}:{int(seconds) % 60:02d}"


class Locator:
    """Localiza trechos citados na transcrição por palavra (usa os tempos por palavra do WhisperX)."""

    def __init__(self, transcript: dict):
        self.words: list[tuple[str, float]] = []
        for seg in transcript["segments"]:
            for w in seg.get("words") or [{"word": t, "start": seg["start"]} for t in seg["text"].split()]:
                for tok in norm(w["word"]).split():
                    self.words.append((tok, w.get("start", seg["start"])))
        self.tokens = [w for w, _ in self.words]
        self.text = transcript

    def locate(self, quote: str, lo: float = 0, hi: float = float("inf")) -> tuple[float, float]:
        """Retorna (similaridade 0–1, segundo inicial) da melhor janela dentro de [lo, hi]."""
        q = norm(quote).split()
        if not q:
            return 0.0, lo
        n, target = len(q), " ".join(q)
        best, best_t = 0.0, lo
        idx = [i for i, (_, t) in enumerate(self.words) if lo - 5 <= (t or 0) <= hi + 5]
        for i in idx[: max(1, len(idx) - n + 1)]:
            window = " ".join(self.tokens[i : i + n])
            if window == target:
                return 1.0, self.words[i][1]
            r = SequenceMatcher(None, target, window).ratio()
            if r > best:
                best, best_t = r, self.words[i][1]
        return round(best, 2), best_t

    def span_text(self, t0: float, t1: float) -> str:
        return " ".join(tok for tok, t in self.words if t0 <= t <= t1)

    def context(self, t: float, radius: float = 25) -> str:
        return " ".join(s["text"] for s in self.text["segments"] if t - radius <= s["start"] <= t + radius)


_norm = norm  # compatibilidade
