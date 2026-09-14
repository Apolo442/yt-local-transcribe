from ytlt.consensus import Stream, _boundary_map, _glossary, canon, decide_region


def test_canon_normalizes_spelling_numbers_and_fillers():
    assert canon("45°") == ["forty", "five", "degrees"]
    assert canon("9-week") == ["nine", "week"]
    assert canon("gonna") == ["going", "to"]
    assert canon("um,") == []


def test_boundary_map_keeps_insertions_inside_region():
    a = Stream("I think you should do this now".split())
    b = Stream("I think you should really do this".split())
    bmap, diffs = _boundary_map(a.tokens, b.tokens)
    assert diffs == [(4, 4), (6, 7)]
    assert [bmap.span(i1, i2) for i1, i2 in diffs] == [(4, 5), (7, 7)]


def _decide(hyps, meta=None):
    keys = {k: "".join(canon(v)) for k, v in hyps.items()}
    e = {"hypotheses": hyps}
    decide_region(e, keys, _glossary(meta or {"title": "", "description": "", "chapters": []}))
    return e


def test_plurality_with_independent_support_wins():
    # caso real (vídeo de bíceps): a maioria simples tem apoio do Parakeet, fonte independente do Whisper
    e = _decide({"whisperx": "degree pre-stric curl", "parakeet": "degree Preacher curl",
                 "youtube": "45° pre-ural", "redecode": "degree preacher curl"})
    assert "preacher" in e["hypotheses"][e["decision"]].lower()
    assert not e.get("uncertain")


def test_spelling_variants_are_clustered():
    # "delta"/"delt" contam como a mesma palavra: não há disputa real, fica o WhisperX
    e = _decide({"whisperx": "delta", "parakeet": "delt", "youtube": "delt", "redecode": "delta"})
    assert e["decision"] == "whisperx"


def test_glossary_spelling_wins():
    meta = {"title": "Best Rows", "description": "", "chapters": [{"title": "Pendlay Row"}]}
    e = _decide({"whisperx": "Pendlay", "parakeet": "penle", "youtube": "penl", "redecode": "pen lay"}, meta)
    assert e["hypotheses"][e["decision"]] == "Pendlay"


def test_tie_on_content_word_is_flagged_uncertain():
    e = _decide({"whisperx": "hit.", "parakeet": "head.", "youtube": "head", "redecode": "hit."})
    assert e["decision"] == "whisperx"
    assert e.get("uncertain") is True
