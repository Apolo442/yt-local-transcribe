from ytlt.profiles.tier_list.analyze import TIER_RE, _score_mention


def test_tier_regex_ignores_legend_and_tier_list():
    assert [m.groups() for m in TIER_RE.finditer("i'm putting the barbell hip thrust into high b tier now")] == [("high", "b", None)]
    assert list(TIER_RE.finditer("We'll go from S tier for super all the way down to F tier for fail.")) == []
    assert list(TIER_RE.finditer("ranking on a tier list")) == []
    assert list(TIER_RE.finditer("drop them back a tier or two")) == []
    assert [m.groups() for m in TIER_RE.finditer("raise it up to S tier plus as the best")] == [(None, "S", "plus")]


def test_context_score_prefers_decisions_over_hypotheses():
    close = _score_mention("these are very close to", "but they are")
    decided = _score_mention("i m going to knock it back to", "plus because of")
    assert decided > close
    cant_take_out = _score_mention("i just can t take the barbell back squat out of", "without a guilty")
    almost = _score_mention("mild anatomy concern almost pulls it back to", "i just can")
    assert cant_take_out > almost
