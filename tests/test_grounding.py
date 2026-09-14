from ytlt.grounding import content_words, dedupe_tips, looks_portuguese, strip_llm_wrappers


def test_strip_llm_wrappers():
    assert strip_llm_wrappers('quote": "and that actually cranks my shoulders a bit.') == "and that actually cranks my shoulders a bit."
    assert strip_llm_wrappers("You don't need a machine. - explicação biomecânica") == "You don't need a machine."
    assert strip_llm_wrappers("First, you never get a big stretch. (opinião)") == "First, you never get a big stretch."


def test_content_words_match_word_variants():
    a = content_words("May allow better glute isolation by reducing quad involvement")
    b = content_words("isolate their glutes more by taking some of the quads out")
    assert {"isola", "glute", "quad"} <= a & b


def test_language_drift_detection():
    assert looks_portuguese(["Exercício não desafia o suficiente em faixa de repetições normal sem adição de peso"])
    assert not looks_portuguese(["Cable provides constant tension, unlike bands"])


def test_dedupe_tips():
    assert dedupe_tips(["Less range of motion than hip thrust"], ["Less range of motion than hip thrust", "Brace against the bench"]) == ["Brace against the bench"]
