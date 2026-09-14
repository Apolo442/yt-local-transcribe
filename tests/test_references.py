from ytlt.references import _unit_numbers, parse_description


def test_parse_description_groups_and_urls():
    desc = "Intro\n\nReferences\n\nHip Thrust Studies\nhttps://pubmed.ncbi.nlm.nih.gov/26214739/\n\nSumo vs Conventional\nhttps://doi.org/10.1/x\n-------\nfooter"
    refs = parse_description(desc)
    assert [(r["group"], r["url"]) for r in refs] == [
        ("Hip Thrust Studies", "https://pubmed.ncbi.nlm.nih.gov/26214739/"),
        ("Sumo vs Conventional", "https://doi.org/10.1/x"),
    ]


def test_unit_numbers_normalize_words_and_hyphens():
    assert _unit_numbers("A recent nine-week study") == {"9 week"}
    assert "9 week" in _unit_numbers("after 9 weeks of training")
