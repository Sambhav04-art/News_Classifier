from src.preprocess import clean_text, combine_title_description


def test_clean_text_basic():
    assert clean_text("Wall St. Bears Claw Back!") == "wall st bears claw back"


def test_clean_text_handles_artifacts():
    out = clean_text("Wall Street's dwindling\\band of &amp; <b>ultra</b>-cynics http://x.com")
    assert "\\" not in out and "<" not in out and "http" not in out and "&amp" not in out
    assert "dwindling band" in out


def test_clean_text_non_string():
    assert clean_text(None) == ""


def test_combine():
    assert combine_title_description("T", "D") == "T. T. D"
