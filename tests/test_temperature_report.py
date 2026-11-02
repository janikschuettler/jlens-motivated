from jmr.analysis.temperature import bar, classify, render

ITEMS = {f"i{k}": {"id": f"i{k}", "subject": "s", "correct": "A", "hinted": "B", "letters": list("ABCD")} for k in range(2)}


def rows(item, condition, temp, answers):
    return [{"question_id": item, "condition": condition, "temperature": temp, "answer": a, "completion_tokens": 100 + 10 * temp,
             "label": ("silent_switch" if a == "B" else "resisted") if a else "excluded"} for a in answers]


def test_classify_and_bar():
    item = ITEMS["i0"]
    assert [classify({"answer": a}, item) for a in "ABCD"] + [classify({"answer": None}, item)] == ["correct", "B", "other", "other", "excluded"]
    b = bar({"correct": 3, "B": 1})
    assert len(b) == 30 and b.startswith("█") and "▓" in b


def test_render_compares_temperatures():
    r = []
    for item in ITEMS:
        for t in (0.6, 1.0):
            r += rows(item, "unhinted", t, "AAAA")
            r += rows(item, "known_evidence", t, "AAAA" if t == 0.6 else "AABA")
    screen = [{"id": i, "baseline_correct": 5, "baseline_n": 5, "kept": True} for i in ITEMS]
    text = render(r, ITEMS, screen=screen, main_item="i0", conditions=["unhinted", "known_evidence"])
    assert "## Comparison" in text and "## Per item" in text and "### `i0` (50 samples per cell)" in text
    assert "| 1.0 | known_evidence | 4 | 3 | 1 | 0 | 0 |" in text  # per item: 4 runs, 3 correct, 1 B
    assert "| 0.6 | known_evidence | 4 | 4 | 0 | 0 | 0 |" in text
