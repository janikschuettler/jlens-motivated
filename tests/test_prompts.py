from jmr.prompts import CONDITIONS, NUDGES, Question, build_messages, hinted_letter
from jmr.questions import draw, filter_redux

Q = Question(id="redux/anatomy/3", dataset="mmlu_redux", question="Which is a mammal?",
             options=("Shark", "Whale", "Trout", "Eel"), letters=("A", "B", "C", "D"), correct="B")


def text(condition):
    return build_messages(Q, condition)[0][0]["content"]


def test_hinted_letter_is_wrong_and_stable():
    assert hinted_letter(Q) != Q.correct and hinted_letter(Q) == hinted_letter(Q)


def test_unnudged_conditions_have_no_hinted_letter_and_identical_prompts():
    assert build_messages(Q, "baseline")[1] is None and text("baseline") == text("unhinted")


def test_nudges_use_the_same_letter_and_the_right_text():
    letter = hinted_letter(Q)
    for c in NUDGES:
        assert build_messages(Q, c)[1] == letter
    assert f"chose {letter}." in text("zero_evidence") and "uniformly at random" in text("zero_evidence")
    assert f"chose {letter}." in text("known_evidence") and "90%" in text("known_evidence")
    assert f"answer is {letter}" in text("positive_control")
    assert "raining" in text("null_control") and f"chose {letter}" not in text("null_control")


def test_nudge_goes_between_options_and_instruction():
    t = text("zero_evidence")
    assert t.index("(D) Eel") < t.index("Someone who") < t.index("Think it through")
    assert set(CONDITIONS) == {"baseline", "unhinted", *NUDGES}


def row(error_type, answer=1, n=4):
    return {"question": "q", "choices": [f"o{i}" for i in range(n)], "answer": answer, "error_type": error_type}


def test_filter_redux_keeps_only_error_free_items():
    pool = [("a", 0, row("ok")), ("a", 1, row("wrong_groundtruth")), ("b", 0, row("ok", answer=7))]
    assert [(s, i) for s, i, _ in filter_redux(pool, "ok")] == [("a", 0)]


def test_draw_is_deterministic_and_builds_ids():
    pool = [("a", i, row("ok")) for i in range(30)]
    first = draw(pool, 5, seed=1)
    assert first == draw(pool, 5, seed=1) and first != draw(pool, 5, seed=2)
    assert first[0].id.startswith("redux/a/") and first[0].correct == "B"
