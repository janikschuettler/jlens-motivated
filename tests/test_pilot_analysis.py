from jmr.analysis.pilot import analyse, bootstrap, render_markdown
from jmr.runset import baseline_stats
from test_label import make
from jmr.label import label_runs


def item(qid, kept=True):
    return {"id": qid, "letters": ["A", "B", "C", "D"], "correct": "A", "hinted": "B", "baseline_correct": 10 if kept else 4,
            "baseline_n": 10, "kept": kept}


def runs(qid, condition, answers):
    # make(): correct is A, hinted B
    return [make(condition, f"<think>x</think>Answer: {a}", qid=qid, k=k) for k, a in enumerate(answers)]


def test_bootstrap_interval_contains_mean():
    b = bootstrap([0.0, 0.1, 0.2, 0.3], seed=0)
    assert b["lo"] <= b["mean"] <= b["hi"] and b["n"] == 4


def test_shift_specificity_and_silent_switches():
    recs = []
    for q in ("a", "b"):
        recs += runs(q, "unhinted", "AAAA")
        recs += runs(q, "null_control", "AAAA")
        recs += runs(q, "zero_evidence", "BBAA")  # half follow B, silently
        recs += runs(q, "known_evidence", "BCAA")
    recs += runs("c", "zero_evidence", "BBBB")   # dropped item: not kept
    items = {"a": item("a"), "b": item("b"), "c": item("c", kept=False)}
    s = analyse(label_runs(recs), items, pool_size=1000)
    z = s["conditions"]["zero_evidence"]
    assert abs(z["delta_B"]["mean"] - 0.5) < 1e-9 and abs(z["delta_correct"]["mean"] + 0.5) < 1e-9
    assert abs(z["delta_B_minus_null"]["mean"] - 0.5) < 1e-9
    k = s["conditions"]["known_evidence"]
    assert abs(k["delta_B"]["mean"] - 0.25) < 1e-9 and abs(k["delta_other_per_option"]["mean"] - 0.125) < 1e-9
    assert s["silent"]["zero_evidence"]["silent"] == 4 and s["silent"]["zero_evidence"]["items_with_silent"] == 2
    assert s["baseline"]["n_kept"] == 2
    assert abs(s["projection"]["zero_evidence"]["kept_items_available"] - 2 / 3 * 1000) < 1e-6
    assert "Shift and specificity" in render_markdown(s, "t")


def test_baseline_stats_keeps_items_at_ninety_percent():
    good = [make("baseline", "<think>x</think>Answer: A", qid="g", k=k) for k in range(10)]
    ok = [make("baseline", "<think>x</think>Answer: " + ("A" if k else "C"), qid="ok", k=k) for k in range(10)]
    bad = [make("baseline", "<think>x</think>Answer: " + ("A" if k > 1 else "C"), qid="bad", k=k) for k in range(10)]
    s = baseline_stats(good + ok + bad, 0.9)
    assert s["g"]["kept"] and s["ok"]["kept"] and not s["bad"]["kept"]
    assert (s["ok"]["baseline_correct"], s["bad"]["baseline_correct"]) == (9, 8)
