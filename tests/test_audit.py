from test_label import make

from jmr.audit import fence, render, select_rows
from jmr.label import label_runs

LONG_COT = "step " * 5000  # far longer than any plausible truncation limit


def records():
    recs = [
        make("zero_evidence", f"<think>{LONG_COT}\nOption B fits. ```code``` end</think>Answer: B", qid=f"q{i}")
        for i in range(3)
    ]
    recs += [make("unhinted", "<think>x</think>Answer: A", qid=f"q{i}") for i in range(3)]
    for r in recs:
        r.messages = [{"role": "user", "content": "Which option? (A) x (B) y"}]
    return recs


def test_fence_survives_backticks():
    block = fence("a ``` b ```` c")
    assert block.startswith("`````") and block.endswith("`````")


def test_select_caps_per_group_and_is_deterministic():
    rows = label_runs(records())
    a = select_rows(rows, 2, seed=0)
    assert a == select_rows(rows, 2, seed=0)
    assert len([r for r in a if r["label"] == "silent_switch"]) == 2
    assert len(select_rows(rows, None, seed=0)) == len(rows)


def test_render_has_summary_and_full_chain_of_thought():
    recs = records()
    text = render(recs, label_runs(recs), per_group=1, seed=0)
    assert "## Summary" in text and "**silent_switch**" in text
    assert "## silent_switch (1 shown of 3)" in text
    assert LONG_COT.strip() in text  # complete, not truncated
    assert "Prompt, user" in text
