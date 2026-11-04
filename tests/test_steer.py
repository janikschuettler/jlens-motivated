"""Steering test helpers (jmr.steer) and the steering hook (jmr.lens.steering, skipped
without torch). CPU only."""

import pytest

from jmr import steer


def test_known_ids_refuses_old_pool_format():
    rows = [{"id": "a", "letters": ["A", "B", "C", "D"], "correct": "B", "n_correct": {"A": 9}},
            {"id": "b", "letters": ["A", "B", "C", "D"], "correct": "C", "n_correct": {"A": 8}}]
    assert steer.known_ids(rows) == ["a"]
    with pytest.raises(ValueError):
        steer.known_ids([{"id": "c", "letters": ["A", "B"], "correct": {"A": 10}, "n_correct": {}}])


def test_select_answer_runs_caps_and_filters():
    labels = []
    for q in ("q1", "q2"):
        for s in range(5):
            labels.append({"run_id": f"I1/{q}/{s}", "arm": "I1_argue", "question_id": q, "outcome": "followed"})
        labels.append({"run_id": f"I1/{q}/x", "arm": "I1_argue", "question_id": q, "outcome": "correct"})
    labels.append({"run_id": "I1/q3/0", "arm": "I1_argue", "question_id": "q3", "outcome": "followed"})
    got = steer.select_answer_runs(labels, "instructed", {"q1", "q2"}, cap=3)
    assert len(got) == 6 and all("/x" not in g and "q3" not in g for g in got)
    assert got == steer.select_answer_runs(labels, "instructed", {"q1", "q2"}, cap=3)  # deterministic
    hint = [{"run_id": "h/0", "arm": "think_qwen_hint_only_ours", "question_id": "q1", "outcome": "switch"}]
    assert steer.select_answer_runs(hint, "hint", {"q1"}) == ["h/0"]
    unh = [{"run_id": "u/0", "arm": "unhinted", "question_id": "q1", "outcome": "correct"}]
    assert steer.select_answer_runs(unh, "unhinted", {"q1"}) == ["u/0"]


def test_target_signs():
    assert steer.target("instructed", "A", "B", True) == ("A", 1)
    assert steer.target("unhinted", "A", "B", True) == ("A", -1)
    assert steer.target("hint", "A", "B", False) == ("B", -1)
    assert steer.target("unhinted", "A", "B", False) == ("B", 1)


def test_answer_outcome():
    assert steer.answer_outcome("A", "A", "B") == "correct"
    assert steer.answer_outcome("B", "A", "B") == "B"
    assert steer.answer_outcome("C", "A", "B") == "other"
    assert steer.answer_outcome(None, "A", "B") == "none"
    assert steer.answer_outcome("B", "A", None) == "other"


def test_repetition_and_rates():
    assert steer.repeated_ngram_share(list(range(100)), 8) == 0.0
    loop = [1, 2, 3, 4] * 50
    assert steer.repeated_ngram_share(loop, 8) > 0.9
    assert steer.letter_rate([5, 6, 5, 7], [5]) == 500.0
    assert steer.non_ascii_share("ab") == 0.0


def test_bootstrap():
    a = {"q1": 1.0, "q2": 1.0, "q3": 0.0}
    b = {"q1": 0.0, "q2": 0.0, "q3": 0.0, "q4": 1.0}
    m = steer.boot_mean(a)
    assert abs(m["mean"] - 2 / 3) < 1e-9 and m["lo"] <= m["mean"] <= m["hi"] and m["n_items"] == 3
    d = steer.boot_diff(a, b)
    assert abs(d["mean"] - 2 / 3) < 1e-9 and d["n_items"] == 3
    rows = [{"question_id": "q1", "v": 1.0}, {"question_id": "q2", "v": 0.0}]
    assert steer.boot_diff_grouped(rows, rows, lambda r: r["v"])["mean"] == 0.0
    im = steer.item_means([{"question_id": "q", "v": 1.0}, {"question_id": "q", "v": 0.0},
                           {"question_id": "r", "v": None}], lambda r: r["v"])
    assert im == {"q": 0.5}


def test_pilot_rule():
    ok, bad = {"accuracy": 0.95, "finished": 1.0}, {"accuracy": 0.5, "finished": 1.0}
    assert steer.pilot_alpha_rule({0.05: ok, 0.1: ok, 0.2: bad, 0.4: ok}) == (0.025, 0.1)
    assert steer.pilot_alpha_rule({0.05: bad, 0.1: ok}) == (0.00625, 0.025)


def test_steerer_hook():
    torch = pytest.importorskip("torch")
    from jmr.lens.steering import Steerer

    class Block(torch.nn.Module):
        def forward(self, x):
            return (x * 1.0,)

    class Inner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([Block(), Block()])

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Inner()

        def forward(self, x):
            for b in self.model.layers:
                x = b(x)[0]
            return x

    m = Model()
    st = Steerer(m, [1])
    x = torch.zeros(2, 3, 4)
    assert m(x).abs().sum() == 0  # no vector set: unchanged
    v = torch.tensor([[1.0, 0, 0, 0], [0, 2.0, 0, 0]])
    st.set({1: v})
    y = m(x)
    assert torch.equal(y[:, -1], v) and y[:, :-1].abs().sum() == 0  # "last": only the last column
    mask = torch.tensor([[False, True, True], [False, False, True]])
    st.set({1: v}, mask)
    y = m(x)
    assert torch.equal(y[0, 1], v[0]) and y[1, 1].abs().sum() == 0 and torch.equal(y[1, 2], v[1])
    st.set({1: torch.zeros(3, 4)})
    with pytest.raises(ValueError):
        m(x)
    st.remove()
    assert m(x).abs().sum() == 0
