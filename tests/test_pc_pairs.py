import importlib.util
from pathlib import Path
import numpy as np

spec = importlib.util.spec_from_file_location("pc", Path(__file__).resolve().parents[1] / "scripts/lens/pc_analyze.py")
pc = importlib.util.module_from_spec(spec); spec.loader.exec_module(pc)


def sel():
    return [
        {"run_id": "a1", "question_id": "q1|A", "group": "aware", "cuts": [10]},
        {"run_id": "a2", "question_id": "q1|A", "group": "aware", "cuts": [20]},
        {"run_id": "u1", "question_id": "q1|A", "group": "unaware", "cuts": [10, 15], "src_cuts": [10, 20]},
        {"run_id": "a3", "question_id": "q2|A", "group": "aware", "cuts": [12]},
        {"run_id": "u2", "question_id": "q2|A", "group": "unaware", "cuts": [12], "src_cuts": [12]},
    ]


def test_pairs_and_auroc():
    P = pc.Pairs(sel())
    assert P.n == 3 and P.nq == 2
    # score = 1 for aware samples, 0 for unaware: AUROC 1; constant: 0.5
    s = np.zeros(len(P.samples))
    for i, (r, k) in enumerate(P.samples):
        s[i] = r.startswith("a")
    assert P.auroc(s) == 1.0
    assert P.auroc(np.zeros(len(P.samples))) == 0.5
    # a2 (cut 20) is paired with u1's second window (cut 15)
    j = [i for i in range(P.n) if P.samples[P.sa[i]][0] == "a2"][0]
    assert P.samples[P.su[j]] == ("u1", 1)
    b = P.boot_auc(s)
    assert np.allclose(b, 1.0)
