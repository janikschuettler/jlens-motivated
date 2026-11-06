"""probe-control PC-L (exploratory): re-score test-(b) pairs after padding the shorter version to the longer one's token count.
  JMR_MODEL_PROFILE=qwen3-14b uv run --project envs/lens python scripts/lens/pc_l_pad.py \
      --dir data/probe_control --out data/probe_control/b/pad_scores.json
Filler = neutral weather words prepended (with a blank line) to the last user message; no eval/test/AI words.
Scores as pc_b_readout.py (Nguyen prompt mean, Hua last token, J-lens/logit-lens eval-word score at the last token).
Sanity: 6 unpadded versions are re-scored and compared with b/scores.json. Rule 3: scalars only."""
import argparse, json
from pathlib import Path
import numpy as np
import torch
from jmr.lens.batch import LENSES, BatchReader
from jmr.lens.load import lens_file, load_lens, load_model
from jmr.tokens import encode_chat

NL = 40
WORDS = " the weather today is mild and the sky stays clear over the quiet hills while a gentle breeze moves across the green fields near the river".split()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", default="data/probe_control"); ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", default="configs/evalaware_tokens.json"); ap.add_argument("--limit", type=int)
    a = ap.parse_args(); d = Path(a.dir)
    vers = {json.loads(l)["vid"]: json.loads(l) for l in open(d / "b/versions.jsonl")}
    pairs = [json.loads(l) for l in open(d / "b/pairs.jsonl")]
    old = {r["vid"]: r for r in json.loads((d / "b/scores.json").read_text())["rows"]}
    hf, tok, lens_model = load_model()
    reader = BatchReader(lens_model, load_lens(lens_file()), tok)
    dirs = torch.load(d / "build/directions.pt"); ng, hu = dirs["nguyen"].to(hf.device), dirs["hua"].to(hf.device)
    cfg = json.loads(Path(a.tokens).read_text()); byt = {t["text"]: t["id"] for t in cfg["tokens"]}
    prim = torch.tensor([byt[" fictional"], byt[" simulation"]], device=hf.device)
    ctrl = torch.tensor([t["id"] for t in cfg["tokens"] if t["groups"] == ["control"]], device=hf.device)
    caps, hooks = {}, []
    for l in range(NL):
        def mk(l):
            def f(mod, args, kwargs):
                caps[l] = (args[0] if args else kwargs["hidden_states"])[0]
            return f
        hooks.append(hf.model.layers[l].register_forward_pre_hook(mk(l), with_kwargs=True))

    def ids_of(msgs):
        return encode_chat(tok, msgs, enable_thinking=True)

    def score(ids):
        with torch.no_grad():
            hf(input_ids=torch.tensor([ids], device=hf.device), use_cache=False)
            keep = torch.tensor([bool(tok.decode([t]).strip()) for t in ids], device=hf.device)
            n = [float((caps[l].float() @ ng[l])[keep].mean()) for l in range(NL)]
            h = [float(caps[l][-1].float() @ hu[l]) for l in range(NL)]
            full, _ = reader._forward(ids, [len(ids) - 1]); sc = {}
            for name in LENSES:
                vals = []
                for layer in reader.layers:
                    x = full[layer][len(ids) - 1:len(ids)].float()
                    if name == "jlens":
                        x = reader.lens.transport(x, layer)
                    lp = torch.log_softmax(lens_model.unembed(x).float(), dim=-1)[0]
                    vals.append(float(lp[prim].mean() - lp[ctrl].mean()))
                sc[name] = float(np.mean(vals))
            del full
        return {"n_tokens": len(ids), "nguyen": n, "hua": h, "jlens": sc["jlens"], "logit": sc["logit"]}

    def padded(v, target):
        msgs = [dict(m) for m in v["messages"]]
        base = msgs[-1]["content"]; k = 0
        while True:
            k += 1
            msgs[-1]["content"] = "".join(WORDS[i % len(WORDS)] for i in range(k)).strip() + "\n\n" + base
            n = len(ids_of(msgs))
            if n >= target:
                return msgs, n
    out = {"sanity": [], "rows": []}
    for vid in list(vers)[:: max(1, len(vers) // 6)][:6]:
        s = score(ids_of(vers[vid]["messages"]))
        out["sanity"].append({"vid": vid, "max_abs_diff_nguyen": float(np.nanmax([abs(x - y) for x, y in zip(s["nguyen"], old[vid]["nguyen"])])),
                              "diff_jlens": s["jlens"] - old[vid]["jlens"], "n_tokens_equal": s["n_tokens"] == old[vid]["n_tokens"]})
    print(out["sanity"], flush=True)
    todo = pairs[: a.limit] if a.limit else pairs
    for i, p in enumerate(todo):
        ne, npl = old[p["eval"]]["n_tokens"], old[p["plain"]]["n_tokens"]
        target = max(ne, npl); short = p["plain"] if npl < ne else p["eval"] if ne < npl else None
        row = {"pair": p["pair"], "item": p["item"], "eval": p["eval"], "plain": p["plain"], "n_eval": ne, "n_plain": npl, "padded": short}
        if short is None:
            row["scores"] = {"eval": {k: old[p["eval"]][k] for k in ("nguyen", "hua", "jlens", "logit")}, "plain": {k: old[p["plain"]][k] for k in ("nguyen", "hua", "jlens", "logit")}}
            row["n_padded"] = target
        else:
            msgs, n = padded(vers[short], target)
            s = score(ids_of(msgs)); row["n_padded"] = n
            other = p["eval"] if short == p["plain"] else p["plain"]
            sc_o = {k: old[other][k] for k in ("nguyen", "hua", "jlens", "logit")}
            sc_s = {k: s[k] for k in ("nguyen", "hua", "jlens", "logit")}
            row["scores"] = {"eval": sc_s if short == p["eval"] else sc_o, "plain": sc_s if short == p["plain"] else sc_o}
        out["rows"].append(row)
        if (i + 1) % 50 == 0:
            print(i + 1, flush=True)
    Path(a.out).write_text(json.dumps(out)); print("done")


if __name__ == "__main__":
    main()
