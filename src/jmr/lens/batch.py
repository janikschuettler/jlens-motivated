"""Batch readouts over saved runs: J-lens, logit lens and the model's own
final-layer distribution, from exactly the saved token IDs.

Per run, one teacher-forced pass captures block outputs (forward hooks on
model.layers[i], the capture point the lens was fit at; jlens's
ActivationRecorder for the full-sequence layers). The full softmax over the
vocabulary is computed in memory in position chunks, and only these leave the
GPU (CLAUDE.md rule 3):

- tracked-token log-probs and ranks at the last `prompt_tail` prompt positions
  and every completion position, per lens x layer, plus the model's own
  (block 35) distribution;
- top-25 token IDs and log-probs at the named anchors only;
- block outputs of all 36 blocks at a few anchors (fp16), for probes;
- the teacher-forced log-prob of every sampled completion token (to compare
  with vLLM's).

Chunking follows jmr.lens.readout.readout (chunks of `position_chunk` from
position 0), so the J-lens and logit-lens numbers are computed with the same
operations and shapes as the M0.2-validated single-run readout.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
from jlens import ActivationRecorder, HFLensModel, JacobianLens

from jmr import config
from jmr.anchors import find_anchors
from jmr.label import extract_answer
from jmr.records import RunRecord

LENSES = ("jlens", "logit")
FORMAT_VERSION = 1


# ---------------------------------------------------------------- run records


def read_runs_lenient(path: Path) -> list[tuple[RunRecord, dict]]:
    """RunRecords from a runs.jsonl, tolerating extra fields (e.g. `arm` and its
    spec in exploration runs). Returns (record, extras) pairs; missing required
    fields still raise."""
    names = {f.name for f in fields(RunRecord)}
    out = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            out.append((RunRecord(**{k: v for k, v in row.items() if k in names}),
                        {k: v for k, v in row.items() if k not in names}))
    return out


# ------------------------------------------------------------ tracked tokens


@dataclass
class Tracked:
    letters: list[str]
    forms: list[str]  # e.g. ["bare", "space", "paren"]
    ids: list[int]  # flat, letter-major: ids[li * len(forms) + fi]
    labels: list[str]  # decoded token per id

    def index(self, letter: str, form: str) -> int:
        return self.letters.index(letter) * len(self.forms) + self.forms.index(form)


def tracked_tokens(tokenizer: Any) -> Tracked:
    """The tracked tokens from configs/lens.yaml, checked against the tokenizer.
    Raises if any form is not exactly one token or its ID differs from the
    config (rule 1)."""
    cfg = config.load("lens")["tracked_tokens"]
    letters, forms = list(cfg["letters"]), list(cfg["forms"])
    ids, labels, problems = [], [], []
    for letter in letters:
        for form in forms:
            text = cfg["forms"][form].replace("{L}", letter)
            enc = tokenizer(text, add_special_tokens=False).input_ids
            want = cfg["ids"][letter][form]
            if len(enc) != 1:
                problems.append(f"{text!r} is {len(enc)} tokens {enc}")
            elif enc[0] != want:
                problems.append(f"{text!r} is ID {enc[0]}, config says {want}")
            elif tokenizer.decode([want]) != text:
                problems.append(f"ID {want} decodes to {tokenizer.decode([want])!r}, expected {text!r}")
            ids.append(want)
            labels.append(text)
    if problems:
        raise ValueError("tracked tokens don't match the tokenizer: " + "; ".join(problems))
    if len(set(ids)) != len(ids):
        raise ValueError(f"tracked token IDs are not distinct: {ids}")
    return Tracked(letters, forms, ids, labels)


def load_content_tokens(path: str | Path, tokenizer: Any) -> dict:
    """Per-item content-word tracked tokens (configs/content_tokens.json, built by
    scripts/lens/content_tokens.py), checked against the tokenizer: every ID must
    be the first token of its form's text and decode to a string that encodes
    back to exactly that ID. Raises on any mismatch (rule 1)."""
    cfg = json.loads(Path(path).read_text())
    problems = []
    for q, item in cfg["items"].items():
        for L, opt in item["options"].items():
            w = opt["word"]
            texts = {"space": " " + w, "bare": w, "cap": " " + w[0].upper() + w[1:]}
            for f, tid in opt["ids"].items():
                first = tokenizer(texts[f], add_special_tokens=False).input_ids[0]
                s = tokenizer.decode([tid])
                if first != tid:
                    problems.append(f"{q} {L} {f}: first token of {texts[f]!r} is {first}, file says {tid}")
                elif tokenizer(s, add_special_tokens=False).input_ids != [tid] or s != opt["labels"][f]:
                    problems.append(f"{q} {L} {f}: ID {tid} decodes to {s!r}, not a single-token round trip")
            if not set(opt["unique_ids"]) <= set(opt["ids"].values()):
                problems.append(f"{q} {L}: unique_ids not among ids")
        all_ids = [i for opt in item["options"].values() for i in opt["unique_ids"]]
        if len(set(all_ids)) != len(all_ids):
            problems.append(f"{q}: content tokens are shared between options")
    if problems:
        raise ValueError(f"content tokens don't match the tokenizer ({len(problems)}): " + "; ".join(problems[:10]))
    return cfg


def run_content_tokens(cfg: dict, record: RunRecord) -> tuple[list[int], list[int]]:
    """(token IDs, option index into record.letters per ID) for one run. Raises
    if the run's item is not in the file or disagrees on letters/correct."""
    item = cfg["items"].get(record.question_id)
    if item is None:
        raise ValueError(f"{record.run_id}: item {record.question_id} has no content tokens")
    if item["letters"] != list(record.letters) or item["correct"] != record.correct_letter:
        raise ValueError(f"{record.run_id}: content-token file disagrees on letters / correct letter")
    ids, opt = [], []
    for li, L in enumerate(record.letters):
        for tid in item["options"][L]["unique_ids"]:
            ids.append(int(tid))
            opt.append(li)
    return ids, opt


# ------------------------------------------------------------------- anchors

TOP_ANCHORS = ["last_prompt", "think_start", "think_end", "pre_first_stated", "pre_answer"]


@dataclass
class RunPlan:
    """Positions for one run, all indexing the full sequence prompt + completion."""

    ids: list[int]
    prompt_len: int
    answer: str | None
    anchors: dict  # jmr.anchors.Anchors.as_dict()
    named: dict[str, int] = field(default_factory=dict)  # anchor name -> position, -1 if absent
    cot_span: tuple[int, int] | None = None  # [start, end) of the reasoning span
    stored_lo: int = 0  # tracked tokens are stored at positions stored_lo .. len(ids) - 1

    @property
    def top_names(self) -> list[str]:
        return list(self.named)


def plan_run(record: RunRecord, tokenizer: Any, *, think_ids: tuple[int, int], prompt_tail: int,
             n_cot: int, resid_anchors: list[str]) -> RunPlan:
    ids = record.token_ids
    P, L = len(record.prompt_token_ids), len(ids)
    answer = extract_answer(record)
    a = find_anchors(tokenizer, ids, P, think_start_id=think_ids[0], think_end_id=think_ids[1], final_letter=answer)

    def in_completion(i):
        return i if i is not None and i >= P else -1

    think_start, think_end = in_completion(a.think_start), in_completion(a.think_end)
    named = {
        "last_prompt": P - 1,
        "think_start": think_start,
        "think_end": think_end,
        # The position whose next-token distribution produces the letter.
        "pre_first_stated": a.answer_first_stated - 1 if a.answer_first_stated is not None else -1,
        "pre_answer": a.final_answer - 1 if a.final_answer is not None else -1,
    }
    # Reasoning span: inside <think>…</think> when thinking happened in the
    # completion; for a truncated CoT, up to the last token; for a non-thinking
    # run (empty think block in the prompt), the completion up to the answer.
    if think_start >= 0:
        start = think_start + 1
        end = think_end if think_end >= 0 else L
    else:
        start = P
        end = named["pre_answer"] + 1 if named["pre_answer"] >= 0 else L
    cot_span = (start, end) if end > start else None
    if cot_span is not None:
        n = min(n_cot, end - start)
        for k, pos in enumerate(np.linspace(start, end - 1, n).round().astype(int)):
            named[f"cot_{k:02d}"] = int(pos)
    for name in resid_anchors:
        if name not in named:
            raise ValueError(f"unknown residual anchor {name!r}")
    return RunPlan(ids=ids, prompt_len=P, answer=answer, anchors=a.as_dict(), named=named,
                   cot_span=cot_span, stored_lo=max(0, P - prompt_tail))


def check_plan(plan: RunPlan, tokenizer: Any) -> list[str]:
    """Anchors decode to the intended tokens (as in M0.4). Returns problems."""
    ids, problems = plan.ids, []

    def dec(i):
        return tokenizer.decode([ids[i]])

    if plan.named["think_start"] >= 0 and dec(plan.named["think_start"]) != "<think>":
        problems.append(f"think_start decodes to {dec(plan.named['think_start'])!r}")
    if plan.named["think_end"] >= 0 and dec(plan.named["think_end"]) != "</think>":
        problems.append(f"think_end decodes to {dec(plan.named['think_end'])!r}")
    for name, key in (("pre_answer", "final_answer"), ("pre_first_stated", "answer_first_stated")):
        pos = plan.anchors[key]
        if pos is not None and (plan.answer is None or plan.answer not in dec(pos)):
            problems.append(f"{key} decodes to {dec(pos)!r}, answer {plan.answer}")
    if plan.answer is not None and plan.anchors["final_answer"] is None:
        problems.append(f"answer {plan.answer} parsed but no final-answer anchor")
    return problems


# ------------------------------------------------------------------- readout


def _min_rank_eager(lp: torch.Tensor, tl: torch.Tensor) -> torch.Tensor:
    """rank[n, k] = 1 + #{v : lp[n, v] > tl[n, k]} (tie rule: min rank)."""
    return (lp.unsqueeze(1) > tl.unsqueeze(2)).sum(-1, dtype=torch.int32) + 1


# Compiled, the compare and the count fuse into one reduction kernel (about 20x
# faster than one masked count per tracked token; same integer results, checked
# against the per-token loop).
_min_rank = torch.compile(_min_rank_eager, dynamic=True)


class BatchReader:
    """Holds the model, the lens (workspace layers on the GPU) and the settings."""

    def __init__(self, lens_model: HFLensModel, lens: JacobianLens, tokenizer: Any, *, cot_resid: bool = False,
                 content_tokens: str | Path | None = None) -> None:
        """cot_resid: also store block outputs (fp16) at the n_cot evenly spaced CoT
        anchors for the workspace layers (optional arrays cot_resid [layers, n_cot, d]
        and cot_resid_pos [n_cot], -1 / NaN where the run has fewer CoT anchors).

        content_tokens: path to a per-item content-token file
        (configs/content_tokens.json). Each run then also gets per-run tracked
        tokens, stored in separate optional arrays (x_ids, x_opt, x_lp, x_rank,
        x_model_lp, x_model_rank); the default arrays are unchanged."""
        self.cot_resid = cot_resid
        self.content = load_content_tokens(content_tokens, tokenizer) if content_tokens else None
        cfg = config.load("lens")
        self.lens_model = lens_model
        self.tokenizer = tokenizer
        self.layers = sorted(cfg["workspace_layers"])
        self.final = lens_model.n_layers - 1
        self.top_k = cfg["top_k"]
        self.chunk = cfg["position_chunk"]
        self.prompt_tail = cfg["batch"]["prompt_tail"]
        self.n_cot = cfg["batch"]["n_cot_positions"]
        self.resid_anchors = list(cfg["batch"]["resid_anchors"])
        self.tracked = tracked_tokens(tokenizer)
        self.device = lens_model.input_device
        # Move only the workspace maps to the GPU once; JacobianLens.transport's
        # .to(device) is then a no-op. Same fp32 values and matmul as before.
        missing = [l for l in self.layers if l not in lens.jacobians]
        if missing:
            raise ValueError(f"lens has no map for workspace layers {missing}")
        lens.jacobians = {l: lens.jacobians[l].to(self.device) for l in self.layers}
        self.lens = lens
        st = config.load("model")["special_tokens"]
        self.think_ids = (tokenizer.convert_tokens_to_ids(st["think_start"]),
                          tokenizer.convert_tokens_to_ids(st["think_end"]))
        self._tracked_t = torch.tensor(self.tracked.ids, device=self.device)

    def plan(self, record: RunRecord) -> RunPlan:
        return plan_run(record, self.tokenizer, think_ids=self.think_ids, prompt_tail=self.prompt_tail,
                        n_cot=self.n_cot, resid_anchors=self.resid_anchors)

    @torch.no_grad()
    def _forward(self, ids: list[int], resid_pos: list[int]):
        """Full block outputs at the workspace layers + the last block, and the
        block outputs of every block at resid_pos (fp16)."""
        input_ids = torch.tensor([ids], device=self.device)
        full_layers = sorted(set(self.layers) | {self.final})
        sel = torch.tensor(resid_pos, device=self.device)
        anchor_resid: dict[int, torch.Tensor] = {}

        def make_hook(i):
            def hook(module, inputs, output):
                t = output if torch.is_tensor(output) else output[0]
                anchor_resid[i] = t[0, sel].float()
            return hook

        handles = [self.lens_model.layers[i].register_forward_hook(make_hook(i))
                   for i in range(self.lens_model.n_layers)]
        try:
            with ActivationRecorder(self.lens_model.layers, at=full_layers) as recorder:
                self.lens_model.forward(input_ids)
                full = {l: recorder.activations[l][0].detach() for l in full_layers}
        finally:
            for h in handles:
                h.remove()
        stacked = torch.stack([anchor_resid[i] for i in range(self.lens_model.n_layers)])  # [36, n, d]
        return full, stacked

    @torch.no_grad()
    def _read_layer(self, resid: torch.Tensor, *, layer: int | None, lo: int, top_pos: torch.Tensor,
                    out_lp: torch.Tensor, out_rank: torch.Tensor, out_top_ids: torch.Tensor,
                    out_top_lp: torch.Tensor, target: torch.Tensor | None = None,
                    out_target: torch.Tensor | None = None, extra: torch.Tensor | None = None,
                    out_xlp: torch.Tensor | None = None, out_xrank: torch.Tensor | None = None) -> None:
        """One lens x layer. layer None: logit lens (J = I). Fills tracked
        log-probs/ranks for positions lo.., top-k at top_pos (-1 = absent), and
        optionally the log-prob of target[t] at each position t."""
        L = resid.shape[0]
        tracked = self._tracked_t
        for start in range(0, L, self.chunk):
            end = min(start + self.chunk, L)
            if end <= lo:
                continue  # whole chunk before the stored range; chunk boundaries unchanged
            h = resid[start:end].float()
            if layer is not None:
                h = self.lens.transport(h, layer)
            lp = torch.log_softmax(self.lens_model.unembed(h).float(), dim=-1)  # [n, V] in memory only
            a = max(lo, start)
            rows = lp[a - start :]
            tl = rows[:, tracked]  # [n, K]
            rank = _min_rank(rows, tl)
            out_lp[a - lo : end - lo] = tl.half()
            out_rank[a - lo : end - lo] = rank
            if extra is not None:  # per-run tracked tokens, same rows and rank rule
                xl = rows[:, extra]
                out_xlp[a - lo : end - lo] = xl.half()
                out_xrank[a - lo : end - lo] = _min_rank(rows, xl)
            m = (top_pos >= start) & (top_pos < end)
            if m.any():
                v, i = lp[top_pos[m] - start].topk(self.top_k, dim=-1)
                out_top_ids[m] = i.int()
                out_top_lp[m] = v.half()
            if out_target is not None:
                stop = min(end, L - 1)
                if stop > start:
                    out_target[start:stop] = lp[torch.arange(stop - start, device=lp.device),
                                                target[start:stop]]

    @torch.no_grad()
    def read(self, record: RunRecord, plan: RunPlan | None = None) -> dict[str, np.ndarray]:
        plan = plan or self.plan(record)
        ids, P, L = plan.ids, plan.prompt_len, len(plan.ids)
        lo, T, K = plan.stored_lo, L - plan.stored_lo, len(self.tracked.ids)
        top_names = plan.top_names
        top_pos = torch.tensor([plan.named[n] for n in top_names], device=self.device)
        A, nL = len(top_names), len(self.layers)
        resid_pos = [plan.named[n] for n in self.resid_anchors]
        full, anchor_resid = self._forward(ids, [p if p >= 0 else 0 for p in resid_pos])

        dev = self.device
        xt, x = None, {}
        if self.content is not None:
            x_ids, x_opt = run_content_tokens(self.content, record)
            xt = torch.tensor(x_ids, device=dev)
            K2 = len(x_ids)
            x = {"lp": torch.zeros((len(LENSES), nL, T, K2), dtype=torch.float16, device=dev),
                 "rank": torch.zeros((len(LENSES), nL, T, K2), dtype=torch.int32, device=dev),
                 "model_lp": torch.zeros((T, K2), dtype=torch.float16, device=dev),
                 "model_rank": torch.zeros((T, K2), dtype=torch.int32, device=dev)}
        lp = torch.zeros((len(LENSES), nL, T, K), dtype=torch.float16, device=dev)
        rank = torch.zeros((len(LENSES), nL, T, K), dtype=torch.int32, device=dev)
        top_ids = torch.full((len(LENSES), nL, A, self.top_k), -1, dtype=torch.int32, device=dev)
        top_lp = torch.full((len(LENSES), nL, A, self.top_k), float("nan"), dtype=torch.float16, device=dev)
        for si, lens_name in enumerate(LENSES):
            for li, layer in enumerate(self.layers):
                self._read_layer(full[layer], layer=layer if lens_name == "jlens" else None, lo=lo,
                                 top_pos=top_pos, out_lp=lp[si, li], out_rank=rank[si, li],
                                 out_top_ids=top_ids[si, li], out_top_lp=top_lp[si, li], extra=xt,
                                 out_xlp=x["lp"][si, li] if x else None, out_xrank=x["rank"][si, li] if x else None)
        m_lp = torch.zeros((T, K), dtype=torch.float16, device=dev)
        m_rank = torch.zeros((T, K), dtype=torch.int32, device=dev)
        m_top_ids = torch.full((A, self.top_k), -1, dtype=torch.int32, device=dev)
        m_top_lp = torch.full((A, self.top_k), float("nan"), dtype=torch.float16, device=dev)
        target = torch.tensor(ids[1:] + [0], device=dev)
        next_lp = torch.zeros(L, dtype=torch.float32, device=dev)
        self._read_layer(full[self.final], layer=None, lo=lo, top_pos=top_pos, out_lp=m_lp, out_rank=m_rank,
                         out_top_ids=m_top_ids, out_top_lp=m_top_lp, target=target, out_target=next_lp,
                         extra=xt, out_xlp=x.get("model_lp"), out_xrank=x.get("model_rank"))

        resid16 = anchor_resid.half()
        if not torch.isfinite(resid16).all():
            raise ValueError(f"{record.run_id}: block outputs overflow fp16 at the residual anchors")
        for j, p in enumerate(resid_pos):
            if p < 0:
                resid16[:, j] = float("nan")

        extra = {}
        if self.cot_resid:
            cpos = [plan.named.get(f"cot_{k:02d}", -1) for k in range(self.n_cot)]
            sel = torch.tensor([p if p >= 0 else 0 for p in cpos], device=dev)
            cres = torch.stack([full[l][sel] for l in self.layers]).half()  # [layers, n_cot, d]
            if not torch.isfinite(cres).all():
                raise ValueError(f"{record.run_id}: block outputs overflow fp16 at the CoT anchors")
            for k, p in enumerate(cpos):
                if p < 0:
                    cres[:, k] = float("nan")
            extra = {"cot_resid_pos": np.array(cpos, dtype=np.int32), "cot_resid": cres.cpu().numpy()}
        if x:
            extra.update({"x_ids": np.array(x_ids, dtype=np.int32), "x_opt": np.array(x_opt, dtype=np.int8),
                          **{f"x_{k}": v.cpu().numpy() for k, v in x.items()}})
        out = {
            "format_version": np.array(FORMAT_VERSION),
            "layers": np.array(self.layers, dtype=np.int16),
            "positions_lo": np.array(lo, dtype=np.int32),
            "n_positions": np.array(T, dtype=np.int32),
            "tracked_ids": np.array(self.tracked.ids, dtype=np.int32),
            "lp": lp.cpu().numpy(),  # [lens, layer, T, K] fp16
            "rank": rank.cpu().numpy(),  # [lens, layer, T, K] int32, 1 = top
            "model_lp": m_lp.cpu().numpy(),  # [T, K]
            "model_rank": m_rank.cpu().numpy(),
            "top_pos": top_pos.cpu().numpy().astype(np.int32),  # [A], -1 = absent
            "top_ids": top_ids.cpu().numpy(),  # [lens, layer, A, k]
            "top_lp": top_lp.cpu().numpy(),
            "model_top_ids": m_top_ids.cpu().numpy(),  # [A, k]
            "model_top_lp": m_top_lp.cpu().numpy(),
            # Teacher-forced log-prob of completion token i (position P - 1 + i).
            "completion_tf_logprob": next_lp[P - 1 : L - 1].cpu().numpy(),
            "resid_pos": np.array(resid_pos, dtype=np.int32),
            "resid": resid16.cpu().numpy(),  # [36 blocks, n resid anchors, d_model] fp16
            **extra,
        }
        return out

    def meta(self, record: RunRecord, extras: dict, plan: RunPlan, arrays: dict) -> dict:
        tf = arrays["completion_tf_logprob"]
        vl = np.asarray(record.completion_logprobs, dtype=np.float64)
        d = np.abs(tf - vl) if len(vl) == len(tf) else None
        return {
            "format_version": FORMAT_VERSION,
            "run_id": record.run_id,
            "run_set": record.run_set,
            "question_id": record.question_id,
            "dataset": record.dataset,
            "condition": record.condition,
            "hint_letter": record.hint_letter,
            "correct_letter": record.correct_letter,
            "letters": record.letters,
            "answer": plan.answer,
            "finish_reason": record.finish_reason,
            "enable_thinking": record.enable_thinking,
            "sample_idx": record.sample_idx,
            "temperature": record.sampling.get("temperature"),
            "prompt_len": plan.prompt_len,
            "completion_len": len(record.completion_token_ids),
            "anchors": plan.anchors,
            "named_positions": plan.named,
            "top_anchor_names": plan.top_names,
            "resid_anchor_names": self.resid_anchors,
            "cot_resid_layers": self.layers if self.cot_resid else None,
            "content_tokens": None if self.content is None else {
                L: {"word": o["word"], "ids": o["unique_ids"]}
                for L, o in self.content["items"][record.question_id]["options"].items()},
            "cot_span": plan.cot_span,
            "lenses": list(LENSES),
            "tracked": {"letters": self.tracked.letters, "forms": self.tracked.forms,
                        "ids": self.tracked.ids, "labels": self.tracked.labels},
            "vllm_logprob_absdiff": None if d is None else {
                "n": int(d.size), "median": float(np.median(d)), "p99": float(np.quantile(d, 0.99)),
                "max": float(d.max())},
            "model": record.model,
            "extras": extras,
        }
