# Normative multiple-choice datasets: HF Hub check (2026-11-02)

Checked through the HF API, dataset cards and papers by a sub-agent of the design survey ([designs.md](report.md)). **[unverified]** marks what it couldn't confirm.

| Dataset | HF id | Size | Format | Licence | Ground truth | Fit for letter tracking |
|---|---|---|---|---|---|---|
| MoralChoice (Scherrer+ 2023) | `ninoscherrer/moralchoice` (raw CSV) | low-ambiguity 687, high 680 | context + action1/action2 | CC-BY-4.0 | low-ambiguity: action1 preferred by construction; 3/3 annotators agree in 90.0%, ≥2/3 in 99.6% | **very good**: native A/B, order can be swapped. Most LLMs pick the commonsense action; exceptions include 3 cheating-in-games items |
| ETHICS commonsense (Hendrycks+ 2021) | `hendrycks/ethics` (Parquet auto-conversion) | val 3,885 (= paper Test), test 3,964 (= Hard Test) | binary Wrong / Not wrong | MIT | crowd label, ≥4/5 agreement | **good** for short items; no `is_short` column on HF, so filter by length. Used by Kim & Flanigan 2026 with Qwen3 models |
| Moral Stories (Emelin+ 2021) | `demelin/moral_stories` (loading script) | 12k | moral vs immoral action | MIT | crowd-authored, validated | good; shuffle into A/B; probably very easy [unverified] |
| MMLU-Redux moral subjects | `edinburgh-dawg/mmlu-redux-2.0` | `ok`: moral_scenarios 97, moral_disputes 96 | 4 options; scenarios bundle 2 judgements per item | CC-BY-4.0 | re-annotated | needs no new code; disputes are mostly facts about ethicists, so not really normative |
| Anthropic sycophancy | `Anthropic/model-written-evals` sycophancy/* | ~10k each | bio + (A)/(B) | CC-BY-4.0 | none (matches user) | not "known"; only the model's own ≥90% view could serve |
| BBQ | `HiTZ/bbq`, `heegyu/bbq` | ~31k | 3 options | CC-BY-4.0 | template key | tests bias and reading, not moral judgement |
| HHH alignment | `HuggingFaceH4/hhh_alignment` | 221 | 2 responses | Apache-2.0 | author-ranked | small |
| Scruples, DailyDilemmas, SafetyBench, ValuePrism | — | — | — | — | noisy, none, withheld or none | poor fit |

## Prior work on moral nudges

- Kim & Flanigan 2026 ([2606.14037](https://arxiv.org/abs/2606.14037)): ETHICS vs TruthfulQA+MMLU, 9 models incl. Qwen3-14B/32B. Strong misleading nudges flip 37.1% of correct moral answers vs 31.8% of factual ones; moral nudges are followed about equally in both directions.
- Blandfort+ 2026 ([2602.22831](https://arxiv.org/abs/2602.22831)): short cues shift moral choice rates by 12–18 pp.
- Huang & Cai 2026 ([2610.26655](https://arxiv.org/abs/2610.26655)): abandoning planted moral verdicts across 741 checkpoints; preference optimisation drives it.
- Cheng+ 2025, ELEPHANT ([2505.13995](https://arxiv.org/abs/2505.13995)): on AITA posts, models side with the user in both the original and flipped versions 48% of the time.
- Turpin+ 2023 ([2305.04388](https://arxiv.org/abs/2305.04388)): BBQ explanations back stereotype-aligned answers and rarely mention the stereotype.

## Top 2 for a pilot

1. **MoralChoice low-ambiguity**: binary, unambiguous key, CSV, CC-BY. Risk: the model may refuse to argue for clearly harmful actions.
2. **ETHICS commonsense, short items** (HF "validation"): thousands of items, direct comparison with Kim & Flanigan. Fallback with no new code: Redux moral_scenarios.
