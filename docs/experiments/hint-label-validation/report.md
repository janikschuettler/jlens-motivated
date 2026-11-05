# Hint-label validation (claim 3): do two LLM judges agree that thinking-mode switches name the hint?

Exploratory, pre-registered section HLV in docs/preregistration.md (commit fc7781b). Summary and numbers: [README.md](README.md); derived numbers: [data/hlv.json](data/hlv.json).

Data: Qwen3-8B, thinking mode, known items. Factual from the gap factorial and wording dissection (octavia), moral from the moral hunt (mayor backup on octavia) and w2_moral_nudge. Population rebuilt with the original keyword rule: 1,231 factual and 5,019 moral switches, 2 and 0 keyword-silent (identical to the original scan).

Sample (seed 0): 382 runs, 380 judged by both gemini-2.5-flash-lite and gpt-5-nano (T 0). Confusion tables, kappas and the implied silent rates are in data/hlv.json.

Headline: judge vs judge on "reasoning mentions the hint" kappa 0.88; keyword rule vs judges on switches 180/180 and 172/180 agree; moral switches 0 of 90 silent by either judge; factual switches 1 of 92 (flash-lite) vs 10 of 92 (gpt-5-nano), the 8 non-trivial extra gpt-5-nano cases mostly quoting the hint. Pre-registered rule: moral met; factual not met under gpt-5-nano (upper bound 13% > 5%), met under flash-lite and both-judges. Unhinted control: 0 and 1 of 100 false mentions.

Reliance question (b): 180/182 (flash-lite) and 170/182 (nano) of sampled switches rest on the hint; 4/98 and 0/98 of resisted runs.
