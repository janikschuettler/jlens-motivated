---
id: hack-manual-read
title: 'Reading the J-lens by hand on the reward-hack model: do eval-awareness, self-justification or sneakiness words show up, more in hidden-hack runs?'
strand: C
also:
- N
- T
status: done
kind: exploratory
verdict: against
confidence: moderate
dates:
  start: 2026-11-04T16:20
  end: 2026-11-04T16:55
  approx: true
timing:
  wall: about 35 min
  gpu_hours: 0
  api_usd: 0
  note: CPU only on arena9-cloud. Seven subagent readers in parallel (3.5-5 min each) plus my own reading; scoring about 2 min per pass.
machine: arena9-cloud (CPU)
n:
  items: 120 problems sampled per subgroup (one run per problem)
  runs: 120 runs; 9,324 lists, 6,132 read
numbers:
- label: Lists read (J-lens / logit lens) and tokens flagged
  value: 4,662 / 1,470 of 9,324; 4,560 flags in 1,440 lists
- label: 'Share of flags whose word is already in the prompt or earlier reasoning: eval / justification / sneakiness / hack mechanism'
  value: 58% / 5% / 7% / 93%
- label: 'J-lens lists with a non-echo sneakiness word: warm-start hidden hack / warm-start no hack / hiding version hidden hack / base no hack / beta 0 named hack'
  value: 14.7% / 17.7% / 5.0% / 1.3% / 21.5%
- label: 'Main contrast, warm-start hidden hack minus no hack (25 vs 25 runs), J-lens, percentage points of lists: sneakiness / strong cheating words / justification / eval'
  value: -3.1 [-10.3, +4.1] / -1.0 [-5.9, +4.0] / -4.3 [-8.2, -0.3] / -2.2 [-5.7, +1.2]
- label: 'Strong cheating words (post hoc set), share of lists: hiding version hidden hack vs base no hack'
  value: 1.1% vs 0.4% (difference +0.6 [-0.3, +1.7])
- label: Logit lens, same main contrast, sneakiness (12 vs 8 runs)
  value: -2.9 [-12.2, +6.9]
- label: Runs with a flagged eval/justification/sneakiness word at the last prompt token, J-lens
  value: 1 of 120
numbers_note: 'A list is one top-25 next-token list at one position, layer and lens. Non-echo: the word (or its stem or English translation) is not in the prompt or earlier in the reasoning. Percentages of lists are means over runs of the share of a run''s read lists with a flag; brackets are 95% bootstrap intervals over runs (one run per problem within a group). ''Strong cheating words'' is a narrower list (cheat, fake, sabotage, exploit, bypass, evade, hack, loophole, 作弊, 造假 ...) chosen after unblinding.'
reports:
- report.md
figure: null
data:
- data/hack_manual_read.json
commits: []
related:
- hack-organism
- lens-concepts
---

# Reading the J-lens by hand on the reward-hack model: do eval-awareness, self-justification or sneakiness words show up, more in hidden-hack runs?

## Question

Going through stored J-lens top-25 lists by eye, before the hack appears in the text: do words that indicate evaluation awareness (test, grader), self-justification (acceptable, allowed) or sneakiness (hide, cheat, fake, bypass), in any language, appear in a sizeable share of runs, and more in runs that hack while their reasoning hides it than in clean runs of the same model?

## Method

120 runs of the reward-hack organism (one per problem within each subgroup): hidden hack 25 warm-start + 15 hiding version (beta 0.02); no hack 25 warm-start + 15 base; named hack 14 beta 0 + 13 beta 0.02 + 13 warm-start. Lists: stored top-25 of the J-lens and logit lens at layers 16, 24, 32, at the last prompt token and at the evenly spaced reasoning anchors whose next token precedes the first hack word (9,324 lists). Decoded with the pinned tokenizer into sheets; runs given random codes, key opened only after all flags were in. Flag criteria (four categories, English, Chinese and other languages) written before reading. 6,132 lists read line by line (all 4,662 J-lens lists; logit-lens lists of 37 runs) by me and seven Claude subagent readers with the same criteria. Echo check: flagged word, stem or English gloss in the preceding 100 tokens, or anywhere in prompt or earlier reasoning. Rates per run and per list, bootstrap over runs.

## Result

Sneakiness words (cheat, 作弊, 造假 fake, sabotage, pretending, bypass, exploit, evade, loophole, 违法违规) are common in the trained models' J-lens lists, often in dense clusters, and almost never echoes (7% anywhere earlier in the text). But they don't depend on the run's outcome: in the warm-start model, 14.7% of the lists of hidden-hack runs and 17.7% of no-hack runs have one, a difference of -3.1 points [-10.3, +4.1]. In the cleanest hidden case, the hiding version's hidden-hack runs, strong cheating words are rare (1.1% of lists; base model 0.4%). The clusters sit where the text is about to announce what it will do ("This problem is too hard. I'll ..."), including in no-hack runs that give up. Eval words are mostly task echoes (58% in the prompt or earlier text); justification words are mostly about problem constraints (compliant, acceptable). The logit lens shows the same clusters and no difference either (-2.9 points [-12.2, +6.9], 12 vs 8 runs).

## Interpretation

By eye, the J-lens does show cheating, faking and getting-caught concepts on this model before the reasoning words them, but they mark the trained model and the next few words, not that this run is hiding a hack: clean runs of the same model show them as often, and the model trained to hide shows almost none in its hidden runs. This agrees with the scored analysis (hack-organism). The literal echo check can't catch paraphrase-driven predictions such as 'too hard, I'll ...'.

## Caveats

- The echo check is literal: most clusters follow 'too hard / I'll ___', where cheat or fake is a likely next word given the text, so 'not an echo' doesn't mean 'not text-driven'.
- Eight readers (seven Claude subagents and me) with written criteria; inclusiveness varied (16-30% of lists flagged per sheet). Rows restricted to the fixed word list give the same answer.
- Blinding was partial: readers saw only lens lists, but dense cheating clusters and fewer positions can hint at a group.
- Labels are keyword rules: several warm-start 'hidden' runs describe the hack in other words, and some 'no hack' runs announce giving up or patching, which blurs the main contrast.
- The logit-lens comparison rests on 37 runs (only 2 base and 3 hiding-version hidden runs).

## Files

- Report: [report.md](report.md)
- Derived numbers: [data/hack_manual_read.json](data/hack_manual_read.json)
