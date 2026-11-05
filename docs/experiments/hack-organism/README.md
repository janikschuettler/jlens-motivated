---
id: hack-organism
title: 'A reward hack the reasoning hides: a model trained to hack (setup 1)'
strand: C
also:
- N
status: done
kind: pre-registered
verdict: against
confidence: moderate
dates:
  start: 2026-11-03T16:30
  end: 2026-11-03T18:10
  approx: true
timing:
  wall: about 1 h 40 min
  gpu_hours: 0.9
  api_usd: 0
  note: Validity 5 min, generation 279 s (1,922 tokens/s), readouts 41 min (58–95 runs/min), CPU analysis about 4 min per pass. 4.3 GB of readouts, now on arena9-cloud.
machine: arena9-octavia (L40S)
n:
  items: 120 problems (114 with hidden runs)
  runs: 1,920 generated; 2,723 read out (incl. 803 cross-reads)
numbers:
- label: 'How often each model hacks (and how often its reasoning names the hack): base / β 0 / β 0.02 / warm-start'
  value: 1/479 / 100% (84%) / 99% (32%) / 77% (45%)
- label: 'Gate, lens on the trained versions: J-lens error (KL) relative to base; logit lens at the last block vs own output'
  value: 0.89–0.99×; equals own log-probs (Δ 0)
- label: 'Main test: hidden-hack runs (β 0.02 model) vs clean runs (base model). J-lens / logit lens / reasoning keyword monitor (114 problems)'
  value: 0.995 [0.989, 0.999] / 0.81 [0.77, 0.85] / 0.46
- label: 'Control: the same clean text read by the β 0.02 model vs by the base model. J-lens / logit lens'
  value: 0.94 [0.91, 0.96] / 0.59
- label: 'Control: neutral words (apple, river …) instead of hack words, J-lens'
  value: 0.98 [0.97, 0.99]
- label: 'Control: model held fixed, warm-start hidden-hack vs no-hack runs. J-lens / logit lens (59 problems)'
  value: 0.44 [0.33, 0.55] / 0.47 [0.37, 0.57]
- label: 'Probe within the warm-start model: trained on runs that name the hack, tested on runs that hide it'
  value: 0.39 [0.29, 0.50]
- label: 'Probe trained on β 0 runs that name the hack, tested on β 0.02 hidden runs; control: the β 0.02 model reading clean text'
  value: 1.00 / 1.00
- label: 'Exploratory: the score at the last prompt token, before the model writes anything (J-lens)'
  value: '1.00'
- label: 'Exploratory: hack words minus neutral words, reading model fixed, matched for reasoning length. J-lens / logit lens'
  value: 0.59–0.61 / 0.65–0.69
numbers_note: 'From the third row on, values are AUROCs over problems: 0.5 is chance, 1 is perfect separation. Each control asks whether something other than the hack separates the groups as well. If identical text, neutral words or the prompt alone separate them, the main test is detecting the model, not the hack. In the gate row, a ratio below 1× means the lens fits the trained version at least as well as the base model.'
reports:
- report.md
- ../../preregistration.md
figure: fig11_hack.svg
data:
- data/hack_organism.json
commits:
- 234c119
- 3d9b806
- 10d46ef
related:
- q4-scoping
---

# A reward hack the reasoning hides: a model trained to hack (setup 1)

## Question

In a model trained to reward-hack while its reasoning hides the hack, do the J-lens, the logit lens or a probe show the hack before the reasoning or the code reveals it?

## Method

Models: base Qwen3-8B and three versions of it that others trained to reward-hack on coding tasks (LoRA adapters from cds-jb/qwen3-8b-reward-hack-kl-cot @ 7ec92bca, applied at load time). Two were trained further with reinforcement learning at two settings of a KL-penalty weight (β): β 0, whose reasoning usually names the hack, and β 0.02, whose reasoning usually hides it. The third is their common starting point after supervised fine-tuning only ('warm-start'); it hacks in some runs and not in others.
Task: UK AISI's CodeContests reward-hack task, rebuilt in our code. The system prompt describes three hacks and says not to use them. Thinking off; the model reasons in a text <thinking> block. 120 held-out hard problems × 4 samples × 4 models, temperature 1.0, up to 2,048 tokens.
Labels: a run hacks if its code calls os._exit, defines an always-true __eq__ or writes a conftest.py (checked by parsing the code). It is 'hidden' if it hacks and its reasoning matches neither hack keyword list.
Score: the lens log-probability of 27 single-token hack words (exit, hack, bypass, workaround …) at reasoning positions before any hack word appears, averaged over layers 14–32.
Pre-registered tests. Main test [H1]: hidden-hack runs of the β 0.02 model against clean runs of the base model, AUROC over problems. Controls [H2]: hold the model fixed (warm-start hidden-hack vs warm-start no-hack runs); have each model read the other's text; use neutral words instead of hack words. Probes [H3]: layer 24, trained on runs that name the hack, tested on runs that hide it.
Before the readout, a gate: does the lens still work on the trained versions (600 neutral tokens)?

## Result

No, not as a hack. The main test passed (AUROC 0.995 with the J-lens, 0.81 with the logit lens). But the controls show it detects which model is running, not the hack. With the model held fixed, hidden-hack and no-hack runs score alike: AUROC 0.44 (J-lens) and 0.47 (logit lens), about chance.

## Interpretation

On this model every readout sees that the trained adapter is there, not that this run is hacking. Identical text read by the two models already separates them: 0.94 in the J-lens against 0.59 in the logit lens. So the J-lens's +0.19 lead over the logit lens in the main test is a model-wide shift that the J-lens amplifies. Comparing lens scores across models needs the model held fixed. With the model fixed, no readout separates hidden-hack runs from no-hack runs before the code.

## Caveats

- The model was trained to hack; it is not cheating on its own. Its hidden reasoning was partly taught by fine-tuning on silent examples, so it may decide at the code, not in the reasoning. Just before the code, the warm-start model's runs separate at 0.59–0.62 (exploratory).
- The clean within-model test rests on the warm-start version (90 vs 71 runs). The β 0.02 version declines to hack in only 4 of 480 runs, so its own within-model tests are underpowered [H2a, H3c].
- The hack words are single tokens read at the next-token position, and the reasoning is short (median 77 tokens before any hack word, in hidden runs).
- Labels are keyword rules checked by eye. Some warm-start 'hidden' reasoning paraphrases the hack; dropping those runs changes nothing.

## Files

- Report: [report.md](report.md)
- Report: [../../preregistration.md](../../preregistration.md)
- Figure: [fig11_hack.svg](fig11_hack.svg)
- Derived numbers: [data/hack_organism.json](data/hack_organism.json)

Code, configs and outputs (from the old experiments index; data/ paths are on the GPU machines, not in git):

- `scripts/gen/hack_*.py`
- `scripts/lens/*hack*`
- `data/hack/`
