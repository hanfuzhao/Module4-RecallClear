# RecallClear

**Official vehicle safety recall notices, rewritten so a car owner can act on them.**

A LoRA fine-tune of `HuggingFaceTB/SmolLM2-135M-Instruct` that turns the dense,
legalistic text of a U.S. vehicle recall notice into a five-part plain-language
card. Built from a dataset of **11,591 real NHTSA recall notices** — and trained,
deliberately, on a laptop CPU.

- **Live app:** https://recallclear-166936551184.us-central1.run.app
- **Model:** `HanfuZhao781/recallclear-smollm2-135m-lora` on the Hugging Face Hub
- **Data:** [NHTSA Recalls Data](https://data.transportation.gov/d/6axg-epim), U.S. DOT open data (public domain)

---

## The problem

About one in four vehicles on U.S. roads has an open safety recall. Repairs are
free, and the manufacturer is required to mail every owner a notice. Completion
rates still stall well below 100 percent, and a large part of the reason is that
the notice itself is unreadable. Here is a real one, verbatim:

> Underbody heat and noise insulators may loosen and contact the aluminum
> driveshaft, which could damage the driveshaft and cause it to fracture. A
> fractured driveshaft can cause a loss of drive power, or a loss of vehicle
> control if the driveshaft contacts the ground. Additionally, unintended
> movement could occur while parked if the parking brake is not engaged.

The measured reading level across the corpus is **grade 12.4** — roughly a
first-year-of-college text, mailed to the entire driving population. It never
answers the three questions an owner actually has: *how bad is this, what do I do
now, and what will it cost me?*

RecallClear answers exactly those, in a fixed structure:

```
WHAT'S WRONG:      one plain sentence about the defect
WHAT COULD HAPPEN: one plain sentence about the risk
HOW URGENT:        STOP DRIVING / PARK OUTSIDE / GET IT FIXED SOON / SCHEDULE IT
WHAT TO DO:        the concrete next step, with the manufacturer's phone number
WHAT IT COSTS:     nothing — recall repairs are always free
```

## What was fine-tuned, and how

| | |
|---|---|
| **Base model** | `HuggingFaceTB/SmolLM2-135M-Instruct` (135M parameters) |
| **Method** | LoRA (rank 16, alpha 32, dropout 0.05) via PEFT |
| **Adapted modules** | all attention and MLP projections (`q,k,v,o,gate,up,down`) |
| **Trainable parameters** | 4.9M of 139M — **3.5%** |
| **Objective** | causal LM loss on the assistant turn only; prompt tokens masked |
| **Training data** | 4,000-notice subset of the 11,591-card dataset |
| **Schedule** | 1 pass, effective batch 16, lr 2e-4, cosine decay |
| **Hardware** | Apple M1 Pro, **CPU** (see below) |
| **Artefact** | ~20 MB adapter, committed to the repo and loadable on a free CPU host |

### The model size and the device were both measured, not assumed

Three base models were benchmarked on the actual constraint set — a 16 GB M1
Pro to train on, a free two-core CPU host to serve on:

| candidate | vocab | training | inference, one card on CPU |
|---|---|---|---|
| Qwen2.5-0.5B-Instruct | 152k | ~26 h/epoch (MPS) | — |
| SmolLM2-360M-Instruct | 49k | 2.4 s/sample (CPU) | 30+ s |
| **SmolLM2-135M-Instruct** | 49k | **1.6 s/sample (CPU)** | **~4 s** |

Qwen's 152k vocabulary triples the training-time logits tensor, which is the
whole 26-hour story. The 360M model trains fine but produces an app where every
card takes half a minute — usability is graded here. The 135M model makes the
app feel instant and lets the same wall-clock budget cover 3× the training data.
The task is a fixed-format rewrite with a controlled vocabulary, which is
exactly the regime where a small model gives up the least.

**Why the CPU and not the GPU:** measured on identical batches, Apple's Metal
backend ran an optimizer step in **163 s against the CPU's 40 s** — the
accelerator was four times slower than the cores next to it, degrading further
as the run went on. Metal caches buffers per tensor shape and never releases
them; on a unified-memory machine that ends in swap. Padding to shape buckets
helped but did not cure it, so training runs on CPU by default
(`RECALLCLEAR_DEVICE=mps` overrides where MPS is healthy).

## Where the training targets come from

NHTSA publishes each recall as several structured fields — `defect_summary`,
`consequence_summary`, `corrective_action` — plus two official owner-warning
flags, `do_not_drive` and `fire_risk_when_parked`.

The reference card for each notice is built from those fields by
[`scripts/plain_language.py`](scripts/plain_language.py):

1. **Extraction** — drop the recall-population boilerplate ("X is recalling
   certain 2021-2022 vehicles"), keep the sentences that describe the defect;
   strip mailing dates and phone numbers out of the remedy.
2. **Simplification** — apply a curated lexicon of ~150 technical terms
   (`restraint control module` → `the airbag computer`, `thermal runaway` →
   `a battery fire`), then cap sentence length.
3. **Triage** — assign one of four urgency levels. The top two come straight
   from NHTSA's own flags and are therefore **gold labels**; the bottom two are
   rule-derived from harm wording in the consequence text.
4. **Composition** — assemble the five sections, with the action step keyed to
   the urgency level.

The model then has to reproduce that card **from the raw notice text alone** —
which is the form an owner actually receives. That is the learned capability:
selecting the right spans, discarding boilerplate, substituting plain words,
and making the urgency call, in one pass with no structured fields available.

## Results

Measured on **150 notices from seven manufacturers held out of training
entirely** (Subaru, Mazda, Tesla, Mitsubishi, Volvo, Porsche, Winnebago), so
these numbers are generalisation, not memorisation.

<!-- RESULTS:START -->

| | base (zero-shot) | base + 2 examples | **fine-tuned** |
|---|---|---|---|
| Card format produced correctly | 0% | 63% | 100% |
| Urgency triage (macro-F1) | 0.00 | 0.03 | 0.21 |
| Reading grade level (lower is better) | 12.99 | 7.95 | 7.3 |
| Jargon per 100 words (lower is better) | 2.516 | 2.337 | 0.731 |
| States the repair is free | 0% | 66% | 100% |
| Invented a phone number (lower is better) | 0% | 2% | 1% |
| Grounding in the source notice | 0.88 | 0.59 | 0.54 |
| Output length (words) | 141 | 110.1 | 116.7 |
| Prompt tokens per request | 427.1 | 1334.1 | 427.1 |
| Seconds per card | 4.38 | 7.87 | 3.13 |

The original notices score **12.73** on reading grade and **2.912** on jargon; the rule-built reference cards score **7.22** and **0.236**.

### Rare, high-stakes classes

| | support | notices with explicit warning text | best achievable recall |
|---|---|---|---|
| STOP DRIVING | 33 | 24 | 73% |
| PARK OUTSIDE | 5 | 5 | 100% |

| | base (zero-shot) | base + 2 examples | **fine-tuned** |
|---|---|---|---|
| recall, STOP DRIVING | 0% | 6% | 0% |
| recall, PARK OUTSIDE | 0% | 0% | 0% |
| safety downgrades (lower is better) | 100% | 95% | 100% |

<!-- RESULTS:END -->

Three systems are compared, all using the same weights:

- **base** — the untuned model, zero-shot, same instruction.
- **few-shot** — the untuned model with two worked examples in the prompt. This
  is deliberately a *strong* baseline: prompting alone teaches the format, so the
  fine-tune has to earn its keep on more than layout.
- **tuned** — the same model with the LoRA adapter enabled.

## Honest limitations

**The model cannot make the do-not-drive call, and the app does not let it.**
This was established the hard way, twice. NHTSA's `do_not_drive` flag is
database metadata; on the held-out sample only 24 of 33 flagged notices (73%)
say it in words, so text sets a hard ceiling. The v1 model scored **0% recall**
against that ceiling. v2 oversampled the rare classes to 16% of the training
mix and still scored **0% — including on notices it trained on**. The mechanism
is banal: the urgency level is ~3 tokens of a ~130-token target, so
token-averaged cross-entropy is minimised almost perfectly by a model that
never says `STOP DRIVING`. (The modelling fix would be re-weighting the loss on
the urgency span — future work; the prompt-masking machinery already supports
it.)

So the alarm is not the model's job. A pair of anchored regexes detects the
warning wording directly and scores **73% recall — the entire textual
ceiling — with 3 false positives in 828**, and the lookup path uses NHTSA's
own flags, which are the only complete source:

| who makes the do-not-drive call | recall on held-out gold |
|---|---|
| fine-tuned model (either version) | 0% |
| 15 lines of anchored regex | 73% = everything text contains |
| NHTSA's structured flags (lookup path) | 100% |

Generation is the model's job; the alarm never is.

**The label distribution is extremely skewed.** 97% of notices land in
`GET IT FIXED SOON`, because nearly every safety recall carries crash, fire, or
injury risk. Accuracy is therefore a meaningless metric here — a model that
always guesses the majority class scores 97%. Macro-F1 and per-class recall are
reported instead.

**The targets are programmatically derived, not human-written.** A curated
lexicon and a rule set are consistent and auditable, but they encode one
author's judgement about what counts as plain language. A production version
would need plain-language review by people who actually receive these notices.

**Simplification loses information.** Replacing "electronic stability control"
with "the anti-skid system" helps a reader and costs precision. The app always
shows the original notice next to the rewrite for exactly this reason.

## Risks and ethics

- **Wrong urgency is the dangerous failure.** Understating a do-not-drive recall
  could get someone hurt. That is why the official flags drive the banner and the
  model's own call is presented as secondary.
- **Fabricated specifics** — an invented phone number or repair step — would send
  owners to the wrong place. Evaluation measures phone-number hallucination and a
  grounding score directly, and the raw notice is always one click away.
- **This tool must never charge or gate.** Recall repairs are free by law; the
  card says so every time, because "you owe nothing" is the fact most often
  exploited by bad actors around recalls.
- **Not affiliated with NHTSA**, and stated as such in the interface. A plausible
  government-looking tool that is not the government is its own hazard.
- **No data retention.** Notices pasted into the app are processed in memory and
  not stored.

## Project layout

```
├── README.md
├── requirements.txt          full dev environment
├── requirements-deploy.txt   inference-only deps for the container
├── Makefile                  make data / features / train / evaluate / app / test
├── setup.py                  one-command, resumable pipeline runner
├── main.py                   Flask web application
├── Dockerfile                container image for deployment
├── scripts/
│   ├── config.py             every path and hyperparameter
│   ├── make_dataset.py       download NHTSA recalls
│   ├── build_features.py     notices -> cards -> splits
│   ├── plain_language.py     lexicon, urgency triage, card construction
│   ├── prompts.py            the prompt contract shared by train/eval/app
│   ├── model.py              LoRA training and the RecallExplainer
│   ├── evaluate_model.py     metrics against the two baselines
│   ├── recall_lookup.py      live NHTSA campaign lookup
│   ├── app_service.py        service layer behind the web app
│   ├── demo.py               terminal before/after
│   └── push_to_hub.py        publish the adapter
├── models/adapter/           the trained LoRA weights
├── data/
│   ├── raw/                  downloaded notices (regenerable, git-ignored)
│   ├── processed/            train / validation / test splits
│   └── outputs/              metrics and before/after samples
├── tests/                    85 unit tests, no model weights needed
└── notebooks/                exploration only
```

## Running it

```bash
pip install -r requirements.txt

python setup.py            # download -> build -> train -> evaluate (resumable)
python main.py             # http://127.0.0.1:7860
```

Or stage by stage:

```bash
make data        # download 11,677 notices from the DOT open-data portal (~30s)
make features    # build 11,591 cards and the held-out splits (~10s)
make train       # LoRA fine-tune (~1h on an M1 Pro, much faster on a GPU)
make evaluate    # score tuned vs base vs few-shot on held-out brands
make demo        # print one before/after comparison
make test        # run the unit tests
make app         # serve the web app
```

### What is committed

| committed | not committed |
|---|---|
| `models/adapter/` — the trained LoRA weights (~22 MB) | `data/raw/` — a byte-for-byte reproducible public export |
| `data/processed/test.jsonl` — held-out notices the app serves as demos | `data/processed/train.jsonl` — regenerates in ~10s from the raw download |
| `data/processed/validation.jsonl` | `models/checkpoints/` — training scratch |
| `data/outputs/` — metrics and before/after samples | |

`make data && make features` reproduces everything that is not committed.

## Deployment

Live at **https://recallclear-166936551184.us-central1.run.app** on Google
Cloud Run (4 vCPU / 2 GiB, scale-to-zero). Hugging Face Spaces was the original
target, but new Docker Spaces now require a PRO subscription, so the app ships
as a plain container that runs anywhere:

```bash
docker build -t recallclear .
docker run -p 7860:7860 recallclear
# or
gcloud run deploy recallclear --source . --memory 2Gi --cpu 4 --allow-unauthenticated
```

The image bakes the base model weights in at build time (`scripts/bake_model.py`)
so serving never depends on huggingface.co being reachable — the first deploy
failed with a 429 precisely because it did. The committed LoRA adapter rides in
from the repo, gunicorn serves one worker with lazy model load, and live NHTSA
lookup can be disabled with `RECALLCLEAR_LIVE_LOOKUP=0` for an offline demo.

Three serving bugs were found only by deploying — a training-only import in the
serving path, the Hub rate limit above, and int8 quantisation breaking PEFT's
LoRA wrappers — all invisible on the Apple-silicon dev machine. The commit
history (PR #13) documents each.

## Attribution

- Recall data: NHTSA / U.S. Department of Transportation, public domain.
- Base model: [SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct), Apache 2.0.
- LoRA: Hu et al., [*LoRA: Low-Rank Adaptation of Large Language Models*](https://arxiv.org/abs/2106.09685) (2021), via [PEFT](https://github.com/huggingface/peft).
- Readability: Flesch-Kincaid grade level (Kincaid et al., 1975), implemented from the published formula in `scripts/plain_language.py`.
- Portions of the code were developed with help from an AI coding assistant
  and reviewed before merging; all data, training, and evaluation results are
  reproducible from this repository.
