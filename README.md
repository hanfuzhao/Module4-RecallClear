# RecallClear

**Official vehicle safety recall notices, rewritten so a car owner can act on them.**

A LoRA fine-tune of `HuggingFaceTB/SmolLM2-360M-Instruct` that turns the dense,
legalistic text of a U.S. vehicle recall notice into a five-part plain-language
card. Trained on **11,448 real NHTSA recall notices**.

- **Live app:** _see Deployment below_
- **Model:** `HanfuZhao781/recallclear-smollm2-360m-lora` on the Hugging Face Hub
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

The measured reading level across the corpus is **grade 11.8** — roughly a
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
| **Base model** | `HuggingFaceTB/SmolLM2-360M-Instruct` (360M parameters) |
| **Method** | LoRA (rank 16, alpha 32, dropout 0.05) via PEFT |
| **Adapted modules** | all attention and MLP projections (`q,k,v,o,gate,up,down`) |
| **Trainable parameters** | 8.7M of 370M — **2.3%** |
| **Objective** | causal LM loss on the assistant turn only; prompt tokens masked |
| **Training data** | 10,064 notices (8,461 after the length filter) |
| **Schedule** | 1 epoch, effective batch 16, lr 2e-4, cosine decay |
| **Hardware** | Apple M1 Pro (MPS), no GPU cluster |
| **Artefact** | ~35 MB adapter, small enough to commit and to load on a free CPU host |

Why a 360M model rather than something larger: the app has to run inference on a
free CPU host, and the whole point is that a targeted adapter lets a small model
beat a much larger one at one specific job. SmolLM2's 49k vocabulary also makes
training tractable on a laptop — a 0.5B model with a 152k vocabulary needed an
estimated 26 hours per epoch on the same machine, almost entirely because of the
size of the loss logits.

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
_Populated by `make evaluate`; see [`data/outputs/evaluation.json`](data/outputs/evaluation.json)._
<!-- RESULTS:END -->

Three systems are compared, all using the same weights:

- **base** — the untuned model, zero-shot, same instruction.
- **few-shot** — the untuned model with two worked examples in the prompt. This
  is deliberately a *strong* baseline: prompting alone teaches the format, so the
  fine-tune has to earn its keep on more than layout.
- **tuned** — the same model with the LoRA adapter enabled.

## Honest limitations

**The rare urgency classes have a hard ceiling.** NHTSA's `do_not_drive` flag is
database metadata, not necessarily words in the notice. Measured on the training
corpus, only **60%** of do-not-drive notices contain any explicit warning
language (against a 1.4% base rate elsewhere); park-outside notices fare better
at 92%. No text-only model can recover the other 40%. The evaluation reports
this ceiling alongside the scores, and **the deployed app takes the do-not-drive
and park-outside banners from NHTSA's flags, never from the model.**

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
├── tests/                    70 unit tests, no model weights needed
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
make features    # build 11,448 cards and the held-out splits (~10s)
make train       # LoRA fine-tune (~1h on an M1 Pro, much faster on a GPU)
make evaluate    # score tuned vs base vs few-shot on held-out brands
make demo        # print one before/after comparison
make test        # run the unit tests
make app         # serve the web app
```

### What is committed

The LoRA adapter and the processed splits are committed so the app runs straight
from a clone. The raw download is git-ignored and regenerated by `make data`.

## Deployment

The app is containerised for Hugging Face Spaces (Docker SDK):

```bash
docker build -t recallclear .
docker run -p 7860:7860 recallclear
```

The container installs CPU-only PyTorch, loads the base model plus the committed
adapter, and serves gunicorn with a single worker. Live NHTSA lookup can be
disabled with `RECALLCLEAR_LIVE_LOOKUP=0` for a fully offline demo.

## Attribution

- Recall data: NHTSA / U.S. Department of Transportation, public domain.
- Base model: [SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct), Apache 2.0.
- LoRA: Hu et al., [*LoRA: Low-Rank Adaptation of Large Language Models*](https://arxiv.org/abs/2106.09685) (2021), via [PEFT](https://github.com/huggingface/peft).
- Readability: Flesch-Kincaid grade level (Kincaid et al., 1975), implemented from the published formula in `scripts/plain_language.py`.
- Built with assistance from Claude (Anthropic) for code scaffolding and review;
  all data, training, and evaluation results are reproducible from this repository.
