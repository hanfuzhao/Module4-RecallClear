# RecallClear — five-minute pitch

Hard stop at 5:00. Timings below are cumulative. Numbers marked `[eval]` come
from `data/outputs/evaluation.json` and should be read off the live file, not
from memory.

---

## 0:00 — 0:45 · The problem, shown not described

Open on a real recall notice on screen. Read one sentence of it aloud:

> "Underbody heat and noise insulators may loosen and contact the aluminum
> driveshaft, which could damage the driveshaft and cause it to fracture."

Then the three facts:

- Roughly **one in four** vehicles on U.S. roads has an open safety recall.
- The repair is **free**, and the manufacturer is legally required to mail the
  owner a notice.
- Measured across 11,677 notices, the average reading level is **grade 11.8** —
  college-level text mailed to the entire driving population.

The line to land: *the information is not withheld, it is unreadable.* The
notice never answers the three questions an owner actually has — how bad is
this, what do I do, what does it cost me.

## 0:45 — 1:30 · What was built

RecallClear turns any recall notice into five labelled lines:

```
WHAT'S WRONG · WHAT COULD HAPPEN · HOW URGENT · WHAT TO DO · WHAT IT COSTS
```

- **Base model:** `SmolLM2-360M-Instruct`, 360M parameters.
- **Method:** LoRA, rank 16, over all attention and MLP projections —
  **8.7M trainable parameters, 2.3% of the model**, one epoch.
- **Data:** **11,448 real NHTSA recall notices**, 2013–2025, from U.S. DOT open
  data.
- **Hardware:** a laptop. No GPU cluster.

Say why the model is small on purpose: it has to serve inference on a free CPU
host, and the claim being tested is that a targeted adapter lets a 360M model
beat a much larger one at one narrow job.

## 1:30 — 2:45 · Before / after, live

Run the demo on a **held-out manufacturer** — say so explicitly, since that is
what makes it evidence rather than a trick.

Toggle "also show the untuned base model" and let the two outputs sit
side by side.

- **Before:** prose. No sections, or invented ones. No urgency call.
- **After:** the five sections, every time, in order.

Then the measured version `[eval]`:

| | base | few-shot | **fine-tuned** |
|---|---|---|---|
| card format produced correctly | | | |
| reading grade | | | |
| urgency macro-F1 | | | |
| prompt tokens per request | | | |

**Do not skip the few-shot column.** Prompting alone teaches the format, so
naming that baseline out loud is what makes the comparison credible. The
fine-tune's argument is that it does the same job at roughly a fifth of the
prompt tokens, which on a CPU host is the difference between usable and not.

## 2:45 — 4:00 · Risks, ethics, and what evaluation cannot see

This is the part most projects hand-wave. Lead with the measurement that changed
the product.

**The finding.** NHTSA's do-not-drive designation is database metadata, not
words in the notice. Measured on the training corpus, only **60%** of
do-not-drive notices contain any explicit warning language, against a **1.4%**
base rate elsewhere. So a text-only model provably cannot recover that label in
about 40% of cases.

**What was done about it.** The app takes the do-not-drive and park-outside
banners from NHTSA's own flags, never from the model. The model writes the
prose; the agency sets the alarm. When someone pastes raw text, where no flag
exists, the interface says so instead of showing a reassuring blank.

Then the shorter points:

- **Accuracy is a trap here.** 97% of notices are one class, so a model that
  always guesses the majority scores 97%. Macro-F1, per-class recall, and a
  standalone count of *safety downgrades* are reported instead.
- **Fluent and wrong is the real hazard.** A confident, readable card that
  invents a phone number is worse than no card. Phone-number hallucination and a
  grounding score are measured directly, and the original notice is always one
  click away.
- **Simplification costs precision.** "Electronic stability control" → "the
  anti-skid system" helps a reader and loses information. That trade is the
  product, and it is why the source text is never hidden.
- **The targets are rule-derived, not human-written.** They are consistent and
  auditable, but they encode one author's idea of plain language. A real
  deployment needs review by people who actually receive these letters.

## 4:00 — 4:40 · Engineering, briefly

Only if the clock allows — this is the part to cut first.

- Two data-collection routes were abandoned before finding a bulk endpoint that
  returns the same corpus in 30 seconds instead of 3 hours.
- The first base model was swapped out after measuring **26 hours per epoch**,
  caused almost entirely by a 152k vocabulary inflating the loss logits.
- Padding batches to a multiple of 64 fixed unbounded memory growth on Apple's
  Metal backend, which caches buffers per tensor shape.

The point of mentioning any of this: the constraint was a laptop, and the design
decisions follow from measurements rather than from taste.

## 4:40 — 5:00 · Close

- **Repo:** `github.com/hanfuzhao/Module4-RecallClear`
- **Model:** `HanfuZhao781/recallclear-smollm2-360m-lora`
- **Live app:** _link_

Closing line: a 35 MB adapter, trained in about an hour on a laptop, takes a
government notice from grade 11.8 to grade 7 — and knows which decision it is
not allowed to make.

---

## Notes for the recording

- Have a `STOP DRIVING` example ready — it is the most vivid demonstration and
  the best lead-in to the ethics section.
- Show the app on a phone-width window for a moment; recall letters get read on
  phones.
- If a live call fails, fall back to `data/outputs/before_after.jsonl`, which has
  every held-out comparison already generated.
