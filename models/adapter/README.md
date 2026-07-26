---
base_model: HuggingFaceTB/SmolLM2-135M-Instruct
library_name: peft
license: apache-2.0
language:
- en
tags:
- lora
- peft
- text2text-generation
- plain-language
- public-safety
---

# RecallClear — plain-language vehicle recall explainer

A LoRA adapter for `HuggingFaceTB/SmolLM2-135M-Instruct` that rewrites official U.S. vehicle safety
recall notices into a five-part summary an owner can act on:

```
WHAT'S WRONG / WHAT COULD HAPPEN / HOW URGENT / WHAT TO DO / WHAT IT COSTS
```

## Training data

4,651 recall notices published by NHTSA between 2013 and 2025
([Recalls Data](https://data.transportation.gov/d/6axg-epim), public domain).
Reference cards were derived from the notices' own structured fields using a
curated plain-language lexicon and a four-level urgency ladder anchored on
NHTSA's official `do_not_drive` and `fire_risk_when_parked` flags.

Seven manufacturers (Subaru, Mazda, Tesla, Mitsubishi, Volvo, Porsche, Winnebago) were held out entirely, so evaluation measures
generalisation to unseen brands.

## Intended use and limits

This is a **reading aid**, not safety or legal advice. The official notice
remains authoritative and should always be shown alongside the rewrite.
Do-not-drive and park-outside warnings should be taken from NHTSA's flags, not
from this model: evaluation found the notice text alone often lacks the
evidence needed to recover them.

## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
model = PeftModel.from_pretrained(base, "HanfuZhao781/recallclear-smollm2-135m-lora")
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
```

Full pipeline, evaluation, and web app: https://github.com/hanfuzhao/Module4-RecallClear
