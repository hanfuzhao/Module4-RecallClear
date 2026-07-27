# RecallClear — 5-minute pitch script

Matched one-to-one with `slides.html` (open it in a browser, press **F** for
fullscreen, **→** to advance). ~610 words at a slow, conversational pace
lands right at 5:00. Numbers in the script match the slides exactly — never
say a number the slide doesn't show.

排练提示：每页台词念完再翻页；慢就是稳。粗体是可以加重语气的词。
三个创新点出现在第 4、6、7 页——这三页最慢、最清楚。

---

## Slide 1 · The hook — 0:00–0:30

> Let me read you one sentence from a real letter. Millions of Americans get
> this in the mail.
>
> *"Underbody heat and noise insulators may loosen and contact the aluminum
> driveshaft, which could damage the driveshaft and cause it to fracture."*
>
> This is a **safety recall**. It's telling you your car might be dangerous.
> And it reads like a legal contract. We measured eleven thousand of these
> letters. The average reading level? **College.**

（读引文时放慢，读完停一拍再说 "This is a safety recall."）

---

## Slide 2 · The problem — 0:30–0:55

> Here's the bigger picture. **One in four** cars on U.S. roads has an open
> safety recall. The repair is **always free** — that's federal law. But
> owners still don't get it fixed. Because the letter never answers the three
> things you actually want to know: how bad is this, what do I do, and what
> does it cost me.

---

## Slide 3 · What we built — 0:55–1:25

> So I built **RecallClear**. It turns any recall letter into five plain
> lines. What's wrong. What could happen. How urgent — as a road-sign answer.
> What to do, with the phone number. And what it costs — which is always
> **"nothing."**
>
> It's a live web app. Paste your letter — or just type the recall number —
> and you get your answer.

---

## Slide 4 · How — 1:25–2:10 ⭐ 创新点 1

> Under the hood: I fine-tuned **SmolLM2** — a 135-million-parameter model —
> with **LoRA**. Only three and a half percent of the weights. The whole
> adapter is **twenty megabytes**.
>
> First innovation: the training data. I didn't hand-label anything. I took
> **eleven thousand five hundred real notices** from the government's open
> database, and built every training target with **auditable rules** — using
> NHTSA's own structured fields, and their official safety flags as gold
> labels.
>
> And here's the fun part: the whole thing trained in **25 minutes, on a
> laptop CPU**. I benchmarked it — on my machine, the CPU beat the GPU by
> **four times**.

---

## Slide 5 · Before and after — 2:10–2:50

> Did it learn? I held out **seven car brands completely** — the model never
> saw a single Subaru or Tesla notice. Then I compared three systems. The
> stock model. The stock model with examples stuffed into the prompt — that's
> the honest baseline. And my fine-tune.
>
> Correct card format: stock, **zero** percent. Prompting, **sixty-three**.
> Fine-tuned: **one hundred percent. Every single letter.** Reading grade
> dropped from thirteen to **seven**. At **one-third** of the prompt tokens.
>
> And you don't have to trust me — the app has **three bays**, one per
> system. Run the race yourself.

---

## Slide 6 · The failure — 2:50–3:50 ⭐ 创新点 2

> Now the result I'm most proud of. It's a **failure**.
>
> Some letters carry the government's highest warning: **do not drive this
> car**. I trained the model to say "STOP DRIVING." Recall on unseen brands:
> **zero percent.** So I oversampled those cases eight times and retrained.
> **Still zero.** Even on letters it had already seen in training.
>
> Why? That warning is **three tokens** out of a 130-token answer. The loss
> function averages over all the tokens — so the model can look nearly
> perfect, and never once sound the alarm.
>
> Class imbalance was not the problem. **Token imbalance was.**

（"Still zero" 之后停两拍。这是全场最重的一句。）

---

## Slide 7 · The fix — 3:50–4:35 ⭐ 创新点 3

> So here's the design answer: **stop asking the model.** One table.
>
> The model: zero percent. **Fifteen lines of rules** that just read the
> letter: seventy-three percent — and seventy-three **is everything**,
> because only seventy-three percent of those letters say the warning in
> words at all. And NHTSA's official flags: one hundred.
>
> The app ships all three layers. The model writes the prose — that's where
> it went zero to a hundred. The rules and the flags own the alarm.
>
> Our shop rule: **generation is the model's job. The alarm never is.**

---

## Slide 8 · Close — 4:35–5:00

> Everything is public. The repo — sixteen reviewed pull requests,
> eighty-five tests. The model, on Hugging Face. And the live app.
>
> A twenty-megabyte adapter, trained on a laptop, takes a government letter
> from grade thirteen to grade seven — and the system knows exactly which
> decision the model is **not allowed to make**.
>
> Thank you.

---

## 录制备忘

- **语速**：全文 ~610 词。感觉快了就在句号处多停，不要删句子。
- **计时锚点**：2:10 必须讲到 Slide 5；3:50 必须讲到 Slide 7。超了就砍
  Slide 4 的最后一句（CPU/GPU 那句），它最可牺牲。
- **演示替代**：如果想插 10 秒实况，就在 Slide 5 时切到应用点 "Race all
  three"，只给镜头不解说，音轨继续念稿。
- 稿子里没有的数字不要现场加；所有数字都来自 `data/outputs/evaluation.json`。
