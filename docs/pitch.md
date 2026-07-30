# RecallClear — 5 分钟 pitch（每页幻灯 + 讲稿）

放映：`slides.pptx`（或浏览器开 `slides.html`，F 全屏，→ 翻页）。
全稿 ~520 词，**慢慢念刚好 5 分钟**。规则只有一条：幻灯片上没有的数字不说。

**Demo 只做一次准备**：录制前打开应用 → 点 glovebox **第一个例子
（Porsche Cars North America）** → 点 **Race all three** → 跑完别动这个标签页。

---

## Slide 1 · 开场 (0:00–0:30)

> Let me read you one sentence from a real letter that millions of Americans
> get in the mail.
>
> *"Underbody heat and noise insulators may loosen and contact the aluminum
> driveshaft, which could damage the driveshaft and cause it to fracture."*
>
> This is a safety recall. Your car might be dangerous — and the warning
> reads like a legal contract. We measured eleven thousand of these letters.
> Average reading level: **college**.

---

## Slide 2 · 问题 (0:30–0:55)

> One in four cars on U.S. roads has an open safety recall. The repair is
> always **free** — that's federal law. But owners still don't get it fixed,
> because the letter never answers three simple questions: how bad is this,
> what do I do, and what does it cost me.

---

## Slide 3 · 产品 (0:55–1:20)

> So I built **RecallClear**. It turns any recall letter into five plain
> lines: what's wrong, what could happen, how urgent, what to do — with the
> phone number — and what it costs, which is always "nothing."
>
> It's a live web app. Paste your letter, or just the recall number.

---

## Slide 4 · 方法 (1:20–2:05) ⭐ 创新点 1

> I fine-tuned **SmolLM2**, a 135-million-parameter model, with **LoRA** —
> three percent of the weights, a **20-megabyte** adapter.
>
> The first innovation is the data. **Nothing was hand-labeled.** I took
> **11,591 real notices** from the government's open database and built every
> training target with auditable rules, using NHTSA's own fields and official
> safety flags as gold labels.
>
> The whole training run took **25 minutes, on my laptop's CPU**.

---

## Slide 5 · 前后对比 + Demo (2:05–2:50) ⏱ 2:05 必须到本页

> Did it learn? I held out **seven brands completely** — the model never saw
> one Porsche or Tesla notice — and compared three systems: the stock model,
> the stock model with examples in the prompt, and my fine-tune.
>
> Format correct: **zero**, sixty-three, **one hundred percent**. Reading
> grade: thirteen down to **seven**. And the fine-tune uses one-third of the
> prompt tokens.

**【切浏览器】** 展示已跑完的三个车位，镜头从左扫到右：

> Here it is live, on that Porsche 918 from the glovebox. Stock: no card.
> Fine-tuned: all five sections. Same engine — twenty megabytes apart.

⚠️ **别滚到页面顶部**——红色横幅是第 7 页的。**【切回 PPT】**

---

## Slide 6 · 失败 (2:50–3:45) ⭐ 创新点 2

> Now the result I'm most proud of — a **failure**.
>
> Some letters carry the government's highest warning: **do not drive**. I
> trained the model to say "STOP DRIVING." Recall: **zero percent**. I
> oversampled those cases eight times and retrained. **Still zero** — even on
> letters it had seen in training. ⏸（停两拍）
>
> Why? That warning is **three tokens** out of a 130-token answer, and the
> loss averages over tokens. The model looks nearly perfect — and never
> sounds the alarm. Class imbalance wasn't the problem. **Token imbalance
> was.**
>
> And that Porsche on screen? It's actually a do-not-drive recall — and the
> model's card said "get it fixed soon."

---

## Slide 7 · 解法 + Demo (3:45–4:35) ⭐ 创新点 3 ⏱ 3:45 必须到本页

> So: **stop asking the model.** One table. The model: zero. **Fifteen lines
> of rules** reading the letter: seventy-three percent — which is everything,
> because only 73% of those letters say the warning in words at all. NHTSA's
> official flags: one hundred.

**【切浏览器，滚到顶部】** 红色禁驾横幅和模型的黄色卡片同框：

> Look at the screen. The red banner — "do not drive" — came from the rules.
> The model missed it; the system caught it.
>
> Our shop rule: **the model writes the prose. The alarm is never its call.**

---

## Slide 8 · 收尾 (4:35–5:00)

> Everything is public — the repo with twenty reviewed pull requests and
> eighty-five tests, the model on Hugging Face, and the live app.
>
> A 20-megabyte adapter, trained on a laptop, takes a government letter from
> grade thirteen to grade seven — and the system knows exactly which decision
> the model is **not allowed to make**. Thank you.

---

## 备忘

- 快了不砍句子，在句号处多停。唯一可砍：Slide 4 最后一句（25 分钟那句）。
- Demo 翻车退路：`data/outputs/before_after.jsonl`（14V816000 在内，全部预生成）。
- 评分点：策略 = Slide 4；before/after = Slide 5；伦理反思 = Slide 6–7。
