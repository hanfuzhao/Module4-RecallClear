# RecallClear — pitch：每页幻灯片 + 对应讲稿

配套文件：`slides.pptx`（PowerPoint 放映用）/ `slides.html`（浏览器版，**F** 全屏 **→** 翻页）。
硬性限时 **5:00**。全稿 ~610 词——**慢慢讲刚好五分钟**，讲快了会提前念完，所以慢就是对。
规则：**幻灯片上没有的数字，嘴里不说**；所有数字出自 `data/outputs/evaluation.json`。

三个创新点在第 4、6、7 页（红色 INNOVATION 标），这三页语速放最慢。

---

## Slide 1 · 开场钩子 (0:00 – 0:30)

**画面**：奶油色信纸 + 红色 SAFETY RECALL 印章，一句真实召回原文；底部一行
"11,591 letters · grade 12.7"。

**讲稿：**

> Let me read you one sentence from a real letter. Millions of Americans get
> this in the mail.
>
> *"Underbody heat and noise insulators may loosen and contact the aluminum
> driveshaft, which could damage the driveshaft and cause it to fracture."*
>
> This is a **safety recall**. It's telling you your car might be dangerous.
> And it reads like a legal contract. We measured eleven thousand of these
> letters. The average reading level? **College.**

（引文放慢读；读完停一拍再说 "This is a safety recall."）

---

## Slide 2 · 问题 (0:30 – 0:55)

**画面**：大标题 "1 IN 4 CARS HAS AN OPEN SAFETY RECALL"，三个要点。

**讲稿：**

> Here's the bigger picture. **One in four** cars on U.S. roads has an open
> safety recall. The repair is **always free** — that's federal law. But
> owners still don't get it fixed. Because the letter never answers the three
> things you actually want to know: how bad is this, what do I do, and what
> does it cost me.

---

## Slide 3 · 产品 (0:55 – 1:25)

**画面**：黄色车牌 RECAL·CLR，五行卡片结构，一句 "Live web app"。

**讲稿：**

> So I built **RecallClear**. It turns any recall letter into five plain
> lines. What's wrong. What could happen. How urgent — as a road-sign answer.
> What to do, with the phone number. And what it costs — which is always
> **"nothing."**
>
> It's a live web app. Paste your letter — or just type the recall number —
> and you get your answer.

---

## Slide 4 · 方法 (1:25 – 2:10) ⭐ 创新点 1：公开数据自建监督信号

**画面**：标题 "LoRA on a 135M model, trained on a laptop CPU"；左侧三个要点
（11,591 条 / 零人工标注 / 3.5% 权重 = 20 MB），右侧实测小表（25 min · CPU 4× · ~4 s）。

**讲稿：**

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

（超时唯一可砍的句子：最后那句 CPU/GPU。若现场被追问"为什么"，一句话答案：
Metal 按张量形状缓存显存、16 GB 统一内存被逼进 swap——细节在 README 和 PR #11。）

---

## Slide 5 · 前后对比 (2:10 – 2:50) ⏱ 锚点：2:10 必须讲到本页

**画面**：大字 "0% → 63% → 100%"，四行对比表（格式 / 阅读等级 / 术语 / prompt tokens）。

**讲稿：**

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

（可选 10 秒实况：切到应用点 "Race all three"，画面给足、音轨照念，不解说。）

---

## Slide 6 · 失败 (2:50 – 3:50) ⭐ 创新点 2：可复现的负结果

**画面**：标题 'We trained it twice to say "STOP DRIVING." It wouldn't.'，四个要点。

**讲稿：**

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

（**"Still zero." 之后停两拍——全场最重的一句。** 最后一句一字一顿。）

---

## Slide 7 · 解法 (3:50 – 4:35) ⭐ 创新点 3：三层安全架构 ⏱ 锚点：3:50 必须讲到本页

**画面**：标题 "STOP ASKING THE MODEL"，三行表（0% / 73% / 100%），
大字口号 "The model writes the prose. The alarm is never its call."

**讲稿：**

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

## Slide 8 · 收尾 (4:35 – 5:00)

**画面**：车牌 + "20 MB. 25 minutes. A laptop. Grade 13 → Grade 7."，三个链接。

**讲稿：**

> Everything is public. The repo — eighteen reviewed pull requests,
> eighty-five tests. The model, on Hugging Face. And the live app.
>
> A twenty-megabyte adapter, trained on a laptop, takes a government letter
> from grade thirteen to grade seven — and the system knows exactly which
> decision the model is **not allowed to make**.
>
> Thank you.

---

## 附：评分点对照（录完自查）

| 作业要求 | 出现在 |
|---|---|
| 微调了什么模型、什么策略 | Slide 4（SmolLM2-135M + LoRA 3.5% / 20 MB） |
| before/after 对比 | Slide 5（0→63→100 表）+ 三车位实况 |
| 风险/伦理/评估反思 | Slide 6–7（负结果 + 三层架构） |

## 附：录制备忘

- 全稿 ~610 词。感觉快了就在句号处多停，**不要删句子**。
- 幻灯片外的数字一个都不加；数字全部出自 `data/outputs/evaluation.json`。
- 实况演示若翻车，退路是 `data/outputs/before_after.jsonl`（150 条对比全部预生成）。
- 手机竖屏给应用一个镜头——召回信本来就是在手机上被读（或被删）的。
