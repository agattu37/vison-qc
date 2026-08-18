# VisionQC — The Complete Build Guide

**A step-by-step, beginner-friendly walkthrough of an industrial-grade computer vision project.**

You do not need prior deep learning experience to follow this. You do need to be
willing to read explanations rather than only copy commands — because the whole
point of this project is that you can *explain* it afterwards.

---

## How to use this guide

Read it in order. Every section follows the same shape:

> **What we're building** → **Why it matters** → **The concepts you need** →
> **The code** → **Run it** → **What to say in an interview**

Two rules that will save you a lot of pain:

1. **Run every command before moving on.** A pipeline you have not executed is
   a pipeline that does not work. This guide is built so you can run something
   within 15 minutes of starting.
2. **When you hit "Interview angle" boxes, actually write your answer down.**
   Not in your head. In a file. You will not be able to improvise these under
   pressure, and the difference between candidates who get offers and those who
   do not is almost never the model — it is the explanation.

### Table of contents

| Part | What you'll do | Time |
|---|---|---|
| [0. Orientation](#part-0--orientation) | Understand what you're building and why | 30 min |
| [1. Setup](#part-1--setup) | Environment, install, first run | 1 hour |
| [2. The problem](#part-2--the-problem-behind-the-project) | Why this isn't "just a CNN project" | 45 min |
| [3. Data](#part-3--data) | Get data, build leak-free splits | Week 1 |
| [4. Supervised path](#part-4--the-supervised-path) | Transfer learning classifier | Week 1–2 |
| [5. Explainability](#part-5--explainability-with-grad-cam) | Grad-CAM from scratch | Week 2 |
| [6. Autoencoder](#part-6--the-unsupervised-path-part-1-autoencoder) | Baseline anomaly detector | Week 3 |
| [7. PaDiM](#part-7--the-unsupervised-path-part-2-padim) | The real anomaly detector | Week 3–4 |
| [8. Evaluation](#part-8--evaluation-the-part-that-earns-the-interview) | Metrics, thresholds, error analysis | Week 4 |
| [9. Decision layer](#part-9--the-decision-layer) | Turn scores into verdicts | Week 5 |
| [10. Serving](#part-10--serving-with-fastapi) | FastAPI + Docker | Week 6–7 |
| [11. Free compute](#part-11--free-compute-and-tooling) | Colab, Kaggle, W&B | anytime |
| [12. Portfolio](#part-12--packaging-it-as-a-portfolio-piece) | README, demo, resume bullet | Week 8 |
| [13. Troubleshooting](#part-13--troubleshooting) | When things break | as needed |
| [Glossary](#glossary) | Every term, in plain English | reference |

There is a companion file, **`INTERVIEW_PREP.md`**, with 40+ practice questions
and model answers. Do not read it until you have finished Part 8 — the answers
will not stick until you have done the work they describe.

---

# Part 0 — Orientation

## 0.1 What you are building

A system that looks at a photograph of a manufactured part and answers three
questions:

1. **Is this part defective?** (a yes/no decision, with a confidence)
2. **Where is the defect?** (a heatmap over the image)
3. **What kind of defect is it?** (scratch, dent, contamination, crack…)

And it answers them **two different ways at once**, because real factories have
two different data situations.

```
                      ┌────────────────────┐
   Part image ───────▶│   Preprocessing    │
                      │  resize, normalise │
                      └─────────┬──────────┘
                                │
              ┌─────────────────┴──────────────────┐
              ▼                                    ▼
   ┌──────────────────────┐          ┌──────────────────────────┐
   │  SUPERVISED PATH     │          │   UNSUPERVISED PATH      │
   │  ResNet18 classifier │          │   PaDiM                  │
   │  trained on labelled │          │   fitted on NORMAL parts │
   │  defect examples     │          │   only — no labels       │
   └──────────┬───────────┘          └────────────┬─────────────┘
              │                                   │
              ▼                                   ▼
   ┌──────────────────────┐          ┌──────────────────────────┐
   │  Grad-CAM heatmap    │          │  Mahalanobis heatmap     │
   │  + defect class      │          │  + anomaly score         │
   └──────────┬───────────┘          └────────────┬─────────────┘
              │                                   │
              └───────────────┬───────────────────┘
                              ▼
                 ┌────────────────────────────┐
                 │      DECISION LAYER        │
                 │  cost-calibrated threshold │
                 │  PASS / FAIL / REVIEW      │
                 └─────────────┬──────────────┘
                               ▼
                 ┌────────────────────────────┐
                 │   FastAPI  ·  Docker       │
                 │   POST /inspect            │
                 └────────────────────────────┘
```

## 0.2 Why this project gets interviews

Be honest with yourself about the competition. A recruiter for an ML internship
sees hundreds of repos titled "Image Classification with CNN". They all look
like this: download a clean dataset, fine-tune ResNet, print 94% accuracy, done.

Those projects demonstrate that you can follow a tutorial. They do not
demonstrate engineering judgement, and judgement is what the job actually needs.

This project is different in five specific ways. Each one is a thing you can
*say* in an interview:

| Most portfolio projects | This project |
|---|---|
| Assume balanced, fully-labelled data | Built around the fact that defects are **rare and expensive to label** |
| One model | **Two complementary models** whose failure modes barely overlap |
| Report accuracy | Report **recall, precision, AUROC, AUPR** and explain why accuracy is misleading here |
| Threshold = 0.5 | Threshold chosen by **minimising business cost**, with a plot to prove it |
| Black box | **Grad-CAM + anomaly heatmaps** so a human can verify the model looked at the right thing |
| Notebook | **Tested, containerised REST API** |
| No failure discussion | **Explicit error analysis** on real false negatives and false positives |

> ### 💡 Interview angle
> The single strongest sentence you can say about this project is:
>
> *"I didn't design it around a dataset, I designed it around a constraint —
> that in real quality control, defects are rare by definition, so you can't
> rely on having labelled examples of them."*
>
> That reframes you from "person who trains models" to "person who thinks about
> problems". Write your own version of that sentence now, in your own words.

## 0.3 What you need to know already

**Required:**
- Basic Python: functions, classes, imports, loops, dictionaries
- Comfort with a terminal: `cd`, running commands, activating a virtualenv

**Not required — this guide teaches it:**
- Neural networks, CNNs, backpropagation
- PyTorch
- Anomaly detection
- FastAPI, Docker
- Any of the maths

If a term is unfamiliar, check the [Glossary](#glossary) at the end. Every
technical word used in this guide has an entry there.

## 0.4 The realistic timeline

The PRD says 6–8 weeks. That is honest for someone learning as they go, working
part-time alongside classes.

| Week | Goal | You'll have |
|---|---|---|
| 1 | Setup, data, first classifier | A model that predicts defect types |
| 2 | Grad-CAM, error analysis | Heatmaps showing *why* |
| 3 | Autoencoder | A working unsupervised baseline |
| 4 | PaDiM + full evaluation | Your headline results table |
| 5 | Decision layer, thresholds | A system, not just models |
| 6 | FastAPI | A live endpoint |
| 7 | Docker, deploy | Something you can send a link to |
| 8 | README, demo, interview prep | A portfolio piece |

**If you have less time**, the minimum viable version is Weeks 1, 3, 4, 6 —
classifier, PaDiM, evaluation, API. Skip the autoencoder and Docker. That is
still a stronger project than 95% of what you are competing against.

**A word of caution about deadlines.** You said this project matters a lot to
you. That is exactly why you should build the minimum version end-to-end
*first*, then improve it. A complete rough system beats a beautiful half-system
every time, because you cannot demo half a system. Get `make all` working in
week 1 even if the numbers are bad, then spend the remaining weeks making the
numbers good.

---

# Part 1 — Setup

## 1.1 Install Python and create an environment

You need **Python 3.10 or newer**. Check:

```bash
python3 --version
```

Create an isolated environment. This matters: installing PyTorch globally will
eventually break some other project on your machine, and "it works on my
machine" is not a thing you want to say in an interview.

```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Your prompt should now start with `(.venv)`. If it does not, the activation did
not work — fix that before continuing, because everything after this installs
into the wrong place.

## 1.2 Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**If you are on a CPU-only machine** (no NVIDIA GPU — this includes all Macs and
most laptops), install PyTorch from its CPU index instead. The default wheel
bundles ~2.5 GB of CUDA libraries you cannot use:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### Why the versions are pinned the way they are

Open `requirements.txt` and look at the format: `torch>=2.13,<3.0`.

This is called a **compatible-release bound**. It says "give me patch fixes and
minor improvements, but never a new major version". Major versions are where
breaking changes live. Without the upper bound, someone cloning your repo in six
months gets `torch 3.0`, half your code fails, and they conclude you write
fragile software.

> ### 💡 Interview angle
> "How do you make a project reproducible?" is a common question. The full
> answer has four parts, and you have all four:
> 1. Pinned dependency ranges (`requirements.txt`)
> 2. A fixed random seed everywhere (`utils.set_seed`)
> 3. Config saved alongside every run (`config_used.yaml`)
> 4. A deterministic data split written to disk (`splits.json`)

## 1.3 Verify the install

```bash
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__)"
```

You should see something like `2.13.0 0.28.0`. If you see an error about a
missing CUDA library, you installed the GPU wheel on a CPU machine — reinstall
using the CPU index above.

## 1.4 Run something immediately

This is the most important 10 minutes of the setup. Do not skip it.

```bash
# 1. Generate a synthetic dataset (~1 min)
make data

# 2. Build the split manifest (~2 sec)
make splits

# 3. Fit the anomaly detector (~1-3 min on CPU)
make padim

# 4. Evaluate (~1 min)
make evaluate
```

Then open `artifacts/synthetic/results.md`. You have a working end-to-end
anomaly detection system with real metrics, and you have not downloaded a single
dataset.

### Why we ship a synthetic dataset

MVTec AD requires a licence click-through and a multi-gigabyte download. Kaggle
needs an API token. Both are easy, but both mean you cannot run your own code on
day one — and beginners who cannot run anything on day one tend to quit.

So `src/visionqc/data/synthetic.py` fabricates brushed-metal discs with four
defect types and pixel-perfect ground-truth masks, in **exactly** the folder
layout MVTec uses. Everything downstream works identically. When you get the
real data, you change one path in a config file.

**Be honest about its limits.** Synthetic defects are easier than real ones —
the noise model is known and the lighting is simpler. Your synthetic numbers
will be optimistic. Use synthetic data to prove the *code* works; report MVTec
numbers as your actual result.

> ### 💡 Interview angle
> "I built a synthetic data generator matching my production schema so I could
> test the pipeline without waiting on data access" is a genuinely senior thing
> to have done. It is the same pattern as fixtures in software testing. Say it
> that way.

## 1.5 Tour of the repository

Spend five minutes here. Knowing where things live makes the rest of the guide
much faster.

```
visionqc/
├── configs/                    # YAML: every knob for a run
│   ├── synthetic.yaml          #   the default, works out of the box
│   ├── fast.yaml               #   tiny + quick, for smoke tests
│   ├── mvtec_bottle.yaml       #   real data
│   └── casting.yaml
│
├── src/visionqc/
│   ├── config.py               # typed config, YAML loading
│   ├── utils.py                # seeding, device, logging, timers
│   │
│   ├── data/
│   │   ├── synthetic.py        # the fake-dataset generator
│   │   ├── splits.py           # ⭐ the leak-free split protocol
│   │   └── datasets.py         # Dataset classes + augmentation
│   │
│   ├── models/
│   │   ├── classifier.py       # ResNet18 transfer learning
│   │   ├── autoencoder.py      # conv AE baseline
│   │   └── padim.py            # ⭐ PaDiM, from scratch
│   │
│   ├── explain/
│   │   ├── gradcam.py          # ⭐ Grad-CAM, from scratch
│   │   └── overlay.py          # heatmap rendering
│   │
│   ├── metrics.py              # ⭐ metrics + cost-based thresholds
│   ├── decision.py             # ⭐ score -> PASS/FAIL verdict
│   ├── inference.py            # ⭐ shared engine (no training/serving skew)
│   │
│   ├── train_classifier.py     # scripts you run
│   ├── train_autoencoder.py
│   ├── fit_padim.py
│   ├── evaluate.py
│   │
│   └── api/
│       ├── main.py             # FastAPI app
│       └── schemas.py          # request/response contracts
│
├── scripts/
│   ├── prepare_dataset.py      # real datasets -> canonical layout
│   └── make_demo_assets.py     # README figures
│
├── tests/                      # 57 tests
├── Dockerfile
├── Makefile
└── requirements.txt
```

The ⭐ files are the ones worth being able to explain line by line. They are
where the interesting decisions live.

**Every source file starts with a long docstring explaining *why* it exists.**
That is not decoration — read them. They contain reasoning that is not repeated
in this guide.

---

# Part 2 — The problem behind the project

Before writing any model code, you need to understand the problem well enough to
defend your design. This section is short but it is the most important part of
the guide for interview purposes.

## 2.1 The situation in a real factory

A production line makes 10,000 parts a day. Around 98–99.5% are fine. Human
inspectors check them, and humans are slow, expensive, and inconsistent — the
same inspector will make different calls at 9am and 5pm.

So the factory wants automation. But here is the catch that shapes everything:

> **Defects are rare by definition.** A healthy production line produces very
> few of them. That is the goal of the production line.

Which means:

- You have **tens of thousands** of images of good parts, effortlessly.
- You have maybe **fifty** images of defects, and someone had to hand-label each.
- New defect types appear over time — a new supplier, a worn tool, a changed
  process — and **nobody has ever labelled those**.

## 2.2 Why a classifier alone is not enough

A supervised classifier learns to recognise the categories it was trained on. It
is very good at that. But it has a specific, dangerous blind spot:

> **A classifier shown a defect type it has never seen will confidently assign
> it to the closest class it knows.** And the closest class is usually "good".

Think about what that means on a factory floor. A new kind of defect appears. It
ships. Your monitoring shows no anomalies, because the model is happily and
confidently wrong. You find out from a customer complaint.

This is not hypothetical — it is the standard failure mode of supervised
inspection systems, and it is why the field cares about anomaly detection.

## 2.3 The anomaly detection idea

Flip the problem around:

> Instead of learning what **defects** look like, learn what **normal** looks
> like. Then flag anything that deviates.

This is a profound reframing and it solves several problems at once:

| Problem | How anomaly detection solves it |
|---|---|
| Defects are rare | Don't need them — train on the abundant normal data |
| Labelling is expensive | Don't need labels at all for training |
| New defect types appear | Anything unlike normal is flagged, including unseen types |
| Class imbalance | No classes; there's one distribution and a distance from it |

The cost: it can tell you *something is wrong* but not *what*. It also fires on
benign variation — a new lighting setup looks "abnormal" too.

## 2.4 So: use both

```
                 Classifier                 Anomaly detector
                 ─────────────────────      ──────────────────────
Knows            the defect types it        what "normal" looks like
                 was trained on

Catches          known defects, and         ANY deviation, including
                 names them                 types nobody labelled

Blind to         unseen defect types        nothing — but it can't
                 (calls them "good")        name what it found

Needs            labelled defects           only normal images
                 (expensive, scarce)        (abundant, free)

Fooled by        novel defects              benign changes
                                            (lighting, new fixture)
```

Their blind spots barely overlap. So flagging when **either** fires catches more
than either alone. That is the dual-path design, and it follows directly from
the problem — not from wanting to use two models.

> ### 💡 Interview angle — this is your core narrative
> Practise saying this out loud until it's natural:
>
> *"The two models fail differently. The classifier is blind to defect types it
> was never trained on — it'll confidently call a novel defect 'good'. PaDiM
> doesn't know defect types at all, it only knows normal, so it catches novel
> defects but can't name them. Because their blind spots barely overlap,
> combining them catches strictly more than either alone. I flag if either
> fires, and I accept the precision cost because in QC a missed defect costs
> roughly ten times a false alarm."*
>
> **Then be ready for the follow-up: "When would you NOT use an OR rule?"**
> Answer: when false positives are expensive or capacity-limited — for example
> fraud review with a fixed-size human team, where every false positive consumes
> a reviewer slot that a real case needed. The rule follows from the cost
> structure, not from ML fashion.

## 2.5 The costs are asymmetric — and that changes everything

One more idea before we code.

- **False negative** (missed defect): part ships → reaches customer → complaint,
  warranty claim, possibly a recall. In safety-critical parts, worse.
- **False positive** (false alarm): part is pulled → an operator looks at it for
  30 seconds → back on the line.

These are not equally bad. Not close. If you had to put a number on it, a missed
defect might cost 10×, 100×, or 1000× a false alarm depending on the industry.

This has a direct consequence for your code: **`score > 0.5` is meaningless.**
0.5 is a convention, not a decision. The right threshold is the one that
minimises expected cost, and you can compute it. We do exactly that in Part 8.

> ### 💡 Interview angle
> When you present results, never lead with accuracy. Lead with:
> *"At my chosen operating point I catch X% of defects with Y% precision, and I
> chose that point by minimising 10×FN + 1×FP on validation — here's the cost
> curve."* That single sentence separates you from almost every other candidate.

---

# Part 3 — Data

## 3.1 The canonical folder layout

Everything in this project expects one layout — the MVTec AD convention:

```
data/<dataset>/
├── train/
│   └── good/              ← NORMAL ONLY. This fits the anomaly model.
│       ├── 000.png
│       └── ...
├── test/
│   ├── good/              ← normal test samples
│   ├── scratch/           ← defective samples, one folder per type
│   ├── dent/
│   └── ...
└── ground_truth/          ← optional pixel masks
    ├── scratch/
    │   ├── 000_mask.png   ← 255 where the defect is, 0 elsewhere
    │   └── ...
    └── ...
```

Two things to notice, because both are load-bearing:

1. **`train/` contains only good parts.** This is not an accident of how MVTec
   was packaged — it is the whole protocol. The anomaly detector must never see
   a defect.
2. **Defect *types* are folder names.** That gives us multiclass labels for free,
   with no CSV to maintain.

## 3.2 Option A — synthetic data (start here)

```bash
python -m visionqc.data.synthetic --root data/synthetic \
    --train-good 220 --test-good 60 --per-defect 25
```

Produces 380 images in about a minute:

```
train/good           220     ← fits PaDiM / trains the autoencoder
test/good             60
test/scratch          25     ← bright thin line
test/dent             25     ← dark soft depression
test/contamination    25     ← scattered dark specks
test/crack            25     ← dark jagged line (hardest class)
```

### How the generator works (worth reading)

Open `src/visionqc/data/synthetic.py`. The interesting trick is in
`_brushed_texture`:

```python
noise = rng.standard_normal((size, size))
streaks = gaussian_filter(noise, sigma=(0.6, 9.0))   # ← different sigma per axis
```

Blurring white noise **much more along x than y** turns random noise into
directional streaks — which is what machined metal looks like. That anisotropic
blur is the whole texture model.

Each defect function returns `(modified_image, binary_mask)`, so the ground
truth is exact by construction. This is one advantage synthetic data genuinely
has over real data: your masks are perfect.

## 3.3 Option B — MVTec AD (your real benchmark)

MVTec AD is *the* standard benchmark for this exact problem. Using it means your
numbers can sit next to published research, which is worth a lot.

**Getting it:**

1. Go to the MVTec Software website and find the Anomaly Detection dataset page.
   Search for "MVTec Anomaly Detection dataset download".
2. It's free for research/educational use; you fill in a short form.
3. Download and extract. It's ~5 GB, 15 object/texture categories.

**Start with ONE category, not all fifteen.** `bottle`, `hazelnut`, or `screw`
are good choices — small, visually clear, fast to train.

```bash
python scripts/prepare_dataset.py mvtec \
    --src ~/Downloads/mvtec_anomaly_detection/bottle \
    --dst data/mvtec/bottle

python -m visionqc.data.splits --root data/mvtec/bottle \
    --out artifacts/mvtec_bottle/splits.json
```

Then run everything with `--config configs/mvtec_bottle.yaml`.

> **Reference numbers, so you know if yours are sane.** The PaDiM paper
> (Defard et al., ICPR 2021) reports roughly **0.89 mean image-level AUROC**
> across all 15 MVTec categories with a ResNet18 backbone and 100 random
> dimensions, and around **0.98** with a much larger WideResNet50 backbone.
> Individual easy categories like `bottle` typically score well above the mean.
> **Verify these against the paper before quoting them in an interview** — do
> not cite a number you got from a guide.

## 3.4 Option C — Kaggle casting dataset

A useful second domain: ~7,300 labelled images of submersible pump impellers,
defective vs OK. Search Kaggle for "Casting Product Image Data for Quality
Inspection".

```bash
python scripts/prepare_dataset.py casting \
    --src ~/Downloads/casting_data --dst data/casting
```

It has **no pixel masks**, so pixel-level AUROC will report `n/a`. That is fine
— say so in your results rather than hiding it.

Why bother with a second dataset? Because "I validated on two domains" is a
meaningfully stronger claim than "I got a good number on one". It's evidence
your approach generalises rather than that you got lucky.

## 3.5 ⭐ The split protocol — the most important 20 minutes in this guide

This is where most portfolio projects have a hidden flaw. Read carefully.

### The problem

Our two models have **incompatible** training requirements:

- The anomaly detector must see **only normal** images.
- The classifier must see **labelled defects**.

The naive fix is to give each model its own split. That is a trap: you then have
no valid way to compare them, because they were scored on different test sets.
Any difference in their numbers might just be a difference in test difficulty.

### The rule

> **There is exactly one test set, and neither model sees any part of it, ever.**

```
┌──────────────────────────────────────────────────────────────┐
│ train/good/          (normal images, no defects present)      │
│   ├─▶ anomaly_fit    100%  — fits PaDiM / trains the AE       │
│   └─▶ sup_train/val  also usable as labelled "good" examples  │
├──────────────────────────────────────────────────────────────┤
│ test/good/ + test/<defect>/   (the only labelled defects)      │
│   ├─▶ HOLDOUT TEST   50%, stratified — untouched until the end│
│   ├─▶ sup_train      ~40%  — the classifier learns from here  │
│   └─▶ sup_val        ~10%  — early stopping AND thresholds    │
└──────────────────────────────────────────────────────────────┘
```

Run it:

```bash
python -m visionqc.data.splits --root data/synthetic --out artifacts/synthetic/splits.json
```

Output:

```
anomaly_fit  n=220   [good=220]
sup_train    n=240   [contamination=10, crack=10, dent=10, good=200, scratch=10]
sup_val      n=62    [contamination=3, crack=3, dent=3, good=50, scratch=3]
test         n=78    [contamination=12, crack=12, dent=12, good=30, scratch=12]
Leak check passed: test set is disjoint from all training splits.
```

### Three things this design gets right

**1. The classifier is deliberately data-starved.** Ten images per defect class.
That is not a limitation of the code — it is the honest reflection of the
problem. Real inspection projects have this little labelled data. Do not "fix"
it by moving test data into training.

**2. Stratification is per class, not global.** With 25 images of a rare defect,
a global random split can easily give you zero of them in test — and then recall
for that class is undefined. We sample the holdout independently from each
class's pool.

**3. The leak check runs automatically, every time.**

```python
def _verify_no_leak(manifest):
    test_paths = {r["path"] for r in splits["test"]}
    for name in ("anomaly_fit", "sup_train", "sup_val"):
        overlap = test_paths & {r["path"] for r in splits[name]}
        if overlap:
            raise AssertionError(f"LEAK: {len(overlap)} images in both ...")
```

A check you have to remember to run is not a check. This one raises on every
build.

> ### 💡 Interview angle — expect this exact question
> **"How did you split your data?"** is asked in almost every ML interview,
> because it is the fastest way to find out whether a candidate understands
> leakage.
>
> Your answer: *"I had a constraint most projects don't — my unsupervised model
> can only train on normal data, and my supervised model needs labelled defects.
> I still wanted one comparable test set, so I froze a stratified 50% of the
> labelled pool as holdout, gave the anomaly model the normal-only training
> folder, and split the remainder for supervised train and validation. I
> stratified per class because some defect types only had 25 images and a global
> split could have left a class with zero test samples. The manifest is written
> to disk with a seed, and there's an assertion that runs on every build
> verifying test is disjoint from every training split."*

### Data augmentation — and why the choices are domain-specific

Open `src/visionqc/data/datasets.py`. Note what we do and do not do:

| Augmentation | Used? | Why |
|---|---|---|
| Rotation (full 360°) | ✅ | A part on a conveyor has no canonical "up" |
| Horizontal/vertical flip | ✅ | Same reason |
| Small translation, slight scale | ✅ | Part placement is not pixel-perfect |
| Brightness / contrast jitter | ✅ | Factory lighting drifts, lamps age |
| **Hue / saturation jitter** | ❌ | Would recolour a rust stain to look like clean metal — destroys the signal |
| **Random erasing / cutout** | ❌ | Punches synthetic holes that look exactly like contamination defects. You'd be teaching the model that defects are normal. |

> ### 💡 Interview angle
> The cutout point is a genuinely strong answer to *"how did you choose your
> augmentations?"*. Most candidates recite a standard recipe. Saying **"I ruled
> out random erasing because it synthesises something visually identical to one
> of my defect classes"** shows you thought about the domain rather than copying
> a config.

**A critical asymmetry:** augmentation is applied to the *training* split only.
Validation and test use a fixed, deterministic pipeline. Applying random
augmentation at evaluation time makes your metrics wobble between runs and is a
genuinely common beginner bug.

**And one more:** the classifier trains *with* rotation, but PaDiM fits
*without* it. Why? PaDiM models a separate distribution per patch position. If
you rotate, the patch at grid cell (3,7) comes from the rim in one image and the
centre in another — the per-position Gaussians blur into one meaningless global
distribution. The same augmentation that helps one model destroys the other.
This is a great small detail to mention.

## 3.6 Look at your data

Before training anything, look at the images. Really. Open the folders. Compare
a good part to each defect type.

Ask yourself:
- Could *you* spot the defect? How long did it take?
- Which defect type looks hardest? (For our synthetic data: `crack` — it's thin
  and low-contrast.)
- Is there anything in the background that correlates with the label? (This is
  how you catch a model that learns "images with a timestamp are defective".)

Your predictions about which classes will be hard are worth writing down now.
Comparing them against your actual results in Part 8 is exactly the kind of
thing that makes a good error-analysis section.

---

# Part 4 — The supervised path

## 4.1 What a CNN actually does (60-second version)

If you already know this, skip to 4.2. If not, here is the minimum you need.

An image is a grid of numbers. A **convolution** slides a small window (say 3×3)
across that grid, multiplies each patch by a set of learned weights, and writes
out a new grid. Different weight sets detect different things: one might respond
to vertical edges, another to a particular texture.

Stack many convolution layers and something useful happens:

```
Layer 1   ──▶  edges, colour blobs           (tiny receptive field)
Layer 2   ──▶  corners, simple textures
Layer 3   ──▶  patterns, object parts
Layer 4   ──▶  whole objects, semantics      (large receptive field)
```

Each layer builds on the one below. This hierarchy is learned, not designed.

At the end, **global average pooling** collapses each feature map to one number,
and a **linear layer** maps those numbers to class scores (**logits**). Softmax
turns logits into probabilities that sum to 1.

Training = show it images, compare its predictions to the truth with a **loss
function**, compute how each weight should change (**backpropagation**), nudge
the weights (**optimiser**), repeat.

## 4.2 Transfer learning — why we don't train from scratch

We have about 40 labelled defect images for training. Training a CNN from random
initialisation on 40 images is hopeless — it will memorise them perfectly and
learn nothing that generalises.

**Transfer learning** solves this. A ResNet18 trained on ImageNet (1.2 million
images, 1000 classes) has already learned, in its early layers, exactly the
things that are expensive to learn: edges, corners, textures, gradients.

Here is the key insight to internalise:

> **A scratch on metal is an edge. A dent is a shading gradient. ImageNet
> contains no machined parts, but the low-level features transfer almost
> perfectly anyway** — because low-level visual structure is universal.

We keep those features and only teach the network what to *do* with them.

## 4.3 ⭐ Staged unfreezing — the detail most tutorials skip

Read `src/visionqc/models/classifier.py`. The training happens in two phases:

```
Phase 1 (epochs 0 to freeze_epochs-1):
    backbone FROZEN, only the new head trains
    → 2,565 trainable parameters

Phase 2 (remaining epochs):
    backbone UNFROZEN, everything fine-tunes
    → 11,179,077 trainable parameters
    but with a 10x smaller learning rate on the backbone
```

### Why phase 1 exists

On epoch 1 your classifier head is **randomly initialised**. It produces large,
meaningless gradients. If the backbone is trainable at that moment, those
garbage gradients flow backwards and damage the pretrained weights you got for
free. This is sometimes called catastrophic forgetting.

So: let the head learn a sane mapping first, *then* unfreeze.

### Why discriminative learning rates

```python
def param_groups(self, lr_head, lr_backbone):
    return [
        {"params": self.net.parameters(),  "lr": lr_backbone},   # 1e-4
        {"params": self.head.parameters(), "lr": lr_head},       # 1e-3
    ]
```

The backbone already knows things — it needs gentle nudges. The head knows
nothing — it needs real learning. A single global learning rate cannot do both:
too high and you wreck the backbone, too low and the head never learns.

### ⚠️ The freezing bug almost everyone hits

```python
def freeze_backbone(self):
    for p in self.net.parameters():
        p.requires_grad = False
    self.net.eval()          # ← THIS LINE
```

Setting `requires_grad = False` stops the *weights* updating. But **BatchNorm
layers keep updating their running mean and variance during any forward pass in
training mode** — those are buffers, not parameters. So your "frozen" backbone
keeps shifting its outputs under the head, and you get mysteriously unstable
training.

`self.net.eval()` stops it. One line, very easy to miss, and there is a test for
it in `tests/test_visionqc.py::test_freeze_actually_freezes`.

> ### 💡 Interview angle
> If asked *"what's a subtle bug you've hit in PyTorch?"* — this is a great
> answer. It's real, it's specific, it demonstrates you understand the
> difference between parameters and buffers, and it shows you write tests for
> things that fail silently.

## 4.4 Handling class imbalance

Our training split: 200 good, 10 of each defect type. A model that predicts
"good" every single time gets **83% accuracy** and catches zero defects.

Fix — inverse-frequency class weights in the loss:

```python
w = counts.sum() / (len(counts) * counts)
w = w / w.mean()      # normalise to mean 1
```

Actual output on our data:

```
{'good': 0.06, 'contamination': 1.23, 'crack': 1.23, 'dent': 1.23, 'scratch': 1.23}
```

One scratch mistake now costs ~20× a good-image mistake. The `w / w.mean()`
normalisation keeps overall loss magnitude comparable to unweighted, so your
learning rate does not silently need retuning.

**Other approaches you should be able to name:**

| Approach | When it's right |
|---|---|
| Class weights (ours) | Simple, no data duplication, works well |
| Oversampling minorities | When you have enough distinct minority samples that duplication adds value |
| Focal loss | Extreme imbalance (1:1000+), heavy easy-negative dominance |
| Undersampling majority | When you have far too much majority data and compute is the constraint |
| **Anomaly detection** | When the minority is *so* rare that supervision fails — **this is our other path** |

That last row is the connection worth making out loud: our dual-path design is
itself a response to class imbalance taken to its logical extreme.

## 4.5 Choosing the model-selection metric

In `train_classifier.py` we select the best checkpoint on **validation macro
recall** — not loss, not accuracy.

- **Accuracy** is dominated by the `good` class. A model that never predicts
  `crack` still scores well.
- **Loss** is a proxy that can improve while the metric you care about gets
  worse — it rewards confidence on easy examples.
- **Macro recall** averages recall over classes with **equal weight**, so
  completely failing one rare defect type is heavily penalised.

> ### 💡 Interview angle
> *"Why did you select on macro recall instead of validation loss?"* →
> *"Because loss and my objective aren't the same thing. In QC, missing an
> entire defect class is the failure I care about, and macro recall is the only
> one of the three that punishes it properly. Selecting on the metric you
> actually care about is nearly free and avoids a whole category of surprise."*

## 4.6 Run it

```bash
make classifier
# or: python -m visionqc.train_classifier --config configs/synthetic.yaml
```

Watch the log for these specific things:

```
Class weights: {'good': 0.06, 'contamination': 1.23, ...}
Phase 1: backbone FROZEN, 2565 trainable params        ← freezing worked
Phase 2: backbone UNFROZEN, 11179077 trainable params  ← unfreezing worked
epoch 00 | train loss 1.4021 rec 0.412 | val loss 1.2210 rec 0.550 f1 0.501
...
Best epoch 9 | val macro-recall 0.9167
```

**On CPU this is slow** — expect 15–40 minutes at 256px. Options:
- Use Google Colab's free GPU (Part 11) — minutes instead of an hour
- Use `configs/fast.yaml` (128px, 2 epochs) to verify the code path first

### Reading the training curve

Open `artifacts/synthetic/plots/clf_history.png`.

| What you see | What it means | What to do |
|---|---|---|
| Both losses fall, val recall rises | Healthy | Nothing |
| Train loss falls, val loss rises | Overfitting | More augmentation, more dropout, fewer epochs |
| Neither moves | Learning rate too low, or something's frozen that shouldn't be | Check the phase logs |
| Loss becomes `nan` | Learning rate too high | Drop `lr_head` 10× |
| Val recall spikes then collapses | Too-aggressive unfreezing | Increase `freeze_epochs` |

---

# Part 5 — Explainability with Grad-CAM

## 5.1 Why this is not optional

Two reasons, and the second one is the one people forget.

**Reason 1 — trust.** A QC operator will not scrap a part because a black box
printed 0.93. They need to see *where*. Without localisation your system is
technically accurate and operationally useless.

**Reason 2 — debugging.** This is the big one. Heatmaps are how you discover
that your model is keyed on the conveyor belt edge, a timestamp overlay, or a
lighting gradient rather than the part.

That failure mode — **a model that is accurate on your test set for entirely the
wrong reason** — is extremely common and completely invisible in your metrics.
There is a well-known class of examples where models learned to detect a scanner
watermark, a hospital's imaging device, or a ruler placed next to a lesion,
rather than the thing they were supposedly detecting.

Grad-CAM is how you catch it before you ship it.

## 5.2 How Grad-CAM works

Take the last convolutional layer. For ResNet18 at 256×256 input, it outputs
**512 feature maps of 8×8**. Each map detects some pattern, and crucially it
still knows *where* that pattern occurred — it hasn't been flattened yet.

We want to know which of those 512 detectors mattered for the prediction.
Backpropagation answers exactly that: the gradient of the class score with
respect to each map says how much nudging that map would change the score.

```
weight_k  =  mean over spatial positions of  d(score_c) / d(A_k)

heatmap   =  ReLU( Σ_k  weight_k · A_k )
```

Then upsample from 8×8 to 256×256.

In three sentences: **average each feature map's gradient into one importance
number; take a weighted sum of the maps; keep only positive evidence.**

The `ReLU` matters — we want evidence *for* the predicted class, not against it.
Without it, regions arguing against the class would show as "important".

## 5.3 The implementation

Open `src/visionqc/explain/gradcam.py`. It's ~60 lines. Two details worth
understanding:

**Hooks.** PyTorch lets you attach a function that fires when a module runs
forward, or when a tensor's gradient is computed:

```python
def forward_hook(_module, _inp, output):
    self.activations = output
    if output.requires_grad:
        output.register_hook(self._save_grad)    # tensor hook for the gradient
```

**We use a context manager**, so hooks are always removed:

```python
with GradCAM(model, model.target_layer()) as cam:
    heatmap, predicted_class = cam(images)
```

Leaked hooks are nasty in a long-running API: they silently keep tensors alive
and leak memory across requests. There's a test asserting they're removed.

### ⚠️ The bug that makes people think Grad-CAM "returns zeros"

```python
with torch.enable_grad():      # ← required, even at inference time
    logits = self.model(images)
    score.backward()
```

Grad-CAM needs a **backward pass**. If you call it inside `torch.no_grad()` —
which you naturally would at inference — there are no gradients, and you get an
all-zero map with no error message.

### Why hook `layer4` and not later

`target_layer()` returns the last residual block. The layer after it is global
average pooling, which **destroys all spatial information**. Hook there and your
heatmap is uniform and useless.

## 5.4 Grad-CAM's honest limitation

At 256×256 input, ResNet18's `layer4` is **8×8**. Upsampling to 256×256 gives a
blurry blob. That's fine for "the defect is in this region" and useless for "the
scratch is exactly these pixels".

> ### 💡 Interview angle — volunteer this before they ask
> *"Grad-CAM's resolution is capped by the layer it hooks — 8×8 for ResNet18 at
> 256px, so it's a coarse blob. For crisp localisation the PaDiM map is much
> better: it works at a 32×32 grid and isn't tied to a predicted class. I built
> both because they answer different questions — Grad-CAM tells me why the
> classifier decided what it did, PaDiM tells me where the image deviates from
> normal."*
>
> Volunteering a limitation before being asked reads as confidence, not
> weakness. Interviewers are trained to probe for weaknesses; getting there
> first changes the dynamic.

Alternatives worth being able to name: **Grad-CAM++** (better for multiple
instances of a class), **Score-CAM** (gradient-free, slower), **Eigen-CAM**,
**LayerCAM** (fuses multiple depths for better resolution).

---

# Part 6 — The unsupervised path, part 1: autoencoder

## 6.1 The idea

Train a network to squeeze an image into a small code and rebuild it, using
**only normal parts**.

It becomes very good at rebuilding normal parts — and *only* normal parts,
because that's all it has ever seen. Show it a scratched part and it rebuilds
the part but not the scratch: it has no vocabulary for scratches.

```
   input          encoder        bottleneck       decoder        output
  ┌──────┐       ┌──────┐        ┌────┐         ┌──────┐       ┌──────┐
  │ part │  ───▶ │ 3→32 │  ───▶  │ 64 │  ───▶   │ 32→3 │ ───▶  │ part │
  │ with │       │ 32→64│        │×16 │         │ 64→32│       │ WITH-│
  │scratch      │64→128│        │×16 │         │128→64│       │ OUT  │
  └──────┘       └──────┘        └────┘         └──────┘       └──────┘
                                                                   │
                    input − output = error map ────────────────────┘
                    (lights up exactly where the scratch is)
```

Subtract reconstruction from input → the leftover error **is** your heatmap. Its
maximum is your anomaly score.

## 6.2 ⭐ The bottleneck is the whole game

> If the latent code is large enough, the autoencoder learns the **identity
> function**. It copies input to output, reconstructs defects perfectly, and
> detects nothing.

This is the number one reason beginner autoencoders fail at anomaly detection,
and *"how did you size the bottleneck?"* is a fair interview question.

Our numbers:
```
input   : 256 × 256 × 3  = 196,608 values
latent  :  16 ×  16 × 64 =  16,384 values     → 12× compression
```

The compression is what forces it to learn a *model of normality* rather than a
copy.

**A counter-intuitive consequence:** lower reconstruction loss is not
automatically better for detection. Train long enough with enough capacity and
it starts reconstructing anything — including defects — and AUROC falls while
loss keeps dropping. If you see that, shrink `latent_channels` or stop earlier.

## 6.3 Two implementation details worth knowing

**1. Upsample + Conv, never ConvTranspose2d.**

```python
nn.Upsample(scale_factor=2, mode="nearest"),
nn.Conv2d(cin, cout, kernel_size=3, padding=1),
```

`ConvTranspose2d` produces **checkerboard artefacts** when kernel size isn't
divisible by stride. In an anomaly detector those artefacts become fake "errors"
scattered across every image — precisely the signal you're trying to read.
Upsample+Conv avoids it entirely at negligible cost.

**2. Blur the error map.**

A single hot pixel is sensor noise, not a defect. Real defects occupy a
**region**. Blurring makes the score respond to spatially coherent evidence and
measurably improves AUROC. Both PaDiM and PatchCore do this; it isn't a hack.

We implement it as two 1-D convolutions rather than one 2-D kernel — a Gaussian
is *separable*, so this is O(k) per pixel instead of O(k²), stays on-device, and
needs no OpenCV.

## 6.4 We train it expecting to beat it

Be upfront about this — it's the honest framing that makes the project credible.

Autoencoders have two known weaknesses for this task:

1. **They generalise too well.** Smooth or low-contrast defects get
   reconstructed anyway. Our `crack` class is exactly this.
2. **Per-pixel L2 error is dominated by texture.** A perfectly normal but
   slightly misaligned edge produces more error than a genuine subtle defect.

PaDiM sidesteps both by comparing **pretrained features** rather than raw pixels.

> ### 💡 Interview angle
> *"Why build the autoencoder if you knew PaDiM would win?"* →
> *"A baseline you can beat is how you demonstrate the upgrade was worth it. If
> I'd only built PaDiM I'd have a number with nothing to compare it to. And the
> comparison told me something specific — the gap was widest on my low-contrast
> crack class, which matches the known weakness of reconstruction-based methods.
> That's a finding, not just a benchmark."*

## 6.5 Run it

```bash
make autoencoder
```

```
Fit on 198 normal images | val on 22 normal images
Autoencoder: 1.11M params | latent 64 x 16 x 16
epoch 00 | train MSE 0.47516 | val MSE 4.70252
epoch 03 | train MSE 0.07200 | val MSE 0.07312
```

Note we hold out normal images for validation. We **cannot** use defect data for
early stopping — that would be exactly the supervision we're pretending not to
have, and it would make the "unsupervised" claim false.

---

# Part 7 — The unsupervised path, part 2: PaDiM

This is the technically strongest part of the project. Take your time.

## 7.1 The algorithm in plain English

**PaDiM = Patch Distribution Modeling** (Defard et al., ICPR 2021).

**Step 1.** Push a normal image through a **frozen, pretrained** ResNet. Grab
feature maps from three depths and stack them channel-wise.

```
layer1 →  64 channels @ 64×64    (fine texture)
layer2 → 128 channels @ 32×32    (patterns)
layer3 → 256 channels @ 16×16    (structure)
                ↓ resize all to a common 32×32 grid, concatenate
        448 channels @ 32×32
```

Every location on that grid is now a **448-dimensional vector describing one
patch** of the image, combining fine texture with coarse structure.

**Step 2.** Do that for all 220 normal training images. For each of the
32×32 = **1024 grid positions**, you now have a cloud of 220 vectors:
*"here is the distribution of what patch (7, 12) normally looks like."*

**Step 3.** Summarise each cloud with a **Gaussian** — a mean vector and a
covariance matrix. Note this is **per position**.

```
   position (0,0)        position (7,12)       position (31,31)
   ┌─────────────┐       ┌─────────────┐       ┌─────────────┐
   │   ·  ·      │       │      · ·    │       │  ·          │
   │ · ·μ₀ ·     │       │    ·  μ₁ ·  │       │ · μ₂  ·     │
   │   · ·       │       │      · ·    │       │   ·  ·      │
   └─────────────┘       └─────────────┘       └─────────────┘
     background            part rim              background
```

**This per-position modelling is why PaDiM beats simpler methods.** The top-left
corner of a part is *expected* to look different from its centre. Methods that
model the whole image with one distribution lose that.

**Step 4.** At test time, compute how far each patch is from **its own
position's** Gaussian, in Mahalanobis distance. Far = anomalous.

## 7.2 ⭐ Why Mahalanobis and not Euclidean

This is the question you're most likely to be asked about, so let's make it
concrete.

Euclidean distance treats every feature dimension as equally important and
independent. They are not. Some feature channels vary wildly across perfectly
normal parts (lighting, texture phase); others barely vary at all.

> A 2-unit deviation in a rock-stable channel is alarming.
> The same 2-unit deviation in a noisy channel is nothing.

Mahalanobis divides by the observed spread, per direction:

```
d(x) = √( (x − μ)ᵀ Σ⁻¹ (x − μ) )
```

**The intuition, in one line:** Euclidean asks *"how far in raw units?"*;
Mahalanobis asks *"how many standard deviations, accounting for how the
dimensions co-vary?"*

Picture it in 2D. Suppose normal data forms a long diagonal ellipse:

```
        Euclidean                    Mahalanobis
     (circular contours)         (contours follow the data)
                                          
         ·  ○  ·                      ·  ╱▔▔╲  ·
       ○   ╱▔╲   ○                  ·  ╱ data ╲  ·
      ○   │data│  ○     vs         · ╱  cloud  ╲ ·
       ○   ╲▁╱   ○                  ·╲         ╱·
         ·  ○  ·                      ·╲▁▁▁▁▁╱·
                                          
    point A and B look        A (off-axis) is correctly
    equally far away          flagged; B (along-axis) is not
```

Point A sits *off* the ellipse's long axis; point B sits along it, the same
Euclidean distance away. Euclidean calls them equally unusual. Mahalanobis
correctly says A is anomalous and B is normal variation. That's the whole idea.

## 7.3 Why random dimension selection

448 channels → a 448×448 covariance per position × 1024 positions ≈ **840 MB**,
and slow to invert.

The paper's finding: **randomly keeping 100 of the 448 dimensions performs as
well as PCA-selecting them, and as well as keeping all 448** — while being ~20×
cheaper.

Why does random beat PCA here? PCA keeps directions of greatest variance in
*normal* data. But those aren't necessarily the directions where **anomalies**
show up. A dimension that's constant across normal parts is *low variance* — PCA
throws it away — yet it's exactly where a defect would stand out most.

> ### 💡 Interview angle
> That PCA explanation is a genuinely sophisticated point and very few
> candidates can make it. Practise it.

## 7.4 Implementation details worth defending

Open `src/visionqc/models/padim.py`.

**Streaming covariance.** We accumulate `Σx` and `Σxxᵀ` per position instead of
storing all embeddings:

```python
sum_x  += e64.sum(dim=1)
sum_xx += torch.einsum("pbd,pbe->pde", e64, e64)
```

Memory then depends on grid size and dimension, **not on how many images you
have** — so this scales to datasets that don't fit in RAM.

**float64 for the accumulation.** Covariance sums lose precision in fp32. Cheap
insurance.

**Ridge regularisation:**

```python
cov = cov + self.reg * avg_var * eye
```

Without this, when the number of images isn't comfortably larger than the
dimension, the covariance is near-singular and the inverse explodes. **This one
line is the difference between "works" and "produces inf".** Scaling by
`avg_var` makes `reg` mean the same thing regardless of feature scale.

**Cholesky instead of an explicit inverse:**

```python
chol = torch.linalg.cholesky(cov)
y = torch.linalg.solve_triangular(self.chol, delta, upper=False)
dist = y.pow(2).sum(dim=1).sqrt()
```

Solving `L y = δ` gives `yᵀy = δᵀ(LLᵀ)⁻¹δ` — exactly the squared Mahalanobis
distance, without ever forming an inverse matrix. More numerically stable and
faster.

**Max, not mean, for the image score:**

```python
return self.anomaly_map(x).flatten(1).max(dim=1).values
```

A defect is small — often under 1% of pixels. Its contribution to a *mean* is
diluted to nothing by the 99% of normal pixels, and image-level AUROC collapses.
Max asks the right question: *"is there anywhere that looks wrong?"*

> ### 💡 Interview angle
> *"Why max instead of mean?"* is a favourite follow-up. The answer above is
> short, concrete, and shows you thought about the geometry of the problem.

## 7.5 PaDiM has no training loop

There is no loss function, no optimiser, no epochs. Nothing is learned by
gradient descent — the backbone stays frozen. Fitting is **one pass** to
accumulate statistics.

```
PaDiM embedding: 448 channels -> 100 random dims, 32x32 grid
PaDiM fitted on 220 normal images (1024 patches, d=100)
PaDiM fit: 18.3s                     ← on a single CPU core
Saved -> artifacts/synthetic/padim.pt (52.5 MB, self-contained)
```

> ### 💡 Interview angle
> *"When a factory adds a new part variant, PaDiM re-fits in under a minute from
> 200 photos, on CPU, with no labelling. That's a completely different
> operational story from 'retrain the classifier overnight and please label 50
> defects first.' The deployment economics are part of why I chose it."*

## 7.6 The self-contained checkpoint

`PaDiM.save()` stores the **backbone weights** alongside the statistics. Costs
~45 MB. Worth it for two reasons:

1. **Correctness.** The Gaussians were estimated in one specific feature space.
   Pair them with different backbone weights and every distance is meaningless.
   Shipping them together makes that mistake impossible.
2. **Deployment.** The container needs no network at startup. On a free tier
   that cold-starts your app, waiting on a weight download is the difference
   between a 3-second and a 90-second first request — or a crash if the CDN is
   unreachable.

There's a test asserting `save()` → `load()` reproduces bit-identical scores.

## 7.7 Why implement it instead of `pip install anomalib`

Using the library is the right call in a job. For a portfolio project it's the
wrong call, for one blunt reason:

> **In an interview you will be asked how it works, and "I called a function"
> ends the conversation.**

The algorithm is ~120 lines. Writing it means you can draw Mahalanobis distance
on a whiteboard. That's a completely different conversation.

Mention `anomalib` anyway — knowing when *not* to hand-roll is also a signal:
*"For production I'd use anomalib, which is Intel-maintained and implements
PaDiM, PatchCore and others with proper benchmarking. I implemented it myself
here because understanding it was the point."*

## 7.8 What PatchCore does differently (know this)

PatchCore is PaDiM's successor and usually scores higher. Expect to be asked.

| | PaDiM | PatchCore |
|---|---|---|
| Normal model | One Gaussian per patch **position** | A **memory bank** of patch features from all positions |
| Scoring | Mahalanobis to that position's Gaussian | Nearest-neighbour distance to the bank |
| Assumes | Features are roughly Gaussian | Nothing parametric |
| Handles misalignment | Poorly (positions must correspond) | Well (position-agnostic) |
| Memory | Grid × d² | Coreset-subsampled bank |

The key trade: PaDiM's per-position model is more precise **when parts are
consistently aligned**, and breaks down when they aren't. PatchCore drops the
positional assumption. Since our parts are conveyor-aligned, PaDiM is a
reasonable fit — but you should say that's *why*.

## 7.9 Run it

```bash
make padim
```

---

# Part 8 — Evaluation: the part that earns the interview

If you only do one part of this project properly, make it this one. Modelling is
commoditised; careful evaluation is not.

## 8.1 Why accuracy is the wrong headline

A line runs at 2% defect rate. A model that stamps PASS on everything is **98%
accurate** and has caught **zero** defects.

Accuracy rewards that model. So we never lead with it.

## 8.2 The four metrics that matter, and why

### Recall (sensitivity) — *of all real defects, how many did we catch?*

```
recall = TP / (TP + FN)
```

**The number the plant manager cares about.** A missed defect leaves the
factory, reaches a customer, and can trigger a recall.

### Precision — *of everything we flagged, how many were really defective?*

```
precision = TP / (TP + FP)
```

Low precision means operators waste time re-inspecting good parts. Within a week
they start ignoring the system. **A model nobody trusts has zero value
regardless of its recall.**

### AUROC — *threshold-free ranking quality*

The probability that a randomly chosen defective image scores higher than a
randomly chosen good one. 0.5 = random, 1.0 = perfect separation.

We report it because it's the **standard MVTec AD metric** — which lets you put
your number next to published results instead of in a vacuum.

### AUPR (average precision) — *AUROC's honest sibling*

AUROC can look flattering under heavy imbalance: a large *absolute* number of
false positives is still a small *rate* when negatives dominate. AUPR has no
such blind spot. **Reporting both is the honest choice.**

> ### 💡 Interview angle
> *"Why report both AUROC and AUPR?"* →
> *"AUROC's false-positive rate has the negative count in the denominator, so
> with 30 good and 48 defective images it behaves reasonably — but on a real
> line at 2% defect rate, a thousand false positives is still a tiny FPR while
> being operationally unacceptable. AUPR is sensitive to that. I report both so
> the number doesn't flatter the model."*

## 8.3 ⭐ The threshold is a business decision

`score > 0.5` is a convention with **no meaning** in QC. The real question is:
what does each mistake cost?

```
total_cost(t) = C_fn × (missed defects) + C_fp × (false alarms)
```

Set `C_fn = 10`, `C_fp = 1` — *"one escaped defect hurts as much as ten
unnecessary re-inspections"* — then pick the threshold minimising total cost.

Our config:
```yaml
decision:
  cost_false_negative: 10.0
  cost_false_positive: 1.0
```

And the output:
```
Anomaly threshold 11.4607 — Minimises 10*FN + 1*FP on the validation split.
At this point: recall=1.000, precision=0.706, FN=0, FP=5, total cost=5.
```

Look at `artifacts/<run>/plots/cost_curve.png`. It shows total cost against
threshold with a clear minimum and your chosen point marked. **That plot is your
answer to "why 11.46?"**

> ### 💡 Interview angle — one of the strongest things you can say
> When asked *"how did you pick your threshold?"*, most candidates say "0.5" or
> "I tuned it". You say:
>
> *"0.5 has no meaning here — it's a convention from balanced binary
> classification. I made it a business decision instead. I assigned a relative
> cost of 10 to a missed defect versus 1 to a false alarm, based on the fact
> that an escaped defect reaches a customer while a false alarm costs an
> operator 30 seconds. Then I swept thresholds on validation and picked the
> cost minimum. I've got the cost curve plotted. And if the plant told me the
> real ratio was 50:1, I'd change one config line and re-derive it — the
> threshold isn't hard-coded, it's a function of the cost structure."*
>
> **Follow-up you should expect: "Where did the 10 come from?"**
> Be honest: *"I chose it as a defensible placeholder. In a real deployment I'd
> get it from the business — warranty cost per escaped defect versus operator
> time per re-inspection. The point is the framework, and that the number is an
> explicit, changeable input rather than buried in code."*

There's a second strategy implemented too — `select_threshold_by_recall` — for
when recall is a hard contract ("we must catch 95%"). It picks the *highest*
threshold meeting that constraint, giving the best precision available subject
to it.

## 8.4 ⭐ Calibrate on validation, evaluate on test

```
Stage 1  CALIBRATE on validation  →  pick thresholds
Stage 2  EVALUATE  on test        →  report numbers, thresholds frozen
```

You may look at validation results as often as you like. You look at test
results **once**, at the end.

If you sweep thresholds on test and report the best, your numbers are
optimistically biased — you fitted a parameter to your test data. It's a small
leak but a real one, and *"how did you pick your threshold?"* is often a probe
for exactly this.

`evaluate.py` enforces the ordering structurally, and logs it loudly:

```
==============================================================
STAGE 1 — calibrating thresholds on the VALIDATION split
==============================================================
...
==============================================================
STAGE 2 — evaluating on the held-out TEST split (thresholds frozen)
==============================================================
```

## 8.5 Pixel-level metrics: is it right for the right reason?

Image-level metrics tell you *whether* the model flagged the image. Pixel-level
metrics tell you whether it flagged **the right pixels**.

> A model can have near-perfect image-level AUROC while its heatmap points at
> the background — because the background happened to correlate with the label.

Three metrics, each answering a different question:

| Metric | Question | Notes |
|---|---|---|
| **Pixel AUROC** | Do defect pixels score higher than normal pixels? | The standard MVTec localisation metric |
| **Peak-hit rate** | Does the single hottest pixel land inside the true defect? | Blunt but very readable; maps to "does the heatmap point at it?" |
| **Mean IoU (top 1%)** | How much of the hottest region overlaps the real defect? | Punishes diffuse maps |

We report all three because IoU alone punishes a correct-but-diffuse map
harshly, while for an operator a correct-but-diffuse pointer is still useful.

Implementation note worth knowing: `localisation_iou` selects the **top k pixels
by score** rather than thresholding at a quantile *value*. Those sound
equivalent but aren't — `amap >= np.quantile(amap, 0.99)` selects *every* pixel
of a map that's 95% zeros, because that map's 99th percentile is itself zero.
Top-k always selects exactly k. (This was a real bug caught by a unit test.)

## 8.6 ⭐ Error analysis

The PRD calls this the most interview-impressive part of a portfolio project.
That's right, and here's why: **anyone can print an AUROC. Almost nobody looks
at their failures.**

`evaluate.py` automatically dumps the worst cases to `artifacts/<run>/failures/`:

```
fn_00_score8.21_crack.png       ← most confidently MISSED defects
fn_01_score9.04_crack.png
fp_00_score21.92_good.png       ← most confident FALSE ALARMS
fp_01_score18.45_good.png
```

We rank by *how badly* the model was wrong, not by how close. A borderline miss
tells you your threshold is tight; a **confident** miss tells you something is
structurally wrong.

### Your job: actually look at them

Open every one and write down what you see. Concretely, ask:

1. **Do the false negatives share a class?** If five of six are `crack`, you've
   found a class-specific weakness — and you can explain it (thin, low-contrast,
   reconstruction-based methods smooth it away).
2. **Do the false positives share a cause?** Unusual lighting? Part sitting at
   an odd angle? A legitimate but rare variation the "normal" model never saw?
3. **Is the heatmap on the defect even when the score was too low?** If yes,
   your *localisation* is fine and only your *threshold* or *scoring aggregation*
   is wrong. That's a very different fix.
4. **Is the model looking at the background?** Then your good number is luck.

### Write this paragraph

This is the deliverable. Something like:

> *"Six of my eight false negatives were the `crack` class. Cracks in my data
> are 1–2 pixels wide and low-contrast, and both my methods smooth spatially —
> the autoencoder via reconstruction, PaDiM via the Gaussian blur on the anomaly
> map. Reducing `smooth_sigma` from 4.0 to 2.0 recovered three of them but added
> eleven false positives, which at my 10:1 cost ratio was a net loss. The real
> fix is a higher-resolution embedding grid, which trades memory quadratically —
> I noted it as future work rather than pretending it was free."*

That paragraph demonstrates: you looked at your data, you formed a hypothesis,
you tested it, you quantified the trade-off, and you made a reasoned decision.
**That is what "senior" looks like in an intern.**

## 8.7 Run it

```bash
make evaluate
```

Produces:

```
artifacts/<run>/
├── thresholds.json       calibrated operating points
├── results.json          every metric, machine-readable
├── results.md            ← the table you paste into your README
├── predictions.csv       per-image scores for your own digging
├── plots/
│   ├── roc.png
│   ├── cost_curve.png            ← justifies your threshold
│   ├── test_anomaly_scores.png   ← the most informative single plot
│   ├── confusion_matrix.png
│   └── clf_history.png
└── failures/             ← the error-analysis panels
```

### The plot to look at first

`test_anomaly_scores.png` — overlapping histograms of good vs defect scores.

AUROC compresses the whole picture into one number. **This plot shows you
*why*** it is what it is: how much the two populations overlap, whether defects
form one cluster or several, and whether any threshold could separate them at
all.

If the histograms overlap heavily, no threshold will save you — you need a
better model, not better tuning. That's a diagnosis you can only make visually.

## 8.8 Your results table

Fill this in from **your** `results.md`. Do not use numbers from this guide.

```markdown
| Model | AUROC | AUPR | Recall | Precision | F1 | FN | FP |
|---|---|---|---|---|---|---|---|
| Classifier (supervised) |  |  |  |  |  |  |  |
| Autoencoder             |  |  |  |  |  |  |  |
| PaDiM                   |  |  |  |  |  |  |  |
| Fusion (OR)             | n/a | n/a |  |  |  |  |  |
```

Note `fusion` has **no AUROC** — it outputs a hard decision, not a score, so
AUROC is undefined. Reporting a made-up number there would be dishonest, and
`evaluate.py` marks it `nan` deliberately. If an interviewer notices that and
you can explain it, that's a win.

### Sanity expectations

- **PaDiM should beat the autoencoder** on image AUROC, usually clearly.
- **Fusion should have the highest recall** and lower precision than the best
  single model. That's the trade working as designed.
- **The classifier may beat PaDiM** on defect types it saw enough of, and lose
  badly on ones it didn't.
- If **PaDiM's AUROC is near 0.5**, something's broken — check for defects in
  `anomaly_fit`, and check you're using pretrained weights.

## 8.9 The comparison write-up

Beyond the table, write 3–4 sentences comparing the paths. Structure:

> *"PaDiM outperformed the autoencoder by X AUROC points, and the gap was
> concentrated in [class] — consistent with reconstruction methods struggling on
> low-contrast defects. The supervised classifier was competitive on [classes]
> where it had enough labelled examples, but [class] with only 10 training
> images was near chance. That's the argument for the dual-path design in one
> sentence: the supervised path is only as good as its label budget, and the
> unsupervised path doesn't have one."*

---

# Part 9 — The decision layer

## 9.1 Why it deserves its own module

Everything upstream produces **numbers**. A factory doesn't act on numbers, it
acts on **decisions**: does this part continue, or get pulled?

Putting that translation in one small, tested file — rather than scattering
`if score > 0.5` through the API — is the difference between a demo and a
system. It's also the piece interviewers probe hardest, because it's where ML
meets the business.

## 9.2 The four verdicts

```python
class Verdict(str, Enum):
    PASS             = "PASS"              # both signals quiet
    FAIL_CLASSIFIED  = "FAIL_CLASSIFIED"   # defect found AND named
    FAIL_ANOMALY     = "FAIL_ANOMALY"      # deviation found, type unknown
    REVIEW           = "REVIEW"            # inconclusive → human decides
```

`FAIL_ANOMALY` is the interesting one. It means: *something is wrong, and it
doesn't match any defect type we know.* Possibly a defect type nobody has
labelled yet.

> **That's exactly where an active-learning loop would hook in** — route these to
> a human, get a label, and you've discovered a new defect class. Say this if
> asked about future work; it shows you're thinking about the system's lifecycle,
> not just its accuracy.

## 9.3 Why the fusion rule is an OR

Covered in Part 2, but here's the code-level statement:

```python
elif primary == "fusion":
    flagged = clf_flag or ano_flag
```

The two models fail in largely **uncorrelated** ways, so OR catches strictly
more than either alone. The cost is precision.

**That trade is right in this domain and you should say why**: a missed defect
ships to a customer; a false alarm costs one operator 30 seconds. When costs are
asymmetric by ~10×, you buy recall with precision.

**And say when it would be wrong**: in a domain where a human reviews every
flag with fixed capacity — fraud review, content moderation — every false
positive consumes a slot a real case needed. There, an AND rule or a learned
combiner is better.

## 9.4 Two defensive design decisions

**1. Missing signals never silently pass.**

```python
if not reasons:
    return Decision(verdict=Verdict.REVIEW, ...)
```

If a model failed to load, we return REVIEW — not PASS. A model-loading failure
must not become a silent stream of PASS verdicts on a production line. **Failing
loudly beats passing everything.**

**2. Display scaling is separate from decision logic.**

`anomaly_normalised` divides the raw score by the 95th percentile of good
validation parts, so `1.0` means "as unusual as the most unusual normal part".
It's a **presentation convenience only** — it never affects the verdict.

Mixing display scaling into decision logic is a great way to ship a bug you
can't reproduce.

> ### 💡 Interview angle
> *"Is `anomaly_normalised` a probability?"* → *"No, and I made sure the API docs
> say so explicitly. It's a raw Mahalanobis distance divided by a reference
> percentile. If a stakeholder reads 0.8 as '80% chance of a defect' they'd be
> wrong, and that misunderstanding leads to bad decisions. If I needed real
> probabilities I'd calibrate — Platt scaling or isotonic regression on a
> held-out set — and I'd validate with a reliability diagram."*

## 9.5 Every decision explains itself

```python
reasons = [
    "classifier P(defect)=0.410 < threshold 0.717",
    "anomaly score=27.400 >= threshold 11.461",
    "deviation from normal with no matching known defect class",
]
```

This ships in the API response. When an operator asks *"why was this pulled?"*,
the answer is in the payload — not in a log file on a server they can't access.

---

# Part 10 — Serving with FastAPI

## 10.1 Why a notebook isn't enough

> A notebook proves you can build a model. An API proves you can build a
> **system**.

The gap between them is where most ML projects die in the real world, and
interviewers know it. Serving forces you to confront things notebooks let you
ignore: load time, memory, latency, error handling, what happens when the input
is garbage.

## 10.2 ⭐ The most important design decision: one inference path

Look at `src/visionqc/inference.py`. Both `evaluate.py` and `api/main.py` use
`InspectionEngine`. Neither has its own preprocessing or scoring code.

**Why this matters:** if evaluation and serving each implement their own
preprocessing, they *will* drift. Someone changes the resize interpolation in
one place, or forgets normalisation, and now your API returns different scores
than the ones you validated and published.

That class of bug is called **training/serving skew**. It's silent. It's one of
the most common ways real ML systems fail.

The fix is boring and effective: **exactly one code path turns an image into a
score.** If they ever disagree, it's a bug in one file, not a mystery.

> ### 💡 Interview angle
> *"What's training/serving skew and how did you prevent it?"* is an
> intermediate-level question that separates people who've deployed something
> from people who haven't. Your answer is concrete: *"I extracted a single
> InspectionEngine that owns preprocessing and scoring. Evaluation and the API
> both call it — neither has its own copy. The transforms are built once from
> the checkpoint's own recorded image_size, mean and std, so serving can't drift
> from what I trained with."*

## 10.3 Load models once, at startup

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    _engine = InspectionEngine(Path(RUN_DIR), get_device(DEVICE))
    yield
    _engine = None
```

Loading a checkpoint takes hundreds of milliseconds to seconds. Do it inside the
request handler and **every request pays that cost**, blowing your latency
budget immediately.

Note `lifespan`, **not** `@app.on_event("startup")` — the latter is deprecated
in current FastAPI/Starlette and will emit warnings.

## 10.4 Degrade, don't crash

If artifacts are missing we log loudly, report `status: "degraded"` on `/health`,
and return a clear **503** from `/inspect` — rather than crashing the process.

**Why:** a container that crash-loops tells an orchestrator nothing useful and
hides the real error behind a restart loop. A container that starts and honestly
reports *"I'm up but I have no model"* is debuggable in ten seconds.

That distinction is a genuine production instinct and worth explaining if asked.

## 10.5 Thread configuration — a non-obvious latency trap

```python
torch.set_num_threads(int(os.environ.get("VISIONQC_TORCH_THREADS", "1")))
```

Without this, PyTorch spawns threads equal to core count **per worker**. On a
small container the workers fight each other and **latency gets worse as you add
workers** — a genuinely confusing bug when you're trying to scale up.

## 10.6 Run it

```bash
make serve
# or: VISIONQC_RUN_DIR=artifacts/synthetic PYTHONPATH=src \
#     uvicorn visionqc.api.main:app --reload --port 8000
```

Open **http://localhost:8000/docs**.

FastAPI generates that page automatically from the Pydantic schemas. **That page
is your demo** — anyone can upload an image and see a response, with zero extra
work from you. It looks far more like a real product than a notebook cell.

### Test it

```bash
curl http://localhost:8000/health

curl -X POST "http://localhost:8000/inspect" \
     -F "file=@data/synthetic/test/scratch/000.png" | python -m json.tool
```

```json
{
  "decision": {
    "verdict": "FAIL_ANOMALY",
    "defect_probability": 0.41,
    "predicted_class": "good",
    "anomaly_score": 27.4,
    "anomaly_normalised": 2.2,
    "classifier_flag": false,
    "anomaly_flag": true,
    "reasons": [
      "classifier P(defect)=0.410 < threshold 0.717",
      "anomaly score=27.400 >= threshold 11.461",
      "deviation from normal with no matching known defect class"
    ]
  },
  "latency_ms": 118.3,
  "heatmap_png_base64": "iVBORw0KGgo..."
}
```

Decode the heatmap:

```bash
curl -s -X POST "http://localhost:8000/inspect" \
     -F "file=@data/synthetic/test/dent/000.png" \
  | python -c "import sys,json,base64; open('out.png','wb').write(base64.b64decode(json.load(sys.stdin)['heatmap_png_base64']))"
```

`out.png` is a three-panel image: **original | anomaly heatmap | Grad-CAM**.

## 10.7 Latency

The PRD budget is **< 2 seconds per image on CPU**. Measured:

```
Latency: mean 111.0 ms | p95 139.4 ms (n=25, cpu)
```

Comfortably inside. Two measurement details that matter:

1. **Warm up first.** The first forward pass pays lazy-init and cache costs. If
   you include it, your mean looks worse than reality.
2. **Time inference only, not the upload.** Otherwise you're measuring the
   client's network and can't compare runs.

> ### 💡 Interview angle
> Report **p95, not just mean**. *"Mean 111 ms, p95 139 ms"* signals you know
> tail latency is what users actually experience. Mean alone hides the case
> where 5% of requests take 3 seconds.

## 10.8 Docker

```bash
make docker        # docker build -t visionqc:latest .
make docker-run    # docker run --rm -p 8000:8000 visionqc:latest
```

Four decisions in the `Dockerfile` worth being able to defend:

**1. Multi-stage build.** Stage 1 installs dependencies; stage 2 copies only the
installed packages and source into a clean image. Build tools and pip caches
never reach the final layer.

**2. CPU-only PyTorch index.**
```dockerfile
--extra-index-url https://download.pytorch.org/whl/cpu
```
The default PyPI wheel bundles ~2.5 GB of CUDA libraries you can't use on a CPU
host. The CPU wheel is ~200 MB. **On a free tier with an image-size cap, this
one line is the difference between deploying and not deploying.**

Note `--extra-index-url`, not `--index-url` — torch comes from the CPU index,
everything else still resolves from PyPI.

**3. Non-root user.** If the container is compromised, the attacker lands as an
unprivileged account. Cheap, and the first thing a security review looks for.

**4. One worker.** Each worker loads its own full copy of the models into RAM.
On a 512 MB free tier, a second worker means an OOM kill, not throughput.

**5. HEALTHCHECK.** Lets an orchestrator distinguish "process is running" from
"app is actually serving". Without it, a hung worker looks healthy.

## 10.9 Deploying free (optional but high-impact)

A live URL in your README is disproportionately valuable — a recruiter can click
it in five seconds.

**Options:** Render, Fly.io, Railway, and Hugging Face Spaces all have free
tiers that accept a Dockerfile. Search for current limits; free tiers change
often.

**Practical constraints you'll hit:**

| Constraint | What to do |
|---|---|
| Image size caps | You already use the CPU wheel and `requirements-serve.txt` |
| 512 MB RAM | One worker; consider `embed_grid: 16` to shrink the PaDiM checkpoint |
| Cold starts | Self-contained checkpoint means no download at boot |
| Checkpoints too big for git | Use a GitHub Release asset, or mount a volume |

**If deployment fights you, don't sink a week into it.** A recorded demo GIF plus
`docker run` instructions is 90% of the value. Ship the rest first.

---

# Part 11 — Free compute and tooling

Everything in this project is free. Here's how to use it well.

## 11.1 Google Colab (free GPU)

Best for: training the classifier and autoencoder, which are the slow parts.

```python
# Cell 1 — check what GPU you got
!nvidia-smi

# Cell 2 — get your code
!git clone https://github.com/<you>/visionqc.git
%cd visionqc
!pip install -q -r requirements.txt

# Cell 3 — mount Drive so checkpoints survive a disconnect
from google.colab import drive
drive.mount('/content/drive')
!mkdir -p /content/drive/MyDrive/visionqc_artifacts
!ln -sfn /content/drive/MyDrive/visionqc_artifacts artifacts

# Cell 4 — run
!make data && make splits
!make padim
!make classifier
!make evaluate
```

**Colab survival rules:**

1. **Symlink `artifacts/` into Drive** (cell 3). Colab disconnects. Without this,
   a 40-minute training run vanishes and you start over.
2. **Runtime → Change runtime type → GPU.** Easy to forget, and everything
   silently runs on CPU.
3. **Don't leave idle tabs open.** Idle sessions burn your quota.
4. **Verify the GPU is actually used**: `torch.cuda.is_available()` should be
   `True`, and the training log should say `Device: cuda`.

## 11.2 Kaggle Notebooks

Roughly 30 GPU-hours/week and **datasets are already mounted** — the casting
dataset is right there with no download.

Better than Colab for: longer runs (up to ~9h), and anything using a Kaggle
dataset.
Worse for: the interface is clunkier; Drive integration is Colab-only.

## 11.3 Experiment tracking

Once you've run 20 experiments you will not remember which config produced which
number.

**Option A — what's already built in (zero setup).** Every run writes
`config_used.yaml`, `results.json`, and `*_history.json` to
`artifacts/<run_name>/`. Use a distinct `run_name` per experiment and you have a
complete, greppable record.

**Option B — Weights & Biases (free tier).** Nicer plots, shareable links.

```python
import wandb
wandb.init(project="visionqc", name=cfg.run_name, config=cfg.to_dict())
wandb.log({"epoch": epoch, "val_macro_recall": score})
```

Ten lines in `train_classifier.py`. A public W&B link in your README is a nice
touch.

**Don't over-invest here.** Tracking is a means, not the project. `run_name` +
`results.json` is genuinely enough for a portfolio piece.

## 11.4 Other free tools worth knowing

| Tool | Use |
|---|---|
| **GitHub Actions** | Run `pytest` on every push — a green badge in your README |
| **Netron** | Drag in a `.pt` file, get an interactive architecture diagram |
| **Excalidraw / draw.io** | Your architecture diagram (hand-drawn style reads well) |
| **ScreenToGif / Kap / peek** | Record the demo GIF |
| **Hugging Face Spaces** | Free hosting for a Gradio/Streamlit demo |

### Minimal CI (copy this into `.github/workflows/test.yml`)

```yaml
name: tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: "3.12"}
      - run: pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
      - run: pip install -r requirements.txt
      - run: pytest -q
```

A passing-tests badge on a portfolio repo is rare and cheap.

---

# Part 12 — Packaging it as a portfolio piece

You've built the thing. Now make it legible in thirty seconds, because that is
how long a recruiter spends.

## 12.1 The README is the product

Nobody clones your repo. They scroll your README. Structure it like this,
in this order:

```markdown
# VisionQC — Automated Visual Quality Inspection

[one-sentence description]
[DEMO GIF — the single highest-value element]

## The problem            ← 3 sentences, the constraint framing
## Architecture           ← the diagram
## Results                ← the table, with your real numbers
## What makes this different  ← the 5-row comparison table
## Error analysis         ← your paragraph
## Quickstart             ← copy-pasteable, actually works
## Design decisions       ← 5-6 bullets with WHY
## Limitations & future work
```

**Put the GIF above the fold.** Someone scrolling GitHub on a phone should see a
defect and a heatmap within one screen.

## 12.2 The demo GIF

```bash
PYTHONPATH=src python scripts/make_demo_assets.py --config configs/synthetic.yaml
```

Gives you `docs/demo_grid.png` and `docs/verdicts.png` — static, but reproducible
and always matching your current model.

For an actual GIF, record 10–15 seconds of:
1. The `/docs` page
2. Uploading a defective image
3. The JSON response appearing
4. The decoded heatmap

Tools: **Kap** (Mac), **ScreenToGif** (Windows), **peek** (Linux).

Keep it under 15 seconds and under ~5 MB or GitHub won't autoplay it smoothly.

## 12.3 The architecture diagram

Redraw the ASCII diagram from Part 0 in **Excalidraw** (free, hand-drawn style
that reads as thoughtful rather than corporate). Export PNG into `docs/`.

Label the two paths clearly and mark which needs labels and which doesn't —
that's the whole story of the project in one image.

## 12.4 The resume bullet

You need **one** bullet, ~2 lines, that survives a 6-second scan. Structure:
*what you built* → *the constraint* → *the number* → *the system*.

**Template:**

> **VisionQC — Visual Defect Detection** · *PyTorch, FastAPI, Docker*
> Built a dual-path industrial inspection system combining a supervised CNN
> classifier with a from-scratch PaDiM anomaly detector trained only on normal
> samples, achieving **[X] AUROC** and **[Y]% defect recall** on [dataset];
> deployed as a containerised REST API with Grad-CAM explanations at
> **[Z] ms** p95 CPU latency.

**Fill in X, Y, Z from your own `results.md`.** Do not invent them. The
follow-up question is always *"walk me through how you got that number"*, and a
fabricated figure ends the interview.

**Why this bullet works:**
- "dual-path" and "trained only on normal samples" signal the constraint framing
- "from-scratch PaDiM" signals depth, not library-calling
- Concrete numbers signal you measured
- "containerised REST API" signals you shipped
- Three technologies, not fifteen — reads as focused

## 12.5 The "walk me through this project" narrative

You'll be asked this in most interviews. **Prepare 90 seconds, not 5 minutes.**

Structure — **Problem → Constraint → Approach → Result → Reflection**:

> *"I built an automated visual quality inspection system for manufacturing
> defect detection.*
>
> *The thing that shaped the design was a constraint rather than a dataset: in
> real QC, defects are rare by definition — a healthy line barely produces any —
> so labelled defect examples are scarce, and new defect types appear that no
> classifier was trained on. A pure supervised approach fails there, because a
> classifier shown an unseen defect confidently calls it 'good'.*
>
> *So I built two complementary paths. A ResNet18 transfer-learning classifier
> for defect types where labels exist, and PaDiM — which I implemented from
> scratch — for the case where they don't. PaDiM models the distribution of
> normal image patches per spatial position and flags anything at high
> Mahalanobis distance, so it only ever needs normal data.*
>
> *I evaluated both against MVTec AD, the standard benchmark, on one shared
> held-out test set so the comparison was valid. PaDiM reached [X] image-level
> AUROC and [Y] pixel-level. I chose my operating threshold by minimising a cost
> function where a missed defect costs 10× a false alarm, rather than defaulting
> to 0.5 — and I calibrated it on validation, never on test.*
>
> *Then I deployed it as a Dockerised FastAPI service returning a verdict plus a
> heatmap, at [Z] ms p95 on CPU.*
>
> *The thing I found most interesting was the error analysis — [your actual
> finding]."*

**Practise this out loud. Time it.** Reading it silently is not preparation.

Then have depth ready for whichever thread they pull. `INTERVIEW_PREP.md` has
40+ follow-ups with answers.

## 12.6 Pre-flight checklist

Before you put the link on your resume:

**Code**
- [ ] `git clone` → `make setup` → `make all` works on a fresh machine
- [ ] `pytest -q` passes
- [ ] No hard-coded absolute paths
- [ ] No datasets or `.pt` files committed (check `.gitignore`)
- [ ] Every module has a docstring explaining *why*

**Docs**
- [ ] README has demo GIF above the fold
- [ ] Results table has **your** real numbers
- [ ] Architecture diagram present
- [ ] Error analysis paragraph written
- [ ] Limitations section is honest

**Demo**
- [ ] `/docs` page works
- [ ] `docker build` and `docker run` both succeed
- [ ] Live URL works (if deployed)

**You**
- [ ] 90-second narrative rehearsed out loud
- [ ] You can explain Mahalanobis distance on a whiteboard
- [ ] You can explain your threshold choice
- [ ] You can name three limitations without being prompted
- [ ] Resume bullet has real numbers

## 12.7 Honest limitations to state (do not hide these)

Volunteering limitations reads as confidence. Hiding them and getting caught
reads as much worse than the limitation itself.

1. **Synthetic data is easier than real data.** Report MVTec numbers as your
   headline; use synthetic to demonstrate the pipeline.
2. **One dataset category is not generalisation.** If you only ran `bottle`, say
   "one category" rather than implying you covered MVTec.
3. **PaDiM assumes aligned parts.** If parts arrive at arbitrary rotation, the
   per-position Gaussians degrade. PatchCore handles that better.
4. **Grad-CAM is coarse** (8×8 upsampled). Fine for regions, not for pixels.
5. **No drift monitoring.** A real deployment needs to detect when "normal"
   itself shifts — new lighting, new supplier, seasonal material variation.
6. **The 10:1 cost ratio is a placeholder**, not a measured business figure.
7. **Small test set.** With ~78 test images, confidence intervals on AUROC are
   wide. Bootstrap them if you want to be rigorous.

> ### 💡 Interview angle
> *"What would you do differently with more time?"* — have three specific
> answers ready, not "more data":
> 1. *"PatchCore instead of PaDiM — drops the alignment assumption and usually
>    scores higher on the benchmark."*
> 2. *"An active-learning loop: route FAIL_ANOMALY cases to human labelling, and
>    you discover new defect classes rather than accumulating unknowns."*
> 3. *"Bootstrap confidence intervals on the metrics. With 78 test images my
>    AUROC has real uncertainty and I'm currently reporting it as a point
>    estimate."*

---

# Part 13 — Troubleshooting

## Installation

**`OSError: libcudart.so.X: cannot open shared object file`**
You installed the CUDA build on a CPU machine.
```bash
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**`ModuleNotFoundError: No module named 'visionqc'`**
The package isn't on the path. Either `export PYTHONPATH=src` or
`pip install -e .`. The Makefile handles this for you.

**Download of pretrained weights fails / 403**
Corporate network or firewall. Try a different network, or set
`pretrained: false` **only to check the code runs** — never for real results.

## Data

**`FileNotFoundError: Dataset root not found`**
Run `make data` first, or check `data.root` in your config.

**`No 'good' folder under test/`**
Your dataset isn't in canonical layout. Use `scripts/prepare_dataset.py`.

**`AssertionError: LEAK: N images in both 'test' and ...`**
Working as intended — it caught a bug. Usually you have duplicate files, or
`train/good` and `test/good` contain the same images.

**`AssertionError: N defective images in anomaly_fit`**
Something defective is in `train/`. In canonical layout, `train/` is
normal-only.

## Training

**Loss becomes `nan`**
Learning rate too high. Divide `lr_head` by 10. If it persists, check for
corrupt images (all-black or all-white files).

**Val accuracy stuck at exactly the majority-class rate**
The model is predicting one class for everything. Set
`use_class_weights: true`, and confirm the printed weights are non-uniform.

**Val loss rises while train loss falls**
Overfitting. Fewer epochs, more augmentation, higher `dropout`, or increase
`freeze_epochs`.

**"Frozen" backbone still seems to change**
The BatchNorm bug from §4.3. Confirm `freeze_backbone()` calls `self.net.eval()`.

**Training is unbearably slow**
You're on CPU. Use Colab (Part 11), or drop `image_size` to 128 and use
`configs/fast.yaml` to verify correctness first.

## PaDiM

**`Covariance is not positive definite`**
Too few fit images relative to dimension. Increase `padim.reg` to `0.05`, reduce
`n_components`, or add more normal images.

**Out of memory during fit**
Memory scales as `embed_grid² × n_components²`. Drop `embed_grid` 32→16 (4×
less) or `n_components` 100→50 (4× less).

**AUROC near 0.5**
Three usual causes, check in this order:
1. `pretrained: false` — random features carry much less signal
2. Defects leaked into `anomaly_fit`
3. Augmentation applied during fit (must be `train=False` — see Part 3.5)

**Anomaly map is uniform**
`smooth_sigma` far too large. Try 2.0.

## Grad-CAM

**Heatmap is all zeros**
You called it inside `torch.no_grad()`. Grad-CAM needs a backward pass — see
§5.3.

**`RuntimeError: Use GradCAM inside a with block`**
Working as intended. Use the context manager so hooks get cleaned up.

**Heatmap is uniform / featureless**
You hooked the wrong layer — probably after global average pooling, which
destroys spatial information. Use `model.target_layer()`.

## Evaluation

**`Threshold selection needs both classes in the validation split`**
Your `sup_val` has only good images. Increase `val_frac`, or reduce
`holdout_frac` so more defects remain for validation.

**AUROC is `nan`**
Only one class present in that slice. Expected for some per-class breakdowns;
`safe_auroc` returns `nan` rather than crashing.

**Pixel metrics say `n/a`**
Your dataset has no ground-truth masks. Expected for the casting dataset —
state it in your results rather than hiding it.

## API

**503 from `/inspect`**
Models didn't load. Check `/health` for what's missing and check
`VISIONQC_RUN_DIR`.

**Checkpoints disagree on image_size**
You trained models with different configs into the same run directory. Retrain
from one config, or use separate `run_name`s.

**Latency much worse than expected**
Set `VISIONQC_TORCH_THREADS=1` (§10.5), and confirm you're warming up before
timing.

**Docker image is enormous**
You're not using the CPU torch index. Check the `--extra-index-url` line in the
Dockerfile.

---

# Glossary

Every technical term used in this guide, in plain English.

**Anomaly detection** — Finding things that deviate from normal, without having
been shown examples of what's wrong.

**AUPR (Average Precision)** — Area under the precision-recall curve. Unlike
AUROC, it doesn't flatter models under heavy class imbalance.

**AUROC** — Area under the ROC curve. The probability a random positive scores
higher than a random negative. 0.5 = random, 1.0 = perfect.

**Augmentation** — Randomly perturbing training images (rotate, flip, adjust
brightness) so the model can't memorise them.

**Autoencoder** — A network that compresses input to a small code and rebuilds
it. Trained on normal data only, it reconstructs normal well and anomalies
badly.

**Backbone** — The feature-extracting body of a network, minus its final
classification layer.

**Backpropagation** — The algorithm computing how much each weight contributed
to the error, so the optimiser knows which direction to nudge them.

**BatchNorm** — A layer normalising activations using batch statistics. It keeps
running mean/variance **buffers** that update in training mode even when weights
are frozen — the source of a classic freezing bug.

**Bottleneck** — The narrowest layer of an autoencoder. If too wide, the network
learns the identity function and detects nothing.

**Buffer (PyTorch)** — Persistent state that isn't a trainable parameter (e.g.
BatchNorm running stats, PaDiM's means). Saved with the model; not updated by
gradients.

**Calibration** — Choosing an operating threshold, on validation data.

**Checkpoint** — A saved file of model weights and metadata.

**Cholesky decomposition** — Factoring a positive-definite matrix as `L Lᵀ`.
Lets you solve linear systems without computing an inverse — faster and more
numerically stable.

**Class weights** — Per-class multipliers in the loss that make rare-class
mistakes cost more.

**Confusion matrix** — Table of true vs predicted classes. Shows exactly which
classes get confused with which.

**Convolution** — Sliding a small learned filter across an image to produce a
feature map.

**Covariance matrix** — Describes how each pair of dimensions varies together.
Central to Mahalanobis distance.

**Discriminative learning rates** — Different learning rates for different parts
of a network; small for pretrained layers, large for new ones.

**Early stopping** — Halting training when validation performance stops
improving, to avoid overfitting.

**Epoch** — One full pass over the training data.

**F1 score** — Harmonic mean of precision and recall.

**False negative (FN)** — A defect the model called good. **In QC this is the
expensive one.**

**False positive (FP)** — A good part the model flagged.

**Feature map** — The output grid of a convolutional layer; one per filter.

**Fine-tuning** — Continuing to train a pretrained model on your own data.

**Freezing** — Setting `requires_grad = False` so weights don't update.

**Fusion** — Combining multiple model outputs into one decision.

**Grad-CAM** — Gradient-weighted Class Activation Mapping. A heatmap showing
which image regions drove a classifier's prediction.

**Ground truth** — The correct answer. For localisation, a pixel mask marking
the real defect.

**Hook (PyTorch)** — A function that fires when a module runs forward, or when a
tensor's gradient is computed. How Grad-CAM captures activations and gradients.

**ImageNet** — A 1.2M-image, 1000-class dataset. Pretraining on it gives
transferable low-level features.

**IoU (Intersection over Union)** — Overlap between predicted and true regions,
divided by their union.

**Label smoothing** — Softening target labels (e.g. 1.0 → 0.95) to stop the
model becoming overconfident and to keep probabilities better calibrated.

**Latent space / latent code** — The compressed representation inside an
autoencoder's bottleneck.

**Leakage** — Test information contaminating training. Makes results look better
than reality. **The most common serious flaw in portfolio projects.**

**Logits** — Raw pre-softmax model outputs. Unbounded; not probabilities.

**Loss function** — Measures how wrong a prediction is. Training minimises it.

**Macro average** — Averaging a metric over classes with equal weight, so rare
classes count as much as common ones.

**Mahalanobis distance** — Distance that accounts for how dimensions vary and
co-vary: `√((x−μ)ᵀΣ⁻¹(x−μ))`. Asks "how many standard deviations away?" rather
than "how far in raw units?"

**MVTec AD** — The standard academic/industrial benchmark for visual anomaly
detection. 15 categories, pixel-level masks.

**Normalisation (image)** — Subtracting a mean and dividing by a standard
deviation per channel, so inputs match what the pretrained model expects.

**Optimiser** — The algorithm updating weights from gradients (SGD, Adam,
AdamW).

**Overfitting** — Memorising training data instead of learning generalisable
patterns. Train loss falls, validation loss rises.

**PaDiM** — Patch Distribution Modeling. Fits a Gaussian per spatial position
over pretrained CNN features; scores by Mahalanobis distance.

**PatchCore** — PaDiM's successor. Uses a nearest-neighbour memory bank instead
of per-position Gaussians; handles misalignment better.

**Pixel AUROC** — AUROC computed over individual pixels. Measures localisation,
not just detection.

**Precision** — Of everything flagged, what fraction was really defective.

**Pretrained weights** — Weights learned on a large dataset, reused as a
starting point.

**Recall (sensitivity)** — Of all real defects, what fraction were caught.

**Regularisation (ridge)** — Adding a small value to a matrix's diagonal to make
it invertible and stable.

**ReLU** — Activation that zeroes negatives, keeps positives.

**ResNet** — A CNN architecture using skip connections, which let gradients flow
through very deep networks.

**ROC curve** — True-positive rate against false-positive rate across all
thresholds.

**Seed** — A number initialising random generators so runs are reproducible.

**Softmax** — Converts logits into probabilities that sum to 1.

**Stratified split** — Splitting so each class's proportion is preserved in
every split.

**Threshold** — The score cut-off above which you flag. **A business decision,
not a default.**

**Training/serving skew** — When training and production preprocess data
differently. Silent and destructive; prevented by sharing one code path.

**Transfer learning** — Reusing a model pretrained on one task as a starting
point for another.

**True negative (TN)** — A good part correctly passed.

**True positive (TP)** — A defect correctly caught.

**Validation set** — Data used to tune choices (epochs, thresholds) during
development. **Never the test set.**

---

# What to do next

1. **Run `make all`** and get your own numbers into `results.md`.
2. **Do the error analysis** in Part 8.6 — actually open the failure panels and
   write the paragraph. This is the highest-value hour in the whole project.
3. **Swap in MVTec AD** and re-run. Those are your headline numbers.
4. **Read `INTERVIEW_PREP.md`** and write your own answers. Not mine — yours.
5. **Record the demo**, write the README, ship it.

You now have something that very few internship applicants have: a project built
around a real constraint, evaluated honestly, deployed as a system, and
explainable end to end.

Good luck.
