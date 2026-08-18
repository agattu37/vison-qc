# Interview Preparation — VisionQC

**Read this only after you have finished Part 8 of `GUIDE.md`.** The answers
will not stick until you have done the work they describe.

---

## How to use this file

These are not scripts to memorise. Memorised answers sound memorised, and
interviewers can tell instantly.

Use them like this:

1. Read the question. **Close the file.** Answer out loud, from memory.
2. Reopen and compare. Note what you missed — usually the *why*, not the *what*.
3. Rewrite the answer **in your own words**, in your own file.
4. Repeat a few days later.

The goal is that you understand the reasoning well enough to reconstruct the
answer under pressure, including for questions not on this list.

**Three habits that matter more than any individual answer:**

- **Say the trade-off, not just the choice.** "I used X" is weak. "I used X
  because Y, and gave up Z" is strong.
- **Volunteer limitations before you're asked.** It reads as confidence.
  Interviewers are trained to probe for weaknesses; getting there first changes
  the dynamic entirely.
- **When you don't know, say so, then reason.** "I don't know, but here's how
  I'd find out" scores far better than a confident guess. Confident wrong
  answers are the fastest way to fail an interview.

---

## Section 1 — The opening question

### Q1. "Walk me through this project."

**You have 90 seconds. Not five minutes.** Structure: Problem → Constraint →
Approach → Result → Reflection.

> *"I built an automated visual quality inspection system for manufacturing
> defect detection.*
>
> *What shaped the design was a constraint rather than a dataset: in real QC,
> defects are rare by definition — a healthy line barely produces any — so
> labelled defect examples are scarce, and new defect types appear that no
> classifier was ever trained on. A pure supervised approach breaks there,
> because a classifier shown an unseen defect confidently calls it 'good'.*
>
> *So I built two complementary paths. A ResNet18 transfer-learning classifier
> for defect types where labels exist, and PaDiM — which I implemented from
> scratch — for when they don't. PaDiM models the distribution of normal image
> patches per spatial position and flags anything at high Mahalanobis distance,
> so it only ever needs normal data.*
>
> *I evaluated both on MVTec AD, the standard benchmark, using one shared
> held-out test set so the comparison was valid. PaDiM reached [X] image-level
> AUROC and [Y] pixel-level. I chose the operating threshold by minimising a
> cost function where a missed defect costs 10× a false alarm, rather than
> defaulting to 0.5 — and I calibrated on validation, never on test.*
>
> *Then I deployed it as a Dockerised FastAPI service returning a verdict plus a
> heatmap, at [Z] ms p95 on CPU.*
>
> *The most interesting part was the error analysis — [your actual finding]."*

**Then stop talking.** Let them pull a thread. Rambling past 90 seconds is the
most common self-inflicted wound in project interviews.

### Q2. "Why this project?"

> *"I wanted a project where the interesting decisions weren't about model
> architecture. Visual QC has a structural constraint — defects are rare, so
> labels are scarce and new defect types appear — that forces you to think about
> the problem rather than just fit a model. And it's a domain where the cost of
> different errors is genuinely asymmetric, which makes threshold selection a
> real decision instead of a default."*

### Q3. "What was the hardest part?"

Do **not** say "getting the data" or "debugging". Pick something technical and
specific. Examples that work:

> *"Designing the split. My two models had incompatible training requirements —
> one needs normal-only data, the other needs labelled defects — but I wanted
> one shared test set so the comparison was valid. Getting that protocol right,
> and adding an assertion that runs on every build to verify no leakage, took
> longer than training either model."*

Or:

> *"The covariance estimation in PaDiM. With 220 images and 100 dimensions per
> patch, the sample covariance is near-singular and the inverse explodes. I had
> to add ridge regularisation scaled to the average variance, and switch from an
> explicit inverse to a Cholesky solve for stability. That's the difference
> between the model working and producing `inf`."*

---

## Section 2 — Problem framing

### Q4. "Why not just train a classifier?"

> *"Because a classifier can only recognise defect types it was trained on, and
> its failure mode on unseen types is dangerous: it doesn't say 'I'm unsure', it
> confidently assigns them to the nearest class it knows — which is usually
> 'good'. On a production line that means a new defect type ships silently, and
> you find out from a customer complaint. Since new defect types genuinely
> appear — new supplier, worn tooling, changed process — I needed something that
> flags deviation rather than recognising categories."*

### Q5. "Why unsupervised rather than just labelling more data?"

> *"Two reasons. First, economics: labelling defects requires a domain expert,
> and you can't manufacture defects on demand to build a dataset — waiting for
> them is the whole problem. Second, and more fundamental: even with unlimited
> labelling budget, you can only label defect types that have already occurred.
> The unsupervised path covers the ones that haven't."*

### Q6. "What does 'anomaly detection' actually mean here?"

> *"Learning what normal looks like and flagging deviation, rather than learning
> what defects look like. Concretely with PaDiM: I take a frozen pretrained
> ResNet, extract patch features from 220 normal images, and fit a Gaussian per
> spatial position. At test time I measure Mahalanobis distance from each patch
> to its own position's Gaussian. High distance means 'this patch doesn't look
> like this part of a normal part'."*

### Q7. "How would you know if this is actually working in production?"

Strong question, and most candidates flounder. Have three layers:

> *"Three things. First, a sampling audit — route a random fraction of PASS
> verdicts to human inspection so I can estimate the false-negative rate I can't
> otherwise see. Second, distribution monitoring: track the anomaly score
> distribution on good parts over time. If it drifts, 'normal' has changed —
> new lighting, new material batch — and I need to re-fit. Third, track the
> FAIL_ANOMALY rate specifically: a spike means either a real process problem or
> that my normal model has gone stale, and either is worth a page."*

---

## Section 3 — Data and splits

### Q8. "How did you split your data?" ⭐ *Asked in almost every ML interview*

This is a leakage probe. Full answer:

> *"I had a constraint most projects don't. My unsupervised model can only train
> on normal data; my supervised model needs labelled defects. But I wanted one
> comparable test set, otherwise any difference in their numbers might just be a
> difference in test difficulty.*
>
> *So: one test set that neither model ever sees. I froze a stratified 50% of
> the labelled pool as holdout. The anomaly model fits on the normal-only
> training folder. The remainder of the labelled data splits into supervised
> train and validation.*
>
> *I stratified per class rather than globally because some defect types only
> had 25 images — a global random split could leave a class with zero test
> samples, and then recall for that class is undefined. The manifest is written
> to disk with a fixed seed, and there's an assertion that runs on every build
> checking test is disjoint from every training split."*

### Q9. "What is data leakage and how did you prevent it?"

> *"Leakage is when information from your test set influences training, making
> results look better than reality. I guarded three forms.*
>
> *Direct overlap — an assertion on every split build that test paths appear in
> no training split.*
>
> *Threshold leakage — I pick my operating threshold on validation, never on
> test. Sweeping thresholds on test and reporting the best means you've fitted a
> parameter to your test data. Subtle, but real, and very common.*
>
> *Contamination of the normal set — a separate assertion that `anomaly_fit`
> contains zero defects. A defect in there wouldn't crash anything; it would
> quietly widen the learned notion of 'normal' to include defects, and detection
> would silently fail."*

### Q10. "How did you handle class imbalance?"

> *"Three ways, at different levels.*
>
> *In the loss: inverse-frequency class weights, normalised to mean 1 so the
> overall loss magnitude stays comparable and I don't have to retune the
> learning rate. With 200 good and 10 per defect type, one defect mistake costs
> about 20× a good-image mistake.*
>
> *In metric choice: I select on macro recall, which averages over classes with
> equal weight, so completely failing a rare class is heavily penalised.
> Accuracy would let me ignore it.*
>
> *Architecturally: the unsupervised path is itself a response to imbalance
> taken to its logical extreme. When the minority class is so rare that
> supervision fails, you stop trying to model it and model the majority
> instead."*

### Q11. "Why those augmentations specifically?" ⭐

> *"I reasoned from the physical setup. Rotation and flips because a part on a
> conveyor has no canonical orientation. Brightness and contrast jitter because
> factory lighting drifts and lamps age. Small translation and scale because
> part placement isn't pixel-perfect.*
>
> *More interesting is what I ruled out. No hue jitter — it would recolour a
> rust stain to look like clean metal, destroying the signal. And no random
> erasing or cutout, which is a standard recipe item, because it punches
> synthetic holes that look visually identical to my contamination defect class.
> I'd literally be teaching the model that defects are normal."*

### Q12. "Do you apply augmentation to both models?"

Excellent detail to have ready:

> *"No, and the difference matters. The classifier trains with rotation. PaDiM
> fits without it.*
>
> *PaDiM models a separate distribution per patch position. If I rotate, the
> patch at grid cell (3,7) comes from the rim in one image and the centre in
> another — the per-position Gaussians blur into one meaningless global
> distribution, and I've thrown away exactly the spatial specificity that makes
> PaDiM work.*
>
> *The same augmentation that helps one model actively destroys the other. The
> right augmentation depends on what the model assumes."*

### Q13. "Why synthetic data?"

> *"To decouple pipeline development from data access. MVTec needs a licence
> form and a 5 GB download; Kaggle needs an API token. I wrote a generator
> producing the exact MVTec folder layout so I could build and test the whole
> pipeline on day one, then swap in real data by changing one path.*
>
> *It's the same pattern as fixtures in software testing. I'm explicit in my
> README that synthetic defects are easier than real ones, so those numbers are
> optimistic — I report MVTec as my actual result."*

---

## Section 4 — The supervised model

### Q14. "Why ResNet18 and not something bigger?"

> *"Three reasons. About 11M parameters trains in minutes on a free-tier GPU, so
> I could run many experiments rather than a few. Its residual stages give clean
> multi-scale feature maps that the PaDiM path reuses directly — one backbone,
> two models. And it runs a 256px image on CPU in about 110ms, inside my 2-second
> latency budget, which a ViT or EfficientNet-B4 would not on free hosting.*
>
> *With more labelled data I'd try a larger backbone. At 40 labelled defects,
> capacity isn't my bottleneck — data is."*

### Q15. "Explain transfer learning."

> *"A network pretrained on ImageNet has already learned the expensive-to-learn
> low-level features: edges, corners, textures, gradients. A scratch on metal is
> an edge; a dent is a shading gradient. ImageNet contains no machined parts, but
> low-level visual structure is universal, so those features transfer almost
> perfectly. I keep them and only teach the network what to do with them."*

### Q16. "Walk me through your training procedure." ⭐

> *"Two phases. First, backbone frozen, head only — about 2,500 trainable
> parameters. Then unfreeze and fine-tune everything, with a 10× smaller
> learning rate on the backbone than the head.*
>
> *Phase 1 exists because on epoch 1 the head is randomly initialised and
> produces large meaningless gradients. If the backbone were trainable then,
> those gradients would damage the pretrained weights I got for free.*
>
> *The discriminative learning rates matter too — the backbone already knows
> things and needs gentle nudges; the head knows nothing and needs real
> learning. One global rate can't do both.*
>
> *I select the checkpoint on validation macro recall, use cosine annealing, and
> clip gradients at norm 5 because with a weighted loss and tiny defect classes
> one badly-scaled batch can undo several epochs."*

### Q17. "What's a subtle bug you hit?" ⭐

> *"BatchNorm during freezing. Setting `requires_grad = False` stops the weights
> updating, but BatchNorm keeps updating its running mean and variance during
> any forward pass in training mode — those are buffers, not parameters. So my
> 'frozen' backbone kept shifting its outputs under the head and training was
> mysteriously unstable.*
>
> *The fix is calling `.eval()` on the backbone as well. One line. I wrote a
> test for it, because it's the kind of thing that fails silently."*

### Q18. "Why select on macro recall instead of validation loss?"

> *"Because loss and my objective aren't the same thing. Loss is a proxy that
> can improve while the metric I care about gets worse — it rewards confidence
> on easy examples. Accuracy is dominated by the 'good' class, so a model that
> never predicts 'crack' still scores well. Macro recall weights classes
> equally, so failing an entire defect class is heavily penalised. That's the
> failure I must not ship."*

### Q19. "What's label smoothing and why use it?"

> *"It softens targets from 1.0 to about 0.95. With only 40 labelled training
> images, the model can drive its logits to infinity and be perfectly confident
> on the training set — a form of overfitting that also destroys the calibration
> of the probabilities I later threshold. Since my whole decision layer depends
> on thresholding a probability, calibration matters."*

---

## Section 5 — Anomaly detection

### Q20. "Explain PaDiM." ⭐ *Be able to whiteboard this*

> *"Four steps.*
>
> *One: push a normal image through a frozen pretrained ResNet, take feature
> maps from three depths, resize to a common grid, concatenate channel-wise. Now
> each of the 1024 grid positions has a 448-dimensional vector describing that
> patch, combining fine texture from early layers with structure from later
> ones.*
>
> *Two: do that for all 220 normal images. Each position now has a cloud of 220
> vectors — 'here's what patch (7,12) normally looks like'.*
>
> *Three: summarise each cloud with a Gaussian — a mean and a covariance. Per
> position, which is the key idea: the corner of a part is expected to look
> different from its centre, and modelling that explicitly is why PaDiM beats
> methods that use one global distribution.*
>
> *Four: at test time, Mahalanobis distance from each patch to its own
> position's Gaussian. Far equals anomalous. Upsample that grid to image size
> and you have the heatmap; take the max for the image score.*
>
> *There's no gradient descent anywhere — the backbone is frozen and fitting is
> one pass to accumulate statistics."*

### Q21. "Why Mahalanobis and not Euclidean distance?" ⭐

> *"Euclidean treats every feature dimension as equally important and
> independent, and they're not. Some CNN feature channels vary wildly across
> perfectly normal parts — lighting, texture phase — and others barely vary at
> all. A 2-unit deviation in a stable channel is alarming; the same deviation in
> a noisy channel is nothing.*
>
> *Mahalanobis divides by the observed spread per direction: square root of
> (x−μ)ᵀ Σ⁻¹ (x−μ). Euclidean asks 'how far in raw units?'; Mahalanobis asks
> 'how many standard deviations, accounting for how the dimensions co-vary?'*
>
> *Picture normal data as a long diagonal ellipse in 2D. Two points the same
> Euclidean distance out — one along the ellipse's axis, one perpendicular to it
> — are equally far by Euclidean, but the perpendicular one is genuinely
> anomalous and the other is normal variation. Mahalanobis gets that right."*

### Q22. "Why random dimension selection, and why not PCA?" ⭐

> *"448 channels means a 448×448 covariance per position across 1024 positions —
> about 840 MB and slow to invert. The paper shows keeping 100 random dimensions
> matches both PCA selection and using all 448, at roughly 20× less cost.*
>
> *The reason random beats PCA is subtle and I think it's the interesting part.
> PCA keeps directions of greatest variance in normal data. But those aren't
> necessarily where anomalies show up. A dimension that's essentially constant
> across normal parts is low-variance, so PCA discards it — yet that's exactly
> the dimension where a defect would stand out most."*

### Q23. "Why max instead of mean for the image score?"

> *"A defect is small — often under 1% of the pixels. Its contribution to a mean
> is diluted to nothing by the 99% of normal pixels, and image-level AUROC
> collapses. Max asks the right question: is there anywhere that looks wrong?"*

### Q24. "Why implement PaDiM yourself instead of using anomalib?" ⭐

> *"In a job I'd use anomalib — it's Intel-maintained, implements PaDiM,
> PatchCore and others, and has proper benchmarking. For a portfolio project it
> would have been the wrong choice, because the point was to understand it. It's
> about 120 lines, and writing it means I can draw Mahalanobis distance on a
> whiteboard rather than say 'I called a function'.*
>
> *I mention anomalib in my docs as the production alternative, because knowing
> when not to hand-roll is also a skill."*

### Q25. "How is PatchCore different?" ⭐

> *"PaDiM fits one Gaussian per spatial position and scores by Mahalanobis
> distance. PatchCore keeps a memory bank of patch features from all positions,
> coreset-subsampled, and scores by nearest-neighbour distance.*
>
> *The trade: PaDiM's per-position model is more precise when parts are
> consistently aligned, but degrades when they aren't, because the positions
> stop corresponding. PatchCore drops that positional assumption entirely, so it
> handles misalignment much better, and it makes no parametric assumption about
> the feature distribution. It generally scores higher on MVTec.*
>
> *My parts are conveyor-aligned, so PaDiM's assumption holds — but that's the
> reason, and if alignment weren't guaranteed I'd have gone with PatchCore."*

### Q26. "Why does the autoencoder underperform?" ⭐

> *"Two known weaknesses. First, autoencoders generalise too well — a
> sufficiently smooth or low-contrast defect gets reconstructed anyway. My crack
> class is exactly that: thin and low-contrast. Second, per-pixel L2 error is
> dominated by high-frequency texture, so a perfectly normal but slightly
> misaligned edge produces more error than a genuine subtle defect.*
>
> *PaDiM sidesteps both by comparing pretrained features rather than raw pixels
> — features are already invariant to the texture noise that swamps pixel
> error."*

### Q27. "Then why build the autoencoder at all?"

> *"A baseline you can beat is how you demonstrate the upgrade was worth it. If
> I'd only built PaDiM I'd have a number with nothing to compare it to. And the
> comparison told me something specific — the gap was widest on my low-contrast
> class, which matches the known theoretical weakness. That's a finding, not
> just a benchmark."*

### Q28. "How did you size the autoencoder bottleneck?"

> *"196,608 input values down to 16,384 in the latent — about 12× compression.
> The bottleneck is the whole game: if the code is large enough the network
> learns the identity function, reconstructs defects perfectly, and detects
> nothing. That's the number one reason beginner autoencoders fail at this.*
>
> *A counter-intuitive consequence: lower reconstruction loss isn't
> automatically better. Train long enough with enough capacity and AUROC falls
> while loss keeps dropping. I log both so the effect is visible."*

### Q29. "Why does the covariance need regularisation?"

> *"With 220 images and 100 dimensions per patch, the sample covariance is
> near-singular — I don't have enough samples relative to the dimension to
> estimate it well. Inverting it amplifies noise enormously and you get infinite
> distances.*
>
> *I add a ridge term to the diagonal, scaled by the average variance so the
> regularisation strength means the same thing regardless of feature scale. And
> I use a Cholesky factorisation with a triangular solve instead of an explicit
> inverse — solving L y = δ gives the same squared Mahalanobis distance without
> ever forming the inverse, which is faster and more numerically stable."*

---

## Section 6 — Evaluation

### Q30. "Why not report accuracy?" ⭐

> *"Because on a line running at 2% defect rate, a model that stamps PASS on
> everything is 98% accurate and catches zero defects. Accuracy rewards it. I
> lead with recall, because that's what a plant manager cares about — a missed
> defect reaches a customer — and precision, because low precision means
> operators waste time on re-inspections and within a week they start ignoring
> the system. A model nobody trusts has zero value regardless of its recall."*

### Q31. "Why both AUROC and AUPR?"

> *"AUROC's false-positive rate has the negative count in the denominator, so
> under heavy imbalance a large absolute number of false positives is still a
> small rate and AUROC looks flattering. AUPR doesn't have that blind spot. On a
> real line at 2% defect rate, a thousand false alarms a day is operationally
> unacceptable but barely moves AUROC. Reporting both keeps me honest."*

### Q32. "How did you choose your threshold?" ⭐ *The highest-value answer here*

> *"0.5 has no meaning in this domain — it's a convention from balanced binary
> classification. I made it a business decision.*
>
> *I assigned a relative cost of 10 to a missed defect versus 1 to a false alarm,
> because an escaped defect reaches a customer and can trigger a warranty claim
> while a false alarm costs an operator about 30 seconds at a re-inspection
> station. Then I swept thresholds on validation and picked the point minimising
> 10×FN + 1×FP. I have the cost curve plotted, which is my answer to 'why that
> number?'.*
>
> *Critically, I calibrated on validation, never on test. And the ratio is a
> config value, not hard-coded — if the plant told me it was really 50:1, I'd
> change one line and re-derive."*

**Expected follow-up: "Where did the 10 come from?"** Be honest:

> *"I chose it as a defensible placeholder. In a real deployment I'd get it from
> the business — warranty cost per escaped defect versus operator time per
> re-inspection. The point is the framework, and that the number is an explicit,
> changeable input rather than buried in code."*

### Q33. "What's the difference between validation and test here?"

> *"Validation is where I make choices — early stopping, model selection,
> threshold calibration. I can look at it as often as I like. Test I look at
> once, at the end, with everything frozen.*
>
> *The reason is that every choice made against a dataset fits a parameter to
> it. If I swept thresholds on test and reported the best, my numbers would be
> optimistically biased even though I never trained on it. My evaluate script
> enforces the ordering structurally and logs the two stages separately, so I
> can't accidentally blur them."*

### Q34. "How do you know the model is right for the right reason?" ⭐

> *"Image-level metrics can't tell you that — a model can have near-perfect
> AUROC while its heatmap points at the background, if the background happens to
> correlate with the label. There's a well-known class of examples where models
> learned to detect a scanner watermark or a ruler next to a lesion rather than
> the thing itself.*
>
> *So I measure localisation three ways: pixel-level AUROC over individual
> pixels, peak-hit rate — does the single hottest pixel land inside the true
> defect — and mean IoU over the top 1% of pixels. And I look at the heatmaps.
> Grad-CAM and the PaDiM map both exist partly as debugging tools, not just as
> user-facing features."*

### Q35. "Tell me about your error analysis." ⭐

**Use your real findings.** The structure:

> *"[N] of my [M] false negatives were the crack class. Cracks in my data are
> 1–2 pixels wide and low-contrast, and both my methods smooth spatially — the
> autoencoder through reconstruction, PaDiM through the Gaussian blur on the
> anomaly map. I tested the hypothesis by dropping smooth_sigma from 4.0 to 2.0,
> which recovered three false negatives but added eleven false positives. At my
> 10:1 cost ratio that's a net loss, so I kept the original setting.*
>
> *The real fix is a higher-resolution embedding grid, but memory scales
> quadratically with grid size, so I noted it as future work rather than
> pretending it was free.*
>
> *My false positives were mostly parts with unusual lighting — legitimate
> variation my normal set underrepresented. More diverse normal data would help
> more than any model change."*

Why this works: hypothesis → test → quantified trade-off → reasoned decision.

### Q36. "Your test set is small. Does that matter?"

> *"Yes, and I should be explicit about it. With around 78 test images my AUROC
> has real uncertainty and I'm reporting it as a point estimate. Bootstrapping
> confidence intervals would be the right fix and I listed it in future work.
> The comparison between models is somewhat more reliable than the absolute
> numbers, because they're evaluated on identical images so the noise is
> correlated."*

---

## Section 7 — System design

### Q37. "What's training/serving skew?" ⭐

> *"When training and production preprocess data differently, so the model sees
> different inputs than it was validated on. It's silent — no error, just
> quietly worse predictions — and it's one of the most common ways real ML
> systems fail.*
>
> *I prevented it structurally: there's exactly one InspectionEngine that owns
> preprocessing and scoring. Evaluation and the API both call it; neither has
> its own copy. The transforms are built from the checkpoint's own recorded
> image size, mean and std, so serving can't drift from training. If they ever
> disagree it's a bug in one file, not a mystery."*

### Q38. "Why load models at startup?"

> *"Loading a checkpoint takes hundreds of milliseconds to seconds. Inside a
> request handler, every request pays that, which blows the 2-second budget
> immediately. I use FastAPI's lifespan handler to load once into a
> module-level singleton. Note lifespan rather than the on_event startup
> decorator, which is deprecated in current FastAPI."*

### Q39. "What happens if a model fails to load?"

> *"The service starts anyway, logs loudly, reports status 'degraded' on
> /health, and returns a 503 from /inspect with the actual startup error.*
>
> *It's deliberate. A container that crash-loops tells an orchestrator nothing
> and hides the real error behind a restart loop. One that starts and honestly
> reports 'I'm up but I have no model' is debuggable in ten seconds.*
>
> *Related: my decision layer returns REVIEW rather than PASS when a signal is
> missing. A model-loading failure must not become a silent stream of PASS
> verdicts on a production line."*

### Q40. "Why is the fusion rule an OR?" ⭐

> *"Because the two models fail in largely uncorrelated ways. The classifier is
> blind to unseen defect types; PaDiM is blind to defect identity but catches
> any deviation. Since their blind spots barely overlap, flagging when either
> fires catches strictly more than either alone. The cost is precision — false
> alarms from both models add up.*
>
> *That trade is right here because costs are asymmetric by about 10×. It would
> be wrong in a domain where a human reviews every flag with fixed capacity —
> fraud review, content moderation — because there every false positive consumes
> a reviewer slot a real case needed. The rule follows from the cost structure,
> not from ML fashion."*

### Q41. "Why not train a learned fusion model?"

> *"Stacking a logistic regression on the two scores is the obvious next step
> and would probably help a little. I didn't, because fitting it honestly needs
> a third held-out split, and with tens of labelled defects that split would be
> too small to trust — I'd be fitting a combiner on noise.*
>
> *Recognising when you don't have the data to justify a more complex method is
> the actual decision. I listed it in future work, which is the right place."*

### Q42. "Walk me through your Dockerfile."

> *"Multi-stage build — stage one installs dependencies, stage two copies only
> the installed packages and source into a clean image, so build tools and pip
> caches never reach the final layer.*
>
> *The important line is the CPU-only PyTorch index. The default PyPI wheel
> bundles about 2.5 GB of CUDA libraries useless on a CPU host; the CPU wheel is
> around 200 MB. On a free tier with an image-size cap that one line is the
> difference between deploying and not.*
>
> *Beyond that: non-root user, a HEALTHCHECK so an orchestrator can distinguish
> 'process running' from 'actually serving', and one worker — each worker loads
> its own copy of the models, so on 512 MB a second worker is an OOM kill rather
> than throughput."*

### Q43. "Your latency numbers?"

> *"Mean 111 ms, p95 139 ms on CPU, against a 2-second budget. I report p95
> rather than just mean because tail latency is what users experience. Two
> measurement details: I warm up before timing, because the first forward pass
> pays lazy-init costs, and I time inference only rather than including the
> upload, otherwise I'd be measuring the client's network."*

### Q44. "How would you scale this to a real line?"

> *"Depends on the bottleneck. For throughput, batch requests — my engine
> already scores batches, the API just exposes single images. For higher volume,
> horizontal scaling behind a load balancer, though on CPU I'd size memory first
> since each replica holds its own models.*
>
> *For lower latency, ONNX Runtime or TorchScript export, plus int8 quantisation
> — PaDiM's forward pass is a frozen ResNet, which quantises well.*
>
> *But the honest first question is whether I need it. A line at one part per
> second needs 1 QPS, and I'm at roughly 9. I'd rather spend the effort on drift
> monitoring, because a model silently going stale is a bigger risk than
> latency."*

### Q45. "What would you do differently with more time?"

Three specific answers. Never "more data":

> *"First, PatchCore instead of PaDiM — it drops the alignment assumption and
> generally scores higher on MVTec.*
>
> *Second, an active-learning loop. My FAIL_ANOMALY verdict already identifies
> images that deviate from normal but match no known defect class — those are
> exactly the images worth human labelling. Route them to a reviewer and you
> discover new defect classes over time rather than accumulating unknowns.*
>
> *Third, bootstrap confidence intervals on my metrics. With 78 test images my
> AUROC has real uncertainty and I'm currently reporting a point estimate."*

---

## Section 8 — Questions that trip people up

### Q46. "What don't you like about your own project?"

They want self-awareness, not false modesty.

> *"The supervised path is undertrained and I knew it would be — ten labelled
> images per defect class isn't enough for the multiclass head to be reliable,
> and my per-class recall shows that. I kept it because the comparison is the
> point: it demonstrates concretely why the unsupervised path is necessary
> rather than me just asserting it. But if someone asked me to ship the defect
> *typing* feature, I'd say it isn't ready and I need more labels."*

### Q47. "How do you know your synthetic data isn't misleading you?"

> *"I don't fully, which is why I report MVTec numbers as my headline and treat
> synthetic as a pipeline test. Synthetic defects are easier — the noise model
> is known, and lighting variation is simpler than a real factory. If my
> synthetic and MVTec numbers were similar I'd be suspicious of the synthetic
> generator being too easy; if MVTec were dramatically worse I'd want to know
> which defect types drove the gap."*

### Q48. "Is `anomaly_normalised` a probability?"

> *"No, and I made the API docs say so explicitly. It's a raw Mahalanobis
> distance divided by the 95th percentile of good validation parts, so 1.0 means
> 'as unusual as the most unusual normal part'. If a stakeholder read 0.8 as
> '80% chance of a defect' they'd be wrong, and that misunderstanding leads to
> bad decisions.*
>
> *If I needed real probabilities I'd calibrate — Platt scaling or isotonic
> regression on a held-out set — and validate with a reliability diagram."*

### Q49. "Suppose your AUROC came back at 0.51. What would you check?"

They want a debugging process, not an answer.

> *"In order of likelihood. First, is the model actually using pretrained
> weights — random features carry much less signal. Second, did defects leak
> into the normal fitting set, which would widen 'normal' to include them.
> Third, is augmentation being applied during PaDiM fitting, which scrambles the
> per-position assumption. Fourth, are labels inverted somewhere — an AUROC of
> 0.49 versus 0.51 versus 0.05 tells you very different things. Fifth, plot the
> score distributions; if the two histograms sit exactly on top of each other
> it's a modelling problem, and if they're separated but the metric says 0.5,
> it's a labelling or indexing bug."*

### Q50. "You said you wrote tests. What did you test and why those?"

> *"I deliberately tested things that fail *silently*, because those cost you a
> week. Does a defect leak into the anomaly training split. Does freezing
> actually freeze, including the BatchNorm buffers. Does save-then-load
> reproduce bit-identical scores. Does the cost threshold actually favour recall
> when I raise the false-negative cost. Does the decision layer return REVIEW
> rather than PASS when a signal is missing.*
>
> *I didn't write tests asserting my loss function computes a number. 57 tests,
> they run in about 7 seconds, and each one corresponds to a bug that would
> otherwise be invisible."*

---

## Final preparation checklist

**Can you, without notes:**

- [ ] Give the 90-second narrative, timed
- [ ] Draw the two-path architecture on a whiteboard
- [ ] Explain Mahalanobis vs Euclidean with the ellipse picture
- [ ] Explain your split protocol and why it's leak-free
- [ ] Justify your threshold with the cost argument
- [ ] Describe one real finding from your error analysis
- [ ] Name three limitations unprompted
- [ ] Explain why the fusion rule is OR, and when it wouldn't be
- [ ] Explain training/serving skew and how you prevented it
- [ ] State the difference between PaDiM and PatchCore

**Have you:**

- [ ] Filled every `[X]` in this file with your own real numbers
- [ ] Rewritten at least ten of these answers in your own words
- [ ] Practised out loud, not just read
- [ ] Re-read your own `results.md` the morning of the interview

---

## One last thing

If you are asked something you do not know, say so.

> *"I don't know. My instinct is [X] because [reasoning], but I'd want to
> [check/measure/read] before committing to that."*

That answer scores well. A confident wrong answer scores worse than admitting
uncertainty — because the interviewer's real question is "can I trust what this
person tells me?", and a candidate who bluffs once will bluff about a production
incident.

You built this. You understand it. Explain it like someone who does.
