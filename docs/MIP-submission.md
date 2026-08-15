# What does an interpretable buyer persona cost? A measurement-first evaluation of customer representations

**Article classification:** Research paper

**Keywords:** Buyer personas, customer segmentation, marketing analytics, interpretability, predictive validation, representation learning

---

## Structured abstract

**Purpose** — Buyer personas direct substantial marketing expenditure, yet the field's own
survey of data-driven persona development names evaluation as an open gap. Existing
predictive validation reduces persona quality to a single accuracy figure, which cannot
express what is sacrificed to obtain the interpretability practitioners choose personas for.
This paper asks what an interpretable persona costs.

**Design/methodology/approach** — Persona quality is reformulated as three separately
falsifiable properties: predictive lift on held-out future behaviour, alignment with named
marketing constructs, and stability across repeated estimation. Eleven representations,
including RFM, PCA, autoencoders and a construct-aligned variational autoencoder, are
compared under a frozen-representation protocol on two public datasets (2,240 customers;
2.55 million users from 110 million events), with leakage guards enforced by automated
tests, six seeds per result, pre-specified performance bars and familywise correction.

**Findings** — Interpretability is costly and the cost is measurable. Non-neural incumbents
set the strongest bars; the construct-aligned model falls significantly below both and
yields markedly less stable personas. Two controls sharpen this: alignment proves to be a
property of a latent subspace rather than of individual named axes, and a zero-training
representation built directly from the constructs achieves comparable measured lift to the
neural model while holding perfect axis purity by construction.

**Originality/value** — The contribution is measurement rather than method: a protocol that
prices interpretability, and evidence that an assumed benefit of neural segmentation does
not survive it.



---

## 1. Introduction

Customer segmentation into buyer personas is a standard instrument of marketing practice,
and data-driven persona development now has a substantial research literature of its own.
That literature's most comprehensive review — 77 articles spanning 2005–2020 — nonetheless
lists evaluation methods among the field's persistent open gaps (Salminen *et al.*, 2021).
Work addressing the gap exists: Hsu *et al.* (2023) build personas with predictive
algorithms, hold out a partition, and validate against real purchase behaviour, showing
that clustering-based personas fail to separate buyers where a predictive persona does.

Their measure is a single accuracy figure. That is the limitation this paper addresses.
Practitioners do not choose personas on accuracy alone; they choose them because a persona
can be named, explained to a marketing team, and relied upon to mean the same thing next
quarter. A one-dimensional criterion cannot express what is given up to obtain those
properties, and therefore cannot tell a manager whether interpretability is worth its
price.

We treat this as a measurement problem. A persona representation is good, we argue, exactly
to the extent that it satisfies three properties, each measurable and each able to fail
independently:

1. **Predictive lift.** A frozen representation should predict real future behaviour at
   least as well as incumbent representations, under a protocol that forbids it from ever
   seeing the labels.
2. **Interpretability.** The latent space should carry named marketing constructs —
   recency-frequency-monetary value (RFM), price sensitivity, category affinity —
   recoverably *and separably*.
3. **Stability.** Persona assignments should persist across repeated estimation.

The vehicle for testing whether all three can hold at once is a construct-aligned
disentangled variational autoencoder (CA-DVAE), whose latent vector is split into a block
trained to predict the named constructs and a free block. Neither the architecture nor the
split is claimed as novel. Sweeping the alignment strength traces what alignment buys and
what it costs.

The findings are largely negative and specific, which is the contribution. Under honest
evaluation the incumbents this literature tends to dismiss are the strongest baselines.
Strong construct alignment costs predictive lift and can reduce stability. Most
consequentially, when
the interpretability claim is itself tested rather than assumed, it proves narrower than
reported: the constructs live in a latent *subspace*, not on individual named axes, and a
representation that simply uses the constructs as coordinates achieves comparable measured
lift without any training at all.

## 2. Background

**Persona evaluation.** Salminen *et al.* (2021) establish the evaluation gap this work
addresses. Hsu *et al.* (2023) are the nearest neighbour: their predictive personas are
validated on a held-out partition and against subsequent coupon redemption. We do not claim
that measuring persona quality by held-out prediction is new — it is theirs. What is new
here is holding three properties as co-equal criteria and measuring the trade-offs among
them, and evaluating a task-agnostic *representation* by transfer to labels it never saw
rather than a persona set defined by one supervised outcome.

Two things called persona stability must be distinguished. Jansen *et al.* (2021) measure
it longitudinally, generating personas over 32 monthly rounds and reporting an average 40%
change as the audience evolves. This paper measures *stochastic reproducibility* — the same
population, repeated estimation — using the bootstrap-and-adjusted-Rand approach that is
already standard in marketing methodology (Dolnicar and Leisch, 2010). The properties are
independent: a representation can be perfectly reproducible on a fixed population and
obsolete within months.

**Interpretable representations.** Constraining designated units to predict named concepts
is the concept bottleneck family (Koh *et al.*, 2020), and the specific arrangement used
here — supervised concepts alongside unsupervised ones in the same layer — is prior art
(Sawada and Nakamura, 2022). In the customer domain, Mancisidor *et al.* (2019) steer a
customer VAE's latent space with a business quantity. Critically, concept-supervised models
with an unsupervised side channel are known to leak: learned concept representations encode
information beyond the named concepts, and the resulting explanations mislead (Mahinpei
*et al.*, 2021; Margeloiu *et al.*, 2021). That documented failure motivates the diagnostic
in Section 4.3, which turns out to be decisive.

**What this study does not claim.** Being explicit about this is necessary, because each
component of the approach has an established antecedent. The variational autoencoder and
the aligned/free latent split are prior art. Validating personas by held-out prediction is
prior art. Assessing segmentation stability by resampling agreement is long-established
marketing methodology. Naming the tension between interpretability and performance in
segmentation is prior art. What is claimed is narrower and, we think, more useful: holding
the three properties as co-equal criteria; measuring the exchange rate between them under a
protocol strict enough that the numbers survive scrutiny; and two methodological findings
that emerge only from doing so — that a standard disentanglement metric selects models in
the wrong direction when business constructs are correlated, and that reproducibility and
performance must be read jointly because degenerate segmentations are maximally stable.

**Neural segmentation and its evaluation.** Deep embedded clustering (Xie *et al.*, 2016)
and autoencoder-plus-k-means pipelines dominate neural segmentation, and published
evaluations typically report internal cluster indices on a single run. Boussebough *et al.*
(2026) and Grigorova *et al.* (2025) both weigh interpretability against performance in
customer segmentation, but resolve the tension qualitatively or with internal indices
rather than held-out prediction. Our baseline findings are also consistent with a broader
result: tree-based models remain state of the art on medium-sized tabular data even under
matched tuning (Grinsztajn *et al.*, 2022).

## 3. Method

**Datasets.** *A*: Customer Personality Analysis, 2,240 customers, 29 survey-style columns;
label is response to the final campaign (14.9% positive); stratified splits redrawn per
seed. *B*: a multi-category e-commerce log, 109,950,743 events over two months aggregated
to 2.55 million users × 27 features; features come from a window strictly preceding the
label window, and train/validation/test are user-disjoint. Labels are future purchase
(3.3% positive), dormancy, and next purchase category.

**Constructs.** Ten construct targets on Dataset A and seventeen on B, covering RFM, a
price-sensitivity index and category-affinity shares. These are computed on the training
split only and standardised without refitting; a perturbation test proves test rows cannot
move the fitted scaler. One property shapes everything that follows: **the constructs are
deterministic functions of the input features.** The alignment task therefore adds no
information and can only shape the geometry of the latent space — which is why the control
in Section 4.4 is the decisive experiment.

**Model.** A dense VAE over customer features, latent dimension 16, whose latent vector is
partitioned into an aligned block trained by a *linear* head to predict the standardised
constructs and a free block carrying a β-weighted Kullback–Leibler term. Linearity is a
design constraint, not a simplification: it is what makes a construct correspond to a
direction. The alignment weight λ and disentanglement weight β are swept over a 6 × 4 grid
× 6 seeds, 144 runs per dataset. λ = 0 recovers a β-VAE; both components are configuration
points, not code branches.

Because alignment and disentanglement are separate configuration values rather than
separate code paths, each can be switched off without touching the implementation: setting
the alignment weight to zero recovers a conventional β-VAE, and holding β at one removes the
additional disentanglement pressure while retaining alignment. Both ablations are therefore
points on the same swept grid, evaluated by the same protocol as the full model.

**Protocol.** Every representation is fit on training data only, then **frozen**; two
downstream heads (L2 logistic regression and gradient-boosted trees) are trained on the
frozen features and evaluated on held-out test data. Labels never touch representation
learning. Baselines — RFM, RFM+k-means, PCA at matched capacity, autoencoder, AE+k-means,
Gaussian mixture on AE embeddings, and deep embedded clustering — were run and the
performance bar fixed **before** any proposed-model run. The raw feature vector is included
as a reference ceiling, excluded from bar selection.

**Guarding against leakage.** The most common way a study like this produces an
attractive but false result is by letting information from the evaluation period reach the
representation. Four guards are enforced as automated tests rather than as convention.
Cleaning statistics, construct scalers and every fitted transform are estimated on training
rows only, and a perturbation test corrupts the held-out rows to prove the fitted values do
not move. On the behavioural dataset an ordering assertion checks that every feature event
precedes every label event, and the user-disjoint split blocks the same customer appearing
on both sides. Downstream labels are introduced only when the heads are fitted, after the
representation is frozen. Finally, a drift guard pins the evaluation heads used during the
sweep to the exact specification used when the bars were set, so the comparison cannot
silently change under the proposed model.

**Selection hygiene.** An audit conducted before any sweep computation found that the
operating point would have been chosen on test metrics — the classic tuned-on-test flaw.
The sweep therefore records validation metrics for every run; the operating point is
selected on validation and reported on test, and a sensitivity analysis confirms the
selection is not an artefact of the tolerance used. Where several downstream tasks exist,
each is compared against the best baseline *for that task* rather than a single global
baseline, so no comparison is made against a straw man.

**Statistics.** Every figure is a mean ± standard deviation over six seeds with paired tests
from two families. We report this limitation plainly: at six seeds the two-sided Wilcoxon
p-value has a discrete floor of 0.031, so every Wilcoxon result here sits at that floor and
none survives Holm correction across a realistic comparison family. Claims therefore rest on
the paired *t*-test, effect sizes and per-seed consistency, with Holm-adjusted values
reported alongside.

**What was fixed in advance.** Two things, and it is worth being exact about which. First,
the performance bars: running all baselines and fixing the bar before any proposed-model run
was a gate in the project protocol, and the baseline campaign completed before the sweep
began. Second, the three properties and the direction expected of each were set out in the
project's design document before any model was run — that the representation would predict
behaviour better than incumbents (E1), that its *latent axes* would align with named
constructs (E2), and that persona assignments would persist across seeds and resamples (E3).
These are design expectations recorded in advance, not a registered analysis plan: no
external pre-registration was filed, and the labels are introduced here for exposition.

## 4. Findings

### 4.1 The bars: non-neural incumbents win

**Table I.** Downstream performance of customer representations (test PR-AUC, gradient-boosted
head, six seeds)

| Representation | Dataset A (base rate 0.149) | Dataset B (base rate 0.033) |
|---|---|---|
| Raw features (ceiling)* | 0.6066 ± 0.0749 | 0.2038 ± 0.0042 |
| **PCA (d = 16)** | **0.5677 ± 0.0729** | 0.1649 ± 0.0073 |
| **RFM (3 features)** | 0.3396 ± 0.0462 | **0.1953 ± 0.0035** |
| Autoencoder (d = 16) | 0.5156 ± 0.0682 | 0.1723 ± 0.0021 |
| Gaussian mixture on AE | 0.3973 ± 0.0615 | 0.1148 ± 0.0048 |
| Deep embedded clustering | 0.3483 ± 0.0465 | 0.0819 ± 0.0051 |
| AE + k-means personas | 0.3343 ± 0.0543 | 0.0672 ± 0.0071 |
| RFM + k-means personas | 0.2988 ± 0.0182 | 0.1073 ± 0.0064 |
| CA-DVAE (validation-selected) | 0.5130 ± 0.0597 | 0.1798 ± 0.0051 |

*Excluded from bar selection. Bold marks each dataset's pre-specified bar.

The representations this literature most often treats as superseded are the strongest. On
survey-style data PCA significantly beats a trained autoencoder (*t*, p = 0.002) and is
statistically indistinguishable from the raw ceiling; on behavioural data three raw RFM
features beat every learned representation. Hard cluster assignment — the standard persona
format — is a major signal destroyer in these experiments: every eight-cluster one-hot
representation loses 40–65% of its parent embedding's PR-AUC. An evaluation reporting only
silhouette scores on a single run would observe none of this.

Undertuned baselines are the most common source of illusory improvement, and the proposed
model receives 24 configurations per seed from the sweep. To keep the comparison fair we ran
a dedicated tuning study for the strongest neural baseline on the survey dataset — six
configurations across six seeds. **The pre-specified bar survives**: the best-tuned
autoencoder reaches 0.5519 under the tree head, still below PCA. For transparency, an
autoencoder at twice the proposed model's latent capacity reaches parity with PCA under the
linear head (0.5731), a difference not statistically distinguishable in either test family.
The bar is unchanged, and the raw ceiling remains above everything.

**E1 is not supported on either dataset.** CA-DVAE at its validation-selected operating
point sits below both bars under the paired *t*-test (Dataset A: −0.055, p = 0.003;
Dataset B: −0.016, p = 3.4 × 10⁻⁵), consistently across every seed. A separate observation
points the same way and is consistent with prior work (Nai *et al.*, 2024): high-β
configurations are the worst on both datasets, because pressure toward dimension-wise
independence costs the informativeness the downstream head needs. This was anticipated
during model development — a posterior-collapse diagnostic recorded before the sweep led to
extending the grid downward in β — and is therefore reported as a confirmed expectation
rather than as a tested hypothesis.

**Figure 1** plots the trade-off surface directly: downstream performance against construct
alignment across the swept grid, with the pre-specified bar marked. It is the central
artefact of the paper — the exchange rate between accuracy and interpretability, drawn from
data rather than asserted.

*[Figure 1 near here: `mip_figures/figure1_tradeoff.png` — interpretability–performance
trade-off surface, Dataset A, with the PCA bar and the validation-selected operating point
marked.]*

### 4.2 Stability is a cost, not a benefit

**E3 is not supported.** Incumbent personas are markedly more stable across seeds than aligned
ones: RFM+k-means reaches cross-seed adjusted Rand index 0.975 (A) and 0.962 (B), AE+k-means
0.703 and 0.745, against 0.467 and 0.583 for CA-DVAE at its selected point. RFM's stability
is unsurprising — it clusters three deterministic features — and stability was never its
weakness.

**Table II.** Cross-seed persona stability (adjusted Rand index, eight clusters, six seeds)

| Persona pipeline | Dataset A | Dataset B |
|---|---|---|
| RFM + k-means | 0.975 ± 0.011 | 0.962 ± 0.033 |
| AE + k-means | 0.703 ± 0.068 | 0.745 ± 0.037 |
| CA-DVAE, moderate alignment | 0.64–0.70 | 0.58–0.73 |
| CA-DVAE at the selected point | 0.467 | 0.583 |

Two qualifications matter for practice. First, a confound that makes naive stability
comparisons meaningless: the *most* stable configurations in both grids are the most
collapsed ones, sitting at the worst downstream performance, because a near-degenerate
assignment is trivially reproducible. Stability must be read jointly with performance,
never alone. Restricting to configurations within the selection slack of the best
validation score, aligned personas reach parity with the neural incumbent (0.697 versus
0.703 on A; 0.725 versus 0.745 on B) — but remain far below the non-neural one. Second,
strong alignment actively destabilises: the effect is non-monotone, with moderate alignment
improving stability over the unaligned ablation and strong alignment degrading it.

### 4.3 Interpretability is a subspace, not a set of axes

The alignment result this project had been reporting — construct R² up to 0.968 — is
produced by a linear read-out of the *whole* aligned block. Testing whether individual axes
are separable gives a different answer.

**Table III.** Concept-leakage diagnostic, Dataset A at the selected operating point
(10 aligned / 6 free dimensions, six seeds)

| Quantity | Value | Reading |
|---|---|---|
| Aligned **block** R², mean over constructs | 0.873 (RFM 0.985–0.989) | The block encodes the constructs almost perfectly |
| Best **single-axis** R², mean | 0.406 | No individual axis carries a construct strongly |
| **Axis purity** (on-target − best off-target R²), mean | **−0.025** | A named axis typically predicts some *other* construct as well or better |
| Constructs with negative purity | 6 of 10 | Worst: frequency (−0.149), sweet-affinity (−0.174) |
| Distinct winning dimensions for 10 constructs | 5.0 | Constructs compete for the same axes |
| Free-block R², mean | 0.102 | Constructs stay inside the block: not side-channel leakage |

*[Figure 2 near here: `mip_figures/figure2_persona_cards.png` — latent traversals along each
best-aligned dimension. To be read with this section: the directions are interpretable in
appearance but are not construct-pure.]*

**E2 splits: supported at block level, not supported at axis level.** The failure is not the
side-channel leakage documented by Mahinpei *et al.* (2021) — the constructs remain inside
the aligned block — but *within-block entanglement*: the block encodes the constructs as a
distributed code rather than as separated directions. The practical consequence is direct.
A persona card built by moving one latent dimension is moving a direction that mixes several
constructs, so its apparent interpretability is weaker evidence than it looks. The defensible
claim is a **named subspace**, not named axes.

A related methodological caution: the Mutual Information Gap, a standard disentanglement
metric, *inverts* here. On Dataset B it is highest with alignment switched off and falls as
alignment strengthens, because aligning 12 dimensions to 17 correlated constructs
necessarily shares information across them. Selecting on it chose a model with no named axes
at all. This is narrower than the established finding that correlated factors degrade
disentanglement (Träuble *et al.*, 2021): the point is that MIG should not be used as the
sole selection criterion when business constructs are correlated, which is the normal case.

### 4.4 The decisive control: interpretability without training

Because the constructs are deterministic functions of the features, a representation with
genuinely named axes is available with no training at all: use the constructs as
coordinates, and append principal components of the residual for the capacity they lack.
This control has alignment R² = 1 and perfect axis purity by construction — precisely the
property CA-DVAE claims and, per Section 4.3, does not deliver.

**Table IV.** Named-axis control versus the swept model, Dataset A (test PR-AUC, six seeds)

| Representation | PR-AUC | Named axes | Training cost |
|---|---|---|---|
| PCA (bar) | 0.5677 ± 0.0729 | none | none |
| **Constructs + residual PCA** | **0.5438 ± 0.0640** | 10, R² = 1 by construction | **none** |
| CA-DVAE | 0.5130 ± 0.0597 | 10, mean axis R² 0.406 | 144-run sweep |
| Constructs alone | 0.4129 ± 0.0650 | 10, R² = 1 by construction | none |

| Paired comparison | Δ | Cohen's *d_z* | *t* | Holm-adj. |
|---|---|---|---|---|
| Control − CA-DVAE | +0.031 | +0.82 | 0.099 | 0.199 |
| Control − constructs alone | +0.131 | +5.46 | 4 × 10⁻⁵ | **1.3 × 10⁻⁴** |

The control is ahead on five of six seeds but the difference is not statistically
distinguishable at this sample size. The correct conclusion is not that the simple control
wins; it is that **a 144-run sweep over a neural architecture fails to demonstrate any
advantage over a representation obtained in closed form from the same features**, on the
dataset where the alignment mechanism performs best — while the control strictly dominates
it on the interpretability property both are competing on.

One finding runs the other way and is robust: constructs *alone* reach only 0.4129, far
below the same constructs with a residual block (*d_z* = 5.46, significant after
correction). Interpretable coordinates alone are insufficient; free capacity alongside them
does necessary work. That supports the aligned/free decomposition — but not the neural machinery
used to obtain it.

### 4.5 Where alignment does pay

A single global bar would misrepresent this comparison, because no baseline wins every task.
On the behavioural dataset RFM wins purchase prediction, the autoencoder wins dormancy, and
autoencoder-based personas win next-category prediction — so each task is judged against its
own strongest competitor.

Against those bars, the aligned model wins dormancy prediction in both test families
(0.8632 versus 0.8620), which in business terms is approximately twelve additional dormant
customers identified per ten thousand contacted. The effect is tight across seeds and
genuinely small, and we report it at exactly that size rather than as evidence of general
superiority. Next-category prediction is a loss, not a parity: the seed-to-seed instability
observed in the unregularised autoencoder on that task is not cured by construct anchoring.

Two results support the design rather than the model. First, the twelve aligned dimensions
alone carry as much downstream signal as the autoencoder's entire unnamed embedding, and on
the survey dataset the aligned block alone beats raw RFM — so the named subspace is
predictively load-bearing, not a label attached to a black box. Second, RFM, dominant on
purchase prediction, collapses to below base rate on next-category prediction: it contains
no category information at all, and stability does not compensate for a representation that
lacks information relevant to the prediction task. Where a persona scheme must carry category
structure, a learned representation remains necessary.

The shape of the trade-off surface is also practically informative. It is smooth rather than
a cliff: on the survey dataset a weakly aligned configuration reaches the PCA bar almost
exactly, but with construct alignment of only R² ≈ 0.35. The lift lost to alignment is
recoverable precisely by giving up the naming, which is the trade-off stated in its most
direct form.

### 4.6 Which component is doing the work

Both components can be switched off as configuration points. Removing alignment entirely
gives the best-performing region on the behavioural dataset — and produces no named
constructs at all, which is precisely the trade-off this study exists to price. Retaining
alignment while removing the extra disentanglement pressure is dominated on both datasets by
configurations with weaker disentanglement, consistent with the finding that informativeness
rather than dimension-wise independence drives downstream performance (Nai *et al.*, 2024).

Neither component is decoration: each changes the measured outcome, and each has its own
ablation. But the direction is unflattering to the design. The alignment component *costs*
primary lift while supplying the interpretability the study is about, and the disentanglement
component costs lift without a compensating measured benefit in this setting. Read alongside
Section 4.4, the honest summary is that the aligned/free *structure* is worth keeping and the
neural machinery used to obtain it does not demonstrate a measurable advantage in this
setting.

## 5. Discussion and managerial implications

**Interpretability has a price, and this study quantifies it.** Named construct structure at
block-level R² ≈ 0.97 costs roughly 0.055 PR-AUC against PCA on survey data, 0.016 against
RFM on behavioural data, and a substantial adjusted-Rand stability gap relative to the
non-neural incumbent. A manager can now make that trade explicitly rather than assuming
it away.

**Choose the representation by the job.** For ranking customers by purchase propensity, use
RFM: more accurate, more stable, free, and requiring no specialist infrastructure. For
persona schemes that must express category structure — which RFM cannot represent at all —
a learned representation is necessary. For maximum accuracy irrespective of explanation,
use the raw feature vector with a tree-based model; the ceiling sits above every
representation tested. And if the goal is *interpretable coordinates*, construct them
directly: our control obtained them with no training, no GPU and no measurable loss of lift
relative to the neural model.

**Do not buy a neural model for interpretability alone.** This is the practical implication
we would emphasise. Interpretability was the stated motivation for construct alignment, and
it is precisely where the neural approach was matched by arithmetic on features the
organisation already has.

**Implications for segmentation research.** The pattern across Table I is consistent enough
to be worth stating as a caution to the field: under leakage-safe evaluation on held-out
future behaviour, the more elaborate the pipeline, the worse it performed. Deep embedded
clustering and autoencoder-plus-k-means personas — the two approaches most commonly proposed
as improvements on RFM — were the weakest representations tested on both datasets. This does
not show that neural segmentation cannot work; it shows that the evidence usually offered for
it, internal cluster indices on a single run, cannot distinguish a representation that
predicts behaviour from one that does not. Studies proposing new segmentation methods should
report held-out predictive validation against a non-neural incumbent, with repeated
estimation, as a matter of course.

**Budget the evaluation, not just the model.** The costliest thing an organisation can do
here is not choosing the wrong representation; it is choosing without measurement and then
being unable to tell. Every result in this study came from infrastructure that is modest by
modern standards — the entire experimental programme, 288 sweep runs included, consumed
under nine GPU-hours — while the measurement discipline that produced the findings costs
nothing but protocol: hold the representation out of the labels, repeat across seeds, fix
the bar before running the proposed method, and report the comparison you pre-specified
rather than the one that looks best afterwards. An organisation that adopts only the
protocol and none of the models will still be better off than one that adopts a persona
scheme on narrative appeal.

**Two cautions for anyone evaluating personas.** Never report stability without
performance: the most reproducible segmentations in both our grids were the most degenerate,
and a segmentation that always returns the same answer because it has stopped
discriminating is worse than an unstable one. And do not accept an aggregate alignment score
as evidence that a dimension is nameable — the two questions come apart, as Section 4.3
shows, and only the second supports a persona card shown to a marketing team.

## 6. Limitations and further research

The two controls of Sections 4.3 and 4.4 have been run on Dataset A only; the behavioural
dataset's seventeen correlated construct targets make within-block entanglement more likely
there, not less, and completing those runs is the first priority. Six seeds cannot support
familywise-corrected Wilcoxon claims, as stated in Section 3. Dataset B covers a single
two-month window treated as one population, so no claim is made about performance under
distribution shift. Dataset A's test split is small (448 rows), making PR-AUC genuinely
variable. The price-sensitivity construct on Dataset B measures browsed price tier rather
than discount responsiveness, as the log carries no discount field.

The most valuable extension follows from Section 2: this study measures stochastic
reproducibility, while Jansen *et al.* (2021) show that temporal change is where data-driven
personas actually fail. Measuring both properties on the same representations — reproducible
today, still valid next quarter — is the natural next study.

## 7. Conclusion

This paper replaces an unmeasured marketing artefact with three falsifiable measurements and
reports what they say. Of the three properties the study set out to establish, none held in
the form originally stated: the representation did not beat incumbents on lift, its named
structure proved to be a subspace rather than the individual axes the design anticipated,
and its personas were less stable than the incumbents'. The interpretability being purchased is real
but narrower than claimed — a named subspace rather than named axes, and one a closed-form
control obtains without training or measurable loss in lift. For practice, the finding is
usable immediately: incumbent representations are hard to beat, interpretable coordinates
are cheap to construct directly, and the cost of an interpretable persona is now a number
rather than an assumption.

@@AVAILABILITY@@

@@AIDISCLOSURE@@

## References

Boussebough, I., Zarour, K., Aouabdia, C. and Boutina, D.S. (2026), "Multi-view customer segmentation in the digital economy: balancing performance and interpretability for actionable insights", *Journal of Telecommunications and the Digital Economy*, Vol. 14 No. 2, pp.58-83.

Chen, R.T.Q., Li, X., Grosse, R. and Duvenaud, D. (2018), "Isolating sources of disentanglement in variational autoencoders", *Advances in Neural Information Processing Systems*.

Dolnicar, S. and Leisch, F. (2010), "Evaluation of structure and reproducibility of cluster solutions using the bootstrap", *Marketing Letters*, Vol. 21 No. 1, pp.83-101.

Grigorova, I., Efremov, A.S. and Karamfilov, A. (2025), "An automated machine learning framework for interpretable customer segmentation in financial services", *International Journal of Financial Studies*, Vol. 13 No. 4, art. 243.

Grinsztajn, L., Oyallon, E. and Varoquaux, G. (2022), "Why do tree-based models still outperform deep learning on tabular data?", *Advances in Neural Information Processing Systems, Datasets and Benchmarks Track*.

Higgins, I., Matthey, L., Pal, A., Burgess, C., Glorot, X., Botvinick, M., Mohamed, S. and Lerchner, A. (2017), "β-VAE: learning basic visual concepts with a constrained variational framework", *International Conference on Learning Representations*.

Holm, S. (1979), "A simple sequentially rejective multiple test procedure", *Scandinavian Journal of Statistics*, Vol. 6 No. 2, pp.65-70.

Hsu, P.-F., Lu, Y.-H., Chen, S.-C. and Kuo, P.-Y. (2023), "Creating and validating predictive personas for target marketing", *International Journal of Human-Computer Studies*, Vol. 181, art. 103147.

Hubert, L. and Arabie, P. (1985), "Comparing partitions", *Journal of Classification*, Vol. 2 No. 1, pp.193-218.

Hughes, A.M. (1994), *Strategic Database Marketing*, Probus, Chicago, IL.

Jansen, B.J., Jung, S., Chowdhury, S.A. and Salminen, J. (2021), "Persona analytics: analyzing the stability of online segments and content interests over time using non-negative matrix factorization", *Expert Systems with Applications*, Vol. 185, art. 115611.

Kingma, D.P. and Welling, M. (2014), "Auto-encoding variational Bayes", *International Conference on Learning Representations*.

Koh, P.W., Nguyen, T., Tang, Y.S., Mussmann, S., Pierson, E., Kim, B. and Liang, P. (2020), "Concept bottleneck models", *International Conference on Machine Learning*.

Kumar, A., Sattigeri, P. and Balakrishnan, A. (2018), "Variational inference of disentangled latent concepts from unlabeled observations", *International Conference on Learning Representations*.

Locatello, F., Bauer, S., Lucic, M., Rätsch, G., Gelly, S., Schölkopf, B. and Bachem, O. (2019), "Challenging common assumptions in the unsupervised learning of disentangled representations", *International Conference on Machine Learning*.

Mahinpei, A., Clark, J., Lage, I., Doshi-Velez, F. and Pan, W. (2021), "Promises and pitfalls of black-box concept learning models", arXiv:2106.13314.

Mancisidor, R.A., Kampffmeyer, M., Aas, K. and Jenssen, R. (2019), "Learning latent representations of bank customers with the variational autoencoder", arXiv:1903.06580.

Margeloiu, A., Ashman, M., Bhatt, U., Chen, Y., Jamnik, M. and Weller, A. (2021), "Do concept bottleneck models learn as intended?", arXiv:2105.04289.

Nai, R., Wen, Z., Li, J., Li, Y. and Gao, Y. (2024), "Revisiting disentanglement in downstream tasks: a study on its necessity for abstract visual reasoning", *AAAI Conference on Artificial Intelligence*.

Salminen, J., Guan, K., Jung, S.G. and Jansen, B.J. (2021), "A survey of 15 years of data-driven persona development", *International Journal of Human-Computer Interaction*, Vol. 37 No. 18, pp.1685-1708.

Sawada, Y. and Nakamura, K. (2022), "Concept bottleneck model with additional unsupervised concepts", *IEEE Access*.

Träuble, F., Creager, E., Kilbertus, N., Locatello, F., Dittadi, A., Goyal, A., Schölkopf, B. and Bauer, S. (2021), "On disentangled representations learned from correlated data", *International Conference on Machine Learning*, PMLR 139, pp.10401-10412.

Xie, J., Girshick, R. and Farhadi, A. (2016), "Unsupervised deep embedding for clustering analysis", *International Conference on Machine Learning*, pp.478-487.

---

## Submission checklist (not part of the manuscript — delete before submitting)

| Requirement | Status |
|---|---|
| Word count ≤ 8,000 incl. abstract, references, tables and figures (280 each) | **~6,900** — see note below |
| Structured abstract ≤ 250 words *including* keywords and classification | **243** ✓ |
| Four mandatory abstract sub-headings | Purpose / Design / Findings / Originality ✓ |
| Keywords, maximum 6 | 6 ✓ |
| Article classification | Research paper ✓ |
| Language consistency | UK English throughout ✓ |
| File format | **Convert this Markdown to .docx before submitting** |
| Figures supplied separately | `docs/mip_figures/` — 2 PNGs, callouts placed in text |
| Generative AI declared in-article **and** at submission | In-article ✓; declare again in the submission form |
| Double-anonymous review | **Action needed** — the public repository carries the author's name. Prepare an anonymised artefact link, or omit the availability statement for review and restore it on acceptance |

**Word-count note.** The total above uses Emerald's stated allowance of 280 words per table
or figure (5 tables + 2 figures = 1,960) plus abstract, body and references. If the editor
instead counts the literal text inside tables *in addition to* the allowance, the figure
rises; there is margin either way, but confirm against the current author guidelines.

**Scope note for the cover letter.** The two controls reported in Sections 4.3 and 4.4 were
run on Dataset A. Their Dataset B counterparts are pending, and this is stated in Section 6
rather than glossed. If a reviewer requires them, they can be produced without retraining —
the diagnostic recomputes from saved model states and the control is an ordinary baseline
run.
