<div align="center">

<h1>What Does an Interpretable Buyer Persona Cost? A Measurement-First Evaluation of Customer Representations</h1>

<p>
  <b>Vittal Muku</b><br>
  <i>Independent Research</i><br>
  vittal.muku@gmail.com
</p>

<p>
  <img src="https://img.shields.io/badge/python-3.12-blue" alt="Python 3.12">
  <img src="https://img.shields.io/badge/PyTorch-2.6.0%2Bcu124-ee4c2c" alt="PyTorch 2.6.0+cu124">
  <img src="https://img.shields.io/badge/Polars-streaming-1f6feb" alt="Polars streaming">
  <img src="https://img.shields.io/badge/config-Hydra%20%2B%20OmegaConf-89b4fa" alt="Hydra">
  <img src="https://img.shields.io/badge/tests-82%20passed%20%7C%205%20skipped-brightgreen" alt="tests">
  <img src="https://img.shields.io/badge/lint%20%2B%20types-ruff%20%7C%20mypy%20clean-brightgreen" alt="ruff mypy">
  <img src="https://img.shields.io/badge/seeds-6%20per%20result-8a2be2" alt="6 seeds">
  <img src="https://img.shields.io/badge/sweep-288%2F288%20runs%20complete-brightgreen" alt="sweep complete">
  <img src="https://img.shields.io/badge/repro-bit--exact%20in%20fresh%20clone-brightgreen" alt="repro">
</p>

</div>

> **Provenance and honest reporting.** Phases 0–7 are complete. Every number below is
> recorded evidence from executed runs: baselines and sanity runs on the reference
> machine (RTX 4050), and the 288-run β×λ sweep on a Kaggle T4 at commit `d867713`
> (`git_dirty=false`, config hashes `973131b7` / `0a53edc0`), analysed by
> `run_phase6`. Reproduction was verified bit-exactly (`atol=1e-9`) on both datasets
> and again inside a fresh clone. Canonical copies of every table live in
> [docs/NOTEBOOK.md](docs/NOTEBOOK.md); the `results/` tree is gitignored and
> regenerable from the committed configs and seeds. Nothing here is projected,
> estimated, or extrapolated. The two pre-specified controls have now been run on Dataset A
> (Sections VI-D2, VI-E2) on the same reference machine; their Dataset B counterparts remain
> **not yet run** and are labelled as such wherever they appear.

---

**_Abstract_ — Buyer personas guide substantial marketing expenditure, yet the field's own
survey of fifteen years of data-driven persona development names evaluation as an open
problem. We reformulate persona quality as three separately measurable and separately
falsifiable properties — downstream predictive lift on real behavioural labels,
interpretability of latent axes with respect to named marketing constructs, and stability
of persona assignments across seeds — and then measure what it costs to hold them at once.
The instrument is a Construct-Aligned Disentangled Variational Autoencoder (CA-DVAE) whose
latent vector is partitioned into a block trained by a linear auxiliary head to predict
named constructs (recency-frequency-monetary value, price sensitivity, category affinity)
and a free block carrying a β-weighted Kullback-Leibler term. Neither the architecture nor
the aligned/free partition is claimed as novel; the contribution is the measurement
reformulation, a leakage-audited evaluation harness, and the resulting trade-off surface. Across two
public datasets (2,240 customers; 2.55 million users aggregated from 110 million events),
288 sweep runs and six seeds per point, the findings are largely negative and specific.
Non-neural incumbents set the bars: PCA reaches test PR-AUC 0.5677 ± 0.0729 on campaign
response and three raw RFM features reach 0.1953 ± 0.0035 on future purchase, both beating
a trained autoencoder. At its validation-selected operating point CA-DVAE reaches
block-level construct alignment R² up to 0.968 but sits significantly below both bars
(0.5130 ± 0.0597 and 0.1798 ± 0.0051), and its personas are markedly less stable than
incumbent personas (ARI 0.467 / 0.583 versus 0.975 / 0.962 for RFM+K-means). Two
pre-specified controls sharpen the negative result. A concept-leakage diagnostic shows the
alignment is a property of the aligned *block*, not of individual axes: mean best-single-axis
R² is 0.406 and mean axis purity is −0.025, so a named axis typically predicts some other
construct as well as its own. And a zero-training control that simply uses the constructs as
coordinates alongside PCA of the residual is statistically indistinguishable from the swept
neural model while holding perfect axis purity by construction. Three positive findings emerge:
against the *neural* incumbent the aligned model matches stability at matched performance
and wins dormancy prediction in both test families; on the behavioural dataset the twelve
aligned dimensions alone carry as much downstream signal as the autoencoder's full unnamed
embedding;
and Mutual-Information-Gap-based selection is shown to reverse the preferred
alignment-strength ordering on the behavioural dataset, because increasing alignment spreads
construct information across correlated axes and shrinks the top-1/top-2 gap the metric
rewards. Interpretability in this setting is a purchase, not a free lunch, and this work
prices it.**

**_Index Terms_ — Buyer personas, concept bottleneck models, customer segmentation,
disentangled representation learning, evaluation methodology, interpretability,
variational autoencoders.**

---

## Table of Contents

- [I. Introduction](#i-introduction)
- [II. Related Work](#ii-related-work)
- [III. Proposed Method: CA-DVAE](#iii-proposed-method-ca-dvae)
- [IV. Datasets and Construct Targets](#iv-datasets-and-construct-targets)
- [V. Evaluation Protocol](#v-evaluation-protocol)
- [VI. Results](#vi-results)
- [VII. Discussion](#vii-discussion)
- [VIII. Threats to Validity](#viii-threats-to-validity)
- [IX. Ethics and Responsible Use](#ix-ethics-and-responsible-use)
- [X. Reproducibility and Engineering Rigor](#x-reproducibility-and-engineering-rigor)
- [XI. Limitations and Future Work](#xi-limitations-and-future-work)
- [XII. Conclusion](#xii-conclusion)
- [Acknowledgment](#acknowledgment)
- [References](#references)
- [Appendix A: Reproduction Guide](#appendix-a-reproduction-guide)
- [Appendix B: Repository Layout](#appendix-b-repository-layout)
- [Appendix C: Result Artifacts](#appendix-c-result-artifacts)

---

## I. Introduction

Customer segmentation into "buyer personas" is a standard instrument of marketing
practice. Data-driven persona development has an active research literature of its own,
but that literature's most comprehensive survey — 77 articles spanning 2005–2020 —
lists **evaluation methods** among the field's persistent open gaps, alongside shared
resources, standardization, and inclusivity [16]. Work addressing the gap exists:
predictive personas have been created and validated by predictive accuracy on new
customers [17]. What remains scarce is a protocol under which two persona
*representations* can be compared on held-out future behaviour with the representation
provably blind to the labels, repeated across seeds, and reported with the
interpretability and stability those representations are actually chosen for.

We treat this as a measurement problem rather than an architecture problem. A persona
representation is *good*, we argue, exactly to the extent that it satisfies three
properties, each measurable and each able to fail independently:

1. **Downstream predictive lift.** A frozen persona representation should predict real
   future behaviour (campaign response, next purchase, dormancy) at least as well as
   incumbent representations, under a protocol that forbids the representation from ever
   seeing the labels.
2. **Interpretability.** The latent space should carry *named* marketing constructs —
   recency-frequency-monetary value (RFM), price sensitivity, and category affinity —
   recoverably and separably, quantified by alignment R² (both from the aligned block and
   from the best single dimension), axis purity, the Mutual Information Gap (MIG), and the
   Separated Attribute Predictability (SAP) score. Section VI-D2 shows why the block and
   axis forms of this property must be measured and reported separately: on Dataset A the
   first holds and the second does not.
3. **Stability.** Persona assignments should persist across random seeds and bootstrap
   resamples, quantified by the adjusted Rand index (ARI) and normalized mutual
   information (NMI).

The instrument for testing whether all three can be held simultaneously is a
Construct-Aligned Disentangled VAE (CA-DVAE): a dense VAE over tabular customer features
whose latent vector is partitioned into an aligned block, trained by a linear auxiliary
head to predict the named constructs, and a free block carrying a β-weighted KL penalty.
Sweeping the alignment weight λ and the disentanglement weight β traces an empirical
trade-off surface over the three properties; the Pareto-optimal subset of that surface is
what we call the frontier.

**What we found.** The result is largely negative, and the negative result is the
contribution. Under leakage-safe evaluation the strongest baselines are the two
representations the neural-segmentation literature most often treats as superseded: PCA
on tabular survey data and three raw RFM features on behavioural data. At its
validation-selected operating point CA-DVAE buys high block-level construct alignment
(R² up to 0.968) at a significant cost in predictive lift *and* a substantial cost in
stability; incumbent personas are far more stable than aligned ones. Three findings run
the other way and are reported with equal weight: against the *neural* incumbent
specifically, the aligned model matches stability at matched performance and wins
dormancy prediction in both test families; the twelve aligned dimensions alone carry as much
downstream signal on the behavioural dataset as the autoencoder's full unnamed embedding,
which counts against the "the names are decoration on a black box" reading without
settling construct purity; and MIG-based selection inverts the preferred
alignment-strength ordering when the named constructs are correlated.

The contributions of this work are as follows:

- A reformulation of persona quality as three measurable, falsifiable properties, with a
  concrete instrument for each and pre-specified bars fixed before any proposed-model
  run. Measuring persona quality by predictive accuracy is not new [17], nor is assessing
  segmentation stability by resampling agreement [28]; what is claimed is holding all
  three properties as co-equal criteria and measuring what each costs the others.
- A leakage-audited evaluation harness — frozen-representation protocol, temporal and
  user-disjoint splits, train-only construct fitting, six seeds per number, paired
  significance tests from two families with familywise correction, and validation-based
  hyperparameter selection — enforced by unit tests rather than by convention.
- An empirical interpretability–performance–stability trade-off surface for construct-aligned
  persona representations, with the cost of interpretability quantified on both axes and
  the operating point selected on validation data only — together with two pre-specified
  controls that test the mechanism rather than assume it: a concept-leakage diagnostic
  separating block-level from axis-level alignment, and a zero-training named-axis control
  that the swept model fails to beat.
- Four findings that hold regardless of how the proposed model fares: learned neural
  embeddings do not automatically beat linear or heuristic incumbents under leakage-safe
  future-behaviour evaluation; hard cluster personas discard a large fraction of their
  parent embedding's predictive signal; stability tracks model simplicity and must never be
  read apart from performance, because near-collapsed models are trivially stable; and
  MIG-based *selection* inverts with alignment strength when constructs are correlated,
  making MIG unsafe as the sole model-selection criterion in this setting.

Section II reviews related work and states what this work is not. Section III specifies
the model, Section IV the data and constructs, Section V the protocol. Section VI reports
results, Section VII discusses them, Sections VIII–IX cover validity threats and ethics,
Section X reproducibility, and Section XI limitations and residual risk.

## II. Related Work

### A. Persona Construction and Evaluation

Data-driven persona development is surveyed comprehensively by Salminen *et al.* [16],
who review 77 articles from 2005–2020 and identify evaluation methods as an open gap —
the premise this work builds on, and the reason the claim here is "evaluation is
underdeveloped", not "personas are never evaluated". Hsu *et al.* [17] are the nearest neighbour on the measurement axis, and the overlap is
substantial enough to state plainly rather than minimize. Their predictive-persona (PP)
method surveys 2,240 customers of a pet-food retailer, **partitions the data 60/40 into
training and testing sets specifically to keep the test rows out of model building**, fits
a logistic regression for purchase, and defines the persona as the predicted-buyer group.
It is validated on the held-out partition by ranking and a decile lift chart (lift 3.2 in
the first decile against a 23% naïve benchmark), and then — more convincingly than any
offline metric — against **real subsequent purchase behaviour**: of respondents given
discount coupons, 6.2% of those predicted to buy actually purchased within two weeks
versus 0.7% of those predicted not to. They benchmark this against a traditional
quantitative persona built by k-means and find the clustering personas cannot separate
buyers (39.6% versus 46.2% purchase rates across the two clusters).

**We therefore do not claim that measuring persona quality by held-out predictive
performance is new. It is theirs**, and their coupon-redemption validation has a claim to
external validity that this paper's offline labels do not.

**What remains, stated against the full text rather than inferred.** Three differences
survive, and they are structural rather than cosmetic.

*First, the object.* A PP persona is **defined by a supervised model of one chosen
outcome** — the persona *is* the positive class of that model. It is therefore
task-specific by construction and cannot be carried to a different downstream question.
This paper's object is a task-agnostic *representation*, learned without any label, frozen,
and transferred to three separate downstream tasks it never saw. That is why the
representation here can lose on purchase prediction while winning on dormancy: a
single-outcome persona has no such degrees of freedom.

*Second, the number of axes.* [17] resolves persona quality onto one metric. This paper's
thesis is that one metric is the problem: persona representations are chosen for
interpretability and relied on for stability, and a one-axis criterion cannot express what
is given up to obtain either. A frontier cannot exist in a one-axis framing.

*Third, the statistical protocol.* [17] uses a single random split with no repetition, and
benchmarks PP against the clustering persona descriptively — comparing percentages and
visually comparing response profiles. There is no repeated resampling, no variance on any
reported number, and no significance test of the persona comparison (the paper's
significance testing concerns logistic-regression coefficients). This work reports six
seeds per number with paired tests from two families, familywise correction, and a
leakage audit enforced by unit tests. Evaluating embeddings by downstream transfer is
itself routine in representation learning, including for customer embeddings
specifically [29]; what is claimed here is that protocol applied to persona
representations against incumbent *marketing* baselines.

**This work also takes up two research directions [17] explicitly proposes.** Their
future-work section suggests extending beyond a single outcome to several constructs —
naming *price-sensitive*, quality-oriented and demand-oriented customers — by "altering
the outcome variables" and repeating the procedure per construct; and it suggests
complementing survey data with **online user behaviour data**. This paper does both:
price sensitivity is one of the named constructs, and Dataset B is 110 million behavioural
events. The methodological difference is that the constructs are held **jointly in one
representation** rather than as several independent single-outcome models, which is what
makes the interpretability–performance trade-off measurable at all.

Naming the tension is also not new. Boussebough *et al.* [30] balance performance against
interpretability in multi-view customer segmentation, resolving it with a context-driven
decision matrix over internal validation indices and business KPIs. They do not measure a
frontier: there is no held-out downstream prediction and no stability analysis. The
contribution claimed here is quantifying the trade-off, not observing that one exists.

### B. Concept Supervision and Named Latent Axes

The construct-alignment head belongs to the concept-supervision family and is **not**
claimed as a novel mechanism. Concept Bottleneck Models [18] predict human-specified
concepts as an intermediate layer and then predict the label from those concepts alone.
Concept whitening [19] modifies a normalization layer so that the axes of the latent
space align with known concepts. Closest of all, CBM-AUC [20] places supervised concepts
*and additional unsupervised concepts* in the same bottleneck, trained jointly — the same
aligned/free partition used here. Partitioned supervised/unsupervised latent spaces recur
across application fields, and the design should be read as conventional.

**The distinction, in one sentence.** A concept bottleneck routes the *task label*
through the concepts and is trained on that label; CA-DVAE routes nothing through them —
it is an unsupervised generative representation whose axes happen to be named, which is
then frozen and transferred to downstream labels it never saw, so the interpretability
and the predictive claim are measured on separate footings.

**The failure mode this design inherits.** Concept-supervised models with an unsupervised
side channel are known to leak: learned concept representations encode information beyond
the pre-defined concepts, and natural mitigations do not fully work [21]; concepts may not
correspond to anything semantically meaningful in input space [22]. Two independent groups
report this, which makes it a structural property of the design rather than a footnote.
It bears directly on the headline interpretability evidence here: a high on-target
alignment R² is the diagonal of a matrix whose off-diagonal must also be shown. Section
V-F specifies the diagnostic we implemented in response, and Section VI-E reports the
attribution evidence that bears on it.

In the customer domain specifically, Mancisidor *et al.* [23] steer a VAE's latent space
with Weight of Evidence so the induced clustering reflects creditworthiness — prior art
for "steer a customer VAE with a business quantity". The delta here is multiple named
constructs held simultaneously, with the alignment strength swept rather than fixed.
Glukhov *et al.* [34] pursue interpretable customer embeddings by construction in a
transactional-banking setting, building each dimension as the distance between a user's
geographic-activity vector and a cluster centre, so that dimensions carry meaning without a
post-hoc explanation step. That is interpretability by *prototype distance*; the mechanism
studied here names axes by regression onto marketing constructs, and the question asked is
what that naming costs — a question that requires sweeping the alignment strength rather
than fixing the construction.

### C. Disentangled Representation Learning

The VAE [1] and β-VAE [2] form the base; FactorVAE [3] and the total-correlation
decomposition of [4] penalize latent dependence directly. Kumar *et al.* [5] introduced
SAP and Chen *et al.* [4] MIG, both adopted here. Locatello *et al.* [7] showed
unsupervised disentanglement is impossible without inductive bias or supervision, and
their follow-up [24] showed weak supervision suffices — so using explicit supervision, as
here, is the field's standard escape rather than a differentiator. Träuble *et al.* [33]
address the case most relevant to business constructs, which are rarely independent: across
4,260 models trained on systematically correlated data, they show the induced correlations
are learned and reflected in the latent representations. Their question is whether
correlated factors end up disentangled; the narrower question asked in Section VI-C is
whether MIG survives as a *model-selection criterion* when alignment strength is swept over
correlated constructs. Importantly for this paper's hypothesis, Nai *et al.* [25] find that
dimension-wise disentanglement is
unnecessary for downstream performance and that *informativeness* is the better predictor,
with prior positive findings explained by the correlation between the two. We use this to
pre-state a directional hypothesis for the β sweep (Section V-E) rather than exploring
blind.

### D. Neural Customer Segmentation and Tabular Baselines

Deep Embedded Clustering [6] and autoencoder-plus-K-means pipelines dominate neural
customer segmentation. Published evaluations typically report internal clustering indices
on a single run; downstream predictive validation against the incumbent RFM
representation [8] on held-out future behaviour, with seeds and significance tests, is
rare. That evaluation gap, rather than any architectural gap, is this project's target.
Our baseline findings are also consistent with a broader result: Grinsztajn *et al.* [26]
show that tree-based models remain state of the art on medium-sized tabular data
(~10k samples) even under matched tuning budgets, attributing the gap to neural
sensitivity to uninformative features, rotation non-invariance, and difficulty fitting
irregular functions. Dataset A (n = 2,240) sits squarely in that regime, so PCA and RFM
beating a trained autoencoder is the *predicted* outcome — this work contributes the
demonstration that the prediction extends to representation learning for segmentation
under a frozen-transfer protocol.

### E. Stability of Segmentation Solutions

The stability axis is **not** a criterion this paper proposes; it is an established
marketing criterion applied here to learned representations. Dolnicar and Leisch [28]
assess segmentation solutions by repeating the analysis on bootstrap samples and
measuring partition agreement with the Rand index adjusted for chance, using
reproducibility to decide whether data contain natural segments, structure without
segments, or no structure at all — and distinguishing *natural*, *reproducible*, and
*constructive* segmentation accordingly. That is the instrument used here, with training
seeds added to bootstrap resamples as a second source of variation, since a learned
representation can be unstable for reasons a fixed clustering algorithm cannot be. ARI [9]
and NMI [10] are the underlying partition-agreement measures, and von Luxburg [11] frames
bootstrap stability as a model-selection criterion in the clustering literature.

**Two different things are called persona stability, and this paper measures only one of
them.** Jansen *et al.* [32] study stability *longitudinally*: they run 32 monthly rounds
of data collection on a major publisher's YouTube channel, generate 15 data-driven personas
each month by non-negative matrix factorization, and measure how the persona set changes
over time. They find an average **40% change in the personas**, with 78% of personas showing
more change than consistency in topic interests. Their notion is temporal — whether a
persona remains representative as the underlying population evolves.

The stability measured here is *stochastic reproducibility*: the same data and the same
population, but different training seeds, plus bootstrap resampling as a separate source of
variation. The two are independent, and the distinction is not pedantic — a representation
can be perfectly reproducible on a fixed population while becoming obsolete within months
as that population changes, and [32] gives an empirical reason to expect exactly that in an
online audience. Nothing in this paper speaks to temporal stability; Dataset B spans two
months and is treated as a single population. This is stated as a limitation in Section VIII
and as the most natural follow-up in Section XI.

Section VI-D reports a confound that this framing makes easy to miss and that bears
directly on [28]'s taxonomy: a degenerate, near-collapsed representation is *trivially*
reproducible, so high stochastic stability can indicate an absence of structure rather than
the presence of natural segments. Stability alone therefore cannot select a model, which is
why Section V-D defines stability at matched performance.

## III. Proposed Method: CA-DVAE

### A. Problem Formulation

Each customer is a feature vector $x \in \mathbb{R}^{D}$ aggregated from raw records
(Section IV). An encoder produces a posterior
$q(z|x) = \mathcal{N}(\mu(x), \sigma^2(x))$ over $z \in \mathbb{R}^{d}$ with $d = 16$; a
dense decoder reconstructs $\hat{x}(z)$. The latent vector is partitioned as

$$z = [\,z_{\text{aligned}} \in \mathbb{R}^{a} \;|\; z_{\text{free}} \in \mathbb{R}^{d-a}\,]$$

where $a$ resolves by rule to $\min(n_{\text{constructs}},\, d - 4)$, keeping at least
four free dimensions (Dataset A: $a = 10$; Dataset B: $a = 12$).

### B. Construct-Alignment Head

A **linear** head $A: \mathbb{R}^{a} \to \mathbb{R}^{C}$ predicts the $C$ standardized
construct targets from the aligned block; the alignment loss is summed mean-squared
error. Linearity is a deliberate design constraint, not a simplification: it forces each
construct to correspond to an interpretable *direction* in latent space, so the per-axis
alignment score and latent-traversal persona cards are well defined. A nonlinear head
would reintroduce the black box the project exists to remove — and would also be the
easiest way to close the lift gap reported in Section VI-B while hollowing out the claim,
which is why it was rejected rather than left as future work.

### C. Loss and Ablation Structure

$$\mathcal{L} = \underbrace{\|x - \hat{x}\|^2}_{\text{recon}} \;+\; \beta \cdot \mathrm{KL}_{\text{free}} \;+\; w_a \cdot \mathrm{KL}_{\text{aligned}} \;+\; \lambda \cdot \mathcal{L}_{\text{align}}$$

with the standard ELBO reduction (sum over dimensions, mean over batch), so β and λ of
order one are meaningful. β weights only the free-block KL; the aligned block keeps a
unit-Gaussian prior at fixed weight $w_a = 1$ so it stays stochastic rather than
collapsing under alignment pressure. The two components are independently ablatable as
pure configuration points, with no separate code paths: λ = 0 recovers a pure β-VAE (the
aligned block is then dissolved), and β = 1 recovers a VAE with alignment but no extra
disentanglement pressure. The frozen embedding used downstream is the posterior mean μ
over all $d$ dimensions, identical in treatment to the autoencoder baseline, so any lift
difference is attributable to the objective rather than the read-out.

The model is a small dense network trained comfortably within 6 GB of VRAM. No
architectural novelty is claimed, and per Section II-B no novelty is claimed for the
aligned/free partition either.

## IV. Datasets and Construct Targets

### A. Dataset A: Customer Personality Analysis

2,240 customers × 29 columns of survey-style tabular data (Kaggle, CC0-1.0) [13]. Label:
response to the final marketing campaign (positive rate 14.9%); the five earlier campaign
flags are legitimate strictly-past features. Cleaning is fit on train only (median
imputation, winsorization at train percentiles, junk-level collapse). Splits are
stratified 60/20/20 on the label, redrawn per seed; the test split is 448 rows
(≈ 67 positives), which is small enough that Section V-C's interval estimates matter.
Final matrix: 2,240 × 34.

### B. Dataset B: eCommerce Behavior (Multi-Category Store)

109,950,743 raw events (October–November 2019) from the REES46 multi-category store [12],
processed with Polars in streaming mode (the ~110M-row frame is never materialized; peak
RAM stays bounded on a 16 GB machine). Users with ≥ 5 events in the feature window form
the universe: **2.55 M users × 27 features**, cached to Parquet once (~200 s) and reused
by every experiment.

Temporal separation is absolute: features aggregate the window [Oct 1, Nov 22); labels
come from [Nov 22, Nov 30]; a unit test asserts every feature event precedes every label
event. A skeptical-numbers pass corrected our own initial rationale here: the purchase
surge in this data is Nov 16–17 (inside the feature window), not Black Friday, so the
label window is a routine, promotion-unconfounded target. Train/val/test are user-disjoint
(60/20/20 by seed-salted hash), blocking identity leakage. Three labels: `purchased`
(primary, 3.3% positive), `churned` (dormancy proxy, 70.8% positive), and `next_category`
(multiclass, first label-window purchase category).

### C. Construct Targets (Alignment Supervision)

| Construct | Dataset A (10 targets) | Dataset B (17 targets) |
|---|---|---|
| RFM (3) | Recency; Σ purchase counts; Σ spend | recency_days; n_purchases; total purchase value |
| Price sensitivity (1) | deal reliance = deals / purchases | −z(mean viewed price), a price-tier proxy* |
| Category affinity | 6 spend shares (sum to 1) | 13 known top-level event shares |

*Honest limitation: the event log has no discount field, so B's price-sensitivity target
measures the price tier a user browses, not discount responsiveness.

All construct targets are computed from pre-standardization tables, split with the same
per-seed partition as the features, and standardized on **train only**; a perturbation
unit test proves val/test rows cannot move the fitted scaler.

**A property that shapes the whole evaluation.** The constructs are *deterministic
functions of the engineered features* — recency, column sums, spend shares. The alignment
head therefore predicts something already present in its own input. This matters twice:
it means the alignment task carries no new information and can only shape the *geometry*
of the latent space, and it means a representation with perfect named axes is available
without any training at all, by using the constructs as coordinates. Section V-F
pre-registers that baseline, and Section VI-E reports the attribution evidence that
addresses the circularity directly.

## V. Evaluation Protocol

### A. Frozen-Representation Protocol

Every method, baseline or proposed, yields a per-user representation fit on train only,
which is then **frozen**. Two downstream heads (L2-regularized logistic regression and a
histogram gradient-boosted tree classifier) are trained on frozen train features and
evaluated on the held-out test set. Downstream labels never touch representation
learning. Metrics: ROC-AUC, PR-AUC (primary; both binary labels are imbalanced),
tie-aware precision@k (deterministic and row-order invariant), and for the multiclass task
top-k accuracy plus macro one-vs-rest AUC.

### B. Baselines (All Implemented, Bar Fixed Before Any Proposed-Model Run)

RFM (continuous, 3 features), RFM + K-means, PCA at matched capacity (d = 16),
autoencoder embedding (the designated hard baseline), AE + K-means, GMM on AE embeddings,
and DEC [6]. The raw standardized feature vector is also run through the same heads as a
reference **ceiling**, excluded from bar selection, answering the reviewer question "do
you need a representation at all?".

**Tuning parity.** The proposed model receives 24 configurations per seed from the sweep;
a single-configuration autoencoder would be an unfairly weak comparison in the proposed
model's favour. Section VI-H reports a dedicated AE mini-sweep run to close that gap.

### C. Statistical Protocol

Every headline number is mean ± std over **six seeds**, with paired significance against
the strongest baseline reported from **both** test families (Wilcoxon signed-rank and
paired t). Six seeds is not arbitrary: the two-sided Wilcoxon p-floor is $2^{-(n-1)}$,
which equals 0.0625 at five seeds and cannot reach 0.05; at six seeds the floor is
0.03125, so both families can carry a claim.

**Multiplicity, stated explicitly.** That floor is also a hard constraint on what six
seeds can support. The comparison family here is large — up to eight challengers × two
heads × three tasks, and again per sweep grid point — and *every* Wilcoxon win in this
paper sits exactly at the floor. Under Holm–Bonferroni correction [27] within a
(task, head, metric) family, two floor-valued comparisons already give an adjusted
p = 0.0625 and none survives at α = 0.05. We therefore report: (i) raw p-values from both
families; (ii) Holm-adjusted values, implemented in `cadvae.eval.stats.holm_bonferroni`;
and (iii) paired effect sizes, because at six seeds a p-value carries little information
about magnitude while a marketing reader needs to know how much lift is lost. Claims in
Section VI that rest on the Wilcoxon floor are marked as such and are supported by the
paired t and the effect size, not by the Wilcoxon alone.

**Seeds are not sampling units.** On Dataset A the split is redrawn per seed, so seed
variance mixes initialization variance with split variance and the paired test pairs
across *different* test sets. `cadvae.eval.stats.bootstrap_ci` supplies percentile
intervals over test rows, which measure a different thing — how much of a number is an
accident of which users landed in the test split — and the two are never conflated in
Section VI.

### D. Interpretability and Stability Instruments

MIG [4] and SAP [5] follow the estimator conventions of [7], with one documented
deviation: because our ground-truth factors (the construct targets) are continuous, both
latents and factors are discretized by quantile (equal-mass) bins, which are robust to the
heavy tails of monetary features. Ground-truth-recovery unit tests require a disentangled
code to outscore a rotation-entangled code of identical information content.

Stability is pairwise cross-seed ARI/NMI of K-means personas on a fixed seed-independent
user sample, plus bootstrap persistence [11]. Because a degenerate representation is
trivially stable (Section VI-D), we also define and report **SMP — Stability at Matched
Performance**: the maximum cross-seed ARI among configurations within the selection slack
of the best validation downstream score. SMP excludes collapsed configurations by
construction and is the honest form of the stability comparison.

### E. Pre-Registered Hypotheses and Falsification Criteria

Fixed before any proposed-model run:

| # | Design expectation, recorded in advance | Would fail if | Outcome |
|---|---|---|---|
| E1 | The representation predicts behaviour better than incumbents | Sweet-spot PR-AUC below the bar under the paired test | **Not supported** on both datasets (VI-B) |
| E2 | Latent *axes* align with named constructs | Best per-axis R² near zero, or no better than an unaligned control | **Split**: supported at *block* level (R² 0.87), **not supported at axis level** — mean purity −0.025 (VI-D2) |
| E3 | Personas persist across seeds better than incumbents | Incumbent personas at least as stable | **Not supported** (VI-D) |

A further directional expectation was formed during model development, from [25]: because informativeness
predicts downstream performance better than dimension-wise disentanglement, increasing β
should *cost* lift rather than buy it.

**Is the negative result publishable?** Yes, and this was decided before running: the bars
are pre-specified, the protocol is leakage-audited, and "the interpretable model loses to
three raw features under honest evaluation" is a result the applied literature needs. The
project was designed so that its value does not depend on the effect existing.

### F. Selection Hygiene and Pre-Registered Additions

An audit before any sweep compute found that the sweet-spot (β, λ) point would have been
selected on test metrics, the classic tuned-on-test flaw. The sweep records validation
metrics per run; selection is on validation, reporting on test, and a drift-guard test
pins the dual-evaluation heads to the exact Phase-3 specification. The same audit added
per-task best-baseline comparisons (avoiding straw-man significance tests) and atomic
record writes.

Two analyses were pre-specified here and have now been **run on Dataset A** (results in
Sections VI-D2 and VI-E2); the Dataset B runs remain pending the raw event logs. They are
described here as designed, before their outcomes, because they were specified in advance:

1. **Named-axis baselines** (`constructs`, `construct_pca` in
   `cadvae.eval.representations`). Because the constructs are deterministic functions of
   the features (Section IV-C), using them directly as coordinates gives a representation
   with alignment R² = 1 by construction, no training, and — with PCA of the construct
   residual appended — the same aligned/free structure CA-DVAE learns. This is the strip
   test for the alignment mechanism: if it matches CA-DVAE, the machinery is decoration.
   It is eligible for the bar, and is given the benefit of the doubt where the construct
   count exceeds the latent budget: on Dataset B all 17 named axes are retained rather than
   truncated to 16, **and the free block keeps a floor of 4 components** (D-064), so the
   control is 21-dimensional there. Without that floor the arithmetic would hand it zero
   free components and collapse it onto `constructs` alone, which VI-E2 measures as by far
   the weaker representation. The floor is inert on Dataset A, where the budget already
   allows 6.
2. **Concept-leakage diagnostic** (`cadvae.eval.interpretability.leakage_diagnostic`).
   Following [21], [22], it reports for each construct the off-target R² of its winning
   axis (purity) and the construct's recoverability from the *free* block alone. A high
   on-target R² with a high free-block R² would mean the named axis is not where the
   information uniquely lives. Ground-truth tests verify that the diagnostic separates a
   clean partition from a leaky one *whose on-target scores are identical* — which is
   precisely the case `axis_alignment` cannot distinguish.

   It is wired into both pipelines. New sweep runs record it per point; more usefully,
   `run_phase6` recomputes it at the selected operating point from the **saved
   state_dicts**, so obtaining it does not require re-running the sweep — a Phase-6
   re-run (minutes, no training) emits `leakage.json` for both datasets. The Phase-5
   artifacts this depends on are preserved off the working tree: all 288 model
   state_dicts and 292 run records remain on the `kaggle-results` branch, from which
   `results/phase5/` can be restored in one command (Appendix A). The only remaining
   prerequisite is the processed data caches.

**Cost of closing these.** The two analyses stage unevenly, and the cheaper stage carries
most of the value. Dataset A needs a single 220 KB CC0-licensed file
(`marketing_campaign.csv`); with it, both the strip test and the leakage diagnostic run in
minutes on the existing artifacts, against the dataset where the alignment mechanism is
strongest (axis R² 0.968) and where the bar is PCA. Dataset B additionally needs ~14.7 GB
of raw event logs and one ~200 s streaming pass, and is where the strip test bites hardest,
since its 17 construct targets subsume the three RFM features that already beat every
learned representation there. Neither analysis requires retraining, a GPU, or re-running
the sweep.

## VI. Results

All numbers are from committed, config-hashed runs with 6 seeds under the audited
protocol. Baselines and repro checks ran on an RTX 4050 (6 GB); the sweep ran on a
Kaggle T4 at commit `d867713`. Canonical tables are mirrored in
[docs/NOTEBOOK.md](docs/NOTEBOOK.md).

### A. The Bars: Non-Neural Incumbents Win

**TABLE I** — Baseline bar, Dataset A, campaign response (6 seeds, mean ± std, base rate 0.149)

| Representation | PR-AUC (logreg) | PR-AUC (gbt) | ROC-AUC (gbt) |
|---|---|---|---|
| raw features (ceiling)* | 0.6148 ± 0.0347 | 0.6066 ± 0.0749 | 0.8929 |
| **PCA (d=16) — the bar** | 0.5613 ± 0.0349 | **0.5677 ± 0.0729** | 0.8746 |
| Autoencoder (d=16) | 0.5600 ± 0.0477 | 0.5156 ± 0.0682 | 0.8346 |
| GMM on AE | 0.3523 ± 0.0402 | 0.3973 ± 0.0615 | 0.7649 |
| RFM (3 features) | 0.3794 ± 0.0381 | 0.3396 ± 0.0462 | 0.7230 |
| DEC | 0.3304 ± 0.0436 | 0.3483 ± 0.0465 | 0.7080 |
| AE + K-means | 0.3384 ± 0.0524 | 0.3343 ± 0.0543 | 0.7315 |
| RFM + K-means | 0.2988 ± 0.0182 | 0.2988 ± 0.0182 | 0.7205 |

*Excluded from bar selection.

**PCA significantly beats the trained autoencoder** (paired t, p = 0.0019; Wilcoxon at the
floor, p = 0.03125), and PCA is statistically indistinguishable from the raw ceiling
(p = 0.19). At n = 2,240 linear compression loses nothing and the neural baseline is not
the bar; PCA is. This *raised* the bar CA-DVAE must meet on Dataset A from 0.532 to 0.568.
Section II-D notes this is the outcome [26] predicts for tabular data at this scale.

**TABLE II** — Baseline bar, Dataset B, future purchase (6 seeds, 200k train subsample,
full 509k-user test set, base rate 0.033)

| Representation | PR-AUC (logreg) | PR-AUC (gbt) | ROC-AUC (gbt) |
|---|---|---|---|
| raw features (ceiling)* | 0.1743 ± 0.0032 | 0.2038 ± 0.0042 | 0.8015 |
| **RFM (3 features) — the bar** | 0.1792 ± 0.0024 | **0.1953 ± 0.0035** | 0.7892 |
| Autoencoder (d=16) | 0.1675 ± 0.0045 | 0.1723 ± 0.0021 | 0.7875 |
| PCA (d=16) | 0.1405 ± 0.0079 | 0.1649 ± 0.0073 | 0.7865 |
| GMM on AE | 0.0743 ± 0.0045 | 0.1148 ± 0.0048 | 0.7480 |
| RFM + K-means | 0.1073 ± 0.0064 | 0.1073 ± 0.0064 | 0.7220 |
| DEC | 0.0708 ± 0.0047 | 0.0819 ± 0.0051 | 0.7061 |
| AE + K-means | 0.0672 ± 0.0071 | 0.0672 ± 0.0071 | 0.6341 |

*Excluded from bar selection.

**On behavioural data, three raw RFM features beat every learned representation.** The raw
ceiling sits significantly above RFM, so non-RFM features carry signal no learned
representation captures. Hard one-hot cluster representations — the standard "persona"
format — lose the largest fraction of any representation tested on both datasets, 40–65% of their parent
embedding's PR-AUC.

**TABLE III** — Per-task best baselines, Dataset B (6 seeds)

| Task | Metric | Best baseline | Value | RFM comparison |
|---|---|---|---|---|
| purchased (primary) | PR-AUC (gbt) | RFM | 0.1953 ± 0.0035 | — |
| churned | PR-AUC (gbt) | AE | 0.8620 ± 0.0006 | beats RFM 0.8498 every seed (t, p = 1.7 × 10⁻⁸) |
| next_category | acc@1 | AE + K-means | 0.5550 (stable across all 6 seeds) | RFM collapses to 0.4070, *below* the 0.5053 base rate (p = 0.017) |

No single baseline wins all three tasks, and the per-task best baseline — not a single
global bar — is what the proposed model is compared against, so that no comparison is
made against a straw man. On `next_category`, AE + K-means at 0.5550 is the real bar, above
raw features (0.4520, p = 0.021) and PCA (p = 0.020); several methods, RFM included, score
*below* the 0.5053 base rate, because the heads optimize log-loss rather than top-1
accuracy. RFM's collapse is structural: it contains no category information at all
(macro-OVR AUC 0.52 versus 0.63–0.66 for the AE family).

The plain AE embedding is **bimodal across seeds** on `next_category` (macro-OVR AUC
0.73 / 0.74 / 0.73 on seeds 0–2 versus 0.54 / 0.52 / 0.52 on seeds 3–5; acc@1 0.58 versus
0.41–0.47), meaning the unregularized AE sometimes fails to allocate latent capacity to
category structure at all. This is the observation that originally motivated construct
anchoring as a stability remedy — a motivation Section VI-D goes on to falsify.

### B. The Frontier: What Interpretability Costs in Lift

The Phase-5 campaign is a 6 β × 4 λ × 6 seed grid, 144 runs per dataset, 288 total, all
complete. The λ = 0 column is the β-VAE ablation; the β = 1 row is the alignment-only
ablation. Operating points are selected on **validation** downstream score and reported on
**test**.

**TABLE IV** — Validation-selected operating points versus the pre-specified bars

| Dataset | Sweet spot (β, λ) | Test PR-AUC (gbt) | Bar | Gap | Wilcoxon | paired t |
|---|---|---|---|---|---|---|
| A | (0.25, 4) | 0.5130 ± 0.0597 | 0.5677 (PCA) | −0.055 | 0.0312 (floor) | 3.0 × 10⁻³ |
| B | (0.10, 1) | 0.1798 ± 0.0051 | 0.1953 (RFM) | −0.016 | 0.0312 (floor) | 3.4 × 10⁻⁵ |

**E1 is not supported on both datasets.** The interpretability cost in predictive lift is
real, quantified, and significant under the paired t; the Wilcoxon values sit at the
n = 6 floor and would not survive familywise correction on their own (Section V-C), but
the t-test and the effect sizes carry the claim, and the direction is consistent across
every seed.

The trade-off surface is smooth and legible rather than a cliff. On Dataset A a weak-alignment
point (β = 0.5, λ = 0.25) reaches 0.5647 — essentially at the bar — but with RFM-axis
R² of only 0.35: the lift is recoverable precisely by giving up the naming. On Dataset B
the best point anywhere in the grid is 0.1827 ± 0.0018 at (β = 0.1, λ = 0), i.e. with the
alignment switched off entirely, and it still does not reach the RFM bar.

**the β expectation is falsified in the predicted direction.** High-β rows (β ∈ {1, 2}) are the worst on
both datasets, confirming the posterior-collapse diagnostic recorded at the Phase-4 gate
(KL_free ≈ 0.02 at β = 1) and matching the expectation set from [25]: pressure toward
dimension-wise independence costs informativeness, and informativeness is what the
downstream head needs.

### C. Construct Alignment and the Limits of MIG-Based Selection

At the selected points the aligned **block** carries the constructs almost perfectly:
Dataset A reaches mean alignment-head R² 0.854 with RFM R² **0.968**; Dataset B reaches
RFM R² up to 0.88 at λ = 4. **This is a block-level result, and Section VI-D2 shows it does
not survive translation into a claim about individual axes.** Persona cards built from latent traversals are visually consistent with the
names: Dataset A's price-sensitivity axis moves +deals / +kidhome / +recency against
−income / −premium spend, which is deal-reliance semantics; every Dataset B affinity axis
moves its own category share as the dominant feature.

**A metric finding, reported rather than hidden.** **MIG-based selection inverts the
preferred alignment-strength ordering in this construct-supervised setting.** On Dataset B,
MIG is *highest* at λ = 0 (0.26–0.35) and *drops* with alignment (0.07–0.10 at λ = 4), so
ranking models by MIG ranks them in the opposite direction to alignment quality. The cause
is structural: aligning 12 latent dimensions to 17 correlated construct targets necessarily
shares construct information across latents, and MIG's top-1-minus-top-2 gap penalizes
exactly that. Selecting on MIG chose a pure β-VAE with no named axes at all as Dataset B's
"sweet spot", producing an ablation table in which the proposed model and its own ablation
were identical rows.

This is a narrower claim than "disentanglement metrics misbehave under correlated factors",
which is already established: Träuble *et al.* [33] train 4,260 models on systematically
correlated data and show the induced correlations are learned and reflected in the latent
representations. Their question is whether correlated factors end up disentangled; ours is
whether MIG remains a valid **model-selection criterion** along a supervised
alignment-strength sweep. The failure reported here is that specific one.

It is also worth stating what MIG was built to do, because the inversion is the metric
working as designed rather than malfunctioning. Chen *et al.* [4] define the gap term so that
a factor captured by a single latent scores highly and one whose information is *also*
carried by other latents is penalised; compactness is the objective. Construct alignment
deliberately distributes each construct across a block, so the two objectives are opposed
here. Chen *et al.* further report that disentanglement remains achievable when the
generative factors are sampled dependently, so this is not a general claim about correlated
factors either. The claim is only that MIG and construct-subspace alignment are different
goals, and that selecting on the former discards the latter.
Selection was moved to mean alignment R², and a sensitivity analysis confirms the failure
is not a slack artifact: MIG-selection picks the degenerate λ = 0 point at every slack
tested, while R²-selection is slack-invariant on A and moves gracefully along the frontier
on B. Both curves are still emitted. MIG is low everywhere on Dataset A (0.02–0.05),
which is itself a caution about its usefulness for continuous, correlated business
constructs.

### D2. Concept Leakage: The Named Axes Do Not Survive the Purity Test

The alignment R² reported above is the diagonal of a matrix, and concept leakage is what
inflates a diagonal [21], [22]. Running the pre-specified diagnostic on Dataset A's
selected point (β = 0.25, λ = 4; 10 aligned / 6 free dims; 6 seeds) separates two claims
the manuscript had been treating as one.

**TABLE VIII** — Concept-leakage diagnostic, Dataset A at the selected point (6 seeds)

| Construct | Best single axis R² | Best off-target R² | Purity | Aligned **block** R² | Free block R² |
|---|---|---|---|---|---|
| rfm_R (recency) | 0.423 | 0.296 | **+0.127** | 0.989 | 0.021 |
| rfm_F (frequency) | 0.467 | 0.616 | **−0.149** | 0.988 | 0.167 |
| rfm_M (monetary) | 0.572 | 0.535 | +0.036 | 0.985 | 0.334 |
| price_sensitivity | 0.547 | 0.538 | +0.009 | 0.963 | 0.130 |
| affinity_wines | 0.532 | 0.382 | **+0.151** | 0.881 | 0.092 |
| affinity_fruits | 0.328 | 0.347 | −0.019 | 0.759 | 0.045 |
| affinity_meat | 0.277 | 0.362 | −0.085 | 0.774 | 0.054 |
| affinity_fish | 0.281 | 0.358 | −0.078 | 0.757 | 0.049 |
| affinity_sweet | 0.274 | 0.448 | **−0.174** | 0.778 | 0.045 |
| affinity_gold | 0.356 | 0.426 | −0.069 | 0.861 | 0.086 |
| **mean** | **0.406** | — | **−0.025** | **0.873** | **0.102** |

**What holds.** The aligned block *jointly* encodes the named constructs almost perfectly —
mean block R² 0.873, and 0.985–0.989 for the three RFM constructs. Any named construct can
be recovered from the ten aligned dimensions by a linear read-out. The block is a genuine
**named subspace**, and Section VI-E confirms it is predictively load-bearing.

**What does not hold.** The claim that individual *axes* are nameable fails on this
evidence. Mean best-single-axis R² is **0.406**, not 0.87, and mean purity is
**−0.025** — the axis that best predicts a given construct predicts some *other* construct
equally well or better, on average. Six of the ten constructs have negative purity;
frequency (−0.149) and sweet-affinity (−0.174) are worst. The mechanism is visible in the
assignment itself: across seeds the ten constructs claim only **5.0 distinct winning
dimensions** on average, so several constructs compete for the same axis.

Free-block recoverability is comparatively low (mean R² 0.102, leakage ratio 0.117), so
the failure is **not** the side-channel leakage of [21] — the constructs stay inside the
aligned block. It is *within-block entanglement*: the block encodes the constructs as a
distributed code rather than as separated directions.

**Consequence for the paper's claims.** Hypothesis E2 is supported at block level and
**falsified at axis level**. Reported alignment-head R² of 0.968 — the number this project
had been citing as its interpretability evidence — is a block-level regression and does not
license "there is a dimension you can point at and call recency". Persona cards built by
traversing a single axis are therefore moving a direction that mixes several constructs,
and the visual plausibility of those cards (Section VI-C) is weaker evidence than it
appears. Restating the interpretability claim as *named subspace, not named axes* is the
honest reading, and it is the measurement framework catching a failure that the standard
reporting concealed.

### D. Stability: A Cost, Not a Benefit — and a Confound

**TABLE V** — Cross-seed persona stability (ARI, K = 8, identical protocol)

| Representation | Dataset A | Dataset B |
|---|---|---|
| RFM + K-means | 0.975 ± 0.011 | 0.962 ± 0.033 |
| AE + K-means | 0.703 ± 0.068 | 0.745 ± 0.037 |
| CA-DVAE, stable region (λ ∈ [0.25, 1], low β) | 0.64–0.70 | 0.58–0.73 |
| CA-DVAE at the selected point | 0.467 | 0.583 |

**E3 is not supported.** Both incumbents produce more stable personas than the selected
aligned model. RFM+K-means is near-perfectly stable — unsurprisingly, since it clusters
three deterministic features, so stability was never RFM's weakness. Strong alignment
actively destabilizes: λ = 4 gives 0.46–0.55 across all β on Dataset A and is erratic on B
(0.29–0.75 with large standard deviations at low β). Moderate alignment (λ ∈ [0.25, 1]) at
low β is the stable-and-performant region, and the effect is non-monotone: on Dataset A,
λ = 0.25 yields ARI 0.644, *above* the β-VAE ablation's 0.541, while λ = 4 yields 0.467,
*below* it. Single-point stability claims would be reviewer-fragile in either direction,
which is why the full 24-point stability surface is reported rather than one number.

**The confound that makes raw stability comparisons meaningless.** The most stable grid
points are the most collapsed ones: Dataset A at (β = 1, λ = 0) reaches ARI 0.775 and
Dataset B at (β = 2, λ = 0) reaches 0.834 — both at the *worst* downstream PR-AUC in their
grids. A near-constant assignment is trivially reproducible. Stability must therefore be
read jointly with downstream performance, never alone. A K-sensitivity check over
K ∈ {5, 8, 12} generalizes the point: the ordering "linear incumbents (RFM/PCA, 0.73–0.99)
≫ neural methods (AE/CA-DVAE, 0.41–0.78)" holds at every K on both datasets, so
**stability tracks model simplicity**.

**SMP — the honest comparison.** Restricting to configurations within the selection slack
of the best validation score, both datasets select (β = 0.05, λ = 1), where cross-seed ARI
is **0.697 on A versus 0.703 for AE+K-means, and 0.725 on B versus 0.745**. At matched
performance, aligned personas are as stable as the neural incumbent's. They remain far
less stable than the non-neural incumbent's.

### E. Attribution: The Named Axes Are Not Decoration

Because the constructs are deterministic functions of the features (Section IV-C), the
sharpest objection to the interpretability claim is circularity: the aligned block may be
merely re-encoding its own supervision targets. Decomposing downstream performance by
latent block addresses this directly.

**TABLE VI** — Downstream PR-AUC (gbt) by latent block, at the selected points

| Dataset | Full latent | Aligned block alone | Free block alone | Reference |
|---|---|---|---|---|
| A (β=0.25, λ=4) | 0.5127 | 0.4429 (10 dims) | 0.4456 (6 dims) | raw RFM 0.3794 |
| B (β=0.10, λ=1) | 0.1798 | **0.1722** (12 dims) | 0.1447 (4 dims) | AE full 0.1723 |

On Dataset B, **the twelve aligned dimensions alone carry as much downstream signal as the
autoencoder's entire unnamed embedding**. On Dataset A the named block alone beats raw
RFM, and the two blocks are complementary rather than redundant — neither alone reaches
the full latent. The named block is therefore predictively load-bearing rather than merely
decorative — though this does not by itself establish construct purity, and the two claims
should not be conflated.

Two caveats are recorded with this table. First, it does not settle *concept leakage*:
showing that the aligned block is informative is not the same as showing that each named
axis carries only its own construct. Section VI-D2 now runs that test, and the distinction
turns out to matter — the block is informative, the individual axes are not separated. Second, the block-restricted heads reproduce the sweep records
bit-exactly on B (2.8 × 10⁻¹⁷) but deviate by 5.3 × 10⁻³ on A — cross-hardware encoding
noise (sweep embeddings from a Kaggle T4 under torch 2.10, re-encoded on the RTX 4050
under torch 2.6) amplified by A's 448-row test set. B's 509k-row pipeline absorbs it. The
magnitude is measured rather than assumed.

### E2. The Strip Test: A Zero-Training Named-Axis Control

The sharpest objection to the alignment mechanism is that the constructs are deterministic
functions of the features (Section IV-C), so a representation with named axes is available
without any training at all. Running the pre-specified control on Dataset A answers it.

**TABLE IX** — Named-axis controls versus the proposed model, Dataset A (6 seeds, gbt head)

| Representation | Test PR-AUC | Named axes | Training |
|---|---|---|---|
| raw features (ceiling)* | 0.6066 ± 0.0749 | none | none |
| **PCA (d=16) — the bar** | **0.5677 ± 0.0729** | none | none |
| `construct_pca` = [C ‖ PCA(residual)] | 0.5438 ± 0.0640 | **10, R² = 1 by construction** | **none** |
| CA-DVAE (β = 0.25, λ = 4) | 0.5130 ± 0.0597 | 10, mean axis R² 0.406 | 144-run sweep |
| `constructs` (C alone, 10 dims) | 0.4129 ± 0.0650 | 10, R² = 1 by construction | none |

*Excluded from bar selection.

Three paired comparisons over the six shared seeds, Holm-corrected within the family:

| Comparison | Δ | Cohen's d_z | Wilcoxon | paired t | Holm-adj. t |
|---|---|---|---|---|---|
| `construct_pca` − CA-DVAE | +0.0307 | +0.82 | 0.094 | 0.099 | 0.199 |
| PCA − `construct_pca` | +0.0239 | +0.54 | 0.438 | 0.241 | 0.241 |
| `construct_pca` − `constructs` | +0.1308 | +5.46 | **0.031** | **4 × 10⁻⁵** | **1.3 × 10⁻⁴** |

**The proposed model does not beat the control.** `construct_pca` is numerically ahead of
CA-DVAE on five of six seeds (+0.031 mean, d_z = 0.82) while being statistically
indistinguishable from it, and it is likewise indistinguishable from the PCA bar. The
correct statement is not "the closed-form control wins" — at n = 6 this study cannot
establish that — but the burden runs the other way: **a 144-run sweep over a neural
architecture fails to demonstrate any advantage over a representation obtained in closed
form from the same features, on the dataset where the alignment mechanism is strongest.**

Read together with Section VI-D2 the position is worse for the proposed model than the
lift numbers alone suggest. `construct_pca` has exactly one axis per construct, with
R² = 1 and perfect purity by construction — precisely the property CA-DVAE claims and, per
Table VIII, does not deliver. On Dataset A the control matches it on downstream lift and
strictly dominates it on the interpretability property both are competing on.

One positive result survives, and it is not trivial: `constructs` alone reaches only
0.4129, far below `construct_pca` (Δ = +0.131, d_z = 5.46, significant under both families
and after correction). The residual block carries real signal that the named constructs do
not. Interpretable coordinates alone are **not** sufficient; the free capacity alongside
them is doing necessary work. That finding holds for CA-DVAE's design too — it is the one
architectural intuition of this project the data support — but it does not require a VAE
to obtain.

**Scope.** This is Dataset A only. Dataset B is where the control should bite hardest,
since its 17 construct targets subsume the three RFM features that already beat every
learned representation there; that run is pending the raw event logs (Section V-F).

### F. Where the Aligned Model Wins

**TABLE VII** — Dataset B multi-task outcome at the selected point (β = 0.10, λ = 1)

| Task | CA-DVAE | Best baseline | Outcome |
|---|---|---|---|
| purchased | 0.1798 ± 0.0051 | RFM 0.1953 | **Loss** (W 0.0312, t 3.4 × 10⁻⁵) |
| churned (dormancy) | 0.8632 | AE 0.8620 | **Win**, both families (W 0.0312, t 3.6 × 10⁻⁴) |
| next_category | 0.45–0.50 acc@1 | AE + K-means 0.5550 | **Loss** (t significant; W 0.0625, one step above the floor) |

The dormancy win is real, tight across seeds, and small. In business units, precision at
the top 10% of the ranked list is 0.9312 versus 0.9300 for the AE — **about 12 additional
dormant users identified per 10,000 targeted** (W 0.0312, t 7.8 × 10⁻³). It is reported at
exactly that size. The `next_category` result is a loss, not parity: the head-level
bimodality observed in the baselines is not cured by construct anchoring.

Taken together with Section VI-D's SMP result, the defensible positive claim is narrow and
specific: **against the neural persona pipeline — the thing this literature actually
proposes — the construct-aligned model matches stability at matched performance, wins
dormancy prediction, and adds a named construct subspace at no measured cost in the primary task relative
to that pipeline.** Against the non-neural incumbents it loses on lift and stability
alike.

### G. Ablations

Both components are ablatable as configuration points. λ = 0 (β-VAE, no alignment) is the
best-performing region on Dataset B and produces no named axes; β = 1 with alignment is
dominated by low-β alignment on both datasets. The ablation therefore shows that the
alignment component *costs* primary lift while supplying the interpretability the paper is
about, and that the disentanglement component costs lift without a compensating measured
benefit in this setting — consistent with [25]. Neither component is decoration in the
strip-test sense: each changes the measured outcome, and each has its own ablation row.

### H. Baseline Tuning Parity

An AE mini-sweep on Dataset A (6 configurations × 6 seeds) was run so that the hard
baseline is not undertuned relative to the proposed model's 24 configurations per seed.
**The pre-specified bar is defended:** the best AE under the tree head reaches 0.5519,
still below PCA's 0.5677. For transparency, a tuned latent-32 AE — twice the CA-DVAE
capacity — reaches 0.5731 on the logistic head, at parity with PCA and not significantly
different in either family (Wilcoxon 0.22 / 0.44; t 0.17 / 0.86). The pre-specified bar
(gbt head) is unchanged, and the raw ceiling of 0.615 remains above everything.

## VII. Discussion

**What the study establishes.** Under a leakage-safe, frozen-representation, multi-seed
protocol on held-out future behaviour, the two representations most often treated as
superseded in the neural-segmentation literature — PCA on tabular survey data and three
raw RFM features on behavioural data — are the strongest baselines, and the fashionable
pipelines (DEC, AE + K-means one-hots) are the weakest. Any evaluation reporting only
silhouette scores on a single seed would never observe this. Hard cluster assignment is
a major signal destroyer in these experiments: every K = 8 one-hot representation loses
40–65% of its parent embedding's PR-AUC.

**Interpretability has a price, and this is what it is.** Named axes at R² ≈ 0.97 cost
0.055 PR-AUC against PCA on Dataset A and 0.016 against RFM on Dataset B, plus roughly
0.3 ARI of persona stability relative to the non-neural incumbent. The trade-off is
three-way, not two-way, and the stability leg is the one the project's own design
originally got backwards. A practitioner can now make the trade explicitly instead of
assuming it away.

**But the price may not buy anything a simpler method cannot.** The two pre-specified
controls, now run on Dataset A, both point the same way. The alignment is a property of the
aligned block rather than of individual axes (mean axis purity −0.025), so the "named axis"
reading the persona cards depend on is not supported; and a closed-form control with genuinely
one axis per construct matches the swept model's downstream lift. Where the sweep buys
something real is the residual capacity — constructs alone reach 0.4129 against 0.5438 with a
residual block — but that is an argument for pairing interpretable coordinates with free
capacity, not an argument for learning them with a VAE.

**When each representation is the right choice.** If the goal is ranking customers by
purchase propensity, use RFM: it is more accurate, more stable, free, and needs no GPU.
If the goal is a persona scheme that names its own axes, supports traversal-based
persona cards, and expresses category structure that RFM cannot represent at all, the
aligned model buys that for a measured and modest price — and does so while matching the
neural persona pipeline it replaces. If the goal is maximum accuracy irrespective of
interpretation, use the raw feature vector with a gradient-boosted head; the ceiling is
above every representation tested.

**A methodological caution for this subfield.** In this setting, MIG should not be used as
the *sole* model-selection criterion when continuous business constructs are correlated. It
inverted with alignment strength for a structural reason (Section VI-C), and selecting on it
here would have produced a paper whose proposed model was silently identical to its own
ablation. We do not claim MIG is invalid in general — only that a single sweep over
alignment strength was enough to make it select against the property it was chosen to
measure, which is reason enough to pair it with a direct alignment score. Similarly, stability must never be reported
apart from performance: the most stable configurations in both grids are the most
collapsed ones.

## VIII. Threats to Validity

**Construct validity.** The constructs are deterministic functions of the features, so the
alignment task adds no information and can only shape geometry (Section IV-C). The
attribution analysis (VI-E) counts against the strongest form of the circularity objection,
and the leakage diagnostic (VI-D2) has now been run on Dataset A: it shows the alignment is
block-level, with individual axes unseparated (mean purity −0.025). The equivalent run on
Dataset B is pending, and its 17 correlated construct targets make within-block entanglement
more likely there, not less. Dataset B's
price-sensitivity target measures browsed price tier, not discount responsiveness.

**Internal validity.** On Dataset A the split is redrawn per seed, so paired tests pair
across different test sets and seed variance mixes two sources (Section V-C). The
448-row test split makes PR-AUC genuinely unstable; bootstrap intervals are implemented
for this reason. Sweep embeddings were produced on different hardware from the
re-encoding used in VI-E, with a measured deviation of 5.3 × 10⁻³ on A.

**Statistical validity.** Every Wilcoxon result sits at the n = 6 discrete floor and none
would survive familywise correction alone; claims rest on the paired t, the effect size,
and per-seed consistency. Extending to more seeds *after* observing these p-values was
deliberately rejected as optional stopping, which would be a worse attack surface than the
one it closes.

**External validity.** The stability reported here is stochastic reproducibility on a fixed
population, and says nothing about whether a persona representation remains valid as the
population evolves. That is a real and measured risk rather than a hypothetical one: [32]
observe an average 40% change in data-driven personas across 32 monthly rounds on an online
audience. Dataset B spans two months and is treated as one population, so temporal stability
is outside this study's reach entirely. One feature window and one label window on
Dataset B — a rolling-origin replication across two or three origins would close the "the
result is the window" objection and is the single most valuable unrun experiment. Two datasets, one
domain (retail e-commerce plus a survey-style customer table), one language, one time
period. The tabular-scale regime of Dataset A is exactly where [26] predicts trees and
linear methods to win, so the A results should not be read as a general claim about
neural representation learning.

**Conclusion validity.** The dormancy win is statistically real and practically small
(≈ 12 users per 10,000). It is not evidence that the method is broadly superior, and is
not presented as such.

## IX. Ethics and Responsible Use

Both datasets are public and were used under their stated terms; Dataset B is used under
an attribution and non-redistribution posture, and neither raw files nor derived caches
are redistributed here. No personal identifiers are used: Dataset B is keyed by
pseudonymous user IDs, and all features are aggregates.

Buyer-persona representations are targeting instruments, and the interpretability this
paper measures cuts both ways: axes named "price sensitivity" make a model easier to audit
*and* easier to use for price discrimination against deal-reliant customers, who in
Dataset A co-vary with lower income and the presence of children in the household. Any
deployment should be assessed for differential treatment across protected or proxy
attributes before use; nothing in this evaluation certifies fairness, and no fairness
metric is reported. Personas derived from behavioural logs also generalize poorly to
people whose behaviour is sparse or atypical — Dataset B's universe requires ≥ 5 events,
excluding the least-active users entirely, which is a selection effect a deployed system
would inherit.

## X. Reproducibility and Engineering Rigor

- **Determinism.** Global seed control; deterministic cuDNN; `CUBLAS_WORKSPACE_CONFIG`
  pinned; DEC's nondeterministic CUDA path rewritten (matmul distance expansion, CPU
  permutation draws); single-threaded BLAS on every sklearn fit that feeds a
  representation; deterministic row order everywhere positional operations occur. Identity
  re-runs are bit-reproducible on the reference machine.
- **Verified reproduction.** `repro_check` re-runs a pinned seed and diffs every numeric
  metric against recorded records: **REPRODUCTION OK at `atol=1e-9` (bit-exact) on both
  datasets**, and again inside a **fresh clone** built from the lockfile (69 tests passing
  in the clone). The Definition-of-Done requirement is demonstrated, not assumed.
- **Leakage guards as unit tests.** Perturbation proofs that train-fitted transforms
  ignore val/test on both datasets; temporal-ordering assertions; user-disjointness;
  label-window exclusion; construct-scaler isolation; a drift guard pinning sweep
  evaluation heads to the Phase-3 specification. Suite: **82 passed, 5 skipped**, ruff and
  mypy clean across 28 source files. The one deselected test is the CUDA-availability
  check, which by design fails on a CPU-only environment. The smoke test now derives its
  expected method set from `ALL_METHODS` rather than restating it, so adding a baseline
  cannot silently desynchronize it — a regression the original hardcoded set did produce
  when the Section V-F baselines were added.
- **Cross-version portability.** Polars is pinned at 1.42.1 because Dataset B's split is a
  seed-salted Polars hash of `user_id` and that hash is not stable across versions — an
  unpinned run would silently drift the splits relative to the recorded bars. scikit-learn
  [14] is pinned at 1.9.0 for identical downstream heads, and PyTorch [15] supplies the
  encoder/decoder and the DEC baseline; both versions are recorded in every run manifest,
  because the sweep ran on a different stack (torch 2.10.0+cu128, Kaggle T4) from the
  baselines (torch 2.6.0+cu124, RTX 4050) and Section VI-E measures what that difference
  costs.
- **Config-driven provenance.** Every run writes its resolved Hydra config, config hash,
  git commit, dirty flag, and seed. Ablations are configuration points, not code branches.
  Sweep record writes are atomic and the sweep is resume-safe.
- **No silent fallbacks.** Requesting CUDA when it is unavailable raises rather than
  degrading to CPU. A live GPU matmul, not `is_available()`, gates the sweep — this caught
  a Kaggle P100 whose driver reported availability while its kernels crashed.
- **Documented decisions.** [docs/DECISIONS.md](docs/DECISIONS.md) (D-001 … D-038),
  [docs/NOTEBOOK.md](docs/NOTEBOOK.md), [docs/CHECKLIST.md](docs/CHECKLIST.md).

**Computational cost.** The full study is small, which matters given that a three-feature
baseline wins: Dataset A's 144-run sweep takes ≈ 20 min (≈ 8 s/run) and Dataset B's
≈ 7.7 h (≈ 190 s/run) on a single T4; the Dataset B feature cache is one ~200 s streaming
pass over 110M events; baselines and Phase-6 analysis add well under an hour. Total
compute for every number in this paper is under nine GPU-hours. RFM costs three column
aggregations.

## XI. Limitations and Future Work

**Limitations.** Dataset B's price-sensitivity construct is a price-tier proxy. The churn
label is a short-horizon dormancy proxy on a two-month log. Dataset B training uses a
200k-user subsample per seed (evaluation always uses the full test split). Section VIII
states the validity threats in full.

**Residual novelty risk, stated plainly.** A dedicated novelty search covering
2018–2026 and extending into the marketing and information-systems literature was run
before submission; the audit trail is in `research/ledger.md`. Its results are already
folded into Section II, and they cost this paper two claims:

- The aligned/free latent partition is prior art [20], and no novelty is claimed for it.
- Measuring persona quality by predictive accuracy is prior art [17], and no novelty is
  claimed for it.
- The stability instrument — bootstrap resampling with the adjusted Rand index — is
  established marketing methodology [28], applied here rather than proposed.
- Naming the interpretability/performance tension in segmentation is prior art [30].

What survives, on the evidence searched, is the three-axis reformulation with a measured
trade-off surface, the leakage-audited multi-seed protocol applied to persona representations, and
two methodological findings for which no prior report was found in this specific form:
MIG-based *selection* inverting the alignment-strength ordering in a construct-supervised
sweep — narrower than, and distinct from, the established result that disentanglement is
degraded by correlated factors [33] — and the collapse–stability confound.

**The largest of these risks is now closed.** [17] was obtained and read in full. It does
use a held-out partition, so the concession in Section II-A is made on the strength of the
paper itself rather than inferred: measuring persona quality on data withheld from model
building is theirs. It does **not** use repeated splits, report variance on any number, or
apply a significance test to the persona comparison, and it measures neither
interpretability nor stability — so the protocol and multi-axis deltas stated in Section
II-A are verified against the text, not argued from the absence of evidence. Its persona
is defined by a supervised model of a single outcome, which is a different object from a
task-agnostic representation and cannot be transferred across tasks. On the evidence, the
relationship is better described as continuation than collision: this work implements the
two research directions [17]'s own future-work section proposes.

A second concern was checked and dismissed. Grigorova *et al.* [31] present an automated
framework for interpretable customer segmentation in financial services, described in
secondary sources in terms close enough to construct alignment to warrant retrieval. The
full text shows the resemblance is superficial: their "alignment" applies the Hungarian
algorithm to match machine-generated *clusters* to RFM *segments*, evaluated with
silhouette, Davies–Bouldin and ARI. That is cluster-to-segment correspondence, not
alignment of latent *axes* with named constructs, and their evaluation uses internal
indices without held-out downstream prediction. It is a useful companion result — they
also report machine learning exposing heterogeneity within dormant customers that RFM
misses, which is consistent with this paper's dormancy finding (Section VI-F) — but it
does not pre-empt the mechanism.

**Future work, in priority order.** (1) Repeat the two pre-specified controls of
Section V-F on **Dataset B**, where both should bite harder: its 17 construct targets subsume
the RFM features that already beat every learned representation, and correlated targets make
within-block entanglement more likely. The Dataset A runs (VI-D2, VI-E2) are complete.
(2) **Joint stochastic and temporal stability** — measuring both properties on the same
representations, since [32] shows the temporal property is where data-driven personas
actually fail (40% average change over 32 monthly rounds) and this study measures only the
stochastic one. (3) Rolling-origin replication on Dataset B, which is the cheapest step
toward (2). (4) A third dataset from a different domain. (5) Significance testing on
stability deltas, which are currently reported as point estimates with seed spread. Not
planned: a nonlinear alignment head, which would close the lift gap by removing the property
the paper measures.

## XII. Conclusion

This work replaces an unmeasured marketing artifact with three falsifiable measurements,
builds the harness to take them honestly, and reports what the measurements say. They say
that construct-aligned personas cost predictive lift and cost stability, that the
incumbents this literature dismisses are hard to beat under honest evaluation, and that the
interpretability being purchased is real but narrower than claimed — a named *subspace*
rather than named axes, and one a closed-form control obtains without training or a
measurable loss in lift. Of the three properties the study set out to establish, none held in
the form originally stated.
The methodological findings — MIG-based selection inverting the alignment-strength ordering
under correlated constructs, and the collapse-stability confound that makes unqualified
stability comparisons meaningless — are offered as cautions to anyone evaluating persona
representations next.

One boundary should be carried forward with the result. The stability measured here is
stochastic reproducibility on a fixed population, a different property from the longitudinal
persona stability studied in prior work [32], where an average 40% change in personas was
observed across 32 monthly rounds. Whether these representations remain valid as customer
populations evolve is unanswered, and measuring both properties on the same representations
is the natural next study.

## Acknowledgment

The author thanks REES46 Marketing Platform for the public eCommerce behavior dataset [12]
and the maintainers of the Customer Personality Analysis dataset [13]. Dataset B is used
under an attribution and non-redistribution posture: raw files and derived caches are never
redistributed with this repository, and readers obtain the data from the original Kaggle
source.

**Generative AI disclosure.** Generative AI (Claude, Anthropic) was used substantively in
this work, and its use is declared here in the article as well as at submission. It was
used for: implementation of the codebase under the phased protocol in
[CLAUDE.md](CLAUDE.md); drafting of documentation; literature retrieval and verification
against source records; and drafting and structuring of this manuscript.

No AI tool is credited with authorship, and none could be: accountability for this work
rests entirely with the human author, as does all research direction, every consequential
scientific decision (recorded individually in docs/DECISIONS.md), and every phase-gate
sign-off. **No statistic in this paper was produced, estimated, or reported by a language
model.** Every number was computed by executed code on real data, logged to a
config-hashed run record, and is reproducible from the committed configurations and seeds;
the reproduction check re-runs a pinned seed and diffs each value against its record. No
figure or image in this work is AI-generated. No personal or sensitive data was submitted
to any AI platform: the datasets were processed locally and on the compute described in
Section VI, and Dataset B is keyed by pseudonymous identifiers in any case.

*Note for submission:* this statement is written to the disclosure requirements of the
target venue (Emerald, which requires AI use to be flagged in the article and at
submission, prohibits AI authorship, and prohibits reporting statistics produced by AI).
Confirm the wording against the venue's current policy at submission time, and adapt it if
submitting elsewhere — an earlier draft of this manuscript was framed to IEEE policy.

## References

[1] D. P. Kingma and M. Welling, "Auto-encoding variational Bayes," in *Proc. Int. Conf. Learn. Representations (ICLR)*, Banff, AB, Canada, 2014.

[2] I. Higgins, L. Matthey, A. Pal, C. Burgess, X. Glorot, M. Botvinick, S. Mohamed, and A. Lerchner, "β-VAE: Learning basic visual concepts with a constrained variational framework," in *Proc. Int. Conf. Learn. Representations (ICLR)*, Toulon, France, 2017.

[3] H. Kim and A. Mnih, "Disentangling by factorising," in *Proc. 35th Int. Conf. Mach. Learn. (ICML)*, Stockholm, Sweden, 2018, pp. 2649–2658.

[4] R. T. Q. Chen, X. Li, R. Grosse, and D. Duvenaud, "Isolating sources of disentanglement in variational autoencoders," in *Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)*, Montréal, QC, Canada, 2018.

[5] A. Kumar, P. Sattigeri, and A. Balakrishnan, "Variational inference of disentangled latent concepts from unlabeled observations," in *Proc. Int. Conf. Learn. Representations (ICLR)*, Vancouver, BC, Canada, 2018.

[6] J. Xie, R. Girshick, and A. Farhadi, "Unsupervised deep embedding for clustering analysis," in *Proc. 33rd Int. Conf. Mach. Learn. (ICML)*, New York, NY, USA, 2016, pp. 478–487.

[7] F. Locatello, S. Bauer, M. Lucic, G. Rätsch, S. Gelly, B. Schölkopf, and O. Bachem, "Challenging common assumptions in the unsupervised learning of disentangled representations," in *Proc. 36th Int. Conf. Mach. Learn. (ICML)*, Long Beach, CA, USA, 2019, pp. 4114–4124.

[8] A. M. Hughes, *Strategic Database Marketing*. Chicago, IL, USA: Probus, 1994.

[9] L. Hubert and P. Arabie, "Comparing partitions," *J. Classification*, vol. 2, no. 1, pp. 193–218, 1985.

[10] A. Strehl and J. Ghosh, "Cluster ensembles: A knowledge reuse framework for combining multiple partitions," *J. Mach. Learn. Res.*, vol. 3, pp. 583–617, 2002.

[11] U. von Luxburg, "Clustering stability: An overview," *Found. Trends Mach. Learn.*, vol. 2, no. 3, pp. 235–274, 2010.

[12] REES46 Marketing Platform, "eCommerce behavior data from multi category store." Kaggle. Accessed: Jul. 9, 2026. [Online]. Available: https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store

[13] A. Patel, "Customer personality analysis." Kaggle. Accessed: Jul. 9, 2026. [Online]. Available: https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis

[14] F. Pedregosa *et al.*, "Scikit-learn: Machine learning in Python," *J. Mach. Learn. Res.*, vol. 12, pp. 2825–2830, 2011.

[15] A. Paszke *et al.*, "PyTorch: An imperative style, high-performance deep learning library," in *Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)*, Vancouver, BC, Canada, 2019.

[16] J. Salminen, K. Guan, S. G. Jung, and B. J. Jansen, "A survey of 15 years of data-driven persona development," *Int. J. Human–Computer Interaction*, vol. 37, no. 18, pp. 1685–1708, 2021, doi: 10.1080/10447318.2021.1908670.

[17] P.-F. Hsu, Y.-H. Lu, S.-C. Chen, and P.-Y. Kuo, "Creating and validating predictive personas for target marketing," *Int. J. Human–Computer Studies*, vol. 181, art. 103147, 2023, doi: 10.1016/j.ijhcs.2023.103147.

[18] P. W. Koh, T. Nguyen, Y. S. Tang, S. Mussmann, E. Pierson, B. Kim, and P. Liang, "Concept bottleneck models," in *Proc. 37th Int. Conf. Mach. Learn. (ICML)*, 2020.

[19] Z. Chen, Y. Bei, and C. Rudin, "Concept whitening for interpretable image recognition," *Nature Machine Intelligence*, vol. 2, pp. 772–782, Dec. 2020.

[20] Y. Sawada and K. Nakamura, "Concept bottleneck model with additional unsupervised concepts," *IEEE Access*, 2022. [Online]. Available: https://arxiv.org/abs/2202.01459

[21] A. Mahinpei, J. Clark, I. Lage, F. Doshi-Velez, and W. Pan, "Promises and pitfalls of black-box concept learning models," 2021. [Online]. Available: https://arxiv.org/abs/2106.13314

[22] A. Margeloiu, M. Ashman, U. Bhatt, Y. Chen, M. Jamnik, and A. Weller, "Do concept bottleneck models learn as intended?", 2021. [Online]. Available: https://arxiv.org/abs/2105.04289

[23] R. A. Mancisidor, M. Kampffmeyer, K. Aas, and R. Jenssen, "Learning latent representations of bank customers with the variational autoencoder," 2019. [Online]. Available: https://arxiv.org/abs/1903.06580

[24] F. Locatello, B. Poole, G. Rätsch, B. Schölkopf, O. Bachem, and M. Tschannen, "Weakly-supervised disentanglement without compromises," in *Proc. 37th Int. Conf. Mach. Learn. (ICML)*, 2020.

[25] R. Nai, Z. Wen, J. Li, Y. Li, and Y. Gao, "Revisiting disentanglement in downstream tasks: A study on its necessity for abstract visual reasoning," in *Proc. AAAI Conf. Artificial Intelligence*, 2024.

[26] L. Grinsztajn, E. Oyallon, and G. Varoquaux, "Why do tree-based models still outperform deep learning on tabular data?", in *Proc. Adv. Neural Inf. Process. Syst. (NeurIPS) Datasets and Benchmarks Track*, 2022.

[27] S. Holm, "A simple sequentially rejective multiple test procedure," *Scandinavian J. Statistics*, vol. 6, no. 2, pp. 65–70, 1979.

[28] S. Dolnicar and F. Leisch, "Evaluation of structure and reproducibility of cluster solutions using the bootstrap," *Marketing Letters*, vol. 21, no. 1, pp. 83–101, 2010, doi: 10.1007/s11002-009-9083-4.

[29] J. H. Bertrand, D. B. Hoffmann, J. P. Gargano, L. Mombaerts, and J. Taws, "Autoencoder-based general purpose representation learning for customer embedding," 2024. [Online]. Available: https://arxiv.org/abs/2402.18164

[30] I. Boussebough, K. Zarour, C. Aouabdia, and D. S. Boutina, "Multi-view customer segmentation in the digital economy: Balancing performance and interpretability for actionable insights," *J. Telecommunications and the Digital Economy*, vol. 14, no. 2, pp. 58–83, 2026, doi: 10.18080/jtde.v14n2.1462.

[31] I. Grigorova, A. S. Efremov, and A. Karamfilov, "An automated machine learning framework for interpretable customer segmentation in financial services," *Int. J. Financial Studies*, vol. 13, no. 4, art. 243, 2025, doi: 10.3390/ijfs13040243.

[32] B. J. Jansen, S. Jung, S. A. Chowdhury, and J. Salminen, "Persona analytics: Analyzing the stability of online segments and content interests over time using non-negative matrix factorization," *Expert Systems with Applications*, vol. 185, art. 115611, 2021, doi: 10.1016/j.eswa.2021.115611.

[33] F. Träuble, E. Creager, N. Kilbertus, F. Locatello, A. Dittadi, A. Goyal, B. Schölkopf, and S. Bauer, "On disentangled representations learned from correlated data," in *Proc. 38th Int. Conf. Mach. Learn. (ICML)*, PMLR 139, 2021, pp. 10401–10412.

[34] G. Glukhov, P. Zhdanov, and E. Shikov, "Interpretable embeddings for geographic transactional activity analysis," *Procedia Computer Science*, vol. 229, pp. 357–366, 2023, doi: 10.1016/j.procs.2023.12.038.

---

## Appendix A: Reproduction Guide

Requires: Windows, an NVIDIA GPU (developed against an RTX 4050 laptop, 6 GB VRAM),
Python 3.12.

```powershell
python -m pip install uv          # or: winget install astral-sh.uv
python -m uv sync                 # creates .venv from uv.lock (exact pinned deps, CUDA 12.4 torch)
.venv\Scripts\python.exe -m cadvae.utils.device   # GPU check — must print cuda_available: true
.venv\Scripts\python.exe -m pytest                # full test suite (leakage guards included)
```

(If `uv run` fails with a trampoline error on Windows, invoke the venv interpreter
directly as above; it is the same environment.)

**Data.** Datasets are **not** committed (see Acknowledgment for the license posture). See
[data/README.md](data/README.md) for download instructions and the expected layout under
`data/raw/`. Then build the caches:

```powershell
# Dataset A cache is built on demand by prepare(); Dataset B needs one streaming pass (~4 min):
.venv\Scripts\python.exe -c "from omegaconf import OmegaConf; from cadvae.data.ecommerce import cache_user_table; cache_user_table(OmegaConf.load('configs/data/ecommerce.yaml'))"
```

**Pipeline, exact commands per phase.** All runs are config-driven (Hydra); every run
directory records the resolved config, its hash, the git commit, dirty flag, and seed.
Seeds: `eval.seeds = [0..5]` (D-023). Dataset B always takes the D-020 protocol overrides
shown below.

```powershell
# Phase 3 — baselines (fixes the bar; multi-task records)
.venv\Scripts\python.exe -m cadvae.eval.run_baselines data=personality
.venv\Scripts\python.exe -m cadvae.eval.run_baselines data=ecommerce eval.train_subsample=200000 model.max_epochs=40

# Phase 4 — single-point sanity run (gate evidence, not a result)
.venv\Scripts\python.exe -m cadvae.eval.run_cadvae_sanity data=personality

# Phase 5 — the beta x lambda sweep (resumable; ~20 min for A, ~7.7 h for B on a T4)
powercfg /change standby-timeout-ac 0     # once, before an overnight run
.venv\Scripts\python.exe -m cadvae.eval.run_cadvae_sweep data=personality
.venv\Scripts\python.exe -m cadvae.eval.run_cadvae_sweep data=ecommerce eval.train_subsample=200000 model.max_epochs=40
# Cloud alternative (Kaggle T4 — the path these results used): see cloud/README.md

# Phase 6 — analysis, trade-off curve, stability, persona cards (no training)
.venv\Scripts\python.exe -m cadvae.eval.run_phase6 data=personality
.venv\Scripts\python.exe -m cadvae.eval.run_phase6 data=ecommerce eval.train_subsample=200000

# Phase 7 — reproduction check (re-runs one pinned seed, diffs vs recorded records)
.venv\Scripts\python.exe -m cadvae.eval.repro_check data=personality
.venv\Scripts\python.exe -m cadvae.eval.repro_check data=ecommerce eval.train_subsample=200000 model.max_epochs=40
```

**Pre-specified controls (Section V-F).** Both have been run on Dataset A; the commands
below reproduce that, and the same commands with `data=ecommerce` complete Dataset B once
its raw logs are present. Neither needs new infrastructure or a re-sweep.

The Phase-5 artifacts live on the `kaggle-results` branch (288 model state_dicts, 292 run
records), not in the working tree. Restore them first — this is a download, not a
computation:

```powershell
mkdir results -Force
git fetch origin kaggle-results
git archive origin/kaggle-results | tar -x -C results
# -> results/phase5/{personality,ecommerce}_sweep/{records,models}/
```

```powershell
# 1. Named-axis baselines - the strip test. Same eval.methods mechanism as any baseline.
.venv\Scripts\python.exe -m cadvae.eval.run_baselines data=personality "eval.methods=[raw,pca,rfm,constructs,construct_pca,ae]"
.venv\Scripts\python.exe -m cadvae.eval.run_baselines data=ecommerce eval.train_subsample=200000 model.max_epochs=40 "eval.methods=[raw,pca,rfm,constructs,construct_pca,ae]"

# 2. Concept-leakage diagnostic - recomputed from the SAVED sweep models, so this is a
#    Phase-6 re-run (minutes, no training), NOT a re-run of the 8-hour sweep.
#    Emits leakage.json next to analysis.json for each dataset.
.venv\Scripts\python.exe -m cadvae.eval.run_phase6 data=personality
.venv\Scripts\python.exe -m cadvae.eval.run_phase6 data=ecommerce eval.train_subsample=200000
```

**Definition of done.** A fresh clone, following this guide, reproduces all reported
numbers (mean ± std over 6 seeds), the trade-off curves, the stability analysis, and every
figure, verified by `repro_check` (exit 0). Verified 2026-07-11 on both datasets and in a
fresh clone. The pipeline is bit-reproducible on a single machine (D-028); use
`repro.atol` for cross-machine tolerance.

## Appendix B: Repository Layout

```
configs/          Hydra configs: root (eval/sweep/phase6/repro blocks), data/, model/
src/cadvae/
  data/           Polars preprocessing, leakage-safe splits, constructs (Phase 1–2)
  models/         ae.py, dec.py, cadvae.py (Phase 3–4)
  eval/           protocol, representations, metrics, stats, interpretability,
                  sweep + analysis runners
  viz/            figures.py — trade-off curve, stability bars, persona cards
  utils/          seeding (determinism), device (no silent CPU fallback), run logging
cloud/            Kaggle/Colab/HF sweep setup (Kaggle T4 path verified — used for the sweep)
data/             raw + interim + processed (gitignored; processed = Parquet caches)
research/         literature ledger, change register, decision log for the manuscript
results/          per-run evidence (gitignored; numbers recorded in docs/ + README)
tests/            pytest — leakage guards, protocol smoke, metric ground-truth, phase-6 e2e
docs/             DECISIONS.md, NOTEBOOK.md, CHECKLIST.md, data cards
```

## Appendix C: Result Artifacts

Regenerated by `run_phase6` into `results/phase6/{personality,ecommerce}_analysis/`
(gitignored; numbers mirrored in [docs/NOTEBOOK.md](docs/NOTEBOOK.md)):

| Artifact | Contents | Paper reference |
|---|---|---|
| `analysis.json` | Aggregated sweep, Pareto frontier, sweet-spot selection, paired tests | VI-B |
| `tradeoff_r2.png` / `tradeoff_mig.png` | Interpretability–performance curves on both x-axes | VI-B, VI-C |
| `stability.png`, `stability_map.{json,png}` | Cross-seed ARI/NMI at the selected point and over all 24 grid points | VI-D |
| `frontier3.png` | Three-way frontier: downstream × alignment R² × cross-seed ARI | VI-B, VI-D |
| `persona_cards.png` | Latent traversals along each best-aligned dimension, standardized deltas with raw-unit annotations (read with VI-D2: these directions are not construct-pure) | VI-C |
| `attribution.json` | Downstream performance by latent block (aligned / free / full) | VI-E |
| `smp.json` | Stability at Matched Performance | VI-D |
| `leakage.json` | Concept-leakage diagnostic at the selected point: axis purity, off-target R², free-block recoverability | **VI-D2 (Dataset A run)** |
| `baseline_stability.json` | Cross-seed ARI for incumbent persona pipelines | VI-D |
| `stability_k_sensitivity.json` | Stability over K ∈ {5, 8, 12} | VI-D |
| `selection_sensitivity.json` | Sweet-spot robustness to the selection slack | VI-C |
| `churn_business.json` | Dormancy win in precision@10% and users per 10,000 | VI-F |
| `ae_minisweep.json` | Baseline tuning-parity check | VI-H |
| `multitask.md`, `ablations.md` | Per-task and ablation tables | VI-F, VI-G |

For the full research protocol and phased execution rules, see [CLAUDE.md](CLAUDE.md).
