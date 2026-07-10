<div align="center">

<h1>Construct-Aligned Disentangled Variational Autoencoders for Interpretable Buyer Personas: A Measurement-First Evaluation Framework</h1>

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
  <img src="https://img.shields.io/badge/tests-68%20passed-brightgreen" alt="tests">
  <img src="https://img.shields.io/badge/lint%20%2B%20types-ruff%20%7C%20mypy%20clean-brightgreen" alt="ruff mypy">
  <img src="https://img.shields.io/badge/seeds-6%20per%20result-8a2be2" alt="6 seeds">
  <img src="https://img.shields.io/badge/status-Phase%205%20sweep%20pending-orange" alt="status">
</p>

</div>

> **Project status (honest reporting).** All code for Phases 0–7 is implemented and tested. Every number below is recorded evidence from real runs (Phases 0–4: environment, data pipeline, constructs, baseline campaign, model sanity runs). The Phase-5 β×λ sweep (144 runs per dataset) and the Phase-6 analysis that depends on it (trade-off curve, stability, persona cards) have **not yet run**; those results are marked *pending* throughout. Nothing in this document is projected, estimated, or extrapolated.

---

**_Abstract_ — Buyer personas drive substantial marketing expenditure, yet in practice they are neither measured nor measurable: practitioners cannot say whether one persona scheme is better than another. This project reformulates persona quality as three falsifiable properties: downstream predictive lift on real behavioral labels, interpretability of latent axes with respect to named marketing constructs, and stability across random seeds and resamples. We implement a Construct-Aligned Disentangled Variational Autoencoder (CA-DVAE) whose latent space is split into a designated block trained by a linear auxiliary head to predict named constructs (recency-frequency-monetary value, price sensitivity, category affinity) and a free block regularized by a β-weighted Kullback-Leibler term. The architecture is deliberately unremarkable; the contribution is the measurement reformulation, the alignment mechanism, and a leakage-audited, multi-seed, multi-task evaluation harness spanning two public datasets (2,240 customers; 2.55 million users aggregated from 110 million events). Baselines fixed before any proposed-model run set demanding bars: PCA attains test PR-AUC 0.5677 ± 0.0729 on campaign response, and a three-feature RFM representation attains 0.1953 ± 0.0035 on future-purchase prediction, exceeding a trained autoencoder. Sanity runs confirm the alignment mechanism works (RFM axis R² = 0.90). The β×λ sweep that will trace the interpretability-performance trade-off curve is in progress.**

**_Index Terms_ — Buyer personas, customer segmentation, disentangled representation learning, interpretability, variational autoencoders.**

---

## Table of Contents

- [I. Introduction](#i-introduction)
- [II. Related Work](#ii-related-work)
- [III. Proposed Method: CA-DVAE](#iii-proposed-method-ca-dvae)
- [IV. Datasets and Construct Targets](#iv-datasets-and-construct-targets)
- [V. Evaluation Protocol](#v-evaluation-protocol)
- [VI. Recorded Results](#vi-recorded-results)
- [VII. Discussion](#vii-discussion)
- [VIII. Reproducibility and Engineering Rigor](#viii-reproducibility-and-engineering-rigor)
- [IX. Conclusion and Roadmap](#ix-conclusion-and-roadmap)
- [Acknowledgment](#acknowledgment)
- [References](#references)
- [Appendix A: Reproduction Guide](#appendix-a-reproduction-guide)
- [Appendix B: Repository Layout](#appendix-b-repository-layout)

---

## I. Introduction

Customer segmentation into "buyer personas" is a standard instrument of marketing practice, yet the instrument itself has no accepted measurement. Two persona schemes for the same customer base cannot be compared on any agreed scale; a persona deck survives or dies on narrative appeal. This is an unusual situation for a quantitative field: the representation at the center of targeting decisions is evaluated by nothing.

We treat this as a measurement problem rather than an architecture problem. A persona representation is *good*, we argue, exactly to the extent that it satisfies three properties, each of which can be measured and each of which can fail:

1. **Downstream predictive lift (primary claim).** A frozen persona representation should predict real future behavior (campaign response, next purchase, churn) at least as well as incumbent representations, under a protocol that forbids the representation from ever seeing the labels.
2. **Interpretability.** Latent axes should align with *named* marketing constructs: recency-frequency-monetary value (RFM), price sensitivity, and category affinity, quantified by per-axis alignment R², the Mutual Information Gap (MIG), and the Separated Attribute Predictability (SAP) score.
3. **Stability.** Persona assignments should persist across random seeds and bootstrap resamples, quantified by the adjusted Rand index (ARI) and normalized mutual information (NMI).

The vehicle for testing whether all three can be held simultaneously is a Construct-Aligned Disentangled VAE (CA-DVAE): a dense VAE over tabular customer features whose latent vector is partitioned into an aligned block, trained by a linear auxiliary head to predict the named constructs, and a free block carrying a β-weighted KL penalty. The VAE itself is standard machinery and is not claimed as a contribution. What the alignment weight λ and disentanglement weight β buy, and what they cost in predictive lift, is precisely the empirical question; sweeping both traces an interpretability-performance trade-off curve that this application area does not currently plot.

The contributions of this work are as follows:

- A reformulation of persona quality as three measurable, falsifiable properties, with a concrete instrument for each.
- A construct-alignment mechanism (designated latent block plus linear head) that makes multiple marketing constructs simultaneously nameable as latent directions, evaluated jointly with a β-VAE disentanglement term as two independently ablatable components.
- A leakage-audited evaluation harness: frozen-representation protocol, temporal and user-disjoint splits, train-only construct fitting, six seeds per number, paired significance tests from two families, and validation-based hyperparameter selection, all enforced by unit tests rather than by convention.
- Honest baseline findings that already complicate the standard neural-segmentation narrative (Section VI): on behavioral data, three raw RFM features beat a trained autoencoder; on tabular survey-style data, PCA beats it.

The remainder of this document is organized as follows. Section II reviews related work. Section III specifies the model and loss. Section IV describes the datasets, splits, and construct targets. Section V defines the evaluation protocol. Section VI reports all recorded results to date. Section VII discusses findings and limitations, Section VIII documents the reproducibility engineering, and Section IX states the roadmap.

## II. Related Work

**Disentangled representation learning.** The VAE [1] and its constrained variant β-VAE [2] form the base of most disentanglement work; FactorVAE [3] and the total-correlation decomposition of [4] penalize statistical dependence between latent dimensions directly. Kumar et al. [5] introduced the SAP score, and Chen et al. [4] the MIG metric adopted here. Locatello et al. [7] showed that unsupervised disentanglement is impossible without inductive bias or supervision; this project sidesteps that impossibility deliberately, since the marketing constructs provide explicit, named supervision for the aligned block, and disentanglement pressure is applied only to the residual free block.

**Neural customer segmentation.** Deep Embedded Clustering [6] and autoencoder-plus-K-means pipelines are the dominant neural approaches to customer segmentation. Published evaluations in this space typically report internal clustering indices (silhouette and relatives) on a single run; downstream predictive validation against the incumbent RFM representation [8] on held-out future behavior, with seeds and significance tests, is rare. That evaluation gap, rather than any architectural gap, is the target of this project.

**Stability of clusterings.** ARI [9] and NMI [10] are permutation-invariant partition-agreement measures; von Luxburg [11] frames bootstrap stability as a model-selection criterion for clustering. We apply both across training seeds and bootstrap resamples of the customer universe.

## III. Proposed Method: CA-DVAE

### A. Problem Formulation

Each customer is a feature vector $x \in \mathbb{R}^{D}$ aggregated from raw records (Section IV). An encoder produces a posterior $q(z|x) = \mathcal{N}(\mu(x), \sigma^2(x))$ over $z \in \mathbb{R}^{d}$ with $d = 16$; a dense decoder reconstructs $\hat{x}(z)$. The latent vector is partitioned as

$$z = [\,z_{\text{aligned}} \in \mathbb{R}^{a} \;|\; z_{\text{free}} \in \mathbb{R}^{d-a}\,]$$

where $a$ resolves by rule to $\min(n_{\text{constructs}},\, d - 4)$, keeping at least four free dimensions (Dataset A: $a = 10$; Dataset B: $a = 12$).

### B. Construct-Alignment Head

A **linear** head $A: \mathbb{R}^{a} \to \mathbb{R}^{C}$ predicts the $C$ standardized construct targets from the aligned block; the alignment loss is summed mean-squared error. Linearity is a deliberate design constraint, not a simplification: it forces each construct to correspond to an interpretable *direction* in latent space, so the per-axis alignment score and latent-traversal persona cards are well defined. A nonlinear head would reintroduce the black box the project exists to remove.

### C. Loss and Ablation Structure

$$\mathcal{L} = \underbrace{\|x - \hat{x}\|^2}_{\text{recon}} \;+\; \beta \cdot \mathrm{KL}_{\text{free}} \;+\; w_a \cdot \mathrm{KL}_{\text{aligned}} \;+\; \lambda \cdot \mathcal{L}_{\text{align}}$$

with the standard ELBO reduction (sum over dimensions, mean over batch), so β and λ of order one are meaningful. β weights only the free-block KL; the aligned block keeps a unit-Gaussian prior at fixed weight $w_a = 1$ so it stays stochastic rather than collapsing under alignment pressure. The two components are independently ablatable as pure configuration points, with no separate code paths: λ = 0 recovers a pure β-VAE (the aligned block is then dissolved), and β = 1 recovers a VAE with alignment but no extra disentanglement pressure. The frozen embedding used downstream is the posterior mean μ over all $d$ dimensions, identical in treatment to the autoencoder baseline, so any lift difference is attributable to the objective rather than the read-out.

The model is a small dense network trained comfortably within 6 GB of VRAM; no architectural novelty is claimed.

## IV. Datasets and Construct Targets

### A. Dataset A: Customer Personality Analysis

2,240 customers × 29 columns of survey-style tabular data (Kaggle, CC0-1.0) [13]. Label: response to the final marketing campaign (positive rate 14.9%); the five earlier campaign flags are legitimate strictly-past features. Cleaning is fit on train only (median imputation, winsorization at train percentiles, junk-level collapse). Splits are stratified 60/20/20 on the label, redrawn per seed. Final matrix: 2,240 × 34.

### B. Dataset B: eCommerce Behavior (Multi-Category Store)

109,950,743 raw events (October-November 2019) from the REES46 multi-category store [12], processed with Polars in streaming mode (the ~110M-row frame is never materialized; peak RAM stays bounded on a 16 GB machine). Users with ≥ 5 events in the feature window form the universe: **2.55 M users × 27 features**, cached to Parquet once (~200 s) and reused by every experiment.

Temporal separation is absolute: features aggregate the window [Oct 1, Nov 22); labels come from [Nov 22, Nov 30]; a unit test asserts every feature event precedes every label event. A skeptical-numbers pass corrected our own initial rationale here: the purchase surge in this data is Nov 16-17 (inside the feature window), not Black Friday, so the label window is a routine, promotion-unconfounded target. Train/val/test are user-disjoint (60/20/20 by seed-salted hash), blocking identity leakage. Three labels: `purchased` (primary, 3.3% positive), `churned` (dormancy proxy, 70.8% positive), and `next_category` (multiclass, first label-window purchase category).

### C. Construct Targets (Alignment Supervision)

| Construct | Dataset A (10 targets) | Dataset B (17 targets) |
|---|---|---|
| RFM (3) | Recency; Σ purchase counts; Σ spend | recency_days; n_purchases; total purchase value |
| Price sensitivity (1) | deal reliance = deals / purchases | −z(mean viewed price), a price-tier proxy* |
| Category affinity | 6 spend shares (sum to 1) | 13 known top-level event shares |

*Honest limitation, documented for the paper: the event log has no discount field, so B's price-sensitivity target measures the price tier a user browses, not discount responsiveness.

All construct targets are computed from pre-standardization tables, split with the same per-seed partition as the features, and standardized on **train only**; a perturbation unit test proves val/test rows cannot move the fitted scaler.

## V. Evaluation Protocol

### A. Frozen-Representation Protocol

Every method, baseline or proposed, yields a per-user representation fit on train only, which is then **frozen**. Two downstream heads (L2-regularized logistic regression and a histogram gradient-boosted tree classifier) are trained on frozen train features and evaluated on the held-out test set. Downstream labels never touch representation learning. Metrics: ROC-AUC, PR-AUC (primary; both binary labels are imbalanced), tie-aware precision@k (deterministic and row-order invariant), and for the multiclass task top-k accuracy plus macro one-vs-rest AUC.

### B. Baselines (All Implemented, Bar Fixed Before Any Proposed-Model Run)

RFM (continuous, 3 features), RFM + K-means, PCA at matched capacity (d = 16), autoencoder embedding (the designated hard baseline), AE + K-means, GMM on AE embeddings, and DEC [6]. The raw standardized feature vector is also run through the same heads as a reference **ceiling**, excluded from bar selection, answering the reviewer question "do you need a representation at all?".

### C. Statistical Rigor

Every headline number is mean ± std over **six seeds**, with paired significance against the strongest baseline reported from **both** test families (Wilcoxon signed-rank and paired t). Six seeds is not arbitrary: the two-sided Wilcoxon p-floor is $2^{-(n-1)}$, which equals 0.0625 at five seeds and cannot reach 0.05; at six seeds the floor is 0.03125, so both families can carry a claim.

### D. Interpretability and Stability Instruments

MIG [4] and SAP [5] follow the estimator conventions of [7], with one documented deviation: because our ground-truth factors (the construct targets) are continuous, both latents and factors are discretized by quantile (equal-mass) bins, which are robust to the heavy tails of monetary features. Ground-truth-recovery unit tests require a disentangled code to outscore a rotation-entangled code of identical information content. Stability (pending sweep): pairwise cross-seed ARI/NMI of K-means personas on a fixed seed-independent user sample, plus bootstrap persistence [11].

### E. Selection Hygiene (Pre-Compute Audit)

An audit before any sweep compute found that the sweet-spot (β, λ) point would have been selected on test metrics, the classic tuned-on-test flaw. The sweep now records validation metrics per run; selection is on validation, reporting on test, and a drift-guard test pins the dual-evaluation heads to the exact Phase-3 specification. The same audit added per-task best-baseline comparisons (avoiding straw-man significance tests) and atomic record writes.

## VI. Recorded Results

All numbers in this section are from committed, config-hashed runs on an RTX 4050 (6 GB), 6 seeds, audited protocol. Result records live under `results/` (gitignored, regenerable); the canonical copies of the tables below are also in `docs/NOTEBOOK.md`.

### A. Dataset A: Campaign Response (Test PR-AUC, Base Rate 0.150)

**TABLE I** — Baseline bar, Dataset A (6 seeds, mean ± std)

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

An honest finding, contrary to the neural-segmentation narrative: **PCA significantly beats the trained autoencoder** (paired t, p = 0.0019; Wilcoxon, p = 0.03125), and PCA is statistically indistinguishable from the raw ceiling (p = 0.19). At n = 2,240, linear compression loses nothing and the neural baseline is not the bar; PCA is. This *raised* the bar CA-DVAE must meet on Dataset A from 0.532 to 0.568.

### B. Dataset B: Future Purchase (Test PR-AUC, Base Rate 0.033)

**TABLE II** — Baseline bar, Dataset B (6 seeds, 200k train subsample, full 509k-user test set)

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

The second honest finding: **on behavioral data, three raw RFM features beat every learned representation** (every method below RFM with both test families significant; Wilcoxon p = 0.03125, t ≤ 1.5 × 10⁻⁵). The raw ceiling sits significantly above RFM, so non-RFM features carry signal that no learned representation currently captures; that headroom is exactly what the sweep probes. Hard one-hot cluster representations (the standard "persona" format) discard the most signal on both datasets.

### C. Dataset B: Multi-Task Records (the Blessed Lift Framing)

**TABLE III** — Per-task best baselines, Dataset B (6 seeds)

| Task | Metric | Best baseline | Value | RFM comparison |
|---|---|---|---|---|
| purchased (primary) | PR-AUC (gbt) | RFM | 0.1953 ± 0.0035 | — |
| churned | PR-AUC (gbt) | AE | 0.8620 ± 0.0006 | beats RFM 0.8498 every seed (t, p = 1.7 × 10⁻⁸) |
| next_category | acc@1 (logreg) | AE | 0.6097 ± 0.0020 | RFM collapses to 0.5058 (near base rate 0.5053) |

No single baseline wins all three tasks: RFM wins purchase prediction, the AE embedding wins churn and category. RFM's collapse on next_category is structural (it contains no category information). A finding directly motivating the stability axis: the plain AE embedding is **bimodal across seeds** on next_category under the tree head (macro-AUC ≈ 0.73 on seeds 0-2 versus ≈ 0.52 on seeds 3-5; acc@1 std ± 0.081), meaning the unregularized AE sometimes fails to allocate latent capacity to category structure at all. Construct anchoring is the designed remedy; Phase-6 ARI/NMI will quantify it.

### D. CA-DVAE Sanity Runs (Gate Evidence, Not Results)

Single arbitrary configuration points (β = 1, λ = 1), deliberately untuned, run only to verify the machinery end to end.

**TABLE IV** — Phase-4 sanity runs, seed 7

| Dataset | Aligned/free dims | Downstream PR-AUC (gbt) | Bar | Alignment R² (mean / RFM) |
|---|---|---|---|---|
| A (200 epochs) | 10 / 6 | 0.479 | 0.568 (PCA) | 0.71 / **0.90** |
| B (50k sample, 40 epochs) | 12 / 4 | 0.150 | 0.195 (RFM) | 0.64 / 0.68 |

Two observations. First, **the alignment mechanism works**: named axes genuinely encode the constructs (Dataset A per-axis R²: recency 0.91, frequency 0.88, monetary 0.90), which is direct evidence for the interpretability claim independent of the lift claim. Second, at this arbitrary point the model sits below both bars, exactly the predicted interpretability-performance cost; per protocol we did not tune to close the gap. Training diagnostics show the free dims collapsed to the prior at β = 1 (KL_free ≈ 0.02 on both datasets), a concrete, logged hypothesis for why the sweep grid extends β down to 0.05. A single full-scale preflight point on B (β = 1, λ = 1, 200k) reached PR-AUC 0.1726 with MIG 0.306, closing the gap to the AE (0.1723) while adding interpretability; this is one un-swept point, recorded as a lead, not a result.

### E. Pending: Sweep, Trade-Off Curve, Stability

The Phase-5 campaign is a 6 β × 4 λ × 6 seed grid (144 runs per dataset; the λ = 0 column is the β-VAE ablation, the β = 1 row is the alignment-only ablation), with validation-based sweet-spot selection and test-set reporting. The Phase-6 outputs (interpretability-performance trade-off curve, cross-seed ARI/NMI stability, bootstrap persistence, persona cards from latent traversals) consume the sweep artifacts. **None of these numbers exist yet**; this section will be populated from `results/phase5/` and `results/phase6/` when the runs land.

## VII. Discussion

**What the baselines already establish.** The evaluation gap this project targets is visible in its own baseline table. Under a frozen-representation protocol with future-behavior labels, the two incumbents most often dismissed in the neural-segmentation literature, PCA and raw RFM, are the strongest baselines on Datasets A and B respectively, and the fashionable pipelines (DEC, AE + K-means cluster one-hots) are the weakest. Any paper in this area that reports only silhouette scores on a single seed would never observe this. We hypothesize that hard cluster assignment is the principal signal destroyer: on both datasets, every K = 8 one-hot representation loses 40-65% of its parent embedding's PR-AUC.

**Implications for the proposed model.** The bars are demanding and non-neural, which we consider a feature of the study design: beating PCA at 0.568 and RFM at 0.195 with a representation that is *simultaneously* interpretable and stable is a genuinely falsifiable claim. The sanity evidence splits cleanly: the interpretability mechanism is confirmed (alignment R² up to 0.90 on named axes), while the lift question is open pending the sweep. It appears probable, from the posterior-collapse diagnostic and the full-scale preflight point, that the interesting region of the curve lies at β < 1; the data will decide.

**Limitations.** Dataset B's price-sensitivity construct is a price-tier proxy, not discount responsiveness (no discount field exists in the log). The churn label is a short-horizon dormancy proxy on a two-month log. Dataset B training uses a 200k-user subsample per seed (evaluation always uses the full test split); subsample sensitivity is a logged robustness item. All numbers were produced on one fixed Windows/RTX 4050 machine, with a committed single-platform lockfile as the authoritative environment; cross-machine reproduction uses a documented numeric tolerance.

## VIII. Reproducibility and Engineering Rigor

The repository is built so that a skeptical reviewer can re-run everything and get the same numbers:

- **Determinism.** Global seed control; deterministic cuDNN; `CUBLAS_WORKSPACE_CONFIG` pinned; DEC's nondeterministic CUDA path rewritten (matmul distance expansion, CPU permutation draws); single-threaded BLAS on every sklearn fit that feeds a representation; deterministic row order everywhere positional operations occur. Identity re-runs are bit-reproducible on the reference machine.
- **Leakage guards as unit tests.** Perturbation proofs that train-fitted transforms ignore val/test on both datasets; temporal-ordering assertions; user-disjointness; label-window exclusion; construct-scaler isolation; a drift guard pinning sweep evaluation heads to the Phase-3 specification. Full suite: **68 tests passing**, ruff and mypy clean.
- **Config-driven provenance.** Every run writes its resolved Hydra config, config hash, git commit, dirty flag, and seed. Ablations are configuration points, not code branches. Sweep record writes are atomic (a power cut cannot forge a completed run), and the sweep is resume-safe.
- **No silent fallbacks.** Requesting CUDA when it is unavailable raises an error rather than degrading to CPU.
- **Documented decisions.** Every consequential choice is a numbered entry with rationale in [docs/DECISIONS.md](docs/DECISIONS.md) (D-001 through D-033); every experiment is logged with command, hash, and observation in [docs/NOTEBOOK.md](docs/NOTEBOOK.md); live phase state is in [docs/CHECKLIST.md](docs/CHECKLIST.md).
- **Verification tooling.** `cadvae.eval.repro_check` re-runs a pinned seed and diffs every numeric metric against recorded records, exiting nonzero on mismatch.

## IX. Conclusion and Roadmap

This project replaces an unmeasured marketing artifact with three falsifiable measurements and builds the harness to take them honestly. The recorded evidence so far fixes hard, pre-registered bars (PCA 0.5677 on Dataset A; RFM 0.1953 on Dataset B), confirms the construct-alignment mechanism (named-axis R² up to 0.90), and surfaces two findings worth reporting regardless of how the proposed model fares: learned neural embeddings do not automatically beat linear or heuristic incumbents under leakage-safe future-behavior evaluation, and hard cluster personas discard most of the predictive signal their parent embeddings contain.

Remaining work, in order: (1) the Phase-5 β×λ sweep (144 runs per dataset; local overnight or Kaggle T4 x2, scripts in `cloud/`); (2) Phase-6 analysis: trade-off curve, sweet spot selected on validation, stability ARI/NMI, bootstrap persistence, persona cards; (3) Phase-7 fresh-clone reproduction verification. A correct negative result at the end of that pipeline will be reported as such.

## Acknowledgment

The author thanks REES46 Marketing Platform for the public eCommerce behavior dataset [12] and the maintainers of the Customer Personality Analysis dataset [13]. Dataset B is used under an attribution and non-redistribution posture: raw files and derived caches are never redistributed with this repository, and readers obtain the data from the original Kaggle source.

**AI disclosure (per IEEE policy on AI-generated content).** Generative AI (Claude, Anthropic) was used substantively in this project: implementation of the codebase under the phased protocol in [CLAUDE.md](CLAUDE.md), drafting of documentation, and drafting and structuring of this manuscript. All research direction, consequential scientific decisions (recorded per-decision in docs/DECISIONS.md), gate sign-offs, and accountability for the content rest with the human author. All reported numbers were produced by executed code on real data and are reproducible from the committed configurations and seeds; none were generated by a language model.

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

[14] F. Pedregosa et al., "Scikit-learn: Machine learning in Python," *J. Mach. Learn. Res.*, vol. 12, pp. 2825–2830, 2011.

[15] A. Paszke et al., "PyTorch: An imperative style, high-performance deep learning library," in *Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)*, Vancouver, BC, Canada, 2019.

---

## Appendix A: Reproduction Guide

Requires: Windows, an NVIDIA GPU (developed against an RTX 4050 laptop, 6 GB VRAM), Python 3.12.

```powershell
python -m pip install uv          # or: winget install astral-sh.uv
python -m uv sync                 # creates .venv from uv.lock (exact pinned deps, CUDA 12.4 torch)
.venv\Scripts\python.exe -m cadvae.utils.device   # GPU check — must print cuda_available: true
.venv\Scripts\python.exe -m pytest                # full test suite (leakage guards included)
```

(If `uv run` fails with a trampoline error on Windows, invoke the venv interpreter directly as above; it is the same environment.)

**Data.** Datasets are **not** committed (see Acknowledgment for the license posture). See [data/README.md](data/README.md) for download instructions and the expected layout under `data/raw/`. Then build the caches:

```powershell
# Dataset A cache is built on demand by prepare(); Dataset B needs one streaming pass (~4 min):
.venv\Scripts\python.exe -c "from omegaconf import OmegaConf; from cadvae.data.ecommerce import cache_user_table; cache_user_table(OmegaConf.load('configs/data/ecommerce.yaml'))"
```

**Pipeline, exact commands per phase.** All runs are config-driven (Hydra); every run directory records the resolved config, its hash, the git commit, dirty flag, and seed. Seeds: `eval.seeds = [0..5]` (D-023). Dataset B always takes the D-020 protocol overrides shown below.

```powershell
# Phase 3 — baselines (fixes the bar; multi-task records)
.venv\Scripts\python.exe -m cadvae.eval.run_baselines data=personality
.venv\Scripts\python.exe -m cadvae.eval.run_baselines data=ecommerce eval.train_subsample=200000 model.max_epochs=40

# Phase 4 — single-point sanity run (gate evidence, not a result)
.venv\Scripts\python.exe -m cadvae.eval.run_cadvae_sanity data=personality

# Phase 5 — the beta x lambda sweep (resumable; ~2 h for A, ~8 h for B on the 4050)
powercfg /change standby-timeout-ac 0     # once, before an overnight run
.venv\Scripts\python.exe -m cadvae.eval.run_cadvae_sweep data=personality
.venv\Scripts\python.exe -m cadvae.eval.run_cadvae_sweep data=ecommerce eval.train_subsample=200000 model.max_epochs=40
# Cloud alternative (Kaggle T4 x2): see cloud/README.md

# Phase 6 — analysis, trade-off curve, stability, persona cards (no training)
.venv\Scripts\python.exe -m cadvae.eval.run_phase6 data=personality
.venv\Scripts\python.exe -m cadvae.eval.run_phase6 data=ecommerce eval.train_subsample=200000

# Phase 7 — reproduction check (re-runs one pinned seed, diffs vs recorded records)
.venv\Scripts\python.exe -m cadvae.eval.repro_check data=personality
.venv\Scripts\python.exe -m cadvae.eval.repro_check data=ecommerce eval.train_subsample=200000 model.max_epochs=40
```

**Definition of done.** A fresh clone, following this guide, reproduces all reported numbers (mean ± std over 6 seeds), the trade-off curve, the stability analysis, and every figure, verified by `repro_check` (exit 0). The pipeline is bit-reproducible on a single machine (D-028); use `repro.atol` for cross-machine tolerance.

## Appendix B: Repository Layout

```
configs/          Hydra configs: root (eval/sweep/phase6/repro blocks), data/, model/
src/cadvae/
  data/           Polars preprocessing, leakage-safe splits, constructs (Phase 1–2)
  models/         ae.py, dec.py, cadvae.py (Phase 3–4)
  eval/           protocol, metrics, stats, interpretability, sweep + analysis runners
  viz/            figures.py — trade-off curve, stability bars, persona cards
  utils/          seeding (determinism), device (no silent CPU fallback), run logging
cloud/            Kaggle/Colab/HF sweep setup (Kaggle T4 x2 path verified)
data/             raw + interim + processed (gitignored; processed = Parquet caches)
results/          per-run evidence (gitignored; numbers recorded in docs/ + README)
tests/            pytest — leakage guards, protocol smoke, metric ground-truth, phase-6 e2e
docs/             DECISIONS.md, NOTEBOOK.md, CHECKLIST.md, data cards
```

For the full research protocol and phased execution rules, see [CLAUDE.md](CLAUDE.md).
