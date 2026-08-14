# Change Register — CA-DVAE manuscript (README.md), 2026-08-14

Red-team of the existing manuscript against the 12 attack surfaces
(`references/design-and-redteam.md`), plus the S9 novelty audit against the ledger.
Classification per the skill: **Fatal** (contribution does not survive as written) ·
**Fixable** (repairable in the manuscript or with runnable code) · **Acknowledge**
(state honestly in Risks/Threats, do not attempt to fix).

Evidence sources: `research/ledger.md` (L-0xx), `docs/NOTEBOOK.md` (Sessions 5–8),
`docs/DECISIONS.md` (D-0xx), `docs/CHECKLIST.md`.

---

## C-01 — FATAL — The manuscript omits its own results

**Attack surface:** 6 (evaluation), 2 (significance).
**Finding.** README's status block asserts the Phase-5 sweep "**has not yet run**" and
Section VI-E says "**None of these numbers exist yet**". `docs/CHECKLIST.md` records
**ALL PHASES COMPLETE 2026-07-11**: sweep 288/288 runs (Kaggle T4, commit d867713,
git_dirty=false), Phase 6 run 2 clean, Phase 7 repro bit-exact in a fresh clone, plus a
seven-item defence pack. The manuscript is ~1 month stale and describes a protocol
proposal where a completed study exists.
**Consequence.** Every reviewer-relevant claim — the trade-off curve, the primary
negative, the churn win, the stability confound — is absent. This is not a weakness of
the research; it is the manuscript failing to report it.
**Fix.** Rewrite Sections VI–IX around the recorded evidence. *Implemented.*

## C-02 — FATAL — A contribution claim the project's own evidence killed

**Attack surface:** 2 (significance), 12 (reviewer 2).
**Finding.** Intro property (3) and the CLAUDE.md design rationale present stability as
something construct anchoring is expected to buy ("Construct anchoring is the designed
remedy"). NOTEBOOK Session 8 records the opposite, verified: `rfm_kmeans` ARI
0.975/0.962, `ae_kmeans` 0.703/0.745, CA-DVAE R²-sweet-spot **0.467/0.583**. Both
incumbents are *more* stable than the selected model. The notebook logs this as
"**DEAD:** construct anchoring buys stability over incumbents."
**Consequence.** Publishing the original framing invites a reviewer to check the
stability table and find it contradicts the introduction.
**Fix.** Restate stability as a *cost axis*, not a benefit: the trade-off is three-way.
Report SMP (stability at matched performance), where the aligned model does reach
parity with the neural incumbent (A 0.697 vs 0.703; B 0.725 vs 0.745). *Implemented.*

## C-03 — FATAL — Nearest architectural neighbour uncited (novelty)

**Attack surface:** 1 (novelty).
**Finding.** The construct-alignment head is a concept bottleneck. The aligned/free
latent partition is CBM-AUC (L-002: supervised concepts **plus additional unsupervised
concepts**, trained jointly). Axis-level concept alignment is concept whitening (L-003).
None of L-001/L-002/L-003 appear in the manuscript. Corroborating [S]: CS-VAE, sisPCA,
EXoN — partitioned supervised/unsupervised latents recur across fields.
**Consequence.** A reviewer who knows the CBM literature reads Component 1 as a
relabelled, weaker version of a 2020–2022 method and rejects on novelty.
**Fix.** New Related-Work subsection naming all three; state the distinction in one
sentence (CBMs route the *task label* through the concepts and are trained on it;
CA-DVAE routes *nothing* through them — it is an unsupervised generative representation
whose axes are named, then frozen and transferred to labels it never saw). *Implemented.*

## C-04 — FATAL — The premise is overstated and the nearest work uncited

**Attack surface:** 1 (novelty), 9 (assumption).
**Finding.** "The representation at the center of targeting decisions is evaluated by
nothing" is false as written. L-006 (Salminen et al. 2021, IJHCI 37(18):1685–1708)
surveys 77 data-driven persona papers and lists **evaluation methods** as an open gap —
which supports a *narrower* claim and is far better evidence than an unsupported
assertion. L-007 (Hsu et al. 2023, IJHCS 181:103147) already creates and validates
personas by predictive accuracy.
**Consequence.** As written, one citation from a reviewer collapses the premise.
**Fix.** Narrow to the defensible claim, cite both, and state the delta explicitly
(frozen-representation transfer, leakage audit, multi-seed statistics, and
interpretability + stability as co-equal measured axes). Log the residual risk: L-007's
full text is paywalled and unread. *Implemented, with the risk stated in §XI.*

## C-05 — FIXABLE — The design's documented failure mode is untested and uncited

**Attack surface:** 6, 9. **Evidence:** L-004 (Mahinpei et al. 2021), L-005 (Margeloiu
et al. 2021) — two independent groups ⇒ structural, not a footnote.
**Finding.** Soft concepts + an unsupervised side channel is the documented
concept-leakage regime: concept representations encode information beyond the named
concepts, so *a high alignment R² does not establish that an axis carries only that
construct*. Alignment R² = 0.968 is exactly the statistic leakage inflates. The
manuscript's headline interpretability evidence is the diagonal of a matrix whose
off-diagonal is never shown.
**Fix (two parts).** (a) Cite and state the caveat. (b) Add a runnable diagnostic —
off-target R² per named axis (purity) and construct recoverability from the *free* block
alone. *Both implemented; the diagnostic is code + tests, marked not-yet-run.*

## C-06 — FIXABLE — Multiplicity is not controlled

**Attack surface:** 7 (statistical).
**Finding.** The comparison family is 8 baselines × 2 heads × 3 tasks, plus per-grid-point
sweep comparisons. Two-sided Wilcoxon at n = 6 has p-floor 0.03125 — so **every**
reported Wilcoxon win sits at or near the floor and none survives even a modest
Holm correction. The manuscript's §V-C defends the choice of 6 seeds but never names the
family.
**Fix.** Declare the family, apply Holm–Bonferroni within each (task, head, metric),
report the paired t alongside, and add test-set bootstrap CIs — seed variance is not
sampling variance. *Implemented in `stats.py` (`holm_bonferroni`, `bootstrap_ci`,
`paired_effect_size`); manuscript states the correction and its consequence.*

## C-07 — FIXABLE — The baseline that decides the paper is missing

**Attack surface:** 4 (baseline), 10 (complexity / strip test). **Highest-value item.**
**Finding.** The construct targets are *deterministic functions of the input features*
(`src/cadvae/data/constructs.py`: recency, column sums, spend shares). Therefore
`[C(x) ‖ PCA(residual)]` is a representation with named axes, **alignment R² = 1 by
construction**, zero training, and no VAE. `ALL_METHODS` in `protocol.py` contains no
such baseline. A reviewer will ask: *if you want named axes, why not just use the
constructs as coordinates?*
**Why it matters both ways.** If this baseline matches CA-DVAE, the alignment mechanism
is decoration and the strip test fails. If CA-DVAE beats it, the paper has a genuine
result — the aligned block is not merely a re-encoding of the targets. The existing
attribution result (B: aligned-12d alone 0.1722 ≈ AE full 0.1724) is suggestive but is
*not* this comparison.
**Fix.** Implement `constructs` and `construct_pca` in the protocol and pre-register the
comparison. *Code implemented + unit-tested on synthetic data; not run (no data/GPU in
this environment) and reported as pre-registered, not as a result.*

## C-08 — FIXABLE — Baseline tuning parity is not stated where the evidence exists

**Attack surface:** 4. **Finding.** CA-DVAE receives 24 configurations per seed from the
sweep; the AE receives one. That asymmetry is the single most common rejection cause.
The project *already ran* the fix (AE mini-sweep, 6 configs × 6 seeds: best AE gbt
0.5519 < pca 0.5677 ⇒ bar defended; a latent-32 AE reaches logreg parity 0.5731, not
significant vs PCA in either family) — but the manuscript never mentions it.
**Fix.** Report the mini-sweep, including the parity caveat. *Implemented.*

## C-09 — FIXABLE — Two findings that explain the results are uncited

**Attack surface:** 3 (method), 12. **Evidence:** L-008, L-009.
**Finding.** (a) Grinsztajn et al. 2022 predicts PCA/RFM + trees beating an AE at
n ≈ 2,240 — citing it converts "surprising honest finding" into "the predicted result,
now shown for representation learning in segmentation". (b) Nai et al. 2024 (AAAI):
informativeness, not dimension-wise disentanglement, predicts downstream performance —
this makes the β sweep a *directional test of a stated hypothesis* rather than an
exploration, and it predicts the observed β behaviour. Also missing: L-011 (Mancisidor
et al.), the prior "steer a customer VAE latent space with a business quantity" work
that CLAUDE.md alludes to but the manuscript never names.
**Fix.** Cite all three; pre-state the β hypothesis. *Implemented.*

## C-10 — FIXABLE — Presentation defects

**Attack surface:** 11 (reproducibility), 6.
Abstract says the sweep "is in progress" while the status block says it has not run
(both now moot per C-01) · ROC-AUC columns carry no dispersion · base rate given as
14.9 % in text and 0.150 in a table caption · references [14]/[15] are never cited ·
no figures at all · no threats-to-validity section · no ethics/privacy statement for
behavioural targeting · no computational-cost table, which matters precisely because a
three-feature baseline wins · "a trade-off curve this area does not currently plot" is
uncalibrated language the skill forbids. *All implemented.*

## C-11 — ACKNOWLEDGE — Single time origin

One feature window and one label window on Dataset B. A rolling-origin check (2–3
origins) would close the "the result is the window" attack, but it needs compute this
environment does not have. Stated in Threats to Validity with the exact experiment.

## C-12 — ACKNOWLEDGE — Seeds are not sampling units

On Dataset A the split is redrawn per seed, so seed variance mixes split variance with
initialization variance and the paired test pairs across *different test sets*. Honest
disclosure + bootstrap CIs is the right response, not a redesign. Stated.

---

## Killed — changes considered and rejected

| Candidate | Kill evidence |
|---|---|
| Add a FactorVAE total-correlation term | Strip test: the β term's effect is already measured and the manuscript's claim survives without TC. Adds a component with no ablation demand. Rejected as decoration (minimum-complexity principle). |
| Extend to 10 seeds for stronger p-values | NOTEBOOK Session 8 already rejected this: "optional stopping after observed significance is a worse attack surface than it closes." Agreed — extending seeds *after* seeing p-values is the flaw, not the fix. |
| Reframe the paper around the churn win as the headline | The primary task is a significant loss on both datasets. Leading with the one win is exactly the selective-reporting attack the project's protocol exists to prevent. Rejected. |
| Add a nonlinear alignment head to close the lift gap | Manuscript §III-B is right: linearity is what makes "named direction" well-defined. Closing the gap this way hollows out the contribution — a fatal objection wearing a disguise. |
| Drop Dataset A (weaker result) | Removing the dataset where the interpretability mechanism works best (R² 0.968) to improve the results table is result-shopping. Rejected. |

---

## Residual risk after this pass

1. **L-007 unread (paywalled).** If Hsu et al. 2023 already report a frozen, held-out,
   multi-seed persona evaluation, contribution (1) narrows to replication + extension.
   Check before submission — this is the highest single novelty risk.
2. **Marketing/IS venues not searched** (JM, JMR, Marketing Science, ICIS). Predictive
   segmentation and segment-retention work there may pre-empt the framing.
3. **"Persona stability across seeds" was not searched**, so the novelty of the
   stability axis is unassessed.
4. **C-05 and C-07 diagnostics are implemented but unrun.** They are reported in the
   manuscript as pre-registered, never as results.
