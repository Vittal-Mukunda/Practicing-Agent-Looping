# Research Ledger — CA-DVAE manuscript review (2026-08-14)

Verification tags: `[V]` = abstract page or full text actually fetched this session ·
`[S]` = search snippet only (discovery/navigation, may not support a novelty claim).

---

### L-001 [V]
Cite:    P. W. Koh, T. Nguyen, Y. S. Tang, S. Mussmann, E. Pierson, B. Kim, P. Liang,
         "Concept Bottleneck Models", ICML 2020.
URL:     https://arxiv.org/abs/2007.04612
Fetched: abstract page
Method:  Two-stage: input → predicted human-specified concepts → label predicted *from
         the concepts only*. Supports test-time concept intervention.
Results: "competitive accuracy with standard end-to-end models"; accuracy improves
         when humans correct concepts at test time.
SUPPORTING: Establishes the canonical framing of "designated units trained to predict
         named concepts" — the manuscript's Component 1 is an instance of this family.
DISCONFIRMING: **This is the manuscript's nearest architectural neighbour and it is not
         cited.** A reviewer who knows CBMs will read the construct-alignment head as a
         relabelled concept bottleneck. Novelty must be argued against it explicitly.

### L-002 [V]
Cite:    Y. Sawada, K. Nakamura, "Concept Bottleneck Model with Additional Unsupervised
         Concepts", 2022 (IEEE Access; arXiv:2202.01459).
URL:     https://arxiv.org/abs/2202.01459
Fetched: abstract page
Method:  CBM-AUC — bottleneck holds **supervised concepts plus additional unsupervised
         concepts** trained jointly (unsupervised part via self-explaining networks).
SUPPORTING: —
DISCONFIRMING: **This is the closest published match to the CA-DVAE latent partition**
         (named/aligned block + free/unsupervised block, trained jointly). The
         *partition itself* is therefore not novel. The manuscript's novelty must rest
         on the application/measurement framing, not the mechanism — which the
         manuscript already half-concedes but does not evidence.

### L-003 [V]
Cite:    Z. Chen, Y. Bei, C. Rudin, "Concept Whitening for Interpretable Image
         Recognition", *Nature Machine Intelligence*, vol. 2, pp. 772–782, Dec. 2020.
URL:     https://arxiv.org/abs/2002.01650
Fetched: abstract page
Method:  Normalize + decorrelate the latent space so that "the axes of the latent space
         are aligned with known concepts of interest". Drop-in for batch norm.
SUPPORTING: —
DISCONFIRMING: A second independent prior mechanism for "make latent *axes* nameable".
         Notably it achieves axis alignment *with decorrelation*, which is what the
         manuscript's β term is trying to buy separately.

### L-004 [V]
Cite:    A. Mahinpei, J. Clark, I. Lage, F. Doshi-Velez, W. Pan, "Promises and Pitfalls
         of Black-Box Concept Learning Models", 2021 (arXiv:2106.13314).
URL:     https://arxiv.org/abs/2106.13314
Fetched: abstract page
Results: Concept representations "encode information beyond the pre-defined concepts,
         and natural mitigation strategies do not fully work, rendering the
         interpretation of the downstream prediction misleading."
SUPPORTING: Gives the manuscript a *known, named, citable failure mode* it can test for
         and thereby strengthen its interpretability claim.
DISCONFIRMING: **Directly attacks the manuscript's headline interpretability evidence.**
         Alignment R² = 0.90 on a named axis is exactly the statistic that concept
         leakage inflates; a high diagonal does not establish that the axis carries
         *only* that construct. Soft concepts + a free side channel is the documented
         leakage regime.

### L-005 [V]
Cite:    A. Margeloiu, M. Ashman, U. Bhatt, Y. Chen, M. Jamnik, A. Weller, "Do Concept
         Bottleneck Models Learn as Intended?", 2021 (arXiv:2105.04289).
URL:     https://arxiv.org/abs/2105.04289
Fetched: listing page + abstract via search of the arXiv record
Results: Learned concepts "do not correspond to anything semantically meaningful in
         input space"; CBMs struggle to meet interpretability/predictability/
         intervenability simultaneously.
SUPPORTING: Second independent group documenting the same structural limitation as
         L-004 → this is a **recurring, structural** property, not a footnote.
DISCONFIRMING: Same attack surface as L-004.

### L-006 [V]
Cite:    J. Salminen, K. Guan, S. G. Jung, B. J. Jansen, "A Survey of 15 Years of
         Data-Driven Persona Development", *Int. J. Human–Computer Interaction*,
         vol. 37, no. 18, pp. 1685–1708, 2021. DOI 10.1080/10447318.2021.1908670
URL:     https://research.tudelft.nl/en/publications/a-survey-of-15-years-of-data-driven-persona-development/
Fetched: bibliographic page with verbatim abstract
Results: Reviews 77 data-driven persona articles 2005–2020; names remaining gaps in
         "(a) shared resources, (b) **evaluation methods**, (c) standardization,
         (d) inclusivity, (e) risk of losing in-depth user insights."
SUPPORTING: **Independent, citable evidence for the manuscript's core premise** — an
         evaluation gap in persona development, stated by the field's own survey.
DISCONFIRMING: Also shows the manuscript's stronger phrasing ("evaluated by nothing")
         is false: an active literature *does* evaluate personas. The premise must be
         narrowed to the defensible version.

### L-007 [V — bibliographic]
Cite:    P.-F. Hsu, Y.-H. Lu, S.-C. Chen, P.-Y. Kuo, "Creating and validating predictive
         personas for target marketing", *Int. J. Human–Computer Studies*, vol. 181,
         art. 103147, 2023. DOI 10.1016/j.ijhcs.2023.103147
URL:     https://api.openalex.org/works/doi:10.1016/j.ijhcs.2023.103147 (metadata);
         publisher page 403s, full text not retrieved.
Fetched: OpenAlex metadata record (authors/venue/volume/year verified). Abstract NOT
         retrieved — content characterization below is `[S]` from search snippets and
         must be checked against the full text before any sharp claim rests on it.
SUPPORTING: —
DISCONFIRMING: **The nearest neighbour to the manuscript's central "measurement
         reformulation" contribution.** Snippets describe a persona method validated by
         *predictive accuracy* on new customers. The manuscript's property (1) is
         therefore not an unprecedented reformulation; its defensible delta is the
         frozen-representation protocol, the leakage audit, multi-seed statistics, and
         the addition of interpretability + stability as co-equal measured axes.

### L-008 [V]
Cite:    L. Grinsztajn, E. Oyallon, G. Varoquaux, "Why do tree-based models still
         outperform deep learning on tabular data?", NeurIPS 2022 Datasets & Benchmarks
         (arXiv:2207.08815).
URL:     https://arxiv.org/abs/2207.08815
Fetched: abstract page
Results: 45 datasets, tuned comparison; "tree-based models remain state-of-the-art on
         medium-sized data (~10K samples)". Causes: NN sensitivity to uninformative
         features, rotation non-invariance, difficulty with irregular functions.
SUPPORTING: **Explains the manuscript's own headline baseline finding** (PCA > AE at
         n = 2,240; RFM + GBT beats learned embeddings). Converts a "surprising honest
         result" into "the predicted result, now demonstrated for representation
         learning in segmentation". Also supplies the tuned-baseline methodology bar.

### L-009 [V]
Cite:    R. Nai, Z. Wen, J. Li, Y. Li, Y. Gao, "Revisiting Disentanglement in Downstream
         Tasks: A Study on Its Necessity for Abstract Visual Reasoning", AAAI 2024
         (arXiv:2403.00352).
URL:     https://arxiv.org/abs/2403.00352
Fetched: abstract page
Results: Dimension-wise disentanglement is unnecessary downstream; "the informativeness
         of representations is a better indicator of downstream performance than
         disentanglement"; prior positive findings explained by the correlation between
         informativeness and disentanglement.
SUPPORTING: Lets the manuscript **pre-register a directional hypothesis** for the β
         sweep (β ↑ ⇒ informativeness ↓ ⇒ lift ↓) instead of exploring blind.
DISCONFIRMING: Weakens any implicit expectation that the β term will *help* lift.

### L-010 [V]
Cite:    F. Locatello, B. Poole, G. Rätsch, B. Schölkopf, O. Bachem, M. Tschannen,
         "Weakly-Supervised Disentanglement Without Compromises", ICML 2020
         (arXiv:2002.02886).
URL:     https://arxiv.org/abs/2002.02886 (record confirmed via ICML/PMLR listing)
Fetched: listing + abstract via search of the arXiv/PMLR record
Results: Weak supervision (pairs sharing factors) suffices for disentanglement; large
         empirical study across benchmarks.
SUPPORTING: The manuscript already cites Locatello et al. 2019 (impossibility) to
         justify supervision; this is the constructive follow-up and completes that
         argument — supervision is the standard escape, so using it is conventional,
         not a differentiator.

### L-011 [V]
Cite:    R. A. Mancisidor, M. Kampffmeyer, K. Aas, R. Jenssen, "Learning Latent
         Representations of Bank Customers With The Variational Autoencoder",
         arXiv:1903.06580, 2019 (journal version: *Expert Systems with Applications*).
URL:     https://arxiv.org/abs/1903.06580
Fetched: abstract page
Method:  VAE whose latent space is **steered** using Weight of Evidence so the induced
         clustering reflects customer creditworthiness.
SUPPORTING: —
DISCONFIRMING: The "steer a customer VAE latent space with a business quantity" idea is
         prior art in the finance/marketing application area. CLAUDE.md refers to it as
         "prior single-axis latent steering" but the manuscript never cites it. Must be
         named for the multi-construct delta to be legible.

---

## Terminology map

CONCEPT: construct-alignment head
  synonyms: concept bottleneck · concept supervision · attribute supervision ·
            latent steering · semantic axis alignment · concept whitening
  formal:   supervised latent subspace / partitioned latent variable model
            (cf. Kingma M2 semi-supervised VAE: label block + style block)
  applied:  "named axes", "interpretable factors", "business constructs"
  adjacent: XAI (CBM stream), neuroscience (constrained-subspace VAE),
            single-cell genomics (supervised independent subspace PCA)

CONCEPT: persona quality
  synonyms: persona validation · persona evaluation · segment quality ·
            predictive persona · segmentation validity
  adjacent: HCI (data-driven personas), marketing science (segment retention),
            representation learning (linear probing / frozen-feature transfer)

## Searches run
- concept bottleneck models interpretable concepts prediction → L-001 (Layer 1)
- concept bottleneck residual side channel unsupervised hybrid → L-002 (Layer 4)
- Chen Bei Rudin concept whitening → L-003 (Layer 4)
- promises and pitfalls black-box concept learning → L-004 (Layer 5, failure-oriented)
- Margeloiu do CBMs learn as intended → L-005 (Layer 5)
- data-driven persona generation evaluation Salminen Jung Jansen → L-006 (Layer 1)
- "Creating and validating predictive personas for target marketing" → L-007 (Layer 3)
- Grinsztajn tree-based models outperform deep learning tabular → L-008 (Layer 5)
- disentangled representations useful downstream limited evidence → L-009 (Layer 5)
- weakly-supervised disentanglement without compromises → L-010 (Layer 2)
- VAE customer segmentation RFM interpretable latent marketing → L-011 (Layer 4)
- supervised latent subspace VAE named constructs tradeoff curve → CS-VAE, sisPCA,
  EXoN [S] (Layer 4; corroborates L-002 — partitioned supervised/unsupervised latent
  spaces are a recurring, multi-field design, not a new one)

## Disconfirming searches for the manuscript's novelty claims
- "is the aligned/free latent partition already published?" → **YES** (L-002, plus
  CS-VAE/sisPCA/EXoN [S]). Kills any mechanism-novelty claim. Manuscript already
  disclaims architectural novelty; it must now disclaim *mechanism* novelty too.
- "has persona quality already been reformulated as predictive accuracy?" → **PARTIALLY**
  (L-007). Does not kill the contribution; narrows it to protocol + multi-axis.
- "does the field already say persona evaluation is unsolved?" → **YES, citably** (L-006).
  Strengthens the premise while forcing weaker phrasing.
- "is there evidence the β term will hurt downstream lift?" → **YES** (L-009). Converts
  the sweep from exploration into a directional test.

## Residual novelty risk (highest first)
1. L-007 full text unread (paywalled). If it already reports a frozen, multi-seed,
   held-out-future evaluation of persona representations, contribution (1) shrinks to
   a replication + extension. **Check before submission.**
2. The marketing/IS literature (JM, JMR, Marketing Science, ICIS) was not searched;
   segment-retention and predictive-segmentation work there may pre-empt the framing.
3. No search was run for "persona stability across seeds", so the stability axis's
   novelty is unassessed.

---

# Dedicated novelty search — 2026-08-14, second pass

Scope as specified: papers ~2018–2026 evaluating customer/buyer persona or segmentation
*representations* by (a) downstream predictive performance, (b) interpretability /
construct alignment, and/or (c) stability across seeds or resampling. Extended beyond
the ML venues of pass 1 into marketing and information-systems literature, which pass 1
did not search and flagged as residual risk #2.

### L-012 [V] — the stability axis has direct marketing prior art
Cite:    S. Dolnicar and F. Leisch, "Evaluation of structure and reproducibility of
         cluster solutions using the bootstrap", *Marketing Letters*, vol. 21, no. 1,
         pp. 83–101, 2010 (online 2009). DOI 10.1007/s11002-009-9083-4
URL:     https://api.openalex.org/works/doi:10.1007/s11002-009-9083-4 (metadata verified)
Method:  Repeated segmentation on **bootstrap samples**, partition agreement measured by
         the **Rand index adjusted for chance**, used to decide whether data contain
         natural segments, mere structure, or no structure, and how many segments to
         extract. Distinguishes natural / reproducible / constructive segmentation.
SUPPORTING: Makes the stability axis legible to a marketing audience and supplies a
         standard, marketing-native instrument for it.
DISCONFIRMING: **Bootstrap + adjusted Rand for segmentation stability is established
         marketing methodology, not a contribution of this work.** The manuscript cites
         only von Luxburg [11] for stability and presents cross-seed ARI as though the
         instrument were being imported from ML. It must be cited, and the stability
         axis reframed as *applying an established marketing criterion to learned
         representations* rather than proposing one.
Related [S]: "Stability of market segmentation with cluster analysis – A methodological
         approach" (2014); "Improving the stability of market segmentation analysis",
         IJCHM (2019); "Segmenting markets by bagged clustering". A sustained stream,
         which strengthens the "recurring, independently documented" reading.

### L-013 [S content / V bibliographic] — nearest neighbour on contribution (1)
Cite:    P.-F. Hsu, Y.-H. Lu, S.-C. Chen, P.-Y. Kuo, "Creating and validating predictive
         personas for target marketing", *IJHCS*, vol. 181, art. 103147, 2023.
Status:  Full text and publisher abstract remain unreachable (ScienceDirect, ACM DL,
         Consensus and ResearchGate all 403; Semantic Scholar holds no abstract).
         The characterization below is assembled from consistent abstract snippets across
         several independent aggregators — **[S], not [V]**. 16 citations (S2).
Content (per snippets, consistent across sources): proposes the **"PP method"**, a
         guideline for building personas with **data-mining predictive algorithms**;
         validates them by **accuracy in predicting target customers**; **benchmarks
         predictive personas against traditional quantitative (cluster-analysis) personas
         using predictive accuracy as "one unified metric"**; discusses when qualitative /
         traditional-quantitative / predictive personas each apply.
DISCONFIRMING: **Contribution (1) as originally worded is substantially anticipated.**
         "Persona quality should be measured by predictive accuracy, and that lets you
         compare persona schemes" is their claim. The manuscript must concede this.
SUPPORTING (what survives, and it is sharper for the concession):
         1. They optimize *one* unified metric; this work's thesis is that a single metric
            is the problem — three axes, and the *trade-off between them*, is the object of
            study. A frontier cannot exist in a one-axis framing.
         2. Their unit is a persona *set* built by a pipeline; ours is a *representation*
            evaluated by frozen transfer to labels it never saw.
         3. No evidence of leakage auditing, temporal splits, multi-seed statistics or
            significance testing appears in the snippets.
         Residual risk: (3) is an argument from absence in snippets. **Still must be
         confirmed against the full text before submission.**

### L-014 [V] — closest published "performance vs interpretability" segmentation paper
Cite:    I. Boussebough, K. Zarour, C. Aouabdia, D. S. Boutina, "Multi-View Customer
         Segmentation in the Digital Economy: Balancing Performance and Interpretability
         for Actionable Insights", *J. Telecommunications and the Digital Economy*,
         vol. 14, no. 2, pp. 58–83, 2026. DOI 10.18080/jtde.v14n2.1462
Method:  K-Means / AHC / DBSCAN over four views; PCA reduction; **internal validation
         indices** benchmarked against strategic KPIs (AOV, conversion) with comparisons
         of means; resolves the trade-off with a "context-driven decision matrix".
SUPPORTING: **Strong evidence for the evaluation gap.** The nearest paper that names the
         performance–interpretability trade-off in customer segmentation resolves it with
         a qualitative decision matrix and internal indices — it does not measure a
         frontier, uses no held-out downstream prediction, and reports no seed stability.
DISCONFIRMING: The *framing* "balance performance and interpretability in customer
         segmentation" is occupied. Novelty must rest on *quantifying* the frontier under
         held-out prediction, not on naming the tension.

### L-015 [V] — frozen-transfer evaluation of customer embeddings is established practice
Cite:    J. H. Bertrand, D. B. Hoffmann, J. P. Gargano, L. Mombaerts, J. Taws,
         "Autoencoder-based General Purpose Representation Learning for Customer
         Embedding", arXiv:2402.18164, 2024.
Method:  DEEPCAE, multi-layer contractive autoencoders for tabular entity embeddings;
         13 datasets; embeddings produced as reusable inputs to downstream models.
DISCONFIRMING: Evaluating customer embeddings by downstream task performance is normal
         practice in the industry/ML literature. The manuscript should not imply the
         frozen protocol is itself novel — the novelty is applying it to *persona*
         representations against *incumbent marketing* baselines with a leakage audit.

### Unresolved
- "An Automated Machine Learning Framework for Interpretable Customer Segmentation in
  Financial Services", *Int. J. Financial Studies*, 13(4):243. Snippets describe
  "RFM-based interpretability benchmarks" and "interpretability alignment measures" —
  potentially very close to the construct-alignment idea. **MDPI returned 403; unread.
  Highest-priority remaining check after L-013.**

## Searches run (pass 2)
- Dolnicar Leisch bootstrap reproducibility cluster solutions → L-012 (Layer 4, marketing)
- market segmentation stability reproducibility repeated clustering → L-012 corroboration
- OpenAlex title search "persona predictive" (44 works) → only L-013 relevant (Layer 3)
- "predictive personas" PP method accuracy credibility → L-013 content (Layer 3)
- deep learning customer segmentation predictive validity holdout 2023–2024 → L-014 (Layer 2)
- interpretability performance trade-off customer segmentation frontier → L-014, IJFS (Layer 4)
- autoencoder embedding customer segmentation frozen representation → L-015 (Layer 4)

**Saturation:** pass-2 queries using new vocabulary (marketing-native terms: "segment
reproducibility", "bagged clustering", "predictive validity", "internal validation
indices") returned ≥80% already-seen work by the final two queries. Layer 4 is saturated
for the vocabularies searched. Layer 3 is NOT closed while L-013's full text is unread.

## Net effect on the novelty argument

| Claim as written | Status after pass 2 |
|---|---|
| Persona quality reformulated as predictive lift | **Anticipated** (L-013) — concede and narrow |
| Three-axis reformulation + measured trade-off | **Survives** — no paper found measuring a frontier over lift + interpretability + stability |
| Frozen-representation protocol | **Not novel as a technique** (L-015); novel in this application, with the leakage audit |
| Stability via cross-seed ARI / bootstrap | **Instrument is marketing prior art** (L-012) — cite, reframe as applying it |
| Naming the interpretability/performance tension in segmentation | **Occupied** (L-014); quantifying it is not |
| MIG degeneracy under correlated constructs | **No prior report found** in the sources searched — the cleanest surviving novelty |
| Collapse–stability confound | **No prior report found** in this application area; L-012's "constructive segmentation" is adjacent and should be cited alongside |

---

# L-013 SUPERSEDED — full text obtained 2026-08-14

### L-013b [V — FULL TEXT]
Cite:    P.-F. Hsu, Y.-H. Lu, S.-C. Chen, P. P.-Y. Kuo, "Creating and validating predictive
         personas for target marketing", *Int. J. Human–Computer Studies*, vol. 181,
         art. 103147, 2024 (online 12 Sep 2023). Institute of Service Science, National
         Tsing Hua University, Taiwan.
Fetched: **full text, 18 pp.**, supplied by the project owner. No OA version exists
         (Unpaywall `is_oa=false`, `oa_status="closed"`, zero locations incl. embargoed),
         so this supersedes the earlier snippet-based `[S]` characterization.

PROBLEM:  Quantitative personas built with traditional statistical methods "may not
          reflect the marketing goals of decision-makers, and they mainly focus on
          categorizing existing users rather than predicting target customers."
METHOD:   7-step guideline. Survey of 2,240 pet-food (DCS) customers; multiple-choice
          items decomposed to dummies; outcome binned to buyer / non-buyer; **random
          60/40 train/test partition** (1,344 / 896) explicitly "to keep a separate
          testing dataset out of the model building process… avoids overfitting"
          (citing Shmueli et al.); **single logistic regression**; persona = the
          predicted-buyer group, described by the significant coefficients.
DATA:     Real survey + real subsequent purchase behaviour (224 coupon redemptions,
          45 buying the focal product).
EXPERIMENT: Held-out ranking + **decile lift chart** (lift 3.2 in decile 1 against a
          0.23 naïve base rate; decile 1 captures 33% of true buyers); confusion matrix;
          **behavioural validation** — 6.2% (12/193) of predicted buyers actually bought
          with the coupon vs 0.7% (5/686) of predicted non-buyers. Benchmark vs a
          k-means "traditional quantitative persona" (TQP): clusters fail to separate
          buyers (39.6% vs 46.2%), and PP vs TQP is compared **descriptively** —
          percentage ratios (87.4/30.9 = 2.83) and a visual comparison of response
          profiles across questions.
RESULTS:  PP separates buyers where clustering personas do not; pseudo-R² = 0.233.
LIMITATIONS (authors'): qualitative work still needed for vivid personas; time/cost not
          much reduced; data-science staffing required.

SUPPORTING (for this project's positioning):
  * **Their future work proposes two directions this project implements.** (a) Extend
    beyond a single outcome to several constructs — they name **price-sensitive**,
    quality-oriented and demand-oriented customers — by "altering the outcome variables"
    and repeating the procedure per construct. (b) "use **online user behaviour data** to
    complement our proposed PP method". This project does both, and does (a) *jointly in
    one representation* rather than as N independent supervised models. Continuation,
    not collision — and citable as such.

DISCONFIRMING (what it costs this project):
  * **Held-out predictive validation of personas is theirs.** Contribution (1) must be —
    and now is — conceded outright rather than hedged.
  * Their coupon-redemption validation has a stronger claim to external validity than
    this project's purely offline labels. The manuscript now says so.

VERIFIED ABSENT (full-text search, so these deltas are evidence-based, not inferred):
  repeated splits / seeds · variance on any reported number · any significance test of
  the PP-vs-TQP comparison (all 15 "significant" hits concern logit coefficient p-values)
  · cross-validation · ROC-AUC or PR-AUC · any interpretability measure · any stability
  or reproducibility analysis · representation learning of any kind · leakage auditing.

STRUCTURAL DISTINCTION (the load-bearing one): a PP persona **is the positive class of a
  supervised model of one chosen outcome**. It is task-specific by construction and cannot
  be transferred to a different downstream question. This project's object is a
  task-agnostic representation, learned label-free, frozen, then transferred to three
  tasks it never saw — which is why it can lose on purchase while winning on dormancy.

## Effect on the novelty table (revising the pass-2 entry)
| Claim | Status after reading the full text |
|---|---|
| Persona quality measured by held-out prediction | **Anticipated — conceded outright** (was: anticipated per snippets) |
| Three-axis reformulation + measured frontier | **Survives, now verified** — [17] measures neither interpretability nor stability |
| Multi-seed statistics, significance testing, leakage audit | **Survives, now verified absent in [17]** rather than assumed absent |
| Task-agnostic representation vs single-outcome persona | **New distinction, only visible from the full text** — the strongest of the three |
| Multi-construct personas + behavioural data | **Proposed as future work by [17] itself** — reframes the relationship as continuation |

## Residual novelty risk after this
Reduced to: (1) the marketing/IS venues were sampled, not systematically swept — a formal
database search (Scopus/WoS with the pass-2 vocabulary) would be the completeness check;
(2) no search was run on "persona stability across seeds" as a phrase, though L-012
establishes the marketing-side prior art for the instrument itself.

---

# Third pass — 2026-08-15 (external review prompted three checks; all three verified)

### L-016 [V — FULL TEXT] — the stability axis has a second, different prior art
Cite:    B. J. Jansen, S. Jung, S. A. Chowdhury, J. Salminen, "Persona analytics: Analyzing
         the stability of online segments and content interests over time using non-negative
         matrix factorization", *Expert Systems with Applications*, vol. 185, art. 115611,
         2021. Qatar Computing Research Institute, HBKU.
Fetched: **full text, 15 pp.**, supplied by the project owner.
Method:  32 monthly rounds of data collection on a major publisher's YouTube channel
         (demographics + content consumption); **15 data-driven personas generated monthly
         by non-negative matrix factorization**; change analysed monthly, yearly, lifetime.
Results: "**an average 40% change in the personas**, and 78% of the personas experience
         more change than consistency for topic interests." Implication drawn by the
         authors: organizations publishing frequently should automate collection and
         re-create personas periodically.
SUPPORTING: Gives the manuscript a sharp, citable **distinction between two things both
         called persona stability** — theirs is TEMPORAL (does a persona stay
         representative as the population evolves), ours is STOCHASTIC REPRODUCIBILITY
         (same data, same population, different seeds + bootstrap). Stating the difference
         explicitly makes the stability positioning much harder to attack.
DISCONFIRMING: The manuscript previously implied its stability axis covered persona
         stability generally. It does not, and [32] shows the axis this paper does NOT
         measure is where data-driven personas empirically fail. Now stated in §II-E,
         §VIII (external validity) and §XII, and promoted to future-work item (2).

### L-017 [V] — bounds the MIG novelty claim
Cite:    F. Träuble, E. Creager, N. Kilbertus, F. Locatello, A. Dittadi, A. Goyal,
         B. Schölkopf, S. Bauer, "On Disentangled Representations Learned from Correlated
         Data", ICML 2021, PMLR 139:10401–10412.
Fetched: PMLR abstract page.
Results: **4,260 models** trained on systematically correlated data; "systematically
         induced correlations in the dataset are being learned and reflected in the latent
         representations"; addresses the gap between idealized independent-factor settings
         and realistic dependent ones; proposes weak supervision / post-hoc correction.
DISCONFIRMING: **The manuscript's MIG claim was too strong.** "MIG is degenerate under
         correlated constructs / no prior report found" invites the reply that
         correlated-factor disentanglement is an established research area with a
         4,260-model study behind it.
SUPPORTING (what survives, narrowed): their question is whether correlated factors end up
         *disentangled*; ours is whether MIG survives as a **model-selection criterion**
         along a supervised alignment-strength sweep — i.e. selection-ordering inversion,
         not metric misbehaviour in general. Claim rewritten in the abstract, contributions
         list, §II-C, §VI-C, §VII and §XI; the universal "MIG should not be used…" softened
         to "should not be used as the *sole* criterion **in this setting**".
NOT VERIFIED: the supplementary-material claim that global disentanglement metrics show no
         clear trend as correlation increases was NOT retrieved; no claim rests on it.

### L-018 [V — bibliographic] / [S — content] — interpretable customer embeddings
Cite:    G. Glukhov, P. Zhdanov, E. Shikov, "Interpretable Embeddings for Geographic
         Transactional Activity Analysis", *Procedia Computer Science*, vol. 229,
         pp. 357–366, 2023. DOI 10.1016/j.procs.2023.12.038. ITMO University.
Fetched: OpenAlex metadata (authors/venue/year/DOI, Diamond OA) + Semantic Scholar record.
         **ScienceDirect 403s; full text not read.** Content below from consistent
         secondary descriptions — `[S]`.
Method (per [S]): motivated by transactional customer embeddings being high-quality but
         hard to interpret. Three steps: compute per-client geographic-activity feature
         vectors, cluster them, then set each embedding coordinate to the **distance
         between the user's activity vector and a cluster centre** — so dimensions carry
         meaning by construction. Evaluated on a partner-bank dataset.
DISCONFIRMING: Another interpretable-customer-embedding line the manuscript did not cite;
         omitting it weakens Related Work.
SUPPORTING: The mechanism is **interpretability by prototype distance**, which is a
         different construction from naming axes by regression onto marketing constructs,
         and it is fixed rather than swept — so it cannot price interpretability. Cited in
         §II-D on the mechanism only. **No claim is made about what they do or do not
         evaluate**, since the full text was not read.

## Effect on the manuscript
| Item | Before | After |
|---|---|---|
| MIG finding | "degenerate for model selection under correlated constructs; no prior report" | "MIG-based *selection* inverts the alignment-strength ordering in this construct-supervised setting"; [33] cited as the established neighbouring result |
| MIG prescription | "should not be used… which is the normal case for business constructs" | "should not be used as the *sole* criterion **in this setting**"; explicit non-claim of general invalidity |
| Stability | one undifferentiated axis | temporal [32] vs stochastic (this work) separated in §II-E, §VIII, §XI, §XII |
| Cluster personas | "the principal signal destroyer" | "a major signal destroyer **in these experiments**" |
| "frontier" | used for the swept grid | grid = "trade-off surface"; "frontier" reserved for the Pareto-optimal subset |
| Related work | no interpretable-embedding line in the customer domain besides [23] | [34] added |

## Residual novelty risk after pass 3
Unchanged and stated in §XI: the marketing/IS literature was sampled, not systematically
swept. A Scopus/Web of Science query using the pass-2 and pass-3 vocabulary remains the
completeness check, and is not reachable from this environment.
