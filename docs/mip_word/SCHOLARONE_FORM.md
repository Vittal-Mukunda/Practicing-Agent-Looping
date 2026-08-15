# ScholarOne submission sheet: Marketing Intelligence & Planning

Copy each block into the matching field. Word counts are shown against the limits the
submission form states, which are stricter than the published author guidelines on two
points: the 20-word title cap and the 100-word-per-section abstract cap.

---

## Title  (13 of 20 words)

What does an interpretable buyer persona cost? A measurement-first evaluation of customer representations

## Manuscript category

**Original Article** for the submission *type*. Where a *category* is also requested,
choose **Research paper**: the study empirically tests a framework on real data.
(The journal page lists only Original Article and Editorial as types.)

## Structured abstract

Total including keywords and classification: **246 of 250**. Each section is within the 100-word cap.

### Purpose  (47 of 100 words)

Buyer personas direct substantial marketing expenditure, yet the field's own survey names persona evaluation as an open gap. Existing predictive validation reduces persona quality to a single accuracy figure, which cannot express what is given up to obtain interpretability. This paper asks what an interpretable persona costs.

### Design/methodology/approach  (66 of 100 words)

Persona quality is reformulated as three falsifiable properties: predictive lift on held-out behaviour, construct alignment, and stability across repeated estimation. Nine representations, including RFM, PCA, autoencoders and a construct-aligned variational autoencoder, are compared on two public datasets (2,240 customers; 2.55 million users from 110 million events), plus two construct-based controls. All use a frozen-representation protocol with automated leakage guards, six seeds, and bars fixed in advance.

### Findings  (53 of 100 words)

Non-neural incumbents set the strongest bars; the construct-aligned model falls below both and yields less stable personas. On the survey dataset, two controls sharpen this: alignment is a subspace property rather than one of individual named axes, and a zero-training representation built from the constructs achieves comparable measured lift with perfect axis purity.

### Practical implications (optional)  (42 of 100 words)

Managers can price interpretability rather than assume it. For ranking customers by purchase propensity, RFM remains more accurate and more stable at no cost. Where interpretable coordinates are the goal, they can be built directly from existing features, without a neural model.

### Originality/value  (25 of 100 words)

The contribution is measurement rather than method: a protocol that prices interpretability, and evidence that an assumed benefit of neural segmentation does not survive it.

**Not used:** Research limitations/implications and Social implications. The first would
duplicate the scoping already stated in Findings ("On the survey dataset") and pushed the
total over 250; the second does not apply. Both are optional.

## Keywords  (6 terms)

Buyer personas, customer segmentation, marketing analytics, interpretability, predictive validation, representation learning

## Plain Language Summary (Kudos, optional)  (100 words)

Companies group customers into personas to decide who to target, but there has been no agreed way to tell whether one set of personas is better than another. We measured three things at once: how well a persona scheme predicts what customers actually do next, whether its categories really mean what they claim to mean, and whether it gives the same answer twice. Simple, decades-old methods beat a modern neural network on all three, and the interpretable labels the neural model was built to provide turned out to be obtainable directly from data the company already has, at no cost.

---

## Files to upload

| Order | File | ScholarOne file designation |
|---|---|---|
| 1 | `MIP_manuscript_ANONYMOUS.docx` | Main Document |
| 2 | `MIP_tables.docx` | Table |
| 3 | `MIP_figures.docx` | Figure |
| 4 | `Figure1.tif`, `Figure2.tif` | Figure (originals, 300 dpi) |
| 5 | `MIP_title_page.docx` | Title Page / Supplementary File not for review |

`MIP_manuscript_AUTHOR_VERSION.docx` is your reading copy. **Do not upload it**: it carries
your name, affiliation and the repository URL, which would break double-anonymous review.

## Compliance check

| Requirement | Status |
|---|---|
| Length 6,000 to 8,000 words | 6,982 at 250 words per table/figure; 7,192 at 280. Inside the band on either count |
| Title within 20 words | 13 |
| Abstract within 250 words including keywords and classification | 246 |
| Each abstract section within 100 words | longest is 66 |
| Four mandatory sub-headings present | Purpose, Design/methodology/approach, Findings, Originality/value |
| Keywords | 6 |
| Headings: level 1 bold, sub-headings italic | applied |
| Tables separate, Roman numerals, position marked in text | applied |
| Figures embedded with captions, plus separate file and TIFF originals | applied |
| No author name anywhere in the anonymous manuscript | verified programmatically |
| Emerald Harvard references | formatted; check each entry yourself |
| Word format | yes |

## Before you click submit

1. **ORCiD** on the title page and in your ScholarOne profile.
2. **Ethics statement** on the title page says the study used public secondary data with no
   human-participant interaction, so no approval was required. Confirm that against your
   institution's policy.
3. **Generative AI declaration.** Declare it in the submission form as well as in the
   article, naming the tool and version, and make it match the whole project history.
4. **Cover letter.** Mention that the two controls in Sections 4.3 and 4.4 were run on one
   dataset and the replication is pending, so the editor is not surprised by Section 6.
5. **References already verified.** All 23 were checked against OpenAlex, Crossref and
   PMLR on 15 Aug 2026; see REFERENCE_VERIFICATION.md. No factual error was found and four
   entries gained confirmed proceedings detail.
