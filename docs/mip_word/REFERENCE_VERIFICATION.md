# Reference verification: all 23 entries

Checked against OpenAlex and Crossref by automated query, with PMLR consulted directly for
ICML proceedings (OpenAlex indexes only the arXiv preprints for those). No entry was passed
on trust.

## Result

**23 of 23 verified.** No factual error found in any reference. Four entries were
*upgraded* with proceedings detail that was verified during the check.

## Per-entry status

| # | Reference | Verified against | Outcome |
|---|---|---|---|
| 1 | Boussebough *et al.* (2026) | OpenAlex, DOI 10.18080/jtde.v14n2.1462 | Year, venue, 14(2), 58-83, 4 authors all match |
| 2 | Chen *et al.* (2018) | OpenAlex title search | Match; NeurIPS 2018 not paginated in the index |
| 3 | Dolnicar and Leisch (2010) | Crossref, DOI 10.1007/s11002-009-9083-4 | 21(1), 83-101 confirmed. **Year checked deliberately:** online 5 Aug 2009, print issue March 2010. Citing 2010 is correct for the issue |
| 4 | Grigorova *et al.* (2025) | OpenAlex, DOI 10.3390/ijfs13040243 | 13(4), art. 243, 3 authors match |
| 5 | Grinsztajn *et al.* (2022) | OpenAlex title search | Match; NeurIPS Datasets and Benchmarks track |
| 6 | Higgins *et al.* (2017) | OpenAlex | Year, venue, all 8 authors match |
| 7 | Holm (1979) | OpenAlex | 6(2), 65-70 match |
| 8 | Hsu *et al.* (2023) | OpenAlex, DOI 10.1016/j.ijhcs.2023.103147 | Vol. 181, art. 103147, 4 authors match. Full text also read |
| 9 | Hubert and Arabie (1985) | OpenAlex, DOI 10.1007/BF01908075 | 2(1), 193-218 match |
| 10 | Hughes (1994) | OpenAlex | Book record confirmed |
| 11 | Jansen *et al.* (2021) | OpenAlex, DOI 10.1016/j.eswa.2021.115611 | Vol. 185, art. 115611, 4 authors match. Full text also read |
| 12 | Kingma and Welling (2014) | OpenAlex | Match |
| 13 | Koh *et al.* (2020) | **PMLR v119 direct** | Match. **Upgraded**: PMLR 119, pp.5338-5348 added |
| 14 | Kumar *et al.* (2018) | OpenAlex | Match |
| 15 | Locatello *et al.* (2019) | **PMLR v97 direct** | Not indexed by OpenAlex beyond the preprint. PMLR confirms 2019, all 7 authors in order. **Upgraded**: PMLR 97, pp.4114-4124 added |
| 16 | Mahinpei *et al.* (2021) | OpenAlex | arXiv:2106.13314, 5 authors match |
| 17 | Mancisidor *et al.* (2019) | OpenAlex | arXiv:1903.06580, 4 authors match |
| 18 | Margeloiu *et al.* (2021) | OpenAlex | arXiv:2105.04289, 6 authors match |
| 19 | Nai *et al.* (2024) | OpenAlex | Match. Third author checked by hand: "Ji Li" is given name Ji, surname Li, so "Li, J." is right |
| 20 | Salminen *et al.* (2021) | OpenAlex, DOI 10.1080/10447318.2021.1908670 | 37(18), 1685-1708, 4 authors match |
| 21 | Sawada and Nakamura (2022) | OpenAlex | IEEE Access 2022 confirmed |
| 22 | Träuble *et al.* (2021) | PMLR v139 | 10401-10412, all 8 authors match. **Upgraded**: proceedings title expanded |
| 23 | Xie *et al.* (2016) | **PMLR v48 direct** | pp.478-487 confirmed, 3 authors match. **Upgraded**: PMLR 48 added |

## Flags that turned out to be artefacts of the checking method

Recorded so they are not re-investigated later:

- **"Venue: arXiv (Cornell University)"** on entries 2, 5, 13, 14, 23 and others. OpenAlex
  frequently indexes the arXiv preprint as a paper's primary location even when the paper was
  published at a conference. The conference attribution in the reference list is correct, and
  PMLR was used directly wherever page numbers were claimed.
- **"Pages 243-243", "103147-103147", "115611-115611"** on entries 4, 8 and 11. These are
  article-number journals; the index repeats the article number as both first and last page.
  The "art. N" form used in the reference list is correct.
- **Kingma indexed to UvA-DARE, Mancisidor to Duo Research Archive.** Institutional
  repository copies indexed as primary. Both references are correct as written.
- **"Author 7: Schoelkopf vs Schölkopf"** on entry 22. An artefact of the checker's
  transliteration; the manuscript spells it correctly.

## Remaining limitation

OpenAlex and Crossref index bibliographic metadata, not the content of each work. This check
establishes that the 23 works exist as described, with the stated authors, venues, volumes
and pages. It does not re-confirm that each claim attributed to a source is faithful to it.
Six of the 23 were read at abstract level or in full during the literature work, and those
are recorded in `research/ledger.md` with verification tags.
