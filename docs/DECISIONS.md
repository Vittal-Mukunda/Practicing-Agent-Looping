# DECISIONS.md — every non-obvious choice + rationale

Format: `D-NNN (date) — decision — rationale — status`.
Status: **locked** (approved / uncontroversial engineering), **provisional** (needs
gate sign-off), **pending** (awaiting owner input).

---

**D-001 (2026-07-09) — `uv` as environment + lockfile manager.**
One tool covers Python-version pinning (`.python-version`), dependency locking
(`uv.lock`, committed), and venv management; invoked as `python -m uv` so it works
without PATH changes. Alternatives (poetry, pip-tools) rejected for no functional
gain and slower resolution. *Status: locked (engineering).*

**D-002 (2026-07-09) — torch pinned to 2.6.0 from the cu124 index.**
Exact pin in `pyproject.toml`; wheel `2.6.0+cu124` (Windows, cp312) via
`download.pytorch.org/whl/cu124`. cu124 chosen over cu126/cu128 for wider driver
tolerance; RTX 4050 (Ada, compute capability 8.9) is fully supported by every
CUDA 12.x build. Verified working by `python -m cadvae.utils.device` (evidence in
NOTEBOOK.md). torchvision/torchaudio omitted — tabular project, not needed.
*Status: locked (engineering).*

**D-003 (2026-07-09) — uv resolution restricted to `sys_platform == 'win32'`.**
The paper's numbers are produced on one fixed machine (Windows / RTX 4050). A
single-platform lockfile is smaller and cannot silently resolve different versions
on other OSes. Cross-OS reproduction is future work; the committed lockfile is the
authoritative environment. *Status: locked (engineering).*

**D-004 (2026-07-09) — repo-local git identity set to "Vittal Muku" <vittal.muku@gmail.com>.**
No global git identity existed; commits require one. Name derived from the email —
**owner should correct via `git config user.name "..."` if wrong.** *Status: pending owner check.*

**D-005 (2026-07-09) — `results/` is gitignored.**
Per-run outputs (checkpoints, metrics, resolved configs) are bulky and regenerable
from committed code + configs + seeds; that is the reproducibility contract.
Curated result tables/figures get promoted into `docs/` (or a tracked
`results/summary/`) deliberately at analysis time (Phase 6). *Status: provisional — Phase 0 gate item.*

**D-006 (2026-07-09) — Dataset B = multi-category store (not the cosmetics shop).**
Owner decision from the kickoff message: the category-affinity construct is
near-vacuous with a single-category store; the multi-category set also carries the
scale story. *Status: locked (owner).*

**D-007 (2026-07-09) — null-means-undecided convention for consequential config values.**
Split ratios/boundaries, stratification column, and aligned-dims are consequential
(CLAUDE.md §6) and await their phase gates. They are explicit `null` in configs —
never silently defaulted — and `tests/test_config.py::test_undecided_phase1_parameters_are_null_not_defaults`
enforces it. Phase 1+ code must refuse to run on nulls. *Status: locked (protocol).*

**D-008 (2026-07-09) — proposed GBT downstream head: sklearn `HistGradientBoostingClassifier`.**
Zero extra dependency, strong tabular default, CPU-fast. XGBoost/LightGBM would add
a dep without an obvious accuracy story at these data sizes. Metric-affecting →
consequential. *Status: provisional — decide no later than Phase 3 gate.*

**D-009 (2026-07-09) — repo location risk: OneDrive-synced folder. RESOLVED.**
Original concern: repo under `C:\Users\vitta\OneDrive\Desktop\VAE` would sync
`.venv` (~5 GB) and `data/raw` (~15 GB). **Resolved 2026-07-09: repo relocated to
`C:\VAE` (non-synced; confirmed `pwd` = `/c/VAE`).** 734 GB free on C:, ample for
the ~15 GB Dataset B. No OneDrive interference. *Status: locked (resolved).*

**D-010 (2026-07-09) — determinism posture.**
`seed_everything` seeds python/numpy/torch(+CUDA), forces deterministic cuDNN,
disables cudnn.benchmark, sets `CUBLAS_WORKSPACE_CONFIG=:4096:8`, and enables
`torch.use_deterministic_algorithms(True, warn_only=True)` so nondeterministic ops
are *surfaced* in logs rather than silently allowed (CLAUDE.md: document unavoidable
GPU nondeterminism). Revisit warn-only vs strict at Phase 4 when the training loop
exists. *Status: locked (engineering), revisit Phase 4.*

**D-011 (2026-07-09) — dataset licenses, now verified first-hand at download.**
Kaggle pages are JS-rendered, so earlier programmatic verification failed; the
Kaggle **CLI**, however, prints the license string at download time.
- **Dataset A → `CC0-1.0` (public domain).** Confirmed first-hand. No restriction;
  no attribution legally required (we will credit the source anyway). RESOLVED.
- **Dataset B → `copyright-authors`** (Kaggle's "Data files © Original Authors"
  label). This is the **actual Kaggle license tag** and it CONTRADICTS the softer
  HF-mirror wording ("free to use for research… please mention the source"). It is
  *not* an open/CC license. Practically, academic-research use of REES46's public
  Kaggle release with attribution is the common and intended use, but the label is
  restrictive, not permissive. **Owner must eyeball the Kaggle page's license
  section and decide** whether the terms are acceptable for a public paper +
  code-release, or whether to (a) proceed with attribution, (b) contact REES46, or
  (c) swap Dataset B. Flagged as a Phase 0 gate item — do not treat as settled.
*Status: A resolved; B pending owner confirmation (gate).*
