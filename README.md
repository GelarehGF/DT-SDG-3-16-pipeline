# Digital Transformation Pathways Across Health and Governance

### A Reproducible Similarity and Knowledge-Network Analysis of SDG 3 and SDG 16

A corpus-based computational study of how digital transformation (DT) research is
organized across **SDG 3 (Good Health and Well-being)** and **SDG 16 (Peace, Justice
and Strong Institutions)**. The revised workflow links paper similarity with
auditable technology–mechanism–outcome candidates and cross-SDG relationships.

The original notebook is retained for provenance. The **terminal Python workflow**
below supports the revised manuscript; the two workflows use different text
representations and graph constructions and are not interchangeable.

## Authors

To be added after peer review.

## Summary

The revised corpus audit reconciles **605 canonical DOCX records**, removes six
duplicates, and retains **599 unique papers**: 241 SDG 3, 269 SDG 16, and 89
mixed-domain papers. These are retained corpus-domain labels, not new
SentenceTransformer classifications. Earlier README totals of 600 papers and
273/241/91 domain counts are superseded by this audit.

In the revised TF-IDF/SVD analysis, median cosine similarity is 0.178 within SDG 3,
0.184 within SDG 16, and 0.156 across the two domains. The undirected union k=10
nearest-neighbour graph has **4,307 edges**, one connected component, and nine
Louvain communities (weighted modularity 0.534). Of its retained edges, 658 (15.3%)
directly connect SDG 3 and SDG 16; this share ranges from 14.7% to 16.7% for k=5–15.
This describes **modular organization with selective cross-domain connections**,
not the earlier claim that the revised full network is fragmented. Connectivity
and edge shares depend on graph construction; they are not causal effect estimates
or evidence of statistically significant integration.

Rule-based extraction identifies **1,654 candidate evidence rows** in 318 papers,
representing 1,200 unique paper–technology–outcome combinations. There are 65
cross-SDG candidates in 44 papers: 11 same-span links and 54 weaker paper-level
co-occurrences. These distinguish what papers discuss from which outcomes are
reported together; neither establishes that technology causes one SDG to affect
the other. See [methods and reporting boundaries](docs/METHODS_REPORTING.md).

## Approach

- **Historical embeddings:** the original notebook uses SentenceTransformer
  `sentence-transformers/all-MiniLM-L6-v2`; retained for historical comparison.
- **Revised similarity:** TF-IDF unigrams/bigrams (up to 25,000 features), truncated
  SVD (200 dimensions), L2 normalization, and all-pair cosine similarity.
- **Corpus and coding:** numbered source collections, duplicate controls, and an
  explicit codebook of 18 technology families, 13 mechanisms, and 24 outcomes.
  Candidate rows retain evidence text, matching rules, and review fields.
- **Network models:** within-domain views, combined kNN graph, filtered bridge
  display, full-network and Louvain supplements, and a heterogeneous
  paper–technology–mechanism–outcome–SDG knowledge network.
- **Structural metrics:** degree/strength, PageRank, approximate betweenness,
  domain participation, Louvain communities/modularity, and k=5/10/15 sensitivity.
  Legacy SCI/core–periphery results are not recomputed by the revised pipeline.
- **Theory anchors:** TAM/UTAUT, STS, TOE, RBV, Dynamic Capabilities, IDT, and
  Stakeholder Theory inform interpretation; network structure alone does not test
  these theories. Discussion separates adoption, organizational capability, and
  institutional governance mechanisms.
- **Reproducibility:** analysis seed 42; network-display seed 42 (SDG 16 panel 43).
  Manifests record parameters and hashes. No generative model or external text API
  is called by the revised analysis or figure scripts. This concerns computation,
  not disclosure of editorial assistance.

## Repository structure

```text
.
├── main.py                           # One command runs the complete revised workflow
├── 3-16_dt_sdg.ipynb                  # Original notebook, preserved unchanged
├── technology_outcome_pipeline/
│   ├── pipeline.py                   # Canonical corpus → auditable master files
│   ├── codebook.json                 # Dictionaries and coding rules
│   ├── build_knowledge_network.py     # Similarity, communities, knowledge graph
│   ├── make_figures.py               # Analytical figures and supporting tables
│   ├── make_manuscript_figures_v09.py # Readable non-network manuscript figures
│   ├── make_network_figures_v10.py    # Within/combined/full/community networks
│   └── figure_fonts.py               # Portable font configuration
├── docs/
│   ├── METHODS_REPORTING.md          # Methods, validation scope, interpretation
│   ├── FIGURE_GUIDE.md               # Manuscript numbering and provenance
│   └── VERIFICATION.md               # Tests and reproduction record
├── tests/                            # Synthetic, copyright-free regression tests
├── requirements.txt                  # Revised workflow
├── requirements-legacy.txt           # Optional notebook dependencies
├── .gitignore
├── data/                             # Local only; git-ignored
│   └── files/
│       ├── SDG3/Words/SDG3-*.docx
│       ├── SDG16/Words/SDG16-*.docx
│       └── mixed/SDGM-*.docx
└── outputs/                          # Generated files; git-ignored
```

## Setup

Use Python 3.12 and a fresh environment:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

The revised workflow does not require Node.js, an application-specific runtime,
API credentials, or a downloaded language model. DejaVu Sans supplied with
Matplotlib is the default figure font. Optional `--regular-font` and
`--bold-font` paths allow a preferred typeface.

## Running

1. Place canonical source DOCX files in the folder structure above. Keep manuscripts,
   exports, and alternative representations outside this input.
2. From the repository root, run **one command**:

```bash
python main.py --source "/path/to/SDG3-16"
```

If your corpus is already in `data/files/`, simply run `python main.py`.
Keep all repository files together: `main.py` calls the five existing scripts in
order using the same Python environment. It runs extraction, paper similarity and
knowledge networks, analytical figures, readable manuscript charts, and the four
network views/supplements. The XLSX master workbook is included by default.

Results go into a fresh timestamped folder under `outputs/`:

```text
outputs/run-<timestamp>/
├── master/               # CSV/JSON master data and XLSX workbook
├── network/              # Similarity matrices, metrics, GraphML/GEXF
├── analysis_figures/     # Analytical PNG/SVG/PDF figures and supporting tables
├── manuscript_figures/   # Readable charts, network figures and supplements
├── logs/                 # One progress/error log per stage
└── run_manifest.json     # Parameters, hashes, package versions and stage status
```

Use `--output outputs/my-run` to choose a **new or empty** run folder,
`--skip-workbook` to omit only XLSX, or `--dry-run` to preview the five commands
without creating files. All options are listed by `python main.py --help`.
The runner refuses to overwrite a populated output folder, stops on a failed
stage, checks required output files, and keeps completed work and logs for diagnosis.
It does not rerun the historical notebook, redo human validation, or rewrite the
manuscript/native Word Figure 1.

<details>
<summary>Advanced: run the five stages individually</summary>

```bash
python technology_outcome_pipeline/pipeline.py \
  --source data/files --output outputs/master \
  --max-evidence-per-pair 3 --skip-workbook

python technology_outcome_pipeline/build_knowledge_network.py \
  --master-dir outputs/master --output outputs/network \
  --primary-k 10 --sensitivity-k 5 10 15 \
  --svd-components 200 --max-features 25000 --min-df 3 --seed 42

python technology_outcome_pipeline/make_figures.py \
  --input outputs/master --output outputs/analysis_figures

python technology_outcome_pipeline/make_manuscript_figures_v09.py \
  --network-dir outputs/network \
  --figure-data-dir outputs/analysis_figures/supporting_data \
  --output outputs/manuscript_figures

python technology_outcome_pipeline/make_network_figures_v10.py \
  --network-dir outputs/network --output outputs/manuscript_figures --seed 42
```

</details>

The numbered corpus layout is preferred. A fallback recursively reads DOCX/PDF/TXT
when none of the canonical collections is found; that is not the manuscript's
canonical-input workflow. The network stage requires the original DOCX paths saved
in `master_papers.csv`; regenerate the master after moving the corpus.

CSV/JSON files are the analysis inputs. The plain tabular XLSX export does not
reproduce the earlier application-specific workbook styling. The individual
scripts can replace same-name files, whereas `main.py` protects existing run
folders. Human review decisions are not automatically merged into reruns.

Run synthetic tests without the private corpus:

```bash
python -m unittest discover -s tests -v
```

For an end-to-end software check using 18 invented papers (no private corpus):

```bash
python tests/smoke_workflow.py --work-dir outputs/synthetic-smoke
```

This executes `main.py` and all five real stages, including XLSX and ten
manuscript PNGs. Use a new work folder each time.

For the **historical notebook only**:

```bash
python -m pip install -r requirements-legacy.txt
jupyter notebook 3-16_dt_sdg.ipynb
```

Its `DATA_DIR` and `OUTPUT_DIR` environment variables remain available; revised
scripts use explicit command-line paths. Notebook outputs and cached embeddings
must not be substituted for revised TF-IDF/SVD outputs.

### Google Colab

Mount your own Drive if needed and pass corpus/output paths to the revised
commands above. For the historical notebook, set `DATA_DIR` / `OUTPUT_DIR` before
running its cells. Do not place source papers or evidence-bearing output files in
a public notebook or repository.

## Notes

- Source papers, manuscripts, human-validation workbooks, and extracted evidence
  are not distributed here. Generated files may contain copyrighted passages and
  local source paths; review them before external sharing.
- Existing completed human validation is retained. Regenerating figures or running
  software tests does not require repeating it. Corpus/domain validation does not
  automatically validate new technology–outcome rows or causal direction; those
  remain candidates unless separately documented as reviewed.
- Figure 1 remains a native, editable Word figure in the manuscript. Python
  generates the analytical figures; none is an AI-generated illustration.
  The [figure guide](docs/FIGURE_GUIDE.md) maps filenames to manuscript numbering.
- Filtering the main network affects display only. Statistics use every retained
  edge; supplements expose the complete graph and saved communities.
- The original notebook is unchanged. Its earlier counts and structural claims
  are historical, not the revised baseline. Floating-point results and layouts can
  vary across dependency/platform versions despite fixed seeds.
- [Verification record](docs/VERIFICATION.md): regression tests and reproduction
  checks against the saved manuscript analysis.

## Citation

If you use this work, please cite:

> Farhadian, G., Saeedi, M., & Ha, J. Digital Transformation Pathways Across Health
> and Governance: A Reproducible Similarity and Knowledge-Network Analysis of
> SDG 3 and SDG 16. Manuscript in preparation.

## License

No license has been selected. A license should be selected by the authors before
licensed distribution; no corpus redistribution permission is supplied here.
