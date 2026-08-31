# Mapping Digital Transformation Across Health and Governance

### Evidence of Asymmetric and Mediated Coupling

A corpus-based computational study of how digital transformation (DT) research is
structurally organized across **SDG 3 (Good Health and Well-being)** and
**SDG 16 (Peace, Justice and Strong Institutions)**. Applied to 600 peer-reviewed
articles, the pipeline classifies papers by dominant DT theory and SDG domain, then
constructs network models to characterize the relationship between the two domains.

## Authors

After peer- reviewed 

## Summary

The corpus comprises 600 articles retrieved from Scopus and Web of Science (Q1–Q2
journals, 2023–2026), screened under the PRISMA framework. Semantic classification
assigns 273 documents to SDG 16 (governance), 241 to SDG 3 (health), and 91 to both.

The analysis reveals a highly modular, fragmented structure: SDG 3 forms a cohesive,
central core while SDG 16 remains dispersed and peripheral. Direct cross-domain
interaction is minimal (edge density ≈ 0.00086), yet a small set of high-betweenness
bridging nodes mediates connection (dependency ≈ 6.21, centrality overlap ≈ 0.994).
DT research thus behaves as an asymmetrically coupled system, integrated not
systemically but through a limited set of bridging studies and shared theoretical
frameworks.

## Approach

- **Embeddings:** SentenceTransformer `sentence-transformers/all-MiniLM-L6-v2`
- **Similarity:** pairwise cosine similarity over structured document text
- **Classification:** semantic assignment by dominant DT theory and SDG domain
- **Network models:** domain-specific, combined, cross-domain interaction, and a
  multi-layer knowledge graph
- **Structural metrics:** eigenvector and betweenness centrality, modularity, Louvain
  community detection, core–periphery decomposition, and a Structural Coupling
  Index (SCI)
- **Theory anchors:** TAM, UTAUT, STS, TOE, RBV, Dynamic Capabilities, IDT,
  Stakeholder Theory
- **Reproducibility:** `SEED = 42` is fixed program-wide

## Repository structure

```
.
├── 3-16_dt_sdg.ipynb     # Main analysis notebook
├── requirements.txt
├── .gitignore
├── data/
│   └── files/            # Place source .docx corpus here (git-ignored)
└── outputs/              # Generated CSVs and figures (git-ignored)
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running

1. Place the source `.docx` corpus in `data/files/`.
2. Open the notebook and run cells top to bottom:

```bash
jupyter notebook 3-16_dt_sdg.ipynb
```

Paths are configurable via environment variables:

```bash
export DATA_DIR=/path/to/corpus
export OUTPUT_DIR=/path/to/outputs
```

### Google Colab

The setup cell includes commented `drive.mount` lines. Uncomment them and point
`DATA_DIR` / `OUTPUT_DIR` at your Drive folders.

## Notes

- The corpus is not included in this repository. Outputs are regenerated on each run.
- Notebook outputs were cleared before commit to keep the repository lightweight.

## Citation

If you use this work, please cite:

> Farhadian, G., Saeedi, M., & Ha, J. Mapping Digital Transformation Across Health
> and Governance: Evidence of Asymmetric and Mediated Coupling.

## License

Specify a license before publishing (e.g., MIT, or a restricted/none license for
unpublished research).
