# Revised methods and reporting boundaries

This document describes the terminal workflow accompanying the network-integrated
manuscript revision. It is not a claim that every automated relationship has been
human-validated, nor a new validation protocol that replaces completed work.

`python main.py --source /path/to/SDG3-16` runs the complete revised workflow.
The runner does not change the analytical methods: it calls the five versioned
scripts in dependency order, using one Python environment, and records parameters,
software/codebook hashes, versions, logs, and completion/failure status. It creates
a new run folder and does not merge or overwrite completed human-review records.

## 1. Preserve the distinction between workflows

The original `3-16_dt_sdg.ipynb` uses MiniLM embeddings, semantic theory/domain
assignments, and several thresholded network constructions. It is preserved
unchanged for provenance. The terminal workflow instead uses retained source-domain
labels, a deduplicated corpus, TF-IDF/SVD similarity, and an undirected union kNN
graph. Results from these representations, thresholds, and denominators cannot be
combined as if they came from one network. In particular, the earlier dependency,
centrality-overlap, SCI, and core–periphery claims are not estimates produced by
the revised scripts.

## 2. Canonical corpus and duplicate control

`pipeline.py` selects numbered DOCX records in `SDG3/Words`, `SDG16/Words`, and
`mixed`, assigning their existing collection labels. Alternative-format copies are
not added to the canonical input. Normalized full-text hashes, candidate DOI
matches, and sufficiently specific normalized titles identify duplicate records;
duplicate rows remain in the manifest but are excluded from relation extraction.

The manuscript baseline contains 605 selected/parsed records, six duplicates,
and 599 eligible papers (241 SDG3, 269 SDG16, 89 Mixed). This is a corpus audit, not
a rerun of the earlier human screening or a new domain classifier. The code's
recursive fallback is convenient for other inputs but must not be substituted for
the stated canonical workflow without reporting that change.

## 3. Rule-based technology–outcome coding

The versioned codebook contains 18 technology families, 13 mechanisms, and 24
outcomes. A candidate requires a technology and outcome in one sentence, or in
adjacent sentences linked by specified discourse/causal cues. Reference sections
and keyword lists are excluded. Candidate rows retain the source path, paper ID,
sentence/span, section, triggering terms, rule identifier, polarity/evidence cues,
mechanism codes, heuristic score, and review fields. Up to three highest-scoring
passages per paper–technology–outcome combination are retained by default.

Scores are rule-based ranking heuristics, **not calibrated probabilities**. Cue
matches for positive effects, empirical evidence, or causal language do not verify
an effect or identify a causal research design. Categories may overlap, and one
passage can generate several technology/outcome combinations. Counts therefore
must specify whether their unit is evidence rows, unique combinations, or papers.
The repository makes no claim that the codebook was preregistered.

Cross-SDG candidates connect the same technology to an outcome in each domain.
Same-span evidence is distinguished from paper-level co-occurrence. Synergy,
trade-off, and direction labels are provisional lexical classifications; outcome
order plus a connector does not establish a causal pathway between SDGs.

The manuscript baseline has 1,654 rows in 318 papers and 1,200 unique combinations.
At score >=0.65 there are 646 rows, 501 combinations, and 176 papers; at >=0.75
there are 157 rows. Of 65 cross-SDG candidates in 44 papers, 11 are same-span and
54 are paper-level co-occurrences. Direction is indeterminate for 61; four have
provisional SDG16→SDG3 cues. These are descriptive candidate counts.

## 4. Paper similarity and graph construction

The network program reads eligible papers and their original DOCX paths from the
master. It prioritizes abstract, introduction/background, discussion, and
conclusion text, excludes references, and caps text at 50,000 characters per paper.
If fewer than 800 characters are found in the target sections, it falls back to
body text before references. The text manifest records this extraction basis,
lengths, and sections for audit.

TF-IDF uses English stop words, unigrams/bigrams, sublinear term frequency,
min_df=3, max_df=0.90, and up to 25,000 features. Randomized truncated SVD uses
200 components (bounded by available dimensions), seven iterations, and seed 42.
Reduced vectors are L2-normalized; their dot products give all-pair cosine
similarity. Summary distributions include **all unordered paper pairs**, not just
the selected network edges. Pairs share papers and are not independent samples.

For each paper, up to k nearest neighbours with nonnegative similarity are
selected. An undirected edge is retained if either endpoint selected the other;
self-loops and reciprocal duplicates are removed. Hence degree can exceed k.
The main graph uses k=10, with sensitivity at k=5 and k=15. This construction
encourages connectivity; connectivity is not independent evidence that the SDGs
are integrated in practice.

Edge weight is cosine similarity. Shortest-path distance for approximate
betweenness is max(1e-6, 1−similarity), using up to 200 sampled sources and seed 42.
The program computes degree, weighted strength, PageRank, participation across
source domains, and weighted Louvain communities/modularity. Bridge rankings
first use opposite-domain similarity strength, then betweenness and total strength.
For mixed-domain papers, opposite-domain strength is the smaller of their SDG3
and SDG16 neighbour-strength sums. High rank can reflect shared adoption or
business vocabulary, not substantive health–governance integration.

## 5. Two different networks

The **paper-similarity graph** has 599 paper nodes and 4,307 edges in the baseline.
The **heterogeneous knowledge network** has 656 nodes and 6,623 edges, combining
paper similarity, coded paper/category associations, and outcome-to-SDG taxonomy.
These are different objects: 4,307 is not the knowledge-network edge total.
Aggregated technology–mechanism–outcome figures count supporting papers and show
selected leading categories. They do not estimate mediation or causal effects.

GraphML/GEXF exports retain edge types so that similarity, coding, and taxonomy
relations can be distinguished in downstream tools. Repeated aggregate links
must not be interpreted as independent observations or sequential causal chains.

## 6. Validation and disclosure

Completed human validation should be reported using its existing sample, scope,
agreement statistics, and adjudication records. This release does not rerun it.
Neither figure regeneration nor software regression tests are human validation.
Corpus/domain validation does not automatically transfer to new technology,
outcome, polarity, or direction fields. Unless matching relation-level validation
records exist, report these outputs as automated candidates. Review fields are
provided for provenance, not evidence that review has happened.

The revised computation and drawing code do not call an LLM, generative-image
service, or text-generation API. This does not imply that no AI assistance was used
in developing code or editing prose. Any editorial-assistance disclosure should
accurately describe the actual work and follow the target journal's requirements.

## 7. Reproducibility and private data

Keep the original corpus and finalized human-review files read-only. Run into a
new output folder; reruns do not merge human decisions. Retain quality reports,
codebook/script hashes, network configuration, text manifest, dependency versions,
and figure manifests alongside the analysis. Portability/diagnostic changes in
this public version change script hashes; they do not retroactively change the
hashes in the saved manuscript run. The coding dictionaries are unchanged.

The private source corpus is required to reproduce its numerical findings.
Synthetic tests establish software behavior only. Do not commit evidence text,
full-text extracts, local source paths, review workbooks, or copyrighted papers.
Manuscripts and generated figures remain local rather than being silently
published with this code update.
