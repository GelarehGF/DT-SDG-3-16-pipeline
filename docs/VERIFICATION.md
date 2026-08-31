# Verification record

Date: 31 August 2026. Environment: macOS, Python 3.12, with the installed
scientific-library versions pinned in `requirements.txt`.

## Software checks

- All 14 synthetic regression tests passed. They cover codebook compilation,
  canonical file selection, reference exclusion, duplicate handling, repeatable
  extraction, source/output separation, literal spreadsheet text, undirected kNN
  union, all-pair similarity summaries, seed propagation, data-derived figure
  statistics, display/analysis separation, repeatable layouts, and invalid edges.
- All analysis/figure modules compiled successfully.
- The main analytical figure generator was run headlessly; a macOS graphical
  backend failure was corrected by selecting the non-interactive Agg backend.
- Six readable non-network manuscript figures and four network views were
  generated. Contact-sheet/individual visual checks identified overlapping node
  labels; label placement was adjusted and the affected network figure rechecked.
- Public figure scripts use portable fonts and command-line input/output paths.
  Network counts are calculated from inputs, not hard-coded manuscript values.

## Private-corpus reproduction

The canonical extraction was rerun locally: 605 parsed records, zero parse errors,
six duplicate records, 599 eligible papers, 1,654 candidate evidence rows in 318
papers, 65 cross-SDG candidates, and 11 same-span cross-SDG candidates.
The paper manifest, relation table, cross-SDG table, technology–outcome matrix,
and flattened codebook matched the saved baseline byte-for-byte.

The revised network program was also rerun from the saved master manifest and
the original DOCX corpus. It reproduced the saved paper-similarity matrix, edge
list, paper metrics, sensitivity table, and heterogeneous network node/edge tables
**byte-for-byte**. Summary: 599 papers, 4,307 similarity edges, nine communities,
and a heterogeneous network of 656 nodes and 6,623 edges.

The display manifest independently reports 658 direct cross-SDG edges, 1,090
edges involving mixed-domain papers, weighted modularity 0.5338513004434338,
and 548 context edges plus 150 direct cross-SDG edges in the filtered main view.
All 599 nodes remain in that display. Full supplements retain all 4,307 edges.

The unchanged codebook SHA-256 is
`ced0053766581204c6d8b53f59e8978bf021c9e17fc134569947077faa21e665`.
Script hashes differ from the earlier local scripts because this release adds
portability, argument checks, and figure-label corrections.

## Preservation and limits

- The original notebook is byte-for-byte unchanged.
- The README retains its section hierarchy and order. Title/subtitle and citation
  now match the revised manuscript; obsolete totals/method claims are corrected.
- Source corpus, manuscripts, extracted evidence, human-review workbooks, local
  paths, and generated figures are not part of this repository update.
- These checks are software/reproduction checks, not repeated human validation.
  They neither establish causal effects nor certify automated relation labels.
- The PDF fallback and legacy MiniLM notebook were not rerun. A fresh installation
  on Windows/Linux was not tested; exact layout equality across platforms is not
  guaranteed. The canonical DOCX route and revised terminal generators were tested.
