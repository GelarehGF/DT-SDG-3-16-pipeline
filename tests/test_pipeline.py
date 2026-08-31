"""Copyright-free software regression tests; not human-validation evidence."""

import contextlib
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET
import zipfile

import networkx as nx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "technology_outcome_pipeline"))
import pipeline
import build_knowledge_network as network
import make_network_figures_v10 as figures


def write_docx(path, paragraphs):
    path.parent.mkdir(parents=True, exist_ok=True)
    document = ET.Element(f"{{{pipeline.W_NS}}}document")
    body = ET.SubElement(document, f"{{{pipeline.W_NS}}}body")
    for text in paragraphs:
        paragraph = ET.SubElement(body, f"{{{pipeline.W_NS}}}p")
        run = ET.SubElement(paragraph, f"{{{pipeline.W_NS}}}r")
        ET.SubElement(run, f"{{{pipeline.W_NS}}}t").text = text
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("word/document.xml", ET.tostring(document))


def fixture_graph():
    metrics = pd.DataFrame({
        "paper_id": ["a", "b", "c", "d"],
        "source_domain": ["SDG3", "SDG3", "SDG16", "Mixed"],
        "degree": [1, 2, 2, 1], "community_id": [1, 1, 2, 2],
        "bridge_rank": [1, 2, 3, 4],
    })
    domains = metrics.set_index("paper_id").source_domain.to_dict()
    edges = pd.DataFrame([
        {"source": a, "target": b, "similarity": weight,
         "source_domain": domains[a], "target_domain": domains[b],
         "domain_pair": network.domain_pair_label(domains[a], domains[b])}
        for a, b, weight in [("a", "b", 0.8), ("b", "c", 0.4), ("c", "d", 0.7)]
    ])
    positions = pd.DataFrame({"paper_id": list("abcd"), "x": [0., 1., 0., -1.], "y": [0., 0., 1., 0.]})
    return edges, metrics, positions


class CodingTests(unittest.TestCase):
    def test_codebook_compiles_and_has_expected_categories(self):
        data = json.loads((ROOT / "technology_outcome_pipeline/codebook.json").read_text())
        for key, size in (("technologies", 18), ("mechanisms", 13), ("outcomes", 24)):
            self.assertEqual(len(data[key]), size)
            pipeline.compile_entries(data[key])
            self.assertEqual(len({entry["code"] for entry in data[key]}), size)

    def test_reference_text_is_excluded(self):
        sentences = pipeline.split_sentences(["Abstract", "Telemedicine improves access to healthcare.", "References", "Blockchain improves accountability."])
        text = " ".join(sentence.text for sentence in sentences)
        self.assertIn("Telemedicine", text)
        self.assertNotIn("Blockchain", text)

    def test_canonical_layout_excludes_unrelated_manuscript(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_docx(root / "SDG3/Words/SDG3-1.docx", ["Synthetic health record"])
            write_docx(root / "manuscript.docx", ["Not a source paper"])
            records = pipeline.discover_canonical_files(root)
            self.assertEqual([(p.stem, d) for p, d in records], [("SDG3-1", "SDG3")])

    def test_end_to_end_deduplication_and_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            health = ["Synthetic health services study of remote care access", "Abstract",
                      "Telemedicine improves access to healthcare through remote consultations.",
                      "Results", "Telemedicine improves patient engagement and health equity."]
            governance = ["Synthetic institutional governance study of accountable public administration",
                          "Abstract", "Blockchain improves transparency and accountability through data sharing.",
                          "Results", "E-government improves public service delivery and citizen participation."]
            write_docx(source / "SDG3/Words/SDG3-1.docx", health)
            write_docx(source / "SDG3/Words/SDG3-2.docx", health)
            write_docx(source / "SDG16/Words/SDG16-1.docx", governance)
            args = ["--source", str(source), "--output", str(root / "run1"), "--skip-workbook"]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(pipeline.main(args), 0)
            report = json.loads((root / "run1/quality_report.json").read_text())
            self.assertEqual((report["files_selected"], report["duplicate_records"], report["analysis_eligible_papers"]), (3, 1, 2))
            self.assertGreater(report["technology_outcome_relations"], 0)
            with (root / "run1/master_technology_outcome_relations.csv").open(encoding="utf-8-sig") as handle:
                rows = list(csv.DictReader(handle))
            self.assertNotIn("SDG3-2", {row["paper_id"] for row in rows})
            args[3] = str(root / "run2")
            with contextlib.redirect_stdout(io.StringIO()):
                pipeline.main(args)
            for filename in ("master_technology_outcome_relations.csv", "cross_sdg_relations.csv"):
                self.assertEqual((root / "run1" / filename).read_bytes(), (root / "run2" / filename).read_bytes())

    def test_nested_output_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(SystemExit, "non-nested"):
                pipeline.main(["--source", directory, "--output", str(Path(directory) / "out"), "--skip-workbook"])

    def test_optional_workbook_treats_formula_like_evidence_as_text(self):
        from openpyxl import load_workbook
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = root / "payload.json"
            payload.write_text(json.dumps({"sheets": {"Evidence": [{"text": "=1+1"}]}}))
            ok, message = pipeline.export_workbook(root, payload, root / "master.xlsx", root)
            self.assertTrue(ok, message)
            workbook = load_workbook(root / "master.xlsx")
            self.assertEqual(workbook["Evidence"]["A2"].value, "=1+1")
            self.assertEqual(workbook["Evidence"]["A2"].data_type, "s")
            workbook.close()


class NetworkTests(unittest.TestCase):
    def test_knn_is_undirected_union_without_self_loops(self):
        embeddings = np.array([[1., 0.], [0.8, 0.6], [0., 1.]])
        edges = network.knn_edges(["a", "b", "c"], embeddings, 1, 0.)
        self.assertEqual({(e.source, e.target) for e in edges}, {("a", "b"), ("b", "c")})
        self.assertEqual(len(edges), 2)

    def test_similarity_matrix_is_symmetric_with_unit_diagonal(self):
        matrix = network.full_similarity_matrix(np.eye(3))
        np.testing.assert_allclose(matrix, matrix.T)
        np.testing.assert_allclose(np.diag(matrix), 1.)

    def test_all_pair_summary_not_just_knn_edges(self):
        papers = pd.DataFrame({"paper_id": list("abcd"), "source_domain": ["SDG3", "SDG3", "SDG16", "Mixed"]})
        pairs, summary = network.similarity_group_data(np.eye(4), papers)
        self.assertEqual(len(pairs), 6)
        self.assertEqual(summary.pair_count.sum(), 6)

    def test_graph_metrics_honors_seed(self):
        graph = nx.path_graph(list("abcd"))
        nx.set_edge_attributes(graph, 0.5, "similarity")
        papers = pd.DataFrame({"paper_id": list("abcd"), "source_domain": ["SDG3", "SDG3", "SDG16", "Mixed"]})
        with patch.object(nx.community, "louvain_communities", wraps=nx.community.louvain_communities) as detect:
            result = network.graph_metrics(graph, papers, 1, seed=7)
            self.assertEqual(detect.call_args.kwargs["seed"], 7)
        self.assertEqual(result["components"], 1)

    def test_figure_counts_are_derived_not_hardcoded(self):
        edges, metrics, positions = fixture_graph()
        figures.validate_data(edges, metrics, positions)
        summary = figures.graph_summary(edges, metrics)
        self.assertEqual((summary["nodes"], summary["edges"], summary["communities"]), (4, 3, 2))
        self.assertEqual((summary["direct_cross_edges"], summary["mixed_edges"]), (1, 1))

    def test_display_filter_does_not_change_analysis(self):
        edges, metrics, _positions = fixture_graph()
        before = edges.copy(deep=True)
        with patch.object(figures, "CROSS_EDGES", 0):
            other, cross = figures.display_edges(edges)
        self.assertEqual(len(cross), 0)
        self.assertLess(len(other), len(edges))
        pd.testing.assert_frame_equal(edges, before)
        self.assertEqual(figures.graph_summary(edges, metrics)["direct_cross_edges"], 1)

    def test_layout_repeats_with_same_seed(self):
        edges, metrics, positions = fixture_graph()
        initial = positions.set_index("paper_id")[["x", "y"]].apply(tuple, axis=1).to_dict()
        first = figures.force_layout(metrics.paper_id.tolist(), edges, initial, 10, 42)
        second = figures.force_layout(metrics.paper_id.tolist(), edges, initial, 10, 42)
        self.assertEqual(first, second)

    def test_bad_edges_are_rejected(self):
        edges, metrics, positions = fixture_graph()
        edges.loc[0, "target"] = "missing"
        with self.assertRaisesRegex(ValueError, "unknown paper"):
            figures.validate_data(edges, metrics, positions)


if __name__ == "__main__":
    unittest.main()
