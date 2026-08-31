#!/usr/bin/env python3
"""Build paper similarity and a multilayer DT-SDG knowledge network.

The similarity model is deterministic and non-generative:

1. Read the cleaned, analysis-eligible paper manifest.
2. Extract abstract/introduction/discussion/conclusion text, with a documented
   body-text fallback when headings are unavailable.
3. Fit TF-IDF (unigrams + bigrams), reduce with truncated SVD, and L2-normalize.
4. Calculate cosine paper similarity and k-nearest-neighbour networks.
5. Combine paper similarity with technology, mechanism, outcome, and SDG
   relationships from the auditable master coding table.

All generated networks are descriptive candidates until the underlying coding
has completed human review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from textwrap import fill
from typing import Any, Iterable, Sequence
from xml.etree import ElementTree as ET

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath
from matplotlib.patches import FancyBboxPatch, PathPatch
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
SPACE_RE = re.compile(r"\s+")
REFERENCE_HEADING_RE = re.compile(
    r"^(?:references|bibliography|works cited|literature cited)(?:\s*[:.]?)$", re.I
)


def compact(value: str) -> str:
    return SPACE_RE.sub(" ", value or "").strip()


def paragraph_text(element: ET.Element) -> str:
    chunks: list[str] = []
    for node in element.iter():
        tag = node.tag.rsplit("}", 1)[-1]
        if tag == "t" and node.text:
            chunks.append(node.text)
        elif tag == "tab":
            chunks.append("\t")
        elif tag in {"br", "cr"}:
            chunks.append("\n")
    return compact("".join(chunks))


def read_document(path: Path) -> tuple[list[str], dict[str, str]]:
    """Read the canonical DOCX corpus with only Python's standard library."""
    if path.suffix.casefold() != ".docx":
        raise ValueError(f"Knowledge-network input must be canonical DOCX: {path}")
    paragraphs: list[str] = []
    metadata: dict[str, str] = {}
    with zipfile.ZipFile(path) as archive:
        root = ET.fromstring(archive.read("word/document.xml"))
        for paragraph in root.iter(f"{{{W_NS}}}p"):
            text = paragraph_text(paragraph)
            if text:
                paragraphs.append(text)
        if "docProps/core.xml" in archive.namelist():
            core = ET.fromstring(archive.read("docProps/core.xml"))
            for child in core:
                key = child.tag.rsplit("}", 1)[-1]
                if child.text:
                    metadata[key] = compact(child.text)
    return paragraphs, metadata


BLUE = "#2F6690"
ORANGE = "#D17A22"
GREEN = "#4C956C"
PURPLE = "#7B6FAE"
TEAL = "#3A8D8F"
RED = "#C44E52"
GOLD = "#C8A24A"
GREY = "#747B84"
LIGHT_GREY = "#E6EAEE"
DARK = "#263238"
DOMAIN_COLORS = {"SDG3": BLUE, "SDG16": ORANGE, "Mixed": GREEN, "Unassigned": GREY}

SECTION_RE = re.compile(
    r"^(?:\d+(?:\.\d+)*[.)]?\s*)?"
    r"(abstract|introduction|background|literature review|theoretical framework|"
    r"methods?|methodology|results?|findings?|discussion|conclusions?|limitations?|"
    r"implications?|recommendations?)\b\s*[:.—-]?\s*(.*)$",
    re.I,
)
TARGET_SECTIONS = {"abstract", "introduction", "background", "discussion", "conclusion", "conclusions"}

TECH_SHORT = {
    "TECH_PLATFORM_GENERAL": "General digital\nplatforms & ICT",
    "TECH_EGOV": "E-government &\npublic services",
    "TECH_TELEHEALTH": "Telehealth &\nremote care",
    "TECH_CYBER_PRIVACY": "Cybersecurity &\nprivacy technologies",
    "TECH_AI_ML": "AI & machine\nlearning",
    "TECH_DPI_INTEROP": "Digital public infrastructure\n& interoperability",
    "TECH_MHEALTH": "Mobile health",
    "TECH_DATA_ANALYTICS": "Big data &\nanalytics",
    "TECH_EHR_HIS": "Electronic records &\nhealth information systems",
    "TECH_BLOCKCHAIN": "Blockchain &\ndistributed ledgers",
}

OUTCOME_SHORT = {
    "SDG3_ACCESS": "Health-service\naccess",
    "SDG3_QUALITY": "Care quality\n& safety",
    "SDG3_CLINICAL": "Clinical & population\nhealth outcomes",
    "SDG3_COST_EFF": "Health-system cost\n& efficiency",
    "SDG3_PATIENT": "Patient engagement\n& self-management",
    "SDG3_EQUITY": "Health equity",
    "SDG3_PREVENT": "Prevention &\npreparedness",
    "SDG3_CONTINUITY": "Continuity &\ncoordination of care",
    "SDG16_TRANSPARENCY": "Transparency",
    "SDG16_ACCOUNT": "Accountability",
    "SDG16_CAPACITY": "Institutional capacity\n& effectiveness",
    "SDG16_SERVICE": "Inclusive public-service\naccess",
    "SDG16_PARTICIPATION": "Participation &\nresponsive governance",
    "SDG16_TRUST": "Public trust &\nlegitimacy",
    "SDG16_CORRUPTION": "Corruption control\n& integrity",
    "SDG16_PRIVACY": "Privacy & data\nprotection",
    "SDG16_CYBER": "Cybersecurity &\ninstitutional resilience",
}


@dataclass(frozen=True)
class SimilarityEdge:
    source: str
    target: str
    similarity: float


def configure_plot_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "axes.edgecolor": "#9AA1A8",
            "axes.linewidth": 0.7,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
            "axes.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def save_figure(fig: plt.Figure, folder: Path, stem: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg", "pdf"):
        fig.savefig(folder / f"{stem}.{extension}", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_section(value: str) -> str:
    return value.casefold().strip().rstrip("s") if value.casefold().strip() not in {"conclusions"} else "conclusion"


def select_similarity_text(paragraphs: Sequence[str], max_chars: int) -> tuple[str, str]:
    """Return focused text and a human-readable extraction basis."""
    selected: list[str] = []
    body: list[str] = []
    active_section = "front matter"
    sections_found: set[str] = set()
    for raw in paragraphs:
        text = compact(raw)
        if not text:
            continue
        if REFERENCE_HEADING_RE.match(text):
            break
        heading = SECTION_RE.match(text)
        if heading and len(text.split()) <= 16:
            active_section = normalize_section(heading.group(1))
            sections_found.add(active_section)
            remainder = compact(heading.group(2))
            if remainder and active_section in TARGET_SECTIONS:
                selected.append(remainder)
            continue
        body.append(text)
        if active_section in TARGET_SECTIONS:
            selected.append(text)

    focused = compact(" ".join(selected))
    if len(focused) >= 800:
        basis = "target sections: " + ", ".join(sorted(sections_found & TARGET_SECTIONS))
        return focused[:max_chars], basis

    fallback = compact(" ".join(body))[:max_chars]
    return fallback, "body-text fallback before references"


def load_similarity_texts(papers: pd.DataFrame, max_chars: int) -> tuple[list[str], pd.DataFrame]:
    texts: list[str] = []
    records: list[dict[str, Any]] = []
    for position, row in papers.iterrows():
        path = Path(str(row["source_path"]))
        try:
            paragraphs, _metadata = read_document(path)
            text, basis = select_similarity_text(paragraphs, max_chars)
            status = "OK" if text else "Empty"
        except Exception as exc:
            text, basis, status = "", "read error", repr(exc)
        texts.append(text)
        records.append(
            {
                "paper_id": row["paper_id"],
                "source_domain": row["source_domain"],
                "source_path": str(path),
                "extraction_basis": basis,
                "similarity_text_chars": len(text),
                "similarity_text_words": len(re.findall(r"\b\w+[\w'-]*\b", text)),
                "similarity_text_sha256": sha256_text(text),
                "status": status,
            }
        )
        if (position + 1) % 50 == 0 or position + 1 == len(papers):
            print(f"Similarity text {position + 1}/{len(papers)}", flush=True)
    return texts, pd.DataFrame(records)


def build_embeddings(
    texts: Sequence[str],
    max_features: int,
    min_df: int,
    svd_components: int,
    seed: int,
) -> tuple[Any, np.ndarray, TfidfVectorizer, TruncatedSVD]:
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        stop_words="english",
        ngram_range=(1, 2),
        min_df=min_df,
        max_df=0.90,
        max_features=max_features,
        sublinear_tf=True,
        norm="l2",
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9_-]{2,}\b",
    )
    tfidf = vectorizer.fit_transform(texts)
    components = min(svd_components, tfidf.shape[0] - 1, tfidf.shape[1] - 1)
    if components < 2:
        raise ValueError(f"Insufficient TF-IDF dimensionality: {tfidf.shape}")
    svd = TruncatedSVD(n_components=components, algorithm="randomized", n_iter=7, random_state=seed)
    reduced = svd.fit_transform(tfidf)
    embeddings = normalize(reduced, norm="l2")
    return tfidf, np.asarray(embeddings), vectorizer, svd


def full_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    matrix = embeddings @ embeddings.T
    matrix = np.clip(matrix, -1.0, 1.0)
    np.fill_diagonal(matrix, 1.0)
    return matrix


def knn_edges(
    paper_ids: Sequence[str],
    embeddings: np.ndarray,
    k: int,
    min_similarity: float,
) -> list[SimilarityEdge]:
    neighbors = min(k + 1, len(paper_ids))
    model = NearestNeighbors(n_neighbors=neighbors, metric="cosine", algorithm="brute")
    model.fit(embeddings)
    distances, indices = model.kneighbors(embeddings)
    edge_values: dict[tuple[str, str], float] = {}
    for row_index, (row_distances, row_indices) in enumerate(zip(distances, indices)):
        source = paper_ids[row_index]
        retained = 0
        for distance, neighbor_index in zip(row_distances, row_indices):
            target = paper_ids[int(neighbor_index)]
            if source == target:
                continue
            similarity = float(1.0 - distance)
            if similarity < min_similarity:
                continue
            key = tuple(sorted((source, target)))
            edge_values[key] = max(edge_values.get(key, -1.0), similarity)
            retained += 1
            if retained >= k:
                break
    return [SimilarityEdge(source, target, round(value, 8)) for (source, target), value in sorted(edge_values.items())]


def make_similarity_graph(papers: pd.DataFrame, edges: Sequence[SimilarityEdge]) -> nx.Graph:
    graph = nx.Graph()
    for _, row in papers.iterrows():
        graph.add_node(
            row["paper_id"],
            label=row["paper_id"],
            title=str(row.get("title", "") or ""),
            source_domain=str(row.get("source_domain", "Unassigned")),
            publication_year=str(row.get("publication_year_candidate", "") or ""),
        )
    for edge in edges:
        distance = max(1e-6, 1.0 - edge.similarity)
        graph.add_edge(edge.source, edge.target, similarity=float(edge.similarity), weight=float(edge.similarity), distance=distance)
    return graph


def domain_pair_label(domain_a: str, domain_b: str) -> str:
    pair = tuple(sorted((domain_a, domain_b)))
    labels = {
        ("SDG3", "SDG3"): "Within SDG 3",
        ("SDG16", "SDG16"): "Within SDG 16",
        ("SDG16", "SDG3"): "Cross SDG 3–SDG 16",
        ("Mixed", "Mixed"): "Within mixed-domain",
    }
    return labels.get(pair, "Involving mixed-domain")


def similarity_group_data(matrix: np.ndarray, papers: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    domains = papers["source_domain"].astype(str).tolist()
    rows: list[dict[str, Any]] = []
    for left in range(len(papers)):
        for right in range(left + 1, len(papers)):
            rows.append(
                {
                    "paper_id_a": papers.iloc[left]["paper_id"],
                    "paper_id_b": papers.iloc[right]["paper_id"],
                    "domain_a": domains[left],
                    "domain_b": domains[right],
                    "comparison_group": domain_pair_label(domains[left], domains[right]),
                    "cosine_similarity": float(matrix[left, right]),
                }
            )
    long = pd.DataFrame(rows)
    summary = (
        long.groupby("comparison_group")["cosine_similarity"]
        .agg(pair_count="size", mean="mean", median="median", std="std", minimum="min", maximum="max")
        .reset_index()
    )
    quantiles = long.groupby("comparison_group")["cosine_similarity"].quantile([0.25, 0.75]).unstack()
    quantiles.columns = ["q25", "q75"]
    summary = summary.merge(quantiles.reset_index(), on="comparison_group", how="left")
    return long, summary


def graph_metrics(graph: nx.Graph, papers: pd.DataFrame, k: int, seed: int = 42) -> dict[str, Any]:
    domains = papers.set_index("paper_id")["source_domain"].astype(str).to_dict()
    components = list(nx.connected_components(graph))
    edge_domains = Counter()
    for source, target in graph.edges():
        edge_domains[domain_pair_label(domains[source], domains[target])] += 1
    if graph.number_of_edges():
        communities = nx.community.louvain_communities(graph, weight="similarity", seed=seed)
        modularity = nx.community.modularity(graph, communities, weight="similarity")
        clustering = nx.average_clustering(graph, weight="similarity")
    else:
        communities, modularity, clustering = [], float("nan"), float("nan")
    return {
        "k": k,
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "density": nx.density(graph),
        "components": len(components),
        "largest_component": max((len(component) for component in components), default=0),
        "isolates": nx.number_of_isolates(graph),
        "mean_degree": float(np.mean([degree for _node, degree in graph.degree()])),
        "average_weighted_clustering": clustering,
        "louvain_communities": len(communities),
        "louvain_modularity": modularity,
        "within_sdg3_edges": edge_domains["Within SDG 3"],
        "within_sdg16_edges": edge_domains["Within SDG 16"],
        "cross_sdg3_sdg16_edges": edge_domains["Cross SDG 3–SDG 16"],
        "involving_mixed_edges": edge_domains["Involving mixed-domain"] + edge_domains["Within mixed-domain"],
        "cross_sdg3_sdg16_share": edge_domains["Cross SDG 3–SDG 16"] / max(1, graph.number_of_edges()),
    }


def participation_coefficient(graph: nx.Graph, node: str, domains: dict[str, str]) -> float:
    strengths = Counter()
    total = 0.0
    for neighbor, attributes in graph[node].items():
        value = float(attributes.get("similarity", 1.0))
        strengths[domains.get(neighbor, "Unassigned")] += value
        total += value
    if total <= 0:
        return 0.0
    return 1.0 - sum((value / total) ** 2 for value in strengths.values())


def paper_network_metrics(graph: nx.Graph, papers: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, list[set[str]]]:
    domains = papers.set_index("paper_id")["source_domain"].astype(str).to_dict()
    titles = papers.set_index("paper_id")["title"].fillna("").astype(str).to_dict()
    pagerank = nx.pagerank(graph, weight="similarity") if graph.number_of_edges() else {node: 0.0 for node in graph}
    sample_size = min(200, graph.number_of_nodes())
    betweenness = (
        nx.betweenness_centrality(graph, k=sample_size, normalized=True, weight="distance", seed=seed)
        if graph.number_of_edges()
        else {node: 0.0 for node in graph}
    )
    communities = nx.community.louvain_communities(graph, weight="similarity", seed=seed) if graph.number_of_edges() else [{node} for node in graph]
    community_by_node = {node: index + 1 for index, community in enumerate(communities) for node in community}

    records: list[dict[str, Any]] = []
    for node in graph.nodes():
        own_domain = domains.get(node, "Unassigned")
        strength_by_domain = Counter()
        total_strength = 0.0
        for neighbor, attributes in graph[node].items():
            similarity = float(attributes.get("similarity", 0.0))
            strength_by_domain[domains.get(neighbor, "Unassigned")] += similarity
            total_strength += similarity
        if own_domain == "SDG3":
            opposite_strength = strength_by_domain["SDG16"]
        elif own_domain == "SDG16":
            opposite_strength = strength_by_domain["SDG3"]
        elif own_domain == "Mixed":
            opposite_strength = min(strength_by_domain["SDG3"], strength_by_domain["SDG16"])
        else:
            opposite_strength = 0.0
        cross_neighbors = sum(1 for neighbor in graph.neighbors(node) if domains.get(neighbor) != own_domain)
        records.append(
            {
                "paper_id": node,
                "title": titles.get(node, ""),
                "source_domain": own_domain,
                "degree": graph.degree(node),
                "weighted_similarity_degree": total_strength,
                "sdg3_neighbor_strength": strength_by_domain["SDG3"],
                "sdg16_neighbor_strength": strength_by_domain["SDG16"],
                "mixed_neighbor_strength": strength_by_domain["Mixed"],
                "opposite_domain_similarity_strength": opposite_strength,
                "cross_domain_neighbor_count": cross_neighbors,
                "cross_domain_neighbor_share": cross_neighbors / max(1, graph.degree(node)),
                "participation_coefficient": participation_coefficient(graph, node, domains),
                "pagerank": pagerank.get(node, 0.0),
                "approximate_betweenness": betweenness.get(node, 0.0),
                "community_id": community_by_node.get(node, 0),
            }
        )
    frame = pd.DataFrame(records).sort_values(
        ["opposite_domain_similarity_strength", "approximate_betweenness", "weighted_similarity_degree"],
        ascending=False,
    )
    frame.insert(0, "bridge_rank", np.arange(1, len(frame) + 1))
    return frame, communities


def codebook_maps(codebook: dict[str, Any]) -> dict[str, dict[str, str]]:
    maps: dict[str, dict[str, str]] = {}
    for group in ("technologies", "mechanisms", "outcomes"):
        maps[group] = {entry["code"]: entry["label"] for entry in codebook[group]}
    return maps


def split_codes(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    return [part.strip() for part in str(value).split("|") if part.strip()]


def build_knowledge_graph(
    papers: pd.DataFrame,
    relations: pd.DataFrame,
    similarity_edges: Sequence[SimilarityEdge],
    codebook: dict[str, Any],
) -> tuple[nx.Graph, pd.DataFrame, pd.DataFrame]:
    graph = nx.Graph()
    maps = codebook_maps(codebook)

    for _, row in papers.iterrows():
        graph.add_node(
            f"paper:{row['paper_id']}",
            node_type="paper",
            label=str(row["paper_id"]),
            title=str(row.get("title", "") or ""),
            source_domain=str(row.get("source_domain", "Unassigned")),
            publication_year=str(row.get("publication_year_candidate", "") or ""),
        )
    for code, label in maps["technologies"].items():
        graph.add_node(f"technology:{code}", node_type="technology", label=label, code=code)
    for code, label in maps["mechanisms"].items():
        graph.add_node(f"mechanism:{code}", node_type="mechanism", label=label, code=code)
    for code, label in maps["outcomes"].items():
        sdg = "SDG3" if code.startswith("SDG3_") else "SDG16"
        graph.add_node(f"outcome:{code}", node_type="outcome", label=label, code=code, sdg=sdg)
    graph.add_node("sdg:SDG3", node_type="sdg", label="SDG 3: Good Health and Well-being", code="SDG3")
    graph.add_node("sdg:SDG16", node_type="sdg", label="SDG 16: Peace, Justice and Strong Institutions", code="SDG16")

    edge_accumulator: dict[tuple[str, str, str], dict[str, Any]] = {}

    def add_coded_edge(source: str, target: str, relation: str, paper_id: str) -> None:
        key = (source, target, relation)
        if key not in edge_accumulator:
            edge_accumulator[key] = {"papers": set(), "evidence_count": 0}
        edge_accumulator[key]["papers"].add(paper_id)
        edge_accumulator[key]["evidence_count"] += 1

    for _, row in relations.iterrows():
        paper_id = str(row["paper_id"])
        paper_node = f"paper:{paper_id}"
        technology_code = str(row["technology_code"])
        outcome_code = str(row["outcome_code"])
        technology_node = f"technology:{technology_code}"
        outcome_node = f"outcome:{outcome_code}"
        add_coded_edge(paper_node, technology_node, "uses_technology", paper_id)
        add_coded_edge(paper_node, outcome_node, "reports_outcome", paper_id)
        add_coded_edge(technology_node, outcome_node, "associated_with_outcome", paper_id)
        for mechanism_code in split_codes(row.get("mechanism_codes")):
            mechanism_node = f"mechanism:{mechanism_code}"
            if mechanism_node not in graph:
                graph.add_node(mechanism_node, node_type="mechanism", label=mechanism_code, code=mechanism_code)
            add_coded_edge(paper_node, mechanism_node, "reports_mechanism", paper_id)
            add_coded_edge(technology_node, mechanism_node, "enables_mechanism", paper_id)
            add_coded_edge(mechanism_node, outcome_node, "linked_to_outcome", paper_id)

    for (source, target, relation), values in edge_accumulator.items():
        graph.add_edge(
            source,
            target,
            relation=relation,
            edge_type="coded_relation",
            weight=int(len(values["papers"])),
            paper_count=int(len(values["papers"])),
            evidence_count=int(values["evidence_count"]),
        )

    outcome_papers: dict[str, set[str]] = defaultdict(set)
    for _, row in relations.iterrows():
        outcome_papers[str(row["outcome_code"])].add(str(row["paper_id"]))
    for outcome_code, paper_ids in outcome_papers.items():
        sdg_code = "SDG3" if outcome_code.startswith("SDG3_") else "SDG16"
        graph.add_edge(
            f"outcome:{outcome_code}",
            f"sdg:{sdg_code}",
            relation="belongs_to_sdg",
            edge_type="taxonomy",
            weight=len(paper_ids),
            paper_count=len(paper_ids),
            evidence_count=len(paper_ids),
        )

    for edge in similarity_edges:
        graph.add_edge(
            f"paper:{edge.source}",
            f"paper:{edge.target}",
            relation="semantically_similar",
            edge_type="paper_similarity",
            weight=float(edge.similarity),
            similarity=float(edge.similarity),
            distance=max(1e-6, 1.0 - float(edge.similarity)),
            paper_count=2,
            evidence_count=1,
        )

    node_rows = []
    for node, attributes in graph.nodes(data=True):
        node_rows.append({"node_id": node, **attributes, "degree": graph.degree(node), "weighted_degree": graph.degree(node, weight="weight")})
    edge_rows = []
    for source, target, attributes in graph.edges(data=True):
        edge_rows.append({"source": source, "target": target, **attributes})
    return graph, pd.DataFrame(node_rows), pd.DataFrame(edge_rows)


def sanitize_graph_attributes(graph: nx.Graph) -> nx.Graph:
    sanitized = graph.copy()
    for _node, attributes in sanitized.nodes(data=True):
        for key, value in list(attributes.items()):
            if value is None or (isinstance(value, float) and math.isnan(value)):
                attributes[key] = ""
            elif isinstance(value, (np.integer,)):
                attributes[key] = int(value)
            elif isinstance(value, (np.floating,)):
                attributes[key] = float(value)
            elif not isinstance(value, (str, int, float, bool)):
                attributes[key] = str(value)
    for _source, _target, attributes in sanitized.edges(data=True):
        for key, value in list(attributes.items()):
            if value is None or (isinstance(value, float) and math.isnan(value)):
                attributes[key] = ""
            elif isinstance(value, (np.integer,)):
                attributes[key] = int(value)
            elif isinstance(value, (np.floating,)):
                attributes[key] = float(value)
            elif not isinstance(value, (str, int, float, bool)):
                attributes[key] = str(value)
    return sanitized


def figure_similarity_map(
    embeddings: np.ndarray,
    papers: pd.DataFrame,
    edges: Sequence[SimilarityEdge],
    metrics: pd.DataFrame,
    output: Path,
    seed: int,
) -> pd.DataFrame:
    positions = PCA(n_components=2, random_state=seed).fit_transform(embeddings)
    position_frame = pd.DataFrame(
        {
            "paper_id": papers["paper_id"].tolist(),
            "x": positions[:, 0],
            "y": positions[:, 1],
            "source_domain": papers["source_domain"].tolist(),
        }
    )
    position_map = position_frame.set_index("paper_id")[["x", "y"]].to_dict("index")
    domains = papers.set_index("paper_id")["source_domain"].astype(str).to_dict()
    cross_edges = [edge for edge in edges if {domains[edge.source], domains[edge.target]} == {"SDG3", "SDG16"}]
    cross_edges = sorted(cross_edges, key=lambda edge: edge.similarity, reverse=True)[:80]

    fig, ax = plt.subplots(figsize=(10.5, 7.7), constrained_layout=True)
    for edge in cross_edges:
        left, right = position_map[edge.source], position_map[edge.target]
        ax.plot([left["x"], right["x"]], [left["y"], right["y"]], color=PURPLE, alpha=0.12, linewidth=0.4 + 1.8 * edge.similarity, zorder=1)
    for domain in ("SDG3", "SDG16", "Mixed"):
        subset = position_frame[position_frame["source_domain"] == domain]
        ax.scatter(subset["x"], subset["y"], s=22, color=DOMAIN_COLORS[domain], alpha=0.72, edgecolors="white", linewidths=0.35, label=domain.replace("SDG", "SDG "), zorder=2)
    top_bridges = metrics.head(10)
    for _, row in top_bridges.iterrows():
        position = position_map.get(row["paper_id"])
        if not position:
            continue
        ax.annotate(
            row["paper_id"],
            (position["x"], position["y"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
            color=DARK,
        )
    ax.set_title("Paper similarity map and strongest cross-domain neighbour links")
    ax.set_xlabel("Latent semantic dimension 1")
    ax.set_ylabel("Latent semantic dimension 2")
    ax.grid(color=LIGHT_GREY, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="best")
    fig.text(0.01, -0.01, "Note. Points are PCA projections of normalized TF–IDF/SVD representations. Purple lines show the 80 strongest SDG 3–SDG 16 kNN edges; spatial distance is descriptive.", fontsize=8, color=GREY)
    save_figure(fig, output, "network_figure1_paper_similarity_map")
    return position_frame


def figure_similarity_groups(long: pd.DataFrame, summary: pd.DataFrame, output: Path, seed: int) -> None:
    order = ["Within SDG 3", "Within SDG 16", "Cross SDG 3–SDG 16", "Involving mixed-domain", "Within mixed-domain"]
    available = [group for group in order if group in set(long["comparison_group"])]
    rng = np.random.default_rng(seed)
    data = []
    for group in available:
        values = long.loc[long["comparison_group"] == group, "cosine_similarity"].to_numpy()
        if len(values) > 30000:
            values = rng.choice(values, 30000, replace=False)
        data.append(values)
    fig, ax = plt.subplots(figsize=(10.5, 5.8), constrained_layout=True)
    violin = ax.violinplot(data, positions=np.arange(1, len(available) + 1), showmeans=False, showmedians=True, showextrema=False, widths=0.8)
    colors = [BLUE, ORANGE, PURPLE, GREEN, GREY]
    for body, color in zip(violin["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor("none")
        body.set_alpha(0.65)
    violin["cmedians"].set_color(DARK)
    violin["cmedians"].set_linewidth(1.3)
    ax.set_xticks(np.arange(1, len(available) + 1), [fill(group, 18) for group in available])
    ax.set_ylabel("Cosine similarity")
    ax.set_title("All-pair paper similarity within and across SDG domains")
    ax.grid(axis="y", color=LIGHT_GREY, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    summary_map = summary.set_index("comparison_group").to_dict("index")
    for index, group in enumerate(available, 1):
        values = summary_map[group]
        ax.text(index, ax.get_ylim()[1], f"n={int(values['pair_count']):,}\nmedian={values['median']:.3f}", ha="center", va="top", fontsize=7.5, color=DARK)
    fig.text(0.01, -0.01, "Note. Similarities are calculated from the full normalized latent semantic representations, not only retained network edges.", fontsize=8, color=GREY)
    save_figure(fig, output, "network_figure2_similarity_by_domain")


def short_tech(code: str) -> str:
    return TECH_SHORT.get(code, fill(code.replace("TECH_", "").replace("_", " ").title(), 18))


def short_outcome(code: str) -> str:
    return OUTCOME_SHORT.get(code, fill(code.replace("SDG3_", "").replace("SDG16_", "").replace("_", " ").title(), 18))


def figure_aggregate_knowledge_network(relations: pd.DataFrame, codebook: dict[str, Any], output: Path) -> dict[str, pd.DataFrame]:
    maps = codebook_maps(codebook)
    unique_relations = relations.drop_duplicates(["paper_id", "technology_code", "outcome_code", "mechanism_codes"]).copy()
    tech_counts = unique_relations.groupby("technology_code")["paper_id"].nunique().sort_values(ascending=False)
    technology_codes = tech_counts.head(8).index.tolist()

    expanded_mechanisms = []
    for _, row in unique_relations.iterrows():
        for mechanism in split_codes(row.get("mechanism_codes")):
            expanded_mechanisms.append({"paper_id": row["paper_id"], "technology_code": row["technology_code"], "mechanism_code": mechanism, "outcome_code": row["outcome_code"]})
    mechanism_frame = pd.DataFrame(expanded_mechanisms)
    if mechanism_frame.empty:
        raise ValueError("No mechanism relationships were available for the aggregate knowledge network")
    mechanism_codes = mechanism_frame.groupby("mechanism_code")["paper_id"].nunique().sort_values(ascending=False).head(7).index.tolist()
    sdg3_outcomes = unique_relations[unique_relations["outcome_sdg"] == "SDG3"].groupby("outcome_code")["paper_id"].nunique().sort_values(ascending=False).head(4).index.tolist()
    sdg16_outcomes = unique_relations[unique_relations["outcome_sdg"] == "SDG16"].groupby("outcome_code")["paper_id"].nunique().sort_values(ascending=False).head(4).index.tolist()
    outcome_codes = sdg3_outcomes + sdg16_outcomes

    tm = (
        mechanism_frame[mechanism_frame["technology_code"].isin(technology_codes) & mechanism_frame["mechanism_code"].isin(mechanism_codes)]
        .groupby(["technology_code", "mechanism_code"])["paper_id"].nunique().reset_index(name="paper_count")
        .sort_values("paper_count", ascending=False).head(26)
    )
    mo = (
        mechanism_frame[mechanism_frame["mechanism_code"].isin(mechanism_codes) & mechanism_frame["outcome_code"].isin(outcome_codes)]
        .groupby(["mechanism_code", "outcome_code"])["paper_id"].nunique().reset_index(name="paper_count")
        .sort_values("paper_count", ascending=False).head(28)
    )

    x_values = {"technology": 0.12, "mechanism": 0.50, "outcome": 0.88}
    positions: dict[str, tuple[float, float]] = {}
    for code, y in zip(technology_codes, np.linspace(0.88, 0.12, len(technology_codes))):
        positions[f"technology:{code}"] = (x_values["technology"], y)
    for code, y in zip(mechanism_codes, np.linspace(0.86, 0.14, len(mechanism_codes))):
        positions[f"mechanism:{code}"] = (x_values["mechanism"], y)
    for code, y in zip(outcome_codes, np.linspace(0.88, 0.12, len(outcome_codes))):
        positions[f"outcome:{code}"] = (x_values["outcome"], y)

    fig, ax = plt.subplots(figsize=(14.2, 8.5), constrained_layout=True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    max_edge = max(1, int(max(tm["paper_count"].max(), mo["paper_count"].max())))

    def curve(source: tuple[float, float], target: tuple[float, float], color: str, width: float) -> None:
        x1, y1 = source
        x2, y2 = target
        path = MplPath(
            [(x1, y1), (x1 + (x2 - x1) * 0.48, y1), (x2 - (x2 - x1) * 0.48, y2), (x2, y2)],
            [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4],
        )
        ax.add_patch(PathPatch(path, facecolor="none", edgecolor=color, linewidth=width, alpha=0.25, zorder=1))

    for _, row in tm.sort_values("paper_count").iterrows():
        curve(positions[f"technology:{row['technology_code']}"] , positions[f"mechanism:{row['mechanism_code']}"] , TEAL, 0.6 + 6 * row["paper_count"] / max_edge)
    for _, row in mo.sort_values("paper_count").iterrows():
        color = BLUE if str(row["outcome_code"]).startswith("SDG3_") else ORANGE
        curve(positions[f"mechanism:{row['mechanism_code']}"] , positions[f"outcome:{row['outcome_code']}"] , color, 0.6 + 6 * row["paper_count"] / max_edge)

    def side_node(x: float, y: float, color: str, label: str, side: str) -> None:
        patch = FancyBboxPatch((x - 0.014, y - 0.026), 0.028, 0.052, boxstyle="round,pad=0.006,rounding_size=0.012", facecolor=color, edgecolor="none", zorder=3)
        ax.add_patch(patch)
        if side == "left":
            ax.text(x - 0.024, y, label, ha="right", va="center", fontsize=8, color=DARK, zorder=4)
        else:
            ax.text(x + 0.024, y, label, ha="left", va="center", fontsize=8, color=DARK, zorder=4)

    def mechanism_node(x: float, y: float, label: str) -> None:
        width = 0.145
        height = 0.062
        patch = FancyBboxPatch(
            (x - width / 2, y - height / 2),
            width,
            height,
            boxstyle="round,pad=0.008,rounding_size=0.014",
            facecolor=PURPLE,
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        ax.add_patch(patch)
        ax.text(x, y, fill(label, 21), ha="center", va="center", fontsize=7.2, color="white", linespacing=0.92, zorder=4)

    for code in technology_codes:
        x, y = positions[f"technology:{code}"]
        side_node(x, y, TEAL, short_tech(code).replace("\n", " "), "left")
    for code in mechanism_codes:
        x, y = positions[f"mechanism:{code}"]
        mechanism_node(x, y, maps["mechanisms"].get(code, code))
    for code in outcome_codes:
        x, y = positions[f"outcome:{code}"]
        side_node(x, y, BLUE if code.startswith("SDG3_") else ORANGE, short_outcome(code).replace("\n", " "), "right")
    ax.text(x_values["technology"], 0.97, "Technologies", ha="center", fontsize=11, fontweight="bold")
    ax.text(x_values["mechanism"], 0.97, "Mechanisms", ha="center", fontsize=11, fontweight="bold")
    ax.text(x_values["outcome"], 0.97, "Reported outcomes", ha="center", fontsize=11, fontweight="bold")
    ax.text(0.83, 0.935, "SDG 3", color=BLUE, fontweight="bold")
    ax.text(0.91, 0.935, "SDG 16", color=ORANGE, fontweight="bold")
    ax.set_title("Aggregate technology–mechanism–outcome knowledge network", pad=20)
    fig.text(0.01, 0.01, "Note. Edge width represents unique papers supporting an automated candidate relationship. Only the strongest links among leading categories are shown.", fontsize=8, color=GREY)
    save_figure(fig, output, "network_figure3_aggregate_knowledge_network")
    return {"technology_mechanism_edges": tm, "mechanism_outcome_edges": mo}


def figure_bridge_papers(metrics: pd.DataFrame, output: Path) -> pd.DataFrame:
    top = metrics.head(15).copy().sort_values("opposite_domain_similarity_strength", ascending=True)

    def compact_title(value: Any, max_chars: int = 62) -> str:
        title = " ".join(str(value).split())
        if len(title) <= max_chars:
            return title
        shortened = title[:max_chars].rsplit(" ", 1)[0]
        return f"{shortened}…"

    labels = [f"{row.paper_id} — {fill(compact_title(row.title), 34)}" for row in top.itertuples()]
    colors = [DOMAIN_COLORS.get(domain, GREY) for domain in top["source_domain"]]
    fig, ax = plt.subplots(figsize=(10.5, 7.2), constrained_layout=True)
    bars = ax.barh(np.arange(len(top)), top["opposite_domain_similarity_strength"], color=colors)
    ax.set_yticks(np.arange(len(top)), labels)
    ax.set_xlabel("Sum of similarity to the opposite SDG domain")
    ax.set_title("Candidate bridge papers with the strongest cross-domain semantic neighbourhoods")
    ax.grid(axis="x", color=LIGHT_GREY, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    maximum = max(1e-6, top["opposite_domain_similarity_strength"].max())
    ax.set_xlim(0, maximum * 1.18)
    for bar, value in zip(bars, top["opposite_domain_similarity_strength"]):
        ax.text(value + maximum * 0.015, bar.get_y() + bar.get_height() / 2, f"{value:.2f}", va="center", fontsize=7.5)
    handles = [mpl.patches.Patch(color=DOMAIN_COLORS[domain], label=domain.replace("SDG", "SDG ")) for domain in ("SDG3", "SDG16", "Mixed")]
    ax.legend(handles=handles, frameon=False, loc="lower right")
    fig.text(
        0.01,
        -0.01,
        "Note. Candidates require corpus/domain screening: high scores may reflect shared adoption or business vocabulary. For SDG 3/16 papers, the score sums similarity to neighbours in the opposite domain; for mixed papers it is the smaller of similarity strength to SDG 3 and SDG 16.",
        fontsize=8,
        color=GREY,
    )
    save_figure(fig, output, "network_figure4_bridge_papers")
    return top.sort_values("opposite_domain_similarity_strength", ascending=False)


def write_similarity_matrix(matrix: np.ndarray, paper_ids: Sequence[str], output: Path) -> None:
    frame = pd.DataFrame(matrix, index=paper_ids, columns=paper_ids)
    frame.index.name = "paper_id"
    frame.to_csv(output / "paper_similarity_matrix.csv", encoding="utf-8-sig", float_format="%.6f")
    np.save(output / "paper_similarity_matrix.npy", matrix)


def write_readme(output: Path, config: dict[str, Any], summary: dict[str, Any]) -> None:
    text = f"""# Paper Similarity and Knowledge Network Outputs

## Method

The analysis is deterministic and does not use a generative language model or
external API. Text from abstract, introduction/background, discussion, and
conclusion sections was represented using TF-IDF unigrams and bigrams. The
matrix was reduced to {config['actual_svd_components']} latent dimensions with
truncated SVD and L2-normalized. Cosine similarity was calculated between all
papers. The primary paper network uses an undirected union of each paper's
{config['primary_k']} nearest neighbours. Sensitivity networks use k values
{config['sensitivity_k']}.

## Key files

- `paper_similarity_matrix.csv/.npy`: full all-pair similarity matrix.
- `paper_similarity_edges.csv`: primary kNN edge list.
- `paper_nearest_neighbors.csv`: directed nearest-neighbour list per paper.
- `paper_network_metrics.csv`: centrality, community, participation, and bridge indicators.
- `similarity_group_summary.csv`: within- and cross-domain similarity summaries.
- `similarity_sensitivity.csv`: k = 5/10/15 network sensitivity.
- `knowledge_network_nodes.csv` and `knowledge_network_edges.csv`: heterogeneous graph tables.
- `knowledge_network.graphml/.gexf`: network files for Gephi, Cytoscape, or NetworkX.
- `figures/`: publication figures in PNG, SVG, and PDF.

## Interpretation guardrails

The paper–paper similarity edges are computational similarities, not citations
or causal relationships. Knowledge-network coding edges remain automated
candidates until human review is finalized. Layout coordinates and visual
proximity are descriptive and should not be treated as statistical evidence.

## Run summary

- Papers: {summary['papers']}
- Primary similarity edges: {summary['primary_similarity_edges']}
- Knowledge-network nodes: {summary['knowledge_network_nodes']}
- Knowledge-network edges: {summary['knowledge_network_edges']}
- TF-IDF features: {summary['tfidf_features']}
- Explained SVD variance: {summary['svd_explained_variance']:.3f}
"""
    (output / "README.md").write_text(text, encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a deterministic paper similarity network and multilayer knowledge graph.")
    parser.add_argument("--master-dir", required=True, type=Path, help="Directory produced by pipeline.py")
    parser.add_argument("--output", required=True, type=Path, help="Writable output directory")
    parser.add_argument("--codebook", type=Path, default=Path(__file__).with_name("codebook.json"))
    parser.add_argument("--primary-k", type=int, default=10)
    parser.add_argument("--sensitivity-k", type=int, nargs="+", default=[5, 10, 15])
    parser.add_argument("--min-similarity", type=float, default=0.0, help="Optional minimum cosine similarity after kNN selection")
    parser.add_argument("--max-features", type=int, default=25000)
    parser.add_argument("--min-df", type=int, default=3)
    parser.add_argument("--svd-components", type=int, default=200)
    parser.add_argument("--max-text-chars", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if min([args.primary_k, *args.sensitivity_k]) < 1:
        raise SystemExit("All k values must be positive.")
    if not 0 <= args.min_similarity <= 1:
        raise SystemExit("--min-similarity must be between 0 and 1.")
    master_dir = args.master_dir.expanduser().resolve()
    output = args.output.expanduser().resolve()
    figures = output / "figures"
    supporting = output / "supporting_data"
    output.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    supporting.mkdir(parents=True, exist_ok=True)

    papers = pd.read_csv(master_dir / "master_papers.csv")
    papers = papers[(papers["analysis_eligible"] == "Yes") & (papers["extraction_status"] == "OK")].copy()
    papers = papers.sort_values("paper_id").reset_index(drop=True)
    if len(papers) < 3 or papers["paper_id"].duplicated().any():
        raise SystemExit("At least three eligible papers with unique IDs are required.")
    relations = pd.read_csv(master_dir / "master_technology_outcome_relations.csv")
    relations = relations[relations["paper_id"].isin(set(papers["paper_id"]))].copy()
    codebook = json.loads(args.codebook.read_text(encoding="utf-8"))

    texts, text_manifest = load_similarity_texts(papers, args.max_text_chars)
    text_manifest.to_csv(output / "paper_similarity_text_manifest.csv", index=False, encoding="utf-8-sig")
    empty = text_manifest[text_manifest["similarity_text_chars"] < 100]
    if len(empty):
        raise RuntimeError(f"{len(empty)} papers have insufficient text; inspect paper_similarity_text_manifest.csv")

    tfidf, embeddings, vectorizer, svd = build_embeddings(texts, args.max_features, args.min_df, args.svd_components, args.seed)
    matrix = full_similarity_matrix(embeddings)
    paper_ids = papers["paper_id"].astype(str).tolist()
    write_similarity_matrix(matrix, paper_ids, output)

    primary_edges = knn_edges(paper_ids, embeddings, args.primary_k, args.min_similarity)
    primary_edge_frame = pd.DataFrame([edge.__dict__ for edge in primary_edges], columns=["source", "target", "similarity"])
    if primary_edge_frame.empty:
        raise SystemExit("No edges survived the similarity cutoff; lower --min-similarity.")
    domains = papers.set_index("paper_id")["source_domain"].astype(str).to_dict()
    if not primary_edge_frame.empty:
        primary_edge_frame["source_domain"] = primary_edge_frame["source"].map(domains)
        primary_edge_frame["target_domain"] = primary_edge_frame["target"].map(domains)
        primary_edge_frame["domain_pair"] = [domain_pair_label(a, b) for a, b in zip(primary_edge_frame["source_domain"], primary_edge_frame["target_domain"])]
        primary_edge_frame["distance"] = 1.0 - primary_edge_frame["similarity"]
    primary_edge_frame.to_csv(output / "paper_similarity_edges.csv", index=False, encoding="utf-8-sig")

    nearest_rows: list[dict[str, Any]] = []
    for source_index, source in enumerate(paper_ids):
        order = np.argsort(matrix[source_index])[::-1]
        rank = 0
        for target_index in order:
            if target_index == source_index:
                continue
            rank += 1
            target = paper_ids[int(target_index)]
            nearest_rows.append(
                {
                    "paper_id": source,
                    "neighbor_rank": rank,
                    "neighbor_paper_id": target,
                    "similarity": float(matrix[source_index, target_index]),
                    "source_domain": domains[source],
                    "neighbor_domain": domains[target],
                }
            )
            if rank >= max(args.sensitivity_k + [args.primary_k]):
                break
    pd.DataFrame(nearest_rows).to_csv(output / "paper_nearest_neighbors.csv", index=False, encoding="utf-8-sig")

    similarity_graph = make_similarity_graph(papers, primary_edges)
    paper_metrics, communities = paper_network_metrics(similarity_graph, papers, args.seed)
    paper_metrics.to_csv(output / "paper_network_metrics.csv", index=False, encoding="utf-8-sig")
    paper_metrics.head(50).to_csv(output / "bridge_papers_top50.csv", index=False, encoding="utf-8-sig")

    pair_long, group_summary = similarity_group_data(matrix, papers)
    group_summary.to_csv(output / "similarity_group_summary.csv", index=False, encoding="utf-8-sig")
    pair_long.to_csv(output / "paper_similarity_pairs_long.csv", index=False, encoding="utf-8-sig", float_format="%.6f")

    sensitivity_rows: list[dict[str, Any]] = []
    sensitivity_edges: dict[int, list[SimilarityEdge]] = {}
    for k in sorted(set(args.sensitivity_k + [args.primary_k])):
        edges = knn_edges(paper_ids, embeddings, k, args.min_similarity)
        sensitivity_edges[k] = edges
        graph = make_similarity_graph(papers, edges)
        sensitivity_rows.append(graph_metrics(graph, papers, k, args.seed))
    sensitivity = pd.DataFrame(sensitivity_rows).sort_values("k")
    sensitivity.to_csv(output / "similarity_sensitivity.csv", index=False, encoding="utf-8-sig")

    knowledge_graph, node_frame, edge_frame = build_knowledge_graph(papers, relations, primary_edges, codebook)
    node_frame.to_csv(output / "knowledge_network_nodes.csv", index=False, encoding="utf-8-sig")
    edge_frame.to_csv(output / "knowledge_network_edges.csv", index=False, encoding="utf-8-sig")
    export_graph = sanitize_graph_attributes(knowledge_graph)
    nx.write_graphml(export_graph, output / "knowledge_network.graphml")
    nx.write_gexf(export_graph, output / "knowledge_network.gexf")

    configure_plot_style()
    positions = figure_similarity_map(embeddings, papers, primary_edges, paper_metrics, figures, args.seed)
    positions.to_csv(supporting / "network_figure1_positions.csv", index=False, encoding="utf-8-sig")
    figure_similarity_groups(pair_long, group_summary, figures, args.seed)
    aggregate_data = figure_aggregate_knowledge_network(relations, codebook, figures)
    for name, frame in aggregate_data.items():
        frame.to_csv(supporting / f"network_figure3_{name}.csv", index=False, encoding="utf-8-sig")
    bridge_data = figure_bridge_papers(paper_metrics, figures)
    bridge_data.to_csv(supporting / "network_figure4_bridge_papers.csv", index=False, encoding="utf-8-sig")

    config = {
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "method": "TF-IDF unigrams/bigrams -> truncated SVD -> L2 normalization -> cosine kNN",
        "primary_k": args.primary_k,
        "sensitivity_k": sorted(set(args.sensitivity_k)),
        "min_similarity": args.min_similarity,
        "max_features": args.max_features,
        "min_df": args.min_df,
        "actual_tfidf_features": int(tfidf.shape[1]),
        "requested_svd_components": args.svd_components,
        "actual_svd_components": int(embeddings.shape[1]),
        "svd_explained_variance": float(svd.explained_variance_ratio_.sum()),
        "max_text_chars": args.max_text_chars,
        "seed": args.seed,
        "corpus_papers": len(papers),
        "pipeline_sha256": sha256_text(Path(__file__).read_text(encoding="utf-8")),
        "codebook_sha256": hashlib.sha256(args.codebook.read_bytes()).hexdigest(),
    }
    summary = {
        "papers": len(papers),
        "primary_similarity_edges": len(primary_edges),
        "knowledge_network_nodes": knowledge_graph.number_of_nodes(),
        "knowledge_network_edges": knowledge_graph.number_of_edges(),
        "tfidf_features": int(tfidf.shape[1]),
        "svd_explained_variance": float(svd.explained_variance_ratio_.sum()),
        "communities": len(communities),
    }
    (output / "network_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (output / "network_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_readme(output, config, summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
