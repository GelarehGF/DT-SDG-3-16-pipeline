#!/usr/bin/env python3
"""Create publication-quality paper-network figures for the manuscript.

The figures are deterministic reconstructions from the finalized k=10 paper
similarity edge list and its reported Louvain assignments.  They deliberately
separate analytical structure from display filtering: the force-directed
coordinates use the complete input graph, while the main combined view
suppresses weaker within-domain edges only to keep cross-domain bridges
legible.  No generative model or external API is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from figure_fonts import configure_fonts, font, font_files


ROOT = Path(__file__).resolve().parents[1]
NETWORK = ROOT / "outputs" / "dt_sdg3_16_knowledge_network"
OUT = ROOT / "outputs" / "manuscript_revision" / "assets_v10"

CONFIG = {"primary_k": 10, "seed": 42}
LAYOUT_SEED = 42
CONTEXT_QUANTILE = 0.85
CROSS_EDGES = 150

BLUE = "#2F6690"
ORANGE = "#D17A22"
GREEN = "#4C956C"
PURPLE = "#7B6FAE"
DARK = "#263238"
GREY = "#707780"
LIGHT = "#E7EAEE"
WHITE = "#FFFFFF"

DOMAIN_COLORS = {"SDG3": BLUE, "SDG16": ORANGE, "Mixed": GREEN}
COMMUNITY_COLORS = [
    "#2F6690",
    "#D17A22",
    "#4C956C",
    "#7B6FAE",
    "#C44E52",
    "#3A8D8F",
    "#C8A24A",
    "#8D6E63",
    "#D46AA4",
]


def rgba(value: str, alpha: int = 255) -> tuple[int, int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[index : index + 2], 16) for index in (0, 2, 4)) + (alpha,)


def new_canvas(width: int, height: int) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGBA", (width, height), rgba(WHITE))
    return image, ImageDraw.Draw(image, "RGBA")


def save(image: Image.Image, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    # Composite transparency onto white before dropping the alpha channel.
    background = Image.new("RGBA", image.size, rgba(WHITE))
    Image.alpha_composite(background, image).convert("RGB").save(OUT / f"{stem}.png", dpi=(450, 450), optimize=True)


def center_text(draw: ImageDraw.ImageDraw, xy: tuple[float, float], value: str, text_font, fill=rgba(DARK), spacing: int = 5) -> None:
    box = draw.multiline_textbbox((0, 0), value, font=text_font, spacing=spacing, align="center")
    draw.multiline_text(
        (xy[0] - (box[2] - box[0]) / 2, xy[1] - (box[3] - box[1]) / 2),
        value,
        font=text_font,
        fill=fill,
        spacing=spacing,
        align="center",
    )


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    edges = pd.read_csv(NETWORK / "paper_similarity_edges.csv", encoding="utf-8-sig")
    metrics = pd.read_csv(NETWORK / "paper_network_metrics.csv", encoding="utf-8-sig")
    positions = pd.read_csv(NETWORK / "supporting_data" / "network_figure1_positions.csv", encoding="utf-8-sig")
    validate_data(edges, metrics, positions)
    return edges, metrics, positions


def validate_data(edges: pd.DataFrame, metrics: pd.DataFrame, positions: pd.DataFrame) -> None:
    ids = set(metrics["paper_id"])
    if not ids or metrics["paper_id"].duplicated().any():
        raise ValueError("Paper IDs must be nonempty and unique.")
    if positions["paper_id"].duplicated().any() or not ids.issubset(set(positions["paper_id"])):
        raise ValueError("Every paper needs exactly one initial coordinate pair.")
    if not set(edges["source"]).union(edges["target"]).issubset(ids):
        raise ValueError("Edges refer to unknown paper IDs.")
    if (edges["source"] == edges["target"]).any():
        raise ValueError("Self-loops are not paper-similarity edges.")
    pairs = [tuple(sorted((a, b))) for a, b in zip(edges.source, edges.target)]
    if len(pairs) != len(set(pairs)):
        raise ValueError("Duplicate undirected edges in input.")
    if not np.isfinite(edges["similarity"]).all() or not edges["similarity"].between(0, 1).all():
        raise ValueError("Similarity weights must be finite and between zero and one.")
    if not np.isfinite(positions[["x", "y"]].to_numpy()).all():
        raise ValueError("Layout coordinates must be finite.")
    domains = metrics.set_index("paper_id")["source_domain"].to_dict()
    if not set(domains.values()).issubset(DOMAIN_COLORS):
        raise ValueError("Expected SDG3, SDG16, or Mixed domain labels.")
    for row in edges.itertuples():
        if row.source_domain != domains[row.source] or row.target_domain != domains[row.target]:
            raise ValueError("Edge and node domain labels disagree.")
        pair = {row.source_domain, row.target_domain}
        expected = "Cross SDG 3–SDG 16" if pair == {"SDG3", "SDG16"} else (
            "Involving mixed-domain" if len(pair) > 1 else {
                "SDG3": "Within SDG 3", "SDG16": "Within SDG 16", "Mixed": "Within mixed-domain"
            }[row.source_domain]
        )
        if row.domain_pair != expected:
            raise ValueError("Edge comparison-group label disagrees with its domains.")
    graph = nx.Graph()
    graph.add_nodes_from(ids)
    graph.add_edges_from(pairs)
    if any(graph.degree(row.paper_id) != row.degree for row in metrics.itertuples()):
        raise ValueError("Saved degrees do not match the complete edge list.")


def graph_summary(edges: pd.DataFrame, metrics: pd.DataFrame) -> dict:
    graph = nx.Graph()
    graph.add_nodes_from(metrics["paper_id"])
    graph.add_weighted_edges_from((row.source, row.target, row.similarity) for row in edges.itertuples())
    communities = [set(frame["paper_id"]) for _key, frame in metrics.groupby("community_id")]
    return {
        "nodes": len(metrics), "edges": len(edges),
        "components": nx.number_connected_components(graph),
        "isolates": nx.number_of_isolates(graph),
        "direct_cross_edges": int((edges["domain_pair"] == "Cross SDG 3–SDG 16").sum()),
        "mixed_edges": int(((edges["source_domain"] == "Mixed") | (edges["target_domain"] == "Mixed")).sum()),
        "communities": len(communities),
        "modularity": nx.community.modularity(graph, communities) if graph.size(weight="weight") else 0.0,
    }


def display_edges(edges: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cross_mask = edges["domain_pair"] == "Cross SDG 3–SDG 16"
    direct_cross = edges[cross_mask].nlargest(CROSS_EDGES, "similarity").copy()
    other = edges[~cross_mask].copy()
    cutoff = other["similarity"].quantile(CONTEXT_QUANTILE)
    return other[other["similarity"] >= cutoff], direct_cross


def normalize_positions(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=float).copy()
    if not len(result):
        return result
    result -= result.mean(axis=0, keepdims=True)
    scale = np.quantile(np.abs(result), 0.97, axis=0)
    scale[scale < 1e-9] = 1.0
    result /= scale
    return np.clip(result, -1.15, 1.15)


def force_layout(
    nodes: list[str],
    edges: pd.DataFrame,
    initial: dict[str, tuple[float, float]],
    iterations: int,
    seed: int = 42,
) -> dict[str, tuple[float, float]]:
    """Small deterministic Fruchterman-Reingold implementation using NumPy."""

    if not nodes:
        return {}
    index = {node: position for position, node in enumerate(nodes)}
    rng = np.random.default_rng(seed)
    start = np.array([initial.get(node, (0.0, 0.0)) for node in nodes], dtype=float)
    start = normalize_positions(start) + rng.normal(0.0, 0.01, size=(len(nodes), 2))
    relevant = edges[edges["source"].isin(index) & edges["target"].isin(index)].copy()
    edge_i = relevant["source"].map(index).to_numpy(dtype=int)
    edge_j = relevant["target"].map(index).to_numpy(dtype=int)
    weights = relevant["similarity"].to_numpy(dtype=float)
    if len(weights):
        low, high = float(weights.min()), float(weights.max())
        weights = 0.55 + 0.9 * (weights - low) / max(high - low, 1e-9)

    pos = start
    n = max(1, len(nodes))
    k = np.sqrt(4.0 / n) * 1.15
    initial_temperature = 0.16

    for step in range(iterations):
        delta = pos[:, None, :] - pos[None, :, :]
        distance_squared = np.sum(delta * delta, axis=2) + 1e-7
        np.fill_diagonal(distance_squared, np.inf)
        repulsion = delta * (k * k / distance_squared)[:, :, None]
        displacement = repulsion.sum(axis=1)

        if len(edge_i):
            difference = pos[edge_i] - pos[edge_j]
            distance = np.sqrt(np.sum(difference * difference, axis=1) + 1e-9)
            attraction = (difference / distance[:, None]) * ((distance * distance / k) * weights)[:, None]
            np.add.at(displacement, edge_i, -attraction)
            np.add.at(displacement, edge_j, attraction)

        displacement -= pos * 0.035
        magnitude = np.sqrt(np.sum(displacement * displacement, axis=1)) + 1e-9
        temperature = initial_temperature * (1.0 - step / max(1, iterations)) ** 1.35 + 0.003
        pos += displacement / magnitude[:, None] * np.minimum(magnitude, temperature)[:, None]
        pos -= pos.mean(axis=0, keepdims=True)

    pos = normalize_positions(pos)
    return {node: (float(pos[index[node], 0]), float(pos[index[node], 1])) for node in nodes}


def to_canvas(position: tuple[float, float], bounds: tuple[int, int, int, int]) -> tuple[float, float]:
    left, top, right, bottom = bounds
    x, y = position
    return (
        left + (x + 1.20) / 2.40 * (right - left),
        bottom - (y + 1.20) / 2.40 * (bottom - top),
    )


def degrees(nodes: list[str], edges: pd.DataFrame) -> dict[str, int]:
    values = {node: 0 for node in nodes}
    for source, target in edges[["source", "target"]].itertuples(index=False, name=None):
        if source in values and target in values:
            values[source] += 1
            values[target] += 1
    return values


def draw_edges(
    draw: ImageDraw.ImageDraw,
    positions: dict[str, tuple[float, float]],
    bounds: tuple[int, int, int, int],
    edges: pd.DataFrame,
    color: str,
    alpha_low: int = 30,
    alpha_high: int = 100,
    width_low: int = 2,
    width_high: int = 5,
) -> None:
    if edges.empty:
        return
    low = float(edges["similarity"].min())
    high = float(edges["similarity"].max())
    span = max(high - low, 1e-9)
    for row in edges.sort_values("similarity").itertuples():
        if row.source not in positions or row.target not in positions:
            continue
        strength = (float(row.similarity) - low) / span
        alpha = round(alpha_low + strength * (alpha_high - alpha_low))
        width = round(width_low + strength * (width_high - width_low))
        draw.line(
            (*to_canvas(positions[row.source], bounds), *to_canvas(positions[row.target], bounds)),
            fill=rgba(color, alpha),
            width=max(1, width),
        )


def draw_nodes(
    draw: ImageDraw.ImageDraw,
    nodes: list[str],
    positions: dict[str, tuple[float, float]],
    bounds: tuple[int, int, int, int],
    size_values: dict[str, int],
    color_values: dict[str, str],
    label_nodes: list[str] | None = None,
) -> None:
    maximum = max(size_values.values(), default=1)
    label_nodes = label_nodes or []
    for node in sorted(nodes, key=lambda value: size_values.get(value, 0)):
        x, y = to_canvas(positions[node], bounds)
        radius = 9 + 12 * np.sqrt(size_values.get(node, 0) / max(maximum, 1))
        color = color_values.get(node, GREY)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=rgba(color, 225), outline=rgba(WHITE), width=2)

    label_font = font(42, True)
    occupied = []
    for label_index, node in enumerate(label_nodes):
        if node not in positions:
            continue
        x, y = to_canvas(positions[node], bounds)
        text_box = draw.textbbox((0, 0), node, font=label_font)
        text_width, text_height = text_box[2] - text_box[0], text_box[3] - text_box[1]
        candidates = [(18, -text_height - 20), (18, 22), (-text_width - 22, -text_height - 20), (-text_width - 22, 22)]
        candidates += [(dx, dy) for dy in (-150, 100, -220, 180) for dx in (20, -text_width - 20)]
        for dx, dy in candidates:
            tx = min(bounds[2] - text_width - 18, max(bounds[0] + 18, x + dx))
            ty = min(bounds[3] - text_height - 15, max(bounds[1] + 15, y + dy))
            rectangle = (tx - 10, ty - 8, tx + text_width + 10, ty + text_height + 8)
            if not any(rectangle[0] < box[2] + 8 and rectangle[2] > box[0] - 8 and rectangle[1] < box[3] + 8 and rectangle[3] > box[1] - 8 for box in occupied):
                break
        occupied.append(rectangle)
        draw.line((x, y, tx + text_width / 2, ty + text_height / 2), fill=rgba(GREY, 190), width=2)
        draw.rounded_rectangle(
            rectangle,
            radius=8,
            fill=rgba(WHITE, 235),
            outline=rgba("#B8BEC5", 220),
            width=2,
        )
        draw.text((tx, ty), node, font=label_font, fill=rgba(DARK), anchor="lt")


def draw_domain_legend(draw: ImageDraw.ImageDraw, x: int, y: int, item_gap: int = 285) -> None:
    for index, (domain, label) in enumerate((("SDG3", "SDG 3"), ("SDG16", "SDG 16"), ("Mixed", "Mixed-domain"))):
        item_x = x + index * item_gap
        draw.ellipse((item_x, y, item_x + 40, y + 40), fill=rgba(DOMAIN_COLORS[domain]))
        draw.text((item_x + 55, y - 9), label, font=font(44), fill=rgba(DARK))


def within_domain_figure(edges: pd.DataFrame, metrics: pd.DataFrame, initial: dict[str, tuple[float, float]]) -> None:
    width, height = 4200, 2350
    image, draw = new_canvas(width, height)
    center_text(draw, (width / 2, 96), "Within-domain paper-similarity networks", font(90, True))
    panel_specs = [
        ("SDG3", "A. SDG 3", (150, 275, 2025, 2050), BLUE),
        ("SDG16", "B. SDG 16", (2175, 275, 4050, 2050), ORANGE),
    ]

    for panel_index, (domain, panel_title, bounds, color) in enumerate(panel_specs):
        nodes = metrics.loc[metrics["source_domain"] == domain, "paper_id"].tolist()
        subset = edges[edges["source"].isin(nodes) & edges["target"].isin(nodes)].copy()
        positions = force_layout(nodes, subset, initial, iterations=125, seed=LAYOUT_SEED + panel_index)
        node_degrees = degrees(nodes, subset)
        graph = nx.Graph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(zip(subset.source, subset.target))
        component_count = nx.number_connected_components(graph)
        isolate_count = nx.number_of_isolates(graph)
        draw.rounded_rectangle(bounds, radius=24, fill=rgba("#FBFCFD"), outline=rgba("#C9CED4"), width=4)
        draw.text((bounds[0] + 42, bounds[1] + 32), panel_title, font=font(68, True), fill=rgba(color))
        stats = f"n={len(nodes)} papers  |  {len(subset):,} within-domain edges  |  {component_count} component{'s' if component_count > 1 else ''}"
        if isolate_count:
            stats += f"\n{isolate_count} paper(s) have no within-domain neighbours"
        draw.multiline_text((bounds[0] + 42, bounds[1] + 118), stats, font=font(40), fill=rgba(GREY), spacing=5)
        graph_bounds = (bounds[0] + 75, bounds[1] + 235, bounds[2] - 70, bounds[3] - 80)
        draw_edges(draw, positions, graph_bounds, subset, GREY, 22, 100, 2, 5)
        color_values = {node: color for node in nodes}
        top_nodes = sorted(nodes, key=lambda node: node_degrees[node], reverse=True)[:4]
        draw_nodes(draw, nodes, positions, graph_bounds, node_degrees, color_values, top_nodes)

    center_text(
        draw,
        (width / 2, 2225),
        f"Node size represents within-domain degree; edge opacity and width represent cosine similarity.\nLayouts use all within-domain k={CONFIG['primary_k']} edges and fixed seeds.",
        font(47),
        fill=rgba(GREY),
        spacing=8,
    )
    save(image, "figure3_within_sdg_networks")


def full_layout(edges: pd.DataFrame, metrics: pd.DataFrame, initial: dict[str, tuple[float, float]]) -> dict[str, tuple[float, float]]:
    nodes = metrics["paper_id"].tolist()
    positions = force_layout(nodes, edges, initial, iterations=165, seed=LAYOUT_SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [{"paper_id": node, "x": xy[0], "y": xy[1]} for node, xy in positions.items()]
    ).to_csv(OUT / "paper_network_force_positions.csv", index=False, encoding="utf-8-sig")
    return positions


def combined_bridge_figure(edges: pd.DataFrame, metrics: pd.DataFrame, positions: dict[str, tuple[float, float]]) -> None:
    width, height = 3900, 2900
    image, draw = new_canvas(width, height)
    center_text(draw, (width / 2, 92), "Combined network: modular domains with selective cross-SDG bridges", font(82, True))
    draw_domain_legend(draw, 2380, 165)
    draw.line((210, 192, 315, 192), fill=rgba(PURPLE), width=7)
    draw.text((345, 163), "Direct cross-SDG links", font=font(42), fill=rgba(GREY))
    bounds = (200, 280, 3700, 2375)
    draw.rounded_rectangle(bounds, radius=28, fill=rgba("#FBFCFD"), outline=rgba("#C9CED4"), width=4)

    visible_other, direct_cross = display_edges(edges)
    draw_edges(draw, positions, bounds, visible_other, GREY, 14, 50, 2, 4)
    draw_edges(draw, positions, bounds, direct_cross, PURPLE, 75, 195, 2, 7)

    nodes = metrics["paper_id"].tolist()
    node_degrees = metrics.set_index("paper_id")["degree"].astype(int).to_dict()
    domains = metrics.set_index("paper_id")["source_domain"].to_dict()
    color_values = {node: DOMAIN_COLORS.get(domains[node], GREY) for node in nodes}
    bridge_labels = metrics.nsmallest(8, "bridge_rank")["paper_id"].tolist()
    draw_nodes(draw, nodes, positions, bounds, node_degrees, color_values, bridge_labels)

    stats = graph_summary(edges, metrics)
    cards = [
        (f"{stats['edges']:,}", f"retained k={CONFIG['primary_k']} edges"),
        (f"{stats['direct_cross_edges']:,}", "direct SDG 3–SDG 16 edges"),
        (f"{stats['mixed_edges']:,}", "edges involving mixed papers"),
        (str(stats['communities']), "Louvain communities"),
        (f"{stats['modularity']:.3f}", "modularity"),
    ]
    card_width = 690
    start_x = (width - len(cards) * card_width) / 2
    for index, (value, label) in enumerate(cards):
        x = start_x + index * card_width
        draw.rounded_rectangle((x + 15, 2440, x + card_width - 15, 2700), radius=20, fill=rgba("#F2F4F6"), outline=rgba("#D5DADF"), width=3)
        center_text(draw, (x + card_width / 2, 2512), value, font(64, True), fill=rgba(DARK))
        center_text(draw, (x + card_width / 2, 2622), label, font(41), fill=rgba(GREY))

    center_text(
        draw,
        (width / 2, 2815),
        f"Full-graph layout; all {len(nodes)} papers shown. Display: top {(1-CONTEXT_QUANTILE)*100:g}% of within/mixed edges (ties included) + up to {CROSS_EDGES} strongest direct cross-SDG edges.",
        font(40),
        fill=rgba(GREY),
    )
    save(image, "figure4_filtered_combined_network")


def full_network_supplement(edges: pd.DataFrame, metrics: pd.DataFrame, positions: dict[str, tuple[float, float]]) -> None:
    width, height = 3900, 2850
    image, draw = new_canvas(width, height)
    center_text(draw, (width / 2, 92), "Full combined paper-similarity network", font(82, True))
    draw_domain_legend(draw, 2380, 165)
    bounds = (190, 275, 3710, 2510)
    draw.rounded_rectangle(bounds, radius=26, fill=rgba("#FBFCFD"), outline=rgba("#C9CED4"), width=4)
    for pair, color, low, high in (
        ("Within SDG 3", BLUE, 9, 32),
        ("Within SDG 16", ORANGE, 9, 32),
        ("Within mixed-domain", GREEN, 12, 42),
        ("Involving mixed-domain", GREEN, 10, 38),
        ("Cross SDG 3–SDG 16", PURPLE, 18, 75),
    ):
        draw_edges(draw, positions, bounds, edges[edges["domain_pair"] == pair], color, low, high, 1, 4)
    nodes = metrics["paper_id"].tolist()
    node_degrees = metrics.set_index("paper_id")["degree"].astype(int).to_dict()
    domains = metrics.set_index("paper_id")["source_domain"].to_dict()
    colors = {node: DOMAIN_COLORS.get(domains[node], GREY) for node in nodes}
    draw_nodes(draw, nodes, positions, bounds, node_degrees, colors)
    center_text(
        draw,
        (width / 2, 2730),
        f"All {len(nodes)} nodes and {len(edges):,} undirected k={CONFIG['primary_k']} edges are displayed. Links represent semantic similarity, not citation or causation.",
        font(43),
        fill=rgba(GREY),
    )
    save(image, "supplementary_figure_s1_full_network")


def community_supplement(edges: pd.DataFrame, metrics: pd.DataFrame, positions: dict[str, tuple[float, float]]) -> None:
    width, height = 3900, 2850
    image, draw = new_canvas(width, height)
    center_text(draw, (width / 2, 92), "Louvain community structure of the full network", font(82, True))
    bounds = (190, 285, 3180, 2500)
    draw.rounded_rectangle(bounds, radius=26, fill=rgba("#FBFCFD"), outline=rgba("#C9CED4"), width=4)
    draw_edges(draw, positions, bounds, edges, GREY, 7, 30, 1, 3)
    nodes = metrics["paper_id"].tolist()
    node_degrees = metrics.set_index("paper_id")["degree"].astype(int).to_dict()
    community = metrics.set_index("paper_id")["community_id"].astype(int).to_dict()
    colors = {node: COMMUNITY_COLORS[(community[node] - 1) % len(COMMUNITY_COLORS)] for node in nodes}
    draw_nodes(draw, nodes, positions, bounds, node_degrees, colors)

    counts = metrics.groupby("community_id").size().sort_index()
    draw.text((3270, 350), "Communities", font=font(56, True), fill=rgba(DARK))
    for offset, (community_id, count) in enumerate(counts.items()):
        y = 445 + offset * min(185, 1900 / max(len(counts), 1))
        color = COMMUNITY_COLORS[(int(community_id) - 1) % len(COMMUNITY_COLORS)]
        draw.ellipse((3300, y, 3350, y + 50), fill=rgba(color))
        draw.text((3380, y - 10), f"Community {int(community_id)}", font=font(43, True), fill=rgba(DARK))
        draw.text((3380, y + 48), f"{int(count)} papers", font=font(37), fill=rgba(GREY))

    center_text(
        draw,
        (width / 2, 2735),
        f"{len(counts)} saved Louvain communities (analysis seed {CONFIG['seed']}; modularity={graph_summary(edges, metrics)['modularity']:.3f}). Membership is algorithmic, not a fixed disciplinary classification.",
        font(43),
        fill=rgba(GREY),
    )
    save(image, "supplementary_figure_s2_communities")


def main() -> None:
    global NETWORK, OUT, CONFIG, LAYOUT_SEED, CONTEXT_QUANTILE, CROSS_EDGES
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42, help="Layout seed; SDG16 panel uses seed + 1")
    parser.add_argument("--context-quantile", type=float, default=0.85)
    parser.add_argument("--cross-edges", type=int, default=150)
    parser.add_argument("--regular-font", type=Path)
    parser.add_argument("--bold-font", type=Path)
    args = parser.parse_args()
    if not 0 <= args.context_quantile <= 1 or args.cross_edges < 0:
        parser.error("Quantile must be in [0, 1]; cross-edge limit must be nonnegative.")
    NETWORK, OUT = args.network_dir, args.output
    CONFIG = json.loads((NETWORK / "network_config.json").read_text(encoding="utf-8"))
    LAYOUT_SEED, CONTEXT_QUANTILE, CROSS_EDGES = args.seed, args.context_quantile, args.cross_edges
    configure_fonts(args.regular_font, args.bold_font)
    edges, metrics, position_frame = load_data()
    initial = position_frame.set_index("paper_id")[["x", "y"]].apply(tuple, axis=1).to_dict()
    within_domain_figure(edges, metrics, initial)
    positions = full_layout(edges, metrics, initial)
    combined_bridge_figure(edges, metrics, positions)
    full_network_supplement(edges, metrics, positions)
    community_supplement(edges, metrics, positions)
    other, cross = display_edges(edges)
    inputs = ["paper_similarity_edges.csv", "paper_network_metrics.csv", "supporting_data/network_figure1_positions.csv", "network_config.json"]
    manifest = {
        "method": "Saved similarity graph; deterministic NumPy force layout; Pillow rendering",
        "analysis": graph_summary(edges, metrics),
        "primary_k": CONFIG["primary_k"], "analysis_seed": CONFIG["seed"],
        "layout_seeds": {"full_and_sdg3": LAYOUT_SEED, "sdg16": LAYOUT_SEED + 1},
        "layout_iterations": {"full": 165, "within_domain": 125},
        "display_filter": {"context_quantile": CONTEXT_QUANTILE, "cross_edge_limit": CROSS_EDGES, "visible_context_edges": len(other), "visible_cross_edges": len(cross)},
        "dpi": 450,
        "font_sha256": {name: hashlib.sha256(Path(path).read_bytes()).hexdigest() for name, path in font_files().items()},
        "input_sha256": {name: hashlib.sha256((NETWORK / name).read_bytes()).hexdigest() for name in inputs},
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (OUT / "network_figure_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
