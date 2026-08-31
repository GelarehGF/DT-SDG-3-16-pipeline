#!/usr/bin/env python3
"""Create publication-ready preliminary figures from the master CSV outputs.

Counts represent automated coding candidates and must not be described as
validated findings until the human-review fields are completed.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from textwrap import fill


import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.path import Path as MplPath
from matplotlib.patches import FancyBboxPatch, PathPatch


BLUE = "#2F6690"
BLUE_LIGHT = "#DCEAF4"
ORANGE = "#D17A22"
ORANGE_LIGHT = "#F7E5D1"
GREEN = "#4C956C"
RED = "#C44E52"
PURPLE = "#7B6FAE"
GOLD = "#C8A24A"
TEAL = "#3A8D8F"
GREY = "#707780"
LIGHT_GREY = "#E7EAEE"
DARK = "#263238"

TECH_SHORT = {
    "TECH_PLATFORM_GENERAL": "General digital\nplatforms & ICT",
    "TECH_EGOV": "E-government &\npublic-service platforms",
    "TECH_TELEHEALTH": "Telehealth &\nremote care",
    "TECH_CYBER_PRIVACY": "Cybersecurity &\nprivacy technologies",
    "TECH_AI_ML": "AI & machine\nlearning",
    "TECH_DPI_INTEROP": "Digital public infrastructure\n& interoperability",
    "TECH_MHEALTH": "Mobile health",
    "TECH_DATA_ANALYTICS": "Big data &\nanalytics",
    "TECH_EHR_HIS": "Electronic records &\nhealth information systems",
    "TECH_BLOCKCHAIN": "Blockchain &\ndistributed ledgers",
    "TECH_CLOUD_EDGE": "Cloud & edge\ncomputing",
    "TECH_IOT_WEARABLE": "IoT, sensors &\nwearables",
    "TECH_OPEN_DATA": "Open-data\nplatforms",
    "TECH_CIVIC_SOCIAL": "Civic technology &\nsocial media",
    "TECH_ROBOTICS_AUTO": "Robotics &\nautomation",
    "TECH_DIGITAL_ID": "Digital identity &\nbiometrics",
    "TECH_FINTECH_PAYMENT": "Digital payments &\nfintech",
}

OUTCOME_SHORT = {
    "SDG3_ACCESS": "Health-service access",
    "SDG3_QUALITY": "Care quality & safety",
    "SDG3_CLINICAL": "Clinical & population\nhealth outcomes",
    "SDG3_COST_EFF": "Health-system cost\n& efficiency",
    "SDG3_PATIENT": "Patient engagement\n& self-management",
    "SDG3_EQUITY": "Health equity",
    "SDG3_PREVENT": "Prevention &\npreparedness",
    "SDG3_WORKFORCE": "Health-workforce\ncapacity",
    "SDG3_MENTAL": "Mental health &\nwell-being",
    "SDG3_CONTINUITY": "Continuity &\ncoordination of care",
    "SDG3_DIAG_TREAT": "Diagnosis & treatment\nperformance",
    "SDG16_TRANSPARENCY": "Transparency",
    "SDG16_ACCOUNT": "Accountability",
    "SDG16_CAPACITY": "Institutional capacity\n& effectiveness",
    "SDG16_SERVICE": "Inclusive public-service\naccess",
    "SDG16_PARTICIPATION": "Participation &\nresponsive governance",
    "SDG16_TRUST": "Public trust &\nlegitimacy",
    "SDG16_CORRUPTION": "Corruption control\n& integrity",
    "SDG16_PRIVACY": "Privacy & data\nprotection",
    "SDG16_CYBER": "Cybersecurity &\ninstitutional resilience",
    "SDG16_JUSTICE": "Rule of law &\naccess to justice",
    "SDG16_COORD": "Interagency coordination\n& policy coherence",
    "SDG16_INCLUSION": "Institutional inclusion\n& equal access",
    "SDG16_POLICY": "Evidence-informed\npolicy",
}


def configure_style() -> None:
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


def save_all(fig: plt.Figure, outdir: Path, stem: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg", "pdf"):
        fig.savefig(outdir / f"{stem}.{extension}", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def unique_pairs(relations: pd.DataFrame) -> pd.DataFrame:
    return relations.drop_duplicates(["paper_id", "technology_code", "outcome_code"]).copy()


def label_technology(code: str) -> str:
    return TECH_SHORT.get(code, fill(code.replace("TECH_", "").replace("_", " ").title(), 22))


def label_outcome(code: str) -> str:
    return OUTCOME_SHORT.get(code, fill(code.replace("SDG3_", "").replace("SDG16_", "").replace("_", " ").title(), 20))


def figure_technology_by_sdg(pairs: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    counts = (
        pairs.groupby(["technology_code", "outcome_sdg"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["SDG3", "SDG16"], fill_value=0)
    )
    counts["Total"] = counts.sum(axis=1)
    counts = counts.sort_values("Total", ascending=False).head(12).sort_values("Total", ascending=True)

    y = np.arange(len(counts))
    height = 0.36
    fig, ax = plt.subplots(figsize=(9.3, 6.5), constrained_layout=True)
    bars3 = ax.barh(y - height / 2, counts["SDG3"], height, color=BLUE, label="SDG 3 outcomes")
    bars16 = ax.barh(y + height / 2, counts["SDG16"], height, color=ORANGE, label="SDG 16 outcomes")
    ax.set_yticks(y, [label_technology(code) for code in counts.index])
    ax.set_xlabel("Unique paper–technology–outcome combinations")
    ax.set_title("Technology families connected to SDG 3 and SDG 16 outcomes")
    ax.grid(axis="x", color=LIGHT_GREY, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.legend(loc="lower right", frameon=False)
    max_value = max(1, counts[["SDG3", "SDG16"]].to_numpy().max())
    ax.set_xlim(0, max_value * 1.20)
    for bars in (bars3, bars16):
        for bar in bars:
            if bar.get_width() > 0:
                ax.text(bar.get_width() + max_value * 0.015, bar.get_y() + bar.get_height() / 2, f"{int(bar.get_width())}", va="center", fontsize=8, color=DARK)
    fig.text(0.01, -0.01, "Note. Automated coding candidates; duplicate evidence rows collapsed within paper × technology × outcome.", fontsize=8, color=GREY)
    save_all(fig, outdir, "figure1_technology_families_by_sdg")
    return counts.reset_index()


def annotate_heatmap(ax: plt.Axes, matrix: np.ndarray, threshold: float) -> None:
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = int(matrix[row, column])
            if value:
                color = "white" if value >= threshold else DARK
                ax.text(column, row, str(value), ha="center", va="center", fontsize=7.4, color=color)


def figure_heatmap(pairs: pd.DataFrame, outdir: Path) -> dict[str, pd.DataFrame]:
    tech_order = pairs.groupby("technology_code").size().sort_values(ascending=False).head(10).index.tolist()
    outcome_orders: dict[str, list[str]] = {}
    for sdg in ("SDG3", "SDG16"):
        outcome_orders[sdg] = (
            pairs.loc[pairs["outcome_sdg"] == sdg]
            .groupby("outcome_code")
            .size()
            .sort_values(ascending=False)
            .head(7)
            .index.tolist()
        )
    pivots: dict[str, pd.DataFrame] = {}
    for sdg in ("SDG3", "SDG16"):
        subset = pairs.loc[(pairs["technology_code"].isin(tech_order)) & (pairs["outcome_code"].isin(outcome_orders[sdg]))]
        pivot = subset.groupby(["technology_code", "outcome_code"]).size().unstack(fill_value=0)
        pivots[sdg] = pivot.reindex(index=tech_order, columns=outcome_orders[sdg], fill_value=0)

    fig, axes = plt.subplots(1, 2, figsize=(15.2, 7.1), sharey=True, constrained_layout=True, gridspec_kw={"width_ratios": [1, 1]})
    cmaps = {
        "SDG3": LinearSegmentedColormap.from_list("sdg3", ["#F7FBFF", BLUE]),
        "SDG16": LinearSegmentedColormap.from_list("sdg16", ["#FFF9F2", ORANGE]),
    }
    for ax, sdg in zip(axes, ("SDG3", "SDG16")):
        data = pivots[sdg].to_numpy()
        vmax = max(1, int(data.max()))
        image = ax.imshow(data, aspect="auto", cmap=cmaps[sdg], vmin=0, vmax=vmax)
        ax.set_title(f"{sdg.replace('SDG', 'SDG ')} outcomes")
        ax.set_xticks(np.arange(len(pivots[sdg].columns)), [label_outcome(code) for code in pivots[sdg].columns], rotation=42, ha="right")
        ax.set_yticks(np.arange(len(tech_order)), [label_technology(code).replace("\n", " ") for code in tech_order])
        ax.set_xticks(np.arange(-0.5, data.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, data.shape[0], 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.2)
        ax.tick_params(which="minor", bottom=False, left=False)
        annotate_heatmap(ax, data, vmax * 0.55)
        cbar = fig.colorbar(image, ax=ax, shrink=0.72, pad=0.02)
        cbar.set_label("Unique paper-level links")
    fig.suptitle("Technology–outcome concentration across the two SDG domains", fontsize=13, fontweight="bold")
    fig.text(0.01, -0.015, "Note. Cells show automated candidate links after collapsing repeated evidence passages.", fontsize=8, color=GREY)
    save_all(fig, outdir, "figure2_technology_outcome_heatmap")
    return pivots


def clean_cross_label(value: str) -> str:
    replacements = {
        "Association requires review": "Unclassified association",
        "Synergy candidate": "Synergy",
        "Trade-off candidate": "Trade-off",
        "Joint-risk candidate": "Joint risk",
        "Same sentence/span": "Same evidence span",
        "Same technology in same paper; different evidence spans": "Paper-level co-occurrence",
        "Undetermined—paper-level co-occurrence": "Undetermined:\npaper-level",
        "Undetermined—human coding required": "Undetermined:\nhuman review",
    }
    return replacements.get(value, value)


def figure_cross_sdg(cross: pd.DataFrame, outdir: Path) -> dict[str, pd.DataFrame]:
    basis = cross["relation_basis"].value_counts()
    types = cross["relationship_type_auto"].value_counts()
    directions = cross["direction_auto"].value_counts()
    panels = [
        (basis, "Evidence basis", {
            "Same sentence/span": BLUE,
            "Same technology in same paper; different evidence spans": GREY,
        }),
        (types, "Provisional relationship type", {
            "Synergy candidate": GREEN,
            "Trade-off candidate": GOLD,
            "Joint-risk candidate": RED,
            "Association requires review": GREY,
        }),
        (directions, "Provisional direction", {
            "SDG16 → SDG3": ORANGE,
            "SDG3 → SDG16": BLUE,
            "Undetermined—human coding required": PURPLE,
            "Undetermined—paper-level co-occurrence": GREY,
        }),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8), constrained_layout=True)
    for ax, (series, title, color_map) in zip(axes, panels):
        plot_series = series.sort_values(ascending=True)
        bar_colors = [color_map.get(str(category), GREY) for category in plot_series.index]
        bars = ax.barh(np.arange(len(plot_series)), plot_series.values, color=bar_colors)
        labels = [clean_cross_label(str(value)) for value in plot_series.index]
        labels = [fill(label, 24) for label in labels]
        ax.set_yticks(np.arange(len(plot_series)), labels)
        ax.set_title(title)
        ax.set_xlabel("Candidate relationships")
        ax.grid(axis="x", color=LIGHT_GREY, linewidth=0.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        max_value = max(plot_series.max(), 1)
        ax.set_xlim(0, max_value * 1.20)
        for bar, value in zip(bars, plot_series.values):
            ax.text(value + max_value * 0.025, bar.get_y() + bar.get_height() / 2, str(int(value)), va="center", fontsize=8)
    fig.suptitle("Structure of preliminary cross-SDG relationship candidates", fontsize=13, fontweight="bold")
    fig.text(0.01, -0.02, "Note. Paper-level co-occurrence does not demonstrate integration or causal direction; all categories require human confirmation.", fontsize=8, color=GREY)
    save_all(fig, outdir, "figure3_cross_sdg_candidates")
    return {
        "basis": basis.rename_axis("category").reset_index(name="count"),
        "type": types.rename_axis("category").reset_index(name="count"),
        "direction": directions.rename_axis("category").reset_index(name="count"),
    }


def percent_table(data: pd.DataFrame, column: str, order: list[str]) -> pd.DataFrame:
    table = pd.crosstab(data["outcome_sdg"], data[column]).reindex(index=["SDG3", "SDG16"], fill_value=0)
    table = table.reindex(columns=order, fill_value=0)
    return table.div(table.sum(axis=1), axis=0).fillna(0) * 100


def stacked_percent(ax: plt.Axes, table: pd.DataFrame, colors: list[str], title: str) -> None:
    left = np.zeros(len(table))
    for color, column in zip(colors, table.columns):
        values = table[column].to_numpy()
        ax.barh(np.arange(len(table)), values, left=left, color=color, label=column, height=0.55)
        for row, (value, start) in enumerate(zip(values, left)):
            if value >= 8:
                ax.text(start + value / 2, row, f"{value:.0f}%", ha="center", va="center", fontsize=7.5, color="white" if color not in {GOLD, LIGHT_GREY} else DARK)
        left += values
    ax.set_yticks(np.arange(len(table)), [label.replace("SDG", "SDG ") for label in table.index])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of candidate evidence rows (%)")
    ax.set_title(title)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", color=LIGHT_GREY, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=2, frameon=False)


def figure_evidence_profile(relations: pd.DataFrame, outdir: Path) -> dict[str, pd.DataFrame]:
    evidence_order = [
        "Causal estimate",
        "Statistical association",
        "Empirically observed or reported",
        "Empirical, relation unclear",
        "Proposed or potential",
        "Descriptive or conceptual",
    ]
    polarity_order = ["Positive", "Negative", "Mixed", "Null or mixed", "Unclear"]
    evidence = percent_table(relations, "evidence_level", evidence_order)
    polarity = percent_table(relations, "effect_polarity", polarity_order)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.7), constrained_layout=True)
    stacked_percent(axes[0], evidence, [TEAL, BLUE, GREEN, PURPLE, GOLD, GREY], "Reported evidence strength")
    stacked_percent(axes[1], polarity, [GREEN, RED, GOLD, PURPLE, GREY], "Reported effect polarity")
    fig.suptitle("Evidence profile of automated technology–outcome candidates", fontsize=13, fontweight="bold")
    fig.text(0.01, -0.02, "Note. Categories are rule-based provisional labels and should be updated after human adjudication.", fontsize=8, color=GREY)
    save_all(fig, outdir, "figure4_evidence_and_polarity_profile")
    return {"evidence": evidence.reset_index(), "polarity": polarity.reset_index()}


def draw_node(ax: plt.Axes, x: float, y: float, width: float, height: float, color: str, label: str, align: str) -> None:
    patch = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.01,rounding_size=0.015",
        linewidth=0,
        facecolor=color,
        alpha=0.95,
        zorder=3,
    )
    ax.add_patch(patch)
    text_x = x + width * 0.62 if align == "left" else x - width * 0.62
    ax.text(text_x, y, label, va="center", ha=align, fontsize=8.2, color=DARK, zorder=4)


def figure_pathway_network(pairs: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    techs = pairs.groupby("technology_code").size().sort_values(ascending=False).head(8).index.tolist()
    sdg3_outcomes = pairs.loc[pairs["outcome_sdg"] == "SDG3"].groupby("outcome_code").size().sort_values(ascending=False).head(4).index.tolist()
    sdg16_outcomes = pairs.loc[pairs["outcome_sdg"] == "SDG16"].groupby("outcome_code").size().sort_values(ascending=False).head(4).index.tolist()
    outcomes = sdg3_outcomes + sdg16_outcomes
    edge_data = (
        pairs.loc[pairs["technology_code"].isin(techs) & pairs["outcome_code"].isin(outcomes)]
        .groupby(["technology_code", "outcome_code", "outcome_sdg"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(28)
    )
    tech_positions = {code: (0.18, y) for code, y in zip(techs, np.linspace(0.88, 0.12, len(techs)))}
    outcome_positions = {code: (0.82, y) for code, y in zip(outcomes, np.linspace(0.88, 0.12, len(outcomes)))}
    max_count = max(1, int(edge_data["count"].max()))

    fig, ax = plt.subplots(figsize=(12.5, 8.0), constrained_layout=True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    for _, row in edge_data.sort_values("count").iterrows():
        x1, y1 = tech_positions[row["technology_code"]]
        x2, y2 = outcome_positions[row["outcome_code"]]
        control1 = (x1 + 0.25, y1)
        control2 = (x2 - 0.25, y2)
        path = MplPath([(x1, y1), control1, control2, (x2, y2)], [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4])
        color = BLUE if row["outcome_sdg"] == "SDG3" else ORANGE
        linewidth = 0.7 + 5.2 * row["count"] / max_count
        ax.add_patch(PathPatch(path, facecolor="none", edgecolor=color, linewidth=linewidth, alpha=0.26, zorder=1))
    for code, (x, y) in tech_positions.items():
        draw_node(ax, x, y, 0.018, 0.048, TEAL, label_technology(code).replace("\n", " "), "right")
    for code, (x, y) in outcome_positions.items():
        color = BLUE if code.startswith("SDG3_") else ORANGE
        draw_node(ax, x, y, 0.018, 0.048, color, label_outcome(code).replace("\n", " "), "left")
    ax.text(0.18, 0.97, "Technology families", ha="center", fontsize=11, fontweight="bold", color=DARK)
    ax.text(0.82, 0.97, "Reported SDG outcomes", ha="center", fontsize=11, fontweight="bold", color=DARK)
    ax.text(0.76, 0.945, "SDG 3", color=BLUE, fontsize=9, fontweight="bold")
    ax.text(0.86, 0.945, "SDG 16", color=ORANGE, fontsize=9, fontweight="bold")
    ax.set_title("Preliminary technology–outcome architecture", pad=18)
    fig.text(0.01, 0.01, "Note. The 28 strongest automated links among the leading technology and outcome families are shown. Edge width represents unique paper-level combinations.", fontsize=8, color=GREY)
    save_all(fig, outdir, "figure5_technology_outcome_pathway_network")
    return edge_data


def write_supporting_data(outdir: Path, datasets: dict[str, pd.DataFrame]) -> None:
    data_dir = outdir / "supporting_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, data in datasets.items():
        data.to_csv(data_dir / f"{name}.csv", index=False, encoding="utf-8-sig")


def write_captions(outdir: Path, summary: dict[str, int]) -> None:
    captions = f"""# Preliminary figure captions

All figures describe automated coding candidates rather than human-validated findings.

**Figure 1. Technology families connected to SDG 3 and SDG 16 outcomes.** Counts represent unique paper–technology–outcome combinations after collapsing repeated evidence passages within a paper.

**Figure 2. Technology–outcome concentration across the two SDG domains.** Heatmap cells show unique paper-level candidate links for the ten most frequently identified technology families and the leading outcomes within each SDG.

**Figure 3. Structure of preliminary cross-SDG relationship candidates.** The panels distinguish same-span evidence from weaker paper-level co-occurrence and summarize provisional relationship type and direction. Co-occurrence alone does not establish integration or causality.

**Figure 4. Evidence profile of automated technology–outcome candidates.** Stacked bars report rule-based evidence-strength and effect-polarity labels separately for SDG 3 and SDG 16 outcome relations.

**Figure 5. Preliminary technology–outcome architecture.** The strongest candidate links among leading technology and outcome families are displayed. Edge width represents unique paper-level technology–outcome combinations; blue edges terminate at SDG 3 outcomes and orange edges at SDG 16 outcomes.

Corpus summary used for these figures: {summary['papers']} analysis-eligible papers, {summary['relations']} retained evidence rows, {summary['cross']} cross-SDG candidates.
"""
    (outdir / "figure_captions.md").write_text(captions, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create preliminary publication figures from the DT-SDG technology-outcome master files.")
    parser.add_argument("--input", required=True, type=Path, help="Pipeline output directory containing master CSVs")
    parser.add_argument("--output", required=True, type=Path, help="Figure output directory")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = args.input.expanduser().resolve()
    output_dir = args.output.expanduser().resolve()
    relations = pd.read_csv(input_dir / "master_technology_outcome_relations.csv")
    cross = pd.read_csv(input_dir / "cross_sdg_relations.csv")
    papers = pd.read_csv(input_dir / "master_papers.csv")
    pairs = unique_pairs(relations)
    configure_style()

    datasets: dict[str, pd.DataFrame] = {}
    datasets["figure1_technology_by_sdg"] = figure_technology_by_sdg(pairs, output_dir)
    heatmaps = figure_heatmap(pairs, output_dir)
    datasets["figure2_heatmap_sdg3"] = heatmaps["SDG3"].reset_index()
    datasets["figure2_heatmap_sdg16"] = heatmaps["SDG16"].reset_index()
    cross_tables = figure_cross_sdg(cross, output_dir)
    for key, table in cross_tables.items():
        datasets[f"figure3_{key}"] = table
    evidence_tables = figure_evidence_profile(relations, output_dir)
    for key, table in evidence_tables.items():
        datasets[f"figure4_{key}"] = table
    datasets["figure5_network_edges"] = figure_pathway_network(pairs, output_dir)
    write_supporting_data(output_dir, datasets)

    summary = {
        "papers": int((papers["analysis_eligible"] == "Yes").sum()),
        "relations": int(len(relations)),
        "unique_pairs": int(len(pairs)),
        "cross": int(len(cross)),
    }
    write_captions(output_dir, summary)
    (output_dir / "figure_manifest.json").write_text(
        json.dumps(
            {
                **summary,
                "input_directory": str(input_dir),
                "note": "Automated coding candidates; no claim of completed relation-level human validation",
                "figure_stems": [
                    "figure1_technology_families_by_sdg",
                    "figure2_technology_outcome_heatmap",
                    "figure3_cross_sdg_candidates",
                    "figure4_evidence_and_polarity_profile",
                    "figure5_technology_outcome_pathway_network",
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
