#!/usr/bin/env python3
"""Rebuild manuscript figures with print-readable typography.

This script uses only deterministic supporting tables already produced by the
analysis pipeline. It changes presentation, not analysis or reported values.
Figures are rendered at publication resolution with Pillow so the script runs
in the bundled document runtime without external plotting dependencies.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from textwrap import wrap

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from figure_fonts import configure_fonts, font


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "manuscript_revision" / "assets_v09"
NETWORK = ROOT / "outputs" / "dt_sdg3_16_knowledge_network"
FIGDATA = ROOT / "outputs" / "dt_sdg3_16_figures" / "supporting_data"

BLUE = "#2F6690"
ORANGE = "#D17A22"
GREEN = "#4C956C"
PURPLE = "#7B6FAE"
RED = "#C44E52"
GOLD = "#C8A24A"
TEAL = "#3A8D8F"
GREY = "#707780"
LIGHT = "#E7EAEE"
DARK = "#263238"
WHITE = "#FFFFFF"

TECH_LABELS = {
    "TECH_PLATFORM_GENERAL": "General digital platforms & ICT",
    "TECH_EGOV": "E-government & public-service platforms",
    "TECH_TELEHEALTH": "Telehealth & remote care",
    "TECH_CYBER_PRIVACY": "Cybersecurity & privacy technologies",
    "TECH_AI_ML": "AI & machine learning",
    "TECH_DPI_INTEROP": "Digital public infrastructure & interoperability",
    "TECH_MHEALTH": "Mobile health",
    "TECH_DATA_ANALYTICS": "Big data & analytics",
    "TECH_EHR_HIS": "Electronic records & health information systems",
    "TECH_BLOCKCHAIN": "Blockchain & distributed ledgers",
    "TECH_CLOUD_EDGE": "Cloud & edge computing",
    "TECH_IOT_WEARABLE": "IoT, sensors & wearables",
}

OUTCOME_LABELS = {
    "SDG3_ACCESS": "Health-service access",
    "SDG3_QUALITY": "Care quality & safety",
    "SDG3_CLINICAL": "Clinical & population health outcomes",
    "SDG3_COST_EFF": "Health-system cost & efficiency",
    "SDG3_PATIENT": "Patient engagement & self-management",
    "SDG3_EQUITY": "Health equity",
    "SDG3_PREVENT": "Prevention & preparedness",
    "SDG3_CONTINUITY": "Continuity & coordination of care",
    "SDG16_TRANSPARENCY": "Transparency",
    "SDG16_ACCOUNT": "Accountability",
    "SDG16_CAPACITY": "Institutional capacity & effectiveness",
    "SDG16_SERVICE": "Inclusive public-service access",
    "SDG16_PARTICIPATION": "Participation & responsive governance",
    "SDG16_TRUST": "Public trust & legitimacy",
    "SDG16_PRIVACY": "Privacy & data protection",
    "SDG16_CYBER": "Cybersecurity & institutional resilience",
    "SDG16_COORD": "Interagency coordination & policy coherence",
    "SDG16_INCLUSION": "Institutional inclusion & equal access",
}

MECHANISM_LABELS = {
    "MECH_DATA_SHARE": "Data sharing and interoperability",
    "MECH_TRANSPARENCY": "Transparency and auditability",
    "MECH_TRUST": "Trust and legitimacy",
    "MECH_SECURITY": "Security, privacy, and consent",
    "MECH_CAPACITY": "Capacity, speed, and efficiency",
    "MECH_ACCESS": "Access and reach",
    "MECH_INCLUSION": "Inclusion and digital capability",
    "MECH_PARTICIPATION": "Participation and engagement",
}


def new_canvas(width: int, height: int) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (width, height), WHITE)
    return image, ImageDraw.Draw(image)


def save(image: Image.Image, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    image.save(OUT / f"{stem}.png", dpi=(450, 450), optimize=True)


def multiline(value: str, width: int) -> str:
    return "\n".join(wrap(value, width=width, break_long_words=False))


def tech_label(code: str, width: int = 26) -> str:
    raw = TECH_LABELS.get(code, code.replace("TECH_", "").replace("_", " ").title())
    return multiline(raw, width)


def outcome_label(code: str, width: int = 20) -> str:
    raw = OUTCOME_LABELS.get(code, code.replace("SDG3_", "").replace("SDG16_", "").replace("_", " ").title())
    return multiline(raw, width)


def title(draw: ImageDraw.ImageDraw, width: int, value: str, y: int = 55, size: int = 72) -> None:
    box = draw.textbbox((0, 0), value, font=font(size, True))
    draw.text(((width - (box[2] - box[0])) / 2, y), value, fill=DARK, font=font(size, True))


def text_center(draw: ImageDraw.ImageDraw, xy: tuple[float, float], value: str, text_font, fill=DARK, spacing=5) -> None:
    box = draw.multiline_textbbox((0, 0), value, font=text_font, spacing=spacing, align="center")
    draw.multiline_text((xy[0] - (box[2] - box[0]) / 2, xy[1] - (box[3] - box[1]) / 2), value, fill=fill, font=text_font, spacing=spacing, align="center")


def color_mix(low: str, high: str, ratio: float) -> tuple[int, int, int]:
    ratio = min(1.0, max(0.0, ratio))
    a = tuple(int(low[i : i + 2], 16) for i in (1, 3, 5))
    b = tuple(int(high[i : i + 2], 16) for i in (1, 3, 5))
    return tuple(round(x + (y - x) * ratio) for x, y in zip(a, b))


def draw_axes_grid(draw: ImageDraw.ImageDraw, bounds: tuple[int, int, int, int], x_ticks: int, y_values: list[float], y_min: float, y_max: float) -> None:
    left, top, right, bottom = bounds
    for index in range(x_ticks + 1):
        x = left + (right - left) * index / x_ticks
        draw.line((x, top, x, bottom), fill=LIGHT, width=3)
    for value in y_values:
        y = bottom - (value - y_min) / (y_max - y_min) * (bottom - top)
        draw.line((left, y, right, y), fill=LIGHT, width=3)
    draw.line((left, bottom, right, bottom), fill="#9AA1A8", width=4)
    draw.line((left, top, left, bottom), fill="#9AA1A8", width=4)


def figure_similarity_map() -> None:
    width, height = 3300, 2500
    image, draw = new_canvas(width, height)
    title(draw, width, "Paper-similarity map and strongest cross-domain links")
    left, top, right, bottom = 310, 230, 3110, 2220
    draw_axes_grid(draw, (left, top, right, bottom), 6, [0, 1, 2, 3, 4, 5, 6], 0, 6)

    positions = pd.read_csv(NETWORK / "supporting_data" / "network_figure1_positions.csv", encoding="utf-8-sig")
    edges = pd.read_csv(NETWORK / "paper_similarity_edges.csv", encoding="utf-8-sig")
    metrics = pd.read_csv(NETWORK / "paper_network_metrics.csv", encoding="utf-8-sig")
    xmin, xmax = positions["x"].min(), positions["x"].max()
    ymin, ymax = positions["y"].min(), positions["y"].max()
    xpad, ypad = (xmax - xmin) * 0.04, (ymax - ymin) * 0.05
    xmin, xmax, ymin, ymax = xmin - xpad, xmax + xpad, ymin - ypad, ymax + ypad

    def point(x: float, y: float) -> tuple[float, float]:
        return left + (x - xmin) / (xmax - xmin) * (right - left), bottom - (y - ymin) / (ymax - ymin) * (bottom - top)

    xy = {row.paper_id: point(row.x, row.y) for row in positions.itertuples()}
    cross = edges.loc[
        ((edges["source_domain"] == "SDG3") & (edges["target_domain"] == "SDG16"))
        | ((edges["source_domain"] == "SDG16") & (edges["target_domain"] == "SDG3"))
    ].nlargest(60, "similarity")
    for row in cross.itertuples():
        if row.source in xy and row.target in xy:
            draw.line((*xy[row.source], *xy[row.target]), fill="#C8C0E2", width=4)

    palette = {"SDG3": BLUE, "SDG16": ORANGE, "Mixed": GREEN}
    for row in positions.itertuples():
        x, y = xy[row.paper_id]
        radius = 11
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=palette[row.source_domain], outline=WHITE, width=2)

    label_font = font(43)
    for row in metrics.nsmallest(6, "bridge_rank").itertuples():
        if row.paper_id in xy:
            x, y = xy[row.paper_id]
            box = draw.textbbox((0, 0), row.paper_id, font=label_font)
            tx, ty = x + 18, y - 58
            draw.rounded_rectangle((tx - 8, ty - 5, tx + box[2] + 8, ty + box[3] + 7), radius=8, fill=WHITE, outline="#B8BEC5", width=2)
            draw.text((tx, ty), row.paper_id, fill=DARK, font=label_font)

    legend_x, legend_y = 2560, 280
    for index, (domain, label) in enumerate((("SDG3", "SDG 3"), ("SDG16", "SDG 16"), ("Mixed", "Mixed"))):
        y = legend_y + index * 72
        draw.ellipse((legend_x, y, legend_x + 28, y + 28), fill=palette[domain])
        draw.text((legend_x + 45, y - 8), label, fill=DARK, font=font(46))
    text_center(draw, ((left + right) / 2, 2390), "Latent semantic dimension 1", font(50))
    save(image, "figure2_similarity_map_readable")


def figure_similarity_distributions() -> None:
    width, height = 3300, 1950
    image, draw = new_canvas(width, height)
    title(draw, width, "All-pair similarity within and across SDG domains")
    left, top, right, bottom = 280, 290, 3150, 1650
    y_min, y_max = -0.10, 1.04
    draw_axes_grid(draw, (left, top, right, bottom), 5, [0, 0.2, 0.4, 0.6, 0.8, 1.0], y_min, y_max)
    pairs = pd.read_csv(NETWORK / "paper_similarity_pairs_long.csv", encoding="utf-8-sig")
    summary = pd.read_csv(NETWORK / "similarity_group_summary.csv", encoding="utf-8-sig").set_index("comparison_group")
    order = ["Within SDG 3", "Within SDG 16", "Cross SDG 3–SDG 16", "Involving mixed-domain", "Within mixed-domain"]
    labels = ["Within\nSDG 3", "Within\nSDG 16", "Cross\nSDG 3–SDG 16", "Involving\nmixed-domain", "Within\nmixed-domain"]
    colors = [BLUE, ORANGE, PURPLE, GREEN, GREY]

    for tick in [0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        y = bottom - (tick - y_min) / (y_max - y_min) * (bottom - top)
        draw.text((90, y - 26), f"{tick:.1f}", fill=DARK, font=font(42))
    x_step = (right - left) / len(order)
    bins = np.linspace(y_min, y_max, 100)
    centres = (bins[:-1] + bins[1:]) / 2
    kernel_x = np.linspace(-3, 3, 21)
    kernel = np.exp(-0.5 * kernel_x**2)
    kernel /= kernel.sum()
    for index, (group, label, color) in enumerate(zip(order, labels, colors)):
        values = pairs.loc[pairs["comparison_group"] == group, "cosine_similarity"].to_numpy()
        density, _ = np.histogram(values, bins=bins, density=True)
        density = np.convolve(density, kernel, mode="same")
        density = density / max(density.max(), 1e-9) * x_step * 0.36
        x_mid = left + x_step * (index + 0.5)
        left_points, right_points = [], []
        for y_value, spread in zip(centres, density):
            y = bottom - (y_value - y_min) / (y_max - y_min) * (bottom - top)
            left_points.append((x_mid - spread, y))
            right_points.append((x_mid + spread, y))
        draw.polygon(left_points + right_points[::-1], fill=color)
        median = float(np.median(values))
        median_y = bottom - (median - y_min) / (y_max - y_min) * (bottom - top)
        draw.line((x_mid - x_step * 0.16, median_y, x_mid + x_step * 0.16, median_y), fill=DARK, width=8)
        text_center(draw, (x_mid, 1765), label, font(43), spacing=1)
        row = summary.loc[group]
        text_center(draw, (x_mid, 240), f"n={int(row['pair_count']):,}\nmedian={row['median']:.3f}", font(40), spacing=1)
    save(image, "figure3_similarity_distributions_readable")


def figure_technology_bars() -> None:
    width, height = 3300, 2850
    image, draw = new_canvas(width, height)
    title(draw, width, "Technology families connected to SDG 3 and SDG 16 outcomes")
    data = pd.read_csv(FIGDATA / "figure1_technology_by_sdg.csv", encoding="utf-8-sig").nlargest(12, "Total").sort_values("Total", ascending=False)
    left, top, right, bottom = 1280, 260, 3100, 2520
    maximum = max(data[["SDG3", "SDG16"]].to_numpy().max(), 1)
    for tick in range(0, int(np.ceil(maximum / 50) * 50) + 1, 50):
        x = left + tick / (maximum * 1.13) * (right - left)
        draw.line((x, top, x, bottom), fill=LIGHT, width=3)
        draw.text((x - 25, bottom + 35), str(tick), fill=DARK, font=font(39))
    row_h = (bottom - top) / len(data)
    bar_h = row_h * 0.30
    for index, row in enumerate(data.itertuples()):
        cy = top + row_h * (index + 0.5)
        label = tech_label(row.technology_code, 29)
        box = draw.multiline_textbbox((0, 0), label, font=font(45), spacing=2, align="right")
        draw.multiline_text((left - 55 - (box[2] - box[0]), cy - (box[3] - box[1]) / 2), label, fill=DARK, font=font(45), spacing=2, align="right")
        for offset, value, color in ((-bar_h * 0.62, row.SDG3, BLUE), (bar_h * 0.62, row.SDG16, ORANGE)):
            bar_right = left + value / (maximum * 1.13) * (right - left)
            draw.rectangle((left, cy + offset - bar_h / 2, bar_right, cy + offset + bar_h / 2), fill=color)
            if value:
                draw.text((bar_right + 18, cy + offset - 25), str(int(value)), fill=DARK, font=font(40))
    draw.line((left, bottom, right, bottom), fill="#9AA1A8", width=4)
    legend_y = 2700
    draw.rectangle((2000, legend_y, 2060, legend_y + 28), fill=BLUE)
    draw.text((2080, legend_y - 12), "SDG 3 outcomes", fill=DARK, font=font(42))
    draw.rectangle((2550, legend_y, 2610, legend_y + 28), fill=ORANGE)
    draw.text((2630, legend_y - 12), "SDG 16 outcomes", fill=DARK, font=font(42))
    save(image, "figure4_technology_bars_readable")


def figure_heatmaps() -> None:
    width, height = 3300, 3900
    image, draw = new_canvas(width, height)
    title(draw, width, "Technology–outcome concentration across SDG domains")
    datasets = [
        ("SDG 3 outcomes", pd.read_csv(FIGDATA / "figure2_heatmap_sdg3.csv", encoding="utf-8-sig").head(9), BLUE, "#F7FBFF"),
        ("SDG 16 outcomes", pd.read_csv(FIGDATA / "figure2_heatmap_sdg16.csv", encoding="utf-8-sig").head(9), ORANGE, "#FFF9F2"),
    ]
    panel_tops = [300, 2180]
    left, right = 970, 3040
    cell_w = (right - left) / 7
    for (panel_title, data, high, low), panel_top in zip(datasets, panel_tops):
        codes = data.pop("technology_code")
        values = data.to_numpy()
        vmax = max(int(values.max()), 1)
        draw.text((80, panel_top - 85), panel_title, fill=DARK, font=font(57, True))
        cell_h = 115
        grid_top = panel_top
        for row_index, code in enumerate(codes):
            cy = grid_top + row_index * cell_h
            label = tech_label(code, 28)
            box = draw.multiline_textbbox((0, 0), label, font=font(39), spacing=1, align="right")
            draw.multiline_text((left - 35 - (box[2] - box[0]), cy + cell_h / 2 - (box[3] - box[1]) / 2), label, fill=DARK, font=font(39), spacing=1, align="right")
            for col_index, value in enumerate(values[row_index]):
                x0, y0 = left + col_index * cell_w, cy
                ratio = value / vmax
                fill_color = color_mix(low, high, ratio)
                draw.rectangle((x0, y0, x0 + cell_w, y0 + cell_h), fill=fill_color, outline=WHITE, width=4)
                if value:
                    text_center(draw, (x0 + cell_w / 2, y0 + cell_h / 2), str(int(value)), font(41), fill=WHITE if ratio >= 0.52 else DARK)
        label_y = grid_top + len(codes) * cell_h + 25
        for col_index, code in enumerate(data.columns):
            text_center(draw, (left + (col_index + 0.5) * cell_w, label_y + 95), outcome_label(code, 16), font(34), spacing=1)
    save(image, "figure5_heatmaps_readable")


def cubic_points(start: tuple[float, float], end: tuple[float, float], steps: int = 45) -> list[tuple[float, float]]:
    x1, y1 = start
    x2, y2 = end
    c1, c2 = (x1 + (x2 - x1) * 0.42, y1), (x2 - (x2 - x1) * 0.42, y2)
    points = []
    for t in np.linspace(0, 1, steps):
        x = (1 - t) ** 3 * x1 + 3 * (1 - t) ** 2 * t * c1[0] + 3 * (1 - t) * t**2 * c2[0] + t**3 * x2
        y = (1 - t) ** 3 * y1 + 3 * (1 - t) ** 2 * t * c1[1] + 3 * (1 - t) * t**2 * c2[1] + t**3 * y2
        points.append((x, y))
    return points


def figure_aggregate_network() -> None:
    width, height = 3300, 3300
    image, draw = new_canvas(width, height)
    title(draw, width, "Aggregate technology–mechanism–outcome network")
    tm = pd.read_csv(NETWORK / "supporting_data" / "network_figure3_technology_mechanism_edges.csv", encoding="utf-8-sig")
    mo = pd.read_csv(NETWORK / "supporting_data" / "network_figure3_mechanism_outcome_edges.csv", encoding="utf-8-sig")
    technologies = tm.groupby("technology_code")["paper_count"].sum().nlargest(7).index.tolist()
    mechanisms = pd.concat([tm.groupby("mechanism_code")["paper_count"].sum(), mo.groupby("mechanism_code")["paper_count"].sum()], axis=1).fillna(0).sum(axis=1).nlargest(7).index.tolist()
    sdg3 = mo.loc[mo["outcome_code"].str.startswith("SDG3_")].groupby("outcome_code")["paper_count"].sum().nlargest(4).index.tolist()
    sdg16 = mo.loc[mo["outcome_code"].str.startswith("SDG16_")].groupby("outcome_code")["paper_count"].sum().nlargest(4).index.tolist()
    outcomes = sdg3 + sdg16
    tm = tm.loc[tm["technology_code"].isin(technologies) & tm["mechanism_code"].isin(mechanisms)].nlargest(24, "paper_count")
    mo = mo.loc[mo["mechanism_code"].isin(mechanisms) & mo["outcome_code"].isin(outcomes)].nlargest(24, "paper_count")
    tech_pos = {code: (860, y) for code, y in zip(technologies, np.linspace(520, 2920, len(technologies)))}
    mech_pos = {code: (1640, y) for code, y in zip(mechanisms, np.linspace(520, 2920, len(mechanisms)))}
    out_pos = {code: (2440, y) for code, y in zip(outcomes, np.linspace(450, 2990, len(outcomes)))}
    maximum = max(tm["paper_count"].max(), mo["paper_count"].max(), 1)
    for row in tm.sort_values("paper_count").itertuples():
        draw.line(cubic_points(tech_pos[row.technology_code], mech_pos[row.mechanism_code]), fill="#B9D9D9", width=4 + round(18 * row.paper_count / maximum))
    for row in mo.sort_values("paper_count").itertuples():
        color = "#BDD1E2" if row.outcome_code.startswith("SDG3_") else "#F1CFAC"
        draw.line(cubic_points(mech_pos[row.mechanism_code], out_pos[row.outcome_code]), fill=color, width=4 + round(18 * row.paper_count / maximum))
    draw.text((610, 285), "Technologies", fill=DARK, font=font(55, True))
    draw.text((1460, 285), "Mechanisms", fill=DARK, font=font(55, True))
    draw.text((2280, 285), "Reported outcomes", fill=DARK, font=font(55, True))
    for code, (x, y) in tech_pos.items():
        draw.rounded_rectangle((x - 28, y - 40, x + 28, y + 40), radius=13, fill=TEAL)
        label = tech_label(code, 26)
        box = draw.multiline_textbbox((0, 0), label, font=font(40), spacing=1, align="right")
        draw.multiline_text((x - 50 - (box[2] - box[0]), y - (box[3] - box[1]) / 2), label, fill=DARK, font=font(40), spacing=1, align="right")
    for code, (x, y) in mech_pos.items():
        draw.rounded_rectangle((x - 245, y - 68, x + 245, y + 68), radius=24, fill=PURPLE)
        text_center(draw, (x, y), multiline(MECHANISM_LABELS.get(code, code.replace("MECH_", "").replace("_", " ").title()), 22), font(38), fill=WHITE, spacing=1)
    for code, (x, y) in out_pos.items():
        color = BLUE if code.startswith("SDG3_") else ORANGE
        draw.rounded_rectangle((x - 28, y - 40, x + 28, y + 40), radius=13, fill=color)
        draw.multiline_text((x + 50, y - 46), outcome_label(code, 25), fill=DARK, font=font(40), spacing=1)
    draw.text((2350, 350), "SDG 3", fill=BLUE, font=font(41, True))
    draw.text((2530, 350), "SDG 16", fill=ORANGE, font=font(41, True))
    save(image, "figure6_aggregate_network_readable")


def clean_cross_label(value: str) -> str:
    return {
        "Same sentence/span": "Same evidence span",
        "Same technology in same paper; different evidence spans": "Paper-level co-occurrence",
        "Association requires review": "Unclassified association",
        "Synergy candidate": "Synergy",
        "Trade-off candidate": "Trade-off",
        "Joint-risk candidate": "Joint risk",
        "Undetermined—paper-level co-occurrence": "Undetermined: paper-level",
        "Undetermined—human coding required": "Undetermined: human review",
    }.get(value, value)


def figure_cross_sdg() -> None:
    width, height = 3300, 3300
    image, draw = new_canvas(width, height)
    title(draw, width, "Structure of cross-SDG relationship candidates")
    files = ["figure3_basis.csv", "figure3_type.csv", "figure3_direction.csv"]
    titles = ["Evidence basis", "Provisional relationship type", "Provisional direction"]
    color_maps = [
        {"Same sentence/span": BLUE, "Same technology in same paper; different evidence spans": GREY},
        {"Synergy candidate": GREEN, "Trade-off candidate": GOLD, "Joint-risk candidate": RED, "Association requires review": GREY},
        {"SDG16 → SDG3": ORANGE, "SDG3 → SDG16": BLUE, "Undetermined—human coding required": PURPLE, "Undetermined—paper-level co-occurrence": GREY},
    ]
    panel_tops = [270, 1250, 2300]
    plot_left, plot_right = 1370, 3040
    for panel_top, filename, panel_title, color_map in zip(panel_tops, files, titles, color_maps):
        data = pd.read_csv(FIGDATA / filename, encoding="utf-8-sig").sort_values("count", ascending=False)
        draw.text((80, panel_top), panel_title, fill=DARK, font=font(56, True))
        row_h = min(180, 690 / max(len(data), 1))
        maximum = max(data["count"].max(), 1)
        for index, row in enumerate(data.itertuples()):
            cy = panel_top + 140 + row_h * (index + 0.5)
            label = multiline(clean_cross_label(str(row.category)), 31)
            box = draw.multiline_textbbox((0, 0), label, font=font(45), spacing=1, align="right")
            draw.multiline_text((plot_left - 45 - (box[2] - box[0]), cy - (box[3] - box[1]) / 2), label, fill=DARK, font=font(45), spacing=1, align="right")
            bar_right = plot_left + row.count / (maximum * 1.16) * (plot_right - plot_left)
            draw.rounded_rectangle((plot_left, cy - 48, bar_right, cy + 48), radius=10, fill=color_map.get(row.category, GREY))
            draw.text((bar_right + 20, cy - 30), str(int(row.count)), fill=DARK, font=font(44))
        baseline = panel_top + 140 + row_h * len(data)
        draw.line((plot_left, baseline, plot_right, baseline), fill="#9AA1A8", width=4)
    save(image, "figure7_cross_sdg_readable")


def draw_stacked_panel(draw: ImageDraw.ImageDraw, data: pd.DataFrame, colors: dict[str, str], panel_top: int, panel_title: str) -> None:
    draw.text((80, panel_top), panel_title, fill=DARK, font=font(56, True))
    data = data.set_index("outcome_sdg").reindex(["SDG3", "SDG16"])
    plot_left, plot_right = 520, 3100
    y_values = [panel_top + 220, panel_top + 440]
    for row_index, (sdg, row) in enumerate(data.iterrows()):
        y = y_values[row_index]
        draw.text((100, y - 35), sdg.replace("SDG", "SDG "), fill=DARK, font=font(48))
        start = plot_left
        for category, value in row.items():
            end = start + float(value) / 100 * (plot_right - plot_left)
            draw.rectangle((start, y - 58, end, y + 58), fill=colors[category])
            if value >= 7:
                text_center(draw, ((start + end) / 2, y), f"{value:.0f}%", font(40), fill=WHITE if colors[category] not in {GOLD, LIGHT} else DARK)
            start = end
    categories = list(data.columns)
    columns = 3
    legend_y = panel_top + 620
    for index, category in enumerate(categories):
        col, row = index % columns, index // columns
        x, y = 520 + col * 820, legend_y + row * 95
        draw.rectangle((x, y, x + 55, y + 35), fill=colors[category])
        draw.text((x + 75, y - 10), multiline(category, 26), fill=DARK, font=font(34), spacing=1)


def figure_evidence_profile() -> None:
    width, height = 3300, 3000
    image, draw = new_canvas(width, height)
    title(draw, width, "Evidence profile of automated technology–outcome candidates")
    evidence = pd.read_csv(FIGDATA / "figure4_evidence.csv", encoding="utf-8-sig")
    polarity = pd.read_csv(FIGDATA / "figure4_polarity.csv", encoding="utf-8-sig")
    evidence_colors = {
        "Causal estimate": TEAL,
        "Statistical association": BLUE,
        "Empirically observed or reported": GREEN,
        "Empirical, relation unclear": PURPLE,
        "Proposed or potential": GOLD,
        "Descriptive or conceptual": GREY,
    }
    polarity_colors = {"Positive": GREEN, "Negative": RED, "Mixed": GOLD, "Null or mixed": PURPLE, "Unclear": GREY}
    draw_stacked_panel(draw, evidence, evidence_colors, 260, "Reported evidence strength")
    draw_stacked_panel(draw, polarity, polarity_colors, 1700, "Reported effect polarity")
    save(image, "figure8_evidence_profile_readable")


def main() -> None:
    global NETWORK, FIGDATA, OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network-dir", type=Path, required=True)
    parser.add_argument("--figure-data-dir", type=Path, required=True, help="supporting_data folder produced by make_figures.py")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--regular-font", type=Path)
    parser.add_argument("--bold-font", type=Path)
    parser.add_argument("--include-legacy-map", action="store_true", help="Also draw the PCA map superseded by Ve10 network views")
    args = parser.parse_args()
    NETWORK, FIGDATA, OUT = args.network_dir, args.figure_data_dir, args.output
    configure_fonts(args.regular_font, args.bold_font)
    if args.include_legacy_map:
        figure_similarity_map()
    figure_similarity_distributions()
    figure_technology_bars()
    figure_heatmaps()
    figure_aggregate_network()
    figure_cross_sdg()
    figure_evidence_profile()
    print(OUT)


if __name__ == "__main__":
    main()
