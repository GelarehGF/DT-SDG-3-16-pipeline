#!/usr/bin/env python3
"""Auditable Technology–Mechanism–Outcome–SDG coding pipeline.

This program performs dictionary- and rule-based computer-assisted content
analysis. It does not call a generative model or an external API. Every
candidate code retains its evidence sentence, triggered term, rule id, and
confidence score so that human coders can validate the final dataset.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence
from xml.etree import ElementTree as ET

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.I)
YEAR_RE = re.compile(r"\b(19\d{2}|20[0-2]\d)\b")
SPACE_RE = re.compile(r"\s+")
SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\[(])")
REFERENCE_HEADING_RE = re.compile(
    r"^(?:references|bibliography|works cited|literature cited)(?:\s*[:.]?)$", re.I
)
KEYWORD_LINE_RE = re.compile(r"^(?:key ?words?|index terms?)\s*[:—-]", re.I)
SECTION_HEADING_RE = re.compile(
    r"^(abstract|introduction|background|literature review|theoretical framework|"
    r"methodology|methods?|materials and methods|results?|findings?|discussion|"
    r"conclusions?|limitations?|implications?|recommendations?)$",
    re.I,
)
GENERIC_TITLE_RE = re.compile(
    r"^(abstract|introduction|research article|original article|article|review|"
    r"special article|open access|editorial)$",
    re.I,
)
TITLE_EXCLUDE_RE = re.compile(
    r"\b(email|university|department|faculty|correspond|copyright|doi|volume|vol\.|"
    r"received|accepted|published|issn|journal|license)\b",
    re.I,
)
TITLE_BOILERPLATE_RE = re.compile(
    r"^(?:contents lists? available at sciencedirect|"
    r"(?:a\s+)?r\s*e\s*s\s*e\s*a\s*r\s*c\s*h\s+a\s*r\s*t\s*i\s*c\s*l\s*e|"
    r"a\s*r\s*t\s*i\s*c\s*l\s*e\s+i\s*n\s*f\s*o|"
    r"o\s*r\s*i\s*g\s*i\s*n\s*a\s*l\s+a\s*r\s*t\s*i\s*c\s*l\s*e|"
    r"jmir public health and surveillance|"
    r"monograph dedicated to .+|"
    r"this document is discoverable and free to researchers.+)$",
    re.I,
)
ADJACENT_LINK_RE = re.compile(
    r"^(?:this|these|such|it|they|the technology|the system|the platform|the tool)\b|"
    r"\b(?:therefore|thus|consequently|as a result|thereby|in turn)\b",
    re.I,
)


@dataclass(frozen=True)
class SentenceRecord:
    index: int
    paragraph_index: int
    section: str
    text: str
    before: str
    after: str


def compact(value: str) -> str:
    return SPACE_RE.sub(" ", value or "").strip()


def normalized_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", compact(value).casefold()).strip()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


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


def extract_docx(path: Path) -> tuple[list[str], dict[str, str]]:
    paragraphs: list[str] = []
    metadata: dict[str, str] = {}
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        root = ET.fromstring(archive.read("word/document.xml"))
        for paragraph in root.iter(f"{{{W_NS}}}p"):
            text = paragraph_text(paragraph)
            if text:
                paragraphs.append(text)
        if "docProps/core.xml" in names:
            core = ET.fromstring(archive.read("docProps/core.xml"))
            for child in core:
                key = child.tag.rsplit("}", 1)[-1]
                if child.text:
                    metadata[key] = compact(child.text)
    return paragraphs, metadata


def extract_pdf(path: Path) -> tuple[list[str], dict[str, str]]:
    from pypdf import PdfReader

    reader = PdfReader(str(path), strict=False)
    pages: list[str] = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    text = "\n".join(pages)
    paragraphs = [compact(v) for v in re.split(r"\n{2,}|\f", text) if compact(v)]
    if len(paragraphs) < 5:
        paragraphs = [compact(v) for v in text.splitlines() if compact(v)]
    metadata: dict[str, str] = {"pages": str(len(reader.pages))}
    if reader.metadata:
        for key, value in reader.metadata.items():
            if value is not None:
                metadata[str(key).lstrip("/")] = compact(str(value))
    return paragraphs, metadata


def extract_txt(path: Path) -> tuple[list[str], dict[str, str]]:
    raw = path.read_bytes()
    encoding_used = "utf-8-replace"
    text = raw.decode("utf-8", errors="replace")
    for encoding in ("utf-8-sig", "utf-8", "utf-16", "cp1252", "latin-1"):
        try:
            text = raw.decode(encoding)
            encoding_used = encoding
            break
        except UnicodeDecodeError:
            continue
    paragraphs = [compact(v) for v in re.split(r"\n{2,}", text) if compact(v)]
    if len(paragraphs) < 5:
        paragraphs = [compact(v) for v in text.splitlines() if compact(v)]
    return paragraphs, {"encoding": encoding_used}


def read_document(path: Path) -> tuple[list[str], dict[str, str]]:
    suffix = path.suffix.casefold()
    if suffix == ".docx":
        return extract_docx(path)
    if suffix == ".pdf":
        return extract_pdf(path)
    if suffix == ".txt":
        return extract_txt(path)
    raise ValueError(f"Unsupported input type: {path.suffix}")


def discover_canonical_files(source: Path) -> list[tuple[Path, str]]:
    """Select one preferred representation per numbered corpus record."""
    preferred: list[tuple[Path, str]] = []
    patterns = [
        (source / "SDG3" / "Words", "SDG3-*.docx", "SDG3"),
        (source / "SDG16" / "Words", "SDG16-*.docx", "SDG16"),
        (source / "mixed", "SDGM-*.docx", "Mixed"),
    ]
    for folder, pattern, domain in patterns:
        if folder.exists():
            preferred.extend((path, domain) for path in sorted(folder.glob(pattern)) if not path.name.startswith("~$"))
    if preferred:
        return preferred

    fallback: list[tuple[Path, str]] = []
    for path in sorted(source.rglob("*")):
        if not path.is_file() or path.name.startswith("~$"):
            continue
        if path.suffix.casefold() not in {".docx", ".pdf", ".txt"}:
            continue
        rel = str(path.relative_to(source)).casefold()
        domain = "Mixed" if "mixed" in rel or path.stem.casefold().startswith("sdgm-") else (
            "SDG3" if "sdg3" in rel or path.stem.casefold().startswith("sdg3-") else (
                "SDG16" if "sdg16" in rel or path.stem.casefold().startswith("sdg16-") else "Unassigned"
            )
        )
        fallback.append((path, domain))
    return fallback


def infer_title(paragraphs: Sequence[str], metadata: dict[str, str]) -> str:
    metadata_title = compact(metadata.get("title", "") or metadata.get("Title", ""))
    if 4 <= len(metadata_title.split()) <= 40:
        return metadata_title
    candidates: list[tuple[float, str]] = []
    for position, text in enumerate(paragraphs[:30]):
        if text.casefold() == "abstract":
            break
        words = text.split()
        if not 5 <= len(words) <= 35 or len(text) > 320:
            continue
        if GENERIC_TITLE_RE.match(text) or TITLE_EXCLUDE_RE.search(text) or TITLE_BOILERPLATE_RE.match(text):
            continue
        if "@" in text or DOI_RE.search(text):
            continue
        alpha = [c for c in text if c.isalpha()]
        upper_share = sum(c.isupper() for c in alpha) / max(1, len(alpha))
        punctuation_penalty = text.count(",") * 0.4 + text.count(";")
        score = 6.0 - position * 0.08 + min(len(words), 18) * 0.04 - punctuation_penalty
        if upper_share > 0.75:
            score += 0.4
        candidates.append((score, text))
    return max(candidates, default=(0.0, ""), key=lambda item: item[0])[1]


def infer_doi(text: str) -> str:
    for candidate in DOI_RE.findall(text[:8000]):
        doi = candidate.rstrip(".,;:)]}").casefold()
        suffix = doi.split("/", 1)[1] if "/" in doi else ""
        # Reject OCR-truncated DOI stems such as "10.1371/journal". They are
        # not reliable deduplication identifiers and commonly appear in cited
        # material rather than the article header.
        if len(suffix) >= 4 and re.search(r"\d", suffix):
            return doi
    return ""


def reliable_title_for_deduplication(title: str) -> bool:
    words = title.split()
    return (
        8 <= len(words) <= 35
        and len(title) <= 320
        and not TITLE_BOILERPLATE_RE.match(title)
        and not GENERIC_TITLE_RE.match(title)
    )


def infer_year(text: str) -> str:
    years = [int(value) for value in YEAR_RE.findall(text[:5000])]
    if not years:
        return ""
    counts = Counter(years)
    year = max(counts, key=lambda value: (counts[value], -years.index(value)))
    return str(year)


def split_sentences(paragraphs: Sequence[str]) -> list[SentenceRecord]:
    staged: list[tuple[int, str, str]] = []
    section = "Front matter"
    for paragraph_index, paragraph in enumerate(paragraphs):
        text = compact(paragraph)
        if not text:
            continue
        if REFERENCE_HEADING_RE.match(text):
            break
        if KEYWORD_LINE_RE.match(text):
            continue
        heading = SECTION_HEADING_RE.match(text.rstrip(":."))
        if heading and len(text.split()) <= 5:
            section = heading.group(1).title()
            continue
        pieces = SENTENCE_BOUNDARY_RE.split(text)
        for piece in pieces:
            sentence = compact(piece)
            if len(sentence) < 25:
                continue
            if len(sentence) > 1800:
                sentence = sentence[:1800]
            staged.append((paragraph_index, section, sentence))
    result: list[SentenceRecord] = []
    for index, (paragraph_index, section, text) in enumerate(staged):
        before = staged[index - 1][2] if index > 0 else ""
        after = staged[index + 1][2] if index + 1 < len(staged) else ""
        result.append(SentenceRecord(index, paragraph_index, section, text, before, after))
    return result


def compile_entries(entries: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    compiled: list[dict[str, Any]] = []
    for entry in entries:
        copy = dict(entry)
        copy["compiled_patterns"] = [re.compile(pattern, re.I) for pattern in entry["patterns"]]
        compiled.append(copy)
    return compiled


def compile_patterns(patterns: Sequence[str]) -> list[re.Pattern[str]]:
    return [re.compile(pattern, re.I) for pattern in patterns]


def match_entries(text: str, entries: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for entry in entries:
        candidates = []
        for pattern in entry["compiled_patterns"]:
            found = pattern.search(text)
            if found:
                candidates.append(found)
        if not candidates:
            continue
        best = min(candidates, key=lambda found: (found.start(), -(found.end() - found.start())))
        matches.append(
            {
                "code": entry["code"],
                "label": entry["label"],
                "sdg": entry.get("sdg", ""),
                "term": best.group(0),
                "start": best.start(),
                "end": best.end(),
            }
        )
    return matches


def first_pattern_match(text: str, patterns: Sequence[re.Pattern[str]]) -> str:
    for pattern in patterns:
        found = pattern.search(text)
        if found:
            return found.group(0)
    return ""


def infer_study_design(text: str, designs: Sequence[dict[str, Any]]) -> tuple[str, str, str]:
    for design in designs:
        match = first_pattern_match(text[:30000], design["compiled_patterns"])
        if match:
            return design["code"], design["label"], match
    return "DESIGN_UNCLEAR", "Unclear or not automatically identified", ""


def classify_evidence(
    text: str,
    design_code: str,
    evidence_patterns: dict[str, list[re.Pattern[str]]],
) -> tuple[str, str]:
    causal_cue = first_pattern_match(text, evidence_patterns["causal"])
    if causal_cue and design_code in {"DESIGN_RANDOM", "DESIGN_QUASI"}:
        return "Causal estimate", causal_cue
    statistical_cue = first_pattern_match(text, evidence_patterns["statistical"])
    if statistical_cue:
        return "Statistical association", statistical_cue
    observed_cue = first_pattern_match(text, evidence_patterns["observed"])
    if observed_cue:
        return "Empirically observed or reported", observed_cue
    proposed_cue = first_pattern_match(text, evidence_patterns["proposed"])
    if proposed_cue:
        return "Proposed or potential", proposed_cue
    if design_code in {"DESIGN_RANDOM", "DESIGN_QUASI", "DESIGN_LONG", "DESIGN_SURVEY", "DESIGN_QUAL", "DESIGN_MIXED"}:
        return "Empirical, relation unclear", ""
    return "Descriptive or conceptual", ""


def classify_polarity(text: str, effect_patterns: dict[str, list[re.Pattern[str]]]) -> tuple[str, str]:
    mixed = first_pattern_match(text, effect_patterns["mixed_or_null"])
    if mixed:
        return "Null or mixed", mixed
    positive = first_pattern_match(text, effect_patterns["positive"])
    negative = first_pattern_match(text, effect_patterns["negative"])
    if positive and negative:
        return "Mixed", f"{positive} | {negative}"
    if negative:
        return "Negative", negative
    if positive:
        return "Positive", positive
    return "Unclear", ""


def score_relation(
    rule_id: str,
    technology_code: str,
    mechanism_count: int,
    polarity: str,
    evidence_level: str,
    section: str,
    has_connector: bool,
) -> float:
    score = 0.55 if rule_id == "R1_SAME_SENTENCE" else 0.43
    if mechanism_count:
        score += 0.08
    if polarity != "Unclear":
        score += 0.07
    if evidence_level in {"Causal estimate", "Statistical association", "Empirically observed or reported"}:
        score += 0.12
    elif evidence_level == "Empirical, relation unclear":
        score += 0.05
    if has_connector:
        score += 0.08
    if section.casefold() in {"abstract", "results", "result", "findings", "discussion", "conclusion", "conclusions"}:
        score += 0.04
    if technology_code == "TECH_PLATFORM_GENERAL":
        score -= 0.10
    return round(max(0.05, min(0.99, score)), 2)


def relation_priority(relation: dict[str, Any]) -> str:
    score = float(relation["confidence_score"])
    if relation["effect_polarity"] in {"Negative", "Mixed", "Null or mixed"}:
        return "High"
    if relation["outcome_sdg"] == "SDG16" and relation["source_domain"] == "SDG3":
        return "High"
    if relation["outcome_sdg"] == "SDG3" and relation["source_domain"] == "SDG16":
        return "High"
    if 0.55 <= score < 0.75 or relation["rule_id"] == "R2_ADJACENT_CONTEXT":
        return "Medium"
    return "Low"


def make_relation(
    paper: dict[str, Any],
    sentence: SentenceRecord,
    technology: dict[str, Any],
    outcome: dict[str, Any],
    mechanisms: Sequence[dict[str, Any]],
    rule_id: str,
    evidence_text: str,
    design_code: str,
    design_label: str,
    evidence_patterns: dict[str, list[re.Pattern[str]]],
    effect_patterns: dict[str, list[re.Pattern[str]]],
    causal_patterns: Sequence[re.Pattern[str]],
) -> dict[str, Any]:
    evidence_level, evidence_cue = classify_evidence(evidence_text, design_code, evidence_patterns)
    polarity, polarity_cue = classify_polarity(evidence_text, effect_patterns)
    connector = first_pattern_match(evidence_text, causal_patterns)
    confidence = score_relation(
        rule_id,
        technology["code"],
        len(mechanisms),
        polarity,
        evidence_level,
        sentence.section,
        bool(connector),
    )
    mechanism_codes = " | ".join(item["code"] for item in mechanisms)
    mechanism_labels = " | ".join(item["label"] for item in mechanisms)
    mechanism_terms = " | ".join(item["term"] for item in mechanisms)
    return {
        "relation_id": "",
        "paper_id": paper["paper_id"],
        "canonical_paper_id": paper["canonical_paper_id"],
        "source_domain": paper["source_domain"],
        "title": paper["title"],
        "doi": paper["doi"],
        "publication_year_candidate": paper["publication_year_candidate"],
        "study_design_code": design_code,
        "study_design_label": design_label,
        "technology_code": technology["code"],
        "technology_label": technology["label"],
        "technology_trigger": technology["term"],
        "mechanism_codes": mechanism_codes,
        "mechanism_labels": mechanism_labels,
        "mechanism_triggers": mechanism_terms,
        "outcome_sdg": outcome["sdg"],
        "outcome_code": outcome["code"],
        "outcome_label": outcome["label"],
        "outcome_trigger": outcome["term"],
        "effect_polarity": polarity,
        "polarity_cue": polarity_cue,
        "evidence_level": evidence_level,
        "evidence_cue": evidence_cue,
        "causal_connector": connector,
        "section": sentence.section,
        "sentence_index": sentence.index,
        "rule_id": rule_id,
        "confidence_score": confidence,
        "auto_include_candidate": "Yes" if confidence >= 0.65 else "No",
        "review_priority": "",
        "evidence_text": compact(evidence_text),
        "context_before": compact(sentence.before),
        "context_after": compact(sentence.after),
        "source_path": paper["source_path"],
        "human_decision": "Pending",
        "human_technology_code": "",
        "human_mechanism_codes": "",
        "human_outcome_code": "",
        "human_effect_polarity": "",
        "human_evidence_level": "",
        "human_coder": "",
        "human_notes": "",
    }


def extract_relations_for_paper(
    paper: dict[str, Any],
    sentences: Sequence[SentenceRecord],
    technologies: Sequence[dict[str, Any]],
    mechanisms: Sequence[dict[str, Any]],
    outcomes: Sequence[dict[str, Any]],
    design_code: str,
    design_label: str,
    evidence_patterns: dict[str, list[re.Pattern[str]]],
    effect_patterns: dict[str, list[re.Pattern[str]]],
    causal_patterns: Sequence[re.Pattern[str]],
) -> list[dict[str, Any]]:
    annotated: list[dict[str, Any]] = []
    for sentence in sentences:
        annotated.append(
            {
                "sentence": sentence,
                "technologies": match_entries(sentence.text, technologies),
                "mechanisms": match_entries(sentence.text, mechanisms),
                "outcomes": match_entries(sentence.text, outcomes),
            }
        )

    candidates: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for index, item in enumerate(annotated):
        sentence: SentenceRecord = item["sentence"]
        sentence_technologies = item["technologies"]
        sentence_outcomes = item["outcomes"]
        if sentence_technologies and sentence_outcomes:
            for technology in sentence_technologies:
                for outcome in sentence_outcomes:
                    key = (paper["paper_id"], technology["code"], outcome["code"], sentence.index, "R1")
                    if key in seen:
                        continue
                    seen.add(key)
                    candidates.append(
                        make_relation(
                            paper,
                            sentence,
                            technology,
                            outcome,
                            item["mechanisms"],
                            "R1_SAME_SENTENCE",
                            sentence.text,
                            design_code,
                            design_label,
                            evidence_patterns,
                            effect_patterns,
                            causal_patterns,
                        )
                    )

        if sentence_technologies and not sentence_outcomes and index + 1 < len(annotated):
            following = annotated[index + 1]
            next_sentence: SentenceRecord = following["sentence"]
            if next_sentence.paragraph_index - sentence.paragraph_index > 1:
                continue
            if not following["outcomes"]:
                continue
            evidence_text = f"{sentence.text} {next_sentence.text}"
            if not (ADJACENT_LINK_RE.search(next_sentence.text) or first_pattern_match(evidence_text, causal_patterns)):
                continue
            combined_mechanisms = {item["code"]: item for item in [*item["mechanisms"], *following["mechanisms"]]}
            for technology in sentence_technologies:
                for outcome in following["outcomes"]:
                    key = (paper["paper_id"], technology["code"], outcome["code"], sentence.index, "R2")
                    if key in seen:
                        continue
                    seen.add(key)
                    candidates.append(
                        make_relation(
                            paper,
                            sentence,
                            technology,
                            outcome,
                            list(combined_mechanisms.values()),
                            "R2_ADJACENT_CONTEXT",
                            evidence_text,
                            design_code,
                            design_label,
                            evidence_patterns,
                            effect_patterns,
                            causal_patterns,
                        )
                    )
    return candidates


def retain_best_evidence(relations: Sequence[dict[str, Any]], per_pair: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for relation in relations:
        grouped[(relation["paper_id"], relation["technology_code"], relation["outcome_code"])].append(relation)
    retained: list[dict[str, Any]] = []
    for group in grouped.values():
        group.sort(
            key=lambda row: (
                float(row["confidence_score"]),
                row["rule_id"] == "R1_SAME_SENTENCE",
                row["evidence_level"] != "Proposed or potential",
                -len(row["evidence_text"]),
            ),
            reverse=True,
        )
        retained.extend(group[:per_pair])
    retained.sort(key=lambda row: (row["paper_id"], row["technology_code"], row["outcome_code"], -float(row["confidence_score"])))
    for index, relation in enumerate(retained, 1):
        relation["relation_id"] = f"REL-{index:06d}"
        relation["review_priority"] = relation_priority(relation)
    return retained


def infer_cross_direction(sdg3_relation: dict[str, Any], sdg16_relation: dict[str, Any], evidence: str, causal_patterns: Sequence[re.Pattern[str]]) -> str:
    lower = evidence.casefold()
    term3 = sdg3_relation["outcome_trigger"].casefold()
    term16 = sdg16_relation["outcome_trigger"].casefold()
    position3 = lower.find(term3)
    position16 = lower.find(term16)
    connectors: list[int] = []
    for pattern in causal_patterns:
        connectors.extend(found.start() for found in pattern.finditer(evidence))
    if position3 >= 0 and position16 >= 0 and connectors:
        left, right = sorted((position3, position16))
        if any(left < position < right for position in connectors):
            return "SDG3 → SDG16" if position3 < position16 else "SDG16 → SDG3"
    return "Undetermined—human coding required"


def cross_type(polarity3: str, polarity16: str) -> str:
    if polarity3 == "Positive" and polarity16 == "Positive":
        return "Synergy candidate"
    if {polarity3, polarity16} & {"Negative", "Mixed", "Null or mixed"} and "Positive" in {polarity3, polarity16}:
        return "Trade-off candidate"
    if polarity3 == "Negative" and polarity16 == "Negative":
        return "Joint-risk candidate"
    return "Association requires review"


def build_cross_sdg_relations(relations: Sequence[dict[str, Any]], causal_patterns: Sequence[re.Pattern[str]]) -> list[dict[str, Any]]:
    by_span: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    by_paper_technology: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for relation in relations:
        by_span[(relation["paper_id"], relation["technology_code"], int(relation["sentence_index"]))].append(relation)
        by_paper_technology[(relation["paper_id"], relation["technology_code"])].append(relation)

    rows: list[dict[str, Any]] = []
    strong_keys: set[tuple[str, str]] = set()
    for (paper_id, technology_code, _sentence_index), group in by_span.items():
        sdg3 = [row for row in group if row["outcome_sdg"] == "SDG3"]
        sdg16 = [row for row in group if row["outcome_sdg"] == "SDG16"]
        if not sdg3 or not sdg16:
            continue
        for relation3 in sdg3:
            for relation16 in sdg16:
                strong_keys.add((paper_id, technology_code))
                evidence = relation3["evidence_text"] if relation3["evidence_text"] == relation16["evidence_text"] else f"{relation3['evidence_text']} || {relation16['evidence_text']}"
                score = round(min(float(relation3["confidence_score"]), float(relation16["confidence_score"])), 2)
                rows.append(
                    {
                        "cross_relation_id": "",
                        "paper_id": paper_id,
                        "source_domain": relation3["source_domain"],
                        "title": relation3["title"],
                        "technology_code": technology_code,
                        "technology_label": relation3["technology_label"],
                        "sdg3_outcome_code": relation3["outcome_code"],
                        "sdg3_outcome_label": relation3["outcome_label"],
                        "sdg3_polarity": relation3["effect_polarity"],
                        "sdg16_outcome_code": relation16["outcome_code"],
                        "sdg16_outcome_label": relation16["outcome_label"],
                        "sdg16_polarity": relation16["effect_polarity"],
                        "relationship_type_auto": cross_type(relation3["effect_polarity"], relation16["effect_polarity"]),
                        "direction_auto": infer_cross_direction(relation3, relation16, evidence, causal_patterns),
                        "relation_basis": "Same sentence/span",
                        "confidence_score": score,
                        "evidence_text": evidence,
                        "source_path": relation3["source_path"],
                        "human_decision": "Pending",
                        "human_relationship_type": "",
                        "human_direction": "",
                        "human_coder": "",
                        "human_notes": "",
                    }
                )

    for key, group in by_paper_technology.items():
        if key in strong_keys:
            continue
        sdg3 = sorted((row for row in group if row["outcome_sdg"] == "SDG3"), key=lambda row: float(row["confidence_score"]), reverse=True)
        sdg16 = sorted((row for row in group if row["outcome_sdg"] == "SDG16"), key=lambda row: float(row["confidence_score"]), reverse=True)
        if not sdg3 or not sdg16:
            continue
        relation3, relation16 = sdg3[0], sdg16[0]
        score = round(max(0.05, min(float(relation3["confidence_score"]), float(relation16["confidence_score"])) - 0.20), 2)
        rows.append(
            {
                "cross_relation_id": "",
                "paper_id": relation3["paper_id"],
                "source_domain": relation3["source_domain"],
                "title": relation3["title"],
                "technology_code": relation3["technology_code"],
                "technology_label": relation3["technology_label"],
                "sdg3_outcome_code": relation3["outcome_code"],
                "sdg3_outcome_label": relation3["outcome_label"],
                "sdg3_polarity": relation3["effect_polarity"],
                "sdg16_outcome_code": relation16["outcome_code"],
                "sdg16_outcome_label": relation16["outcome_label"],
                "sdg16_polarity": relation16["effect_polarity"],
                "relationship_type_auto": cross_type(relation3["effect_polarity"], relation16["effect_polarity"]),
                "direction_auto": "Undetermined—paper-level co-occurrence",
                "relation_basis": "Same technology in same paper; different evidence spans",
                "confidence_score": score,
                "evidence_text": f"SDG3 evidence: {relation3['evidence_text']} || SDG16 evidence: {relation16['evidence_text']}",
                "source_path": relation3["source_path"],
                "human_decision": "Pending",
                "human_relationship_type": "",
                "human_direction": "",
                "human_coder": "",
                "human_notes": "",
            }
        )
    rows.sort(key=lambda row: (row["paper_id"], -float(row["confidence_score"]), row["technology_code"]))
    for index, row in enumerate(rows, 1):
        row["cross_relation_id"] = f"CROSS-{index:05d}"
    return rows


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        if fields:
            writer.writeheader()
            writer.writerows(rows)


def flatten_codebook(codebook: dict[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for group in ("technologies", "mechanisms", "outcomes", "study_designs"):
        for entry in codebook[group]:
            rows.append(
                {
                    "group": group,
                    "code": entry["code"],
                    "sdg": entry.get("sdg", ""),
                    "label": entry["label"],
                    "definition": entry.get("definition", ""),
                    "patterns": " | ".join(entry["patterns"]),
                }
            )
    return rows


def build_matrices(relations: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    unique_pairs: set[tuple[str, str, str]] = set()
    technology_counts: Counter[str] = Counter()
    outcome_counts: Counter[str] = Counter()
    technology_labels: dict[str, str] = {}
    outcome_labels: dict[str, str] = {}
    matrix: Counter[tuple[str, str]] = Counter()
    for row in relations:
        pair = (row["paper_id"], row["technology_code"], row["outcome_code"])
        if pair in unique_pairs:
            continue
        unique_pairs.add(pair)
        technology_counts[row["technology_code"]] += 1
        outcome_counts[row["outcome_code"]] += 1
        matrix[(row["technology_code"], row["outcome_code"])] += 1
        technology_labels[row["technology_code"]] = row["technology_label"]
        outcome_labels[row["outcome_code"]] = row["outcome_label"]
    top_technologies = [
        {"technology_code": code, "technology_label": technology_labels[code], "paper_outcome_pairs": count}
        for code, count in technology_counts.most_common()
    ]
    top_outcomes = [
        {"outcome_code": code, "outcome_label": outcome_labels[code], "paper_technology_pairs": count}
        for code, count in outcome_counts.most_common()
    ]
    matrix_rows = [
        {
            "technology_code": technology_code,
            "technology_label": technology_labels[technology_code],
            "outcome_code": outcome_code,
            "outcome_label": outcome_labels[outcome_code],
            "unique_paper_pairs": count,
        }
        for (technology_code, outcome_code), count in sorted(matrix.items(), key=lambda item: (-item[1], item[0]))
    ]
    return top_technologies, top_outcomes, matrix_rows


def build_payload(
    summary: dict[str, Any],
    papers: Sequence[dict[str, Any]],
    relations: Sequence[dict[str, Any]],
    cross_relations: Sequence[dict[str, Any]],
    codebook_rows: Sequence[dict[str, Any]],
    matrix_rows: Sequence[dict[str, Any]],
    top_technologies: Sequence[dict[str, Any]],
    top_outcomes: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    review_queue = sorted(
        relations,
        key=lambda row: (
            {"High": 0, "Medium": 1, "Low": 2}.get(row["review_priority"], 3),
            abs(float(row["confidence_score"]) - 0.65),
        ),
    )
    readme_rows = [
        {"item": "Method", "value": "Dictionary- and rule-based computer-assisted content analysis; no generative AI/API"},
        {"item": "Unit", "value": "Technology–outcome relationship supported by a sentence or linked adjacent sentence pair"},
        {"item": "Validation", "value": "human_decision remains Pending until reviewed; automated candidates must not be treated as final codes"},
        {"item": "Cross-SDG rule", "value": "Same-span relations are candidates; paper-level co-occurrence is weaker and cannot establish causality"},
        {"item": "Duplicate policy", "value": "Exact normalized-text, DOI, and normalized-title duplicates are retained in Papers but excluded from analysis candidates"},
        {"item": "Reproducibility", "value": f"Codebook version {summary['codebook_version']}; generated {summary['generated_at_utc']}"},
    ]
    return {
        "summary": summary,
        "top_technologies": list(top_technologies),
        "top_outcomes": list(top_outcomes),
        "sheets": {
            "README": readme_rows,
            "Papers": list(papers),
            "Relations": list(relations),
            "Cross-SDG": list(cross_relations),
            "Review Queue": list(review_queue),
            "Tech-Outcome Matrix": list(matrix_rows),
            "Codebook": list(codebook_rows),
        },
    }


def export_workbook(script_dir: Path, payload_path: Path, workbook_path: Path, preview_dir: Path) -> tuple[bool, str]:
    """Optional portable tabular export; CSV/JSON remain the analysis inputs.

    The two unused path arguments retain compatibility with the original caller.
    Text is explicitly stored as text, never as an executable spreadsheet formula.
    This export does not claim to reproduce the earlier styled workbook previews.
    """
    try:
        from openpyxl import Workbook
        from openpyxl.cell import WriteOnlyCell
    except ImportError:
        return False, "Install openpyxl or use --skip-workbook; CSV/JSON were created."
    try:
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        workbook = Workbook(write_only=True)
        for name, records in payload["sheets"].items():
            sheet = workbook.create_sheet(name)
            headers = list(dict.fromkeys(key for row in records for key in row))
            sheet.append(headers)
            for record in records:
                cells = []
                for key in headers:
                    value = record.get(key, "")
                    if isinstance(value, (dict, list)):
                        value = json.dumps(value, ensure_ascii=False)
                    cell = WriteOnlyCell(sheet, value=value)
                    if isinstance(value, str):
                        cell.data_type = "s"
                    cells.append(cell)
                sheet.append(cells)
        workbook.save(workbook_path)
    except Exception as error:
        return False, f"Workbook export failed: {error}; CSV/JSON remain available."
    return True, "Portable tabular workbook exported (no visual previews)."


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract auditable Technology–Mechanism–Outcome–SDG coding candidates from a document corpus."
    )
    parser.add_argument("--source", required=True, type=Path, help="Read-only SDG3-16 corpus root")
    parser.add_argument("--output", required=True, type=Path, help="Writable output directory")
    parser.add_argument(
        "--codebook",
        type=Path,
        default=Path(__file__).with_name("codebook.json"),
        help="Editable JSON codebook",
    )
    parser.add_argument("--limit", type=int, default=0, help="Process only the first N canonical files (testing only)")
    parser.add_argument("--max-evidence-per-pair", type=int, default=3, help="Maximum evidence spans retained per paper/technology/outcome")
    parser.add_argument("--skip-workbook", action="store_true", help="Create CSV/JSON outputs but not the XLSX workbook")
    parser.add_argument("--fail-on-workbook-error", action="store_true", help="Return an error if XLSX export fails")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    codebook_path = args.codebook.expanduser().resolve()
    if not source.is_dir():
        raise SystemExit(f"Source directory does not exist: {source}")
    if not codebook_path.is_file():
        raise SystemExit(f"Codebook does not exist: {codebook_path}")
    if source == output or output.is_relative_to(source) or source.is_relative_to(output):
        raise SystemExit("Keep source and output directories separate and non-nested.")
    if args.limit < 0:
        raise SystemExit("--limit must be nonnegative")
    if args.max_evidence_per_pair < 1:
        raise SystemExit("--max-evidence-per-pair must be at least 1")
    output.mkdir(parents=True, exist_ok=True)

    codebook = json.loads(codebook_path.read_text(encoding="utf-8"))
    technologies = compile_entries(codebook["technologies"])
    mechanisms = compile_entries(codebook["mechanisms"])
    outcomes = compile_entries(codebook["outcomes"])
    designs = compile_entries(codebook["study_designs"])
    effect_patterns = {key: compile_patterns(values) for key, values in codebook["effect_cues"].items()}
    evidence_patterns = {key: compile_patterns(values) for key, values in codebook["evidence_cues"].items()}
    causal_patterns = compile_patterns(codebook["causal_connectors"])

    discovered = discover_canonical_files(source)
    if not discovered:
        raise SystemExit("No canonical documents found; check the corpus folder layout.")
    ids = [path.stem for path, _domain in discovered]
    if len(ids) != len(set(ids)):
        raise SystemExit("Paper IDs (filename stems) must be unique across the corpus.")
    if args.limit:
        discovered = discovered[: args.limit]
    print(f"Canonical files selected: {len(discovered)}", flush=True)

    papers: list[dict[str, Any]] = []
    paper_sentences: dict[str, list[SentenceRecord]] = {}
    errors: list[dict[str, str]] = []
    full_texts: dict[str, str] = {}
    doi_owner: dict[str, str] = {}
    title_owner: dict[str, str] = {}
    text_owner: dict[str, str] = {}

    for position, (path, source_domain) in enumerate(discovered, 1):
        paper_id = path.stem
        try:
            paragraphs, metadata = read_document(path)
            full_text = "\n".join(paragraphs)
            title = infer_title(paragraphs, metadata)
            doi = infer_doi(full_text)
            year = infer_year(full_text)
            text_hash = sha256_text(normalized_text(full_text)) if full_text else ""
            title_key = normalized_text(title)
            duplicate_of = ""
            duplicate_basis = ""
            if text_hash and text_hash in text_owner:
                duplicate_of = text_owner[text_hash]
                duplicate_basis = "Exact normalized full text"
            elif doi and doi in doi_owner:
                duplicate_of = doi_owner[doi]
                duplicate_basis = "DOI"
            elif reliable_title_for_deduplication(title) and title_key and title_key in title_owner:
                duplicate_of = title_owner[title_key]
                duplicate_basis = "Normalized title"
            canonical_paper_id = duplicate_of or paper_id
            if not duplicate_of:
                if text_hash:
                    text_owner[text_hash] = paper_id
                if doi:
                    doi_owner[doi] = paper_id
                if reliable_title_for_deduplication(title) and title_key:
                    title_owner[title_key] = paper_id
            design_code, design_label, design_cue = infer_study_design(full_text, designs)
            sentences = split_sentences(paragraphs)
            record = {
                "paper_id": paper_id,
                "canonical_paper_id": canonical_paper_id,
                "source_domain": source_domain,
                "title": title,
                "doi": doi,
                "publication_year_candidate": year,
                "study_design_code": design_code,
                "study_design_label": design_label,
                "study_design_cue": design_cue,
                "source_path": str(path),
                "file_type": path.suffix.casefold().lstrip("."),
                "word_count": len(re.findall(r"\b\w+[\w'-]*\b", full_text)),
                "sentence_count": len(sentences),
                "text_sha256": text_hash,
                "duplicate_of": duplicate_of,
                "duplicate_basis": duplicate_basis,
                "analysis_eligible": "No" if duplicate_of else "Yes",
                "extraction_status": "OK",
                "relation_count": 0,
                "cross_sdg_relation_count": 0,
                "human_screen_status": "Pending",
                "human_include": "",
                "human_exclusion_reason": "",
                "human_coder": "",
                "human_notes": "",
            }
            papers.append(record)
            full_texts[paper_id] = full_text
            paper_sentences[paper_id] = sentences
        except Exception as exc:
            error = {"paper_id": paper_id, "source_path": str(path), "error": repr(exc)}
            errors.append(error)
            papers.append(
                {
                    "paper_id": paper_id,
                    "canonical_paper_id": paper_id,
                    "source_domain": source_domain,
                    "title": "",
                    "doi": "",
                    "publication_year_candidate": "",
                    "study_design_code": "",
                    "study_design_label": "",
                    "study_design_cue": "",
                    "source_path": str(path),
                    "file_type": path.suffix.casefold().lstrip("."),
                    "word_count": 0,
                    "sentence_count": 0,
                    "text_sha256": "",
                    "duplicate_of": "",
                    "duplicate_basis": "",
                    "analysis_eligible": "No",
                    "extraction_status": repr(exc),
                    "relation_count": 0,
                    "cross_sdg_relation_count": 0,
                    "human_screen_status": "Pending",
                    "human_include": "",
                    "human_exclusion_reason": "",
                    "human_coder": "",
                    "human_notes": "",
                }
            )
        if position % 50 == 0 or position == len(discovered):
            print(f"Read {position}/{len(discovered)} files", flush=True)

    all_relations: list[dict[str, Any]] = []
    for position, paper in enumerate(papers, 1):
        if paper["analysis_eligible"] != "Yes" or paper["extraction_status"] != "OK":
            continue
        relations = extract_relations_for_paper(
            paper,
            paper_sentences[paper["paper_id"]],
            technologies,
            mechanisms,
            outcomes,
            paper["study_design_code"],
            paper["study_design_label"],
            evidence_patterns,
            effect_patterns,
            causal_patterns,
        )
        all_relations.extend(relations)
        if position % 50 == 0 or position == len(papers):
            print(f"Coded {position}/{len(papers)} paper records", flush=True)

    relations = retain_best_evidence(all_relations, args.max_evidence_per_pair)
    cross_relations = build_cross_sdg_relations(relations, causal_patterns)
    relation_counts = Counter(row["paper_id"] for row in relations)
    cross_counts = Counter(row["paper_id"] for row in cross_relations)
    for paper in papers:
        paper["relation_count"] = relation_counts[paper["paper_id"]]
        paper["cross_sdg_relation_count"] = cross_counts[paper["paper_id"]]

    top_technologies, top_outcomes, matrix_rows = build_matrices(relations)
    codebook_rows = flatten_codebook(codebook)
    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    summary = {
        "generated_at_utc": generated_at,
        "source_root": str(source),
        "codebook_version": codebook["metadata"]["version"],
        "files_selected": len(discovered),
        "files_parsed": sum(row["extraction_status"] == "OK" for row in papers),
        "parse_errors": len(errors),
        "duplicate_records": sum(bool(row["duplicate_of"]) for row in papers),
        "analysis_eligible_papers": sum(row["analysis_eligible"] == "Yes" for row in papers),
        "papers_with_relations": len({row["paper_id"] for row in relations}),
        "technology_outcome_relations": len(relations),
        "high_confidence_relations": sum(float(row["confidence_score"]) >= 0.75 for row in relations),
        "cross_sdg_candidates": len(cross_relations),
        "same_span_cross_sdg_candidates": sum(row["relation_basis"] == "Same sentence/span" for row in cross_relations),
        "method": codebook["metadata"]["method"],
    }

    write_csv(output / "master_papers.csv", papers)
    write_csv(output / "master_technology_outcome_relations.csv", relations)
    write_csv(output / "cross_sdg_relations.csv", cross_relations)
    write_csv(output / "technology_outcome_matrix.csv", matrix_rows)
    write_csv(output / "codebook_flat.csv", codebook_rows)
    write_csv(output / "parse_errors.csv", errors, ["paper_id", "source_path", "error"])
    (output / "quality_report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    pipeline_hash = sha256_text(Path(__file__).read_text(encoding="utf-8"))
    codebook_hash = hashlib.sha256(codebook_path.read_bytes()).hexdigest()
    manifest = {
        **summary,
        "pipeline_sha256": pipeline_hash,
        "codebook_sha256": codebook_hash,
        "arguments": {
            "limit": args.limit,
            "max_evidence_per_pair": args.max_evidence_per_pair,
            "skip_workbook": args.skip_workbook,
        },
        "outputs": [
            "master_papers.csv",
            "master_technology_outcome_relations.csv",
            "cross_sdg_relations.csv",
            "technology_outcome_matrix.csv",
            "codebook_flat.csv",
            "quality_report.json",
        ],
    }
    (output / "pipeline_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    payload = build_payload(summary, papers, relations, cross_relations, codebook_rows, matrix_rows, top_technologies, top_outcomes)
    payload_path = output / "workbook_payload.json"
    payload_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    workbook_ok = True
    workbook_message = "Workbook export skipped."
    if not args.skip_workbook:
        workbook_ok, workbook_message = export_workbook(
            Path(__file__).resolve().parent,
            payload_path,
            output / "DT_SDG3_16_Technology_Outcome_Master.xlsx",
            output / "workbook_previews",
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(workbook_message)
    if not workbook_ok and args.fail_on_workbook_error:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
