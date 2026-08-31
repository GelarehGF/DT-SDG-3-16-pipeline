"""Run all five real stages on 18 invented papers, without the private corpus.

Run from the repository root:
    python tests/smoke_workflow.py --work-dir outputs/synthetic-smoke
This is a software integration check, not a research dataset or human validation.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys
from xml.etree import ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[1]
W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def write_docx(path: Path, paragraphs: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    document = ET.Element(f"{{{W_NS}}}document")
    body = ET.SubElement(document, f"{{{W_NS}}}body")
    for value in paragraphs:
        paragraph = ET.SubElement(body, f"{{{W_NS}}}p")
        run = ET.SubElement(paragraph, f"{{{W_NS}}}r")
        ET.SubElement(run, f"{{{W_NS}}}t").text = value
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("word/document.xml", ET.tostring(document))


def create_corpus(source: Path) -> None:
    examples = {
        "SDG3": [
            "Telemedicine improves healthcare access through data sharing and interoperability.",
            "Mobile health improves patient engagement and health equity through digital literacy.",
            "Artificial intelligence improves quality of care and patient safety through decision support.",
            "Health services require remote consultations, clinical care and patient communication.",
        ],
        "SDG16": [
            "Blockchain improves transparency and accountability through auditability.",
            "E-government improves public trust and citizen participation through digital literacy.",
            "Artificial intelligence improves institutional capacity and privacy through data governance.",
            "Public administration requires institutional coordination, transparent reporting and oversight.",
        ],
        "Mixed": [
            "Telemedicine improves healthcare access and public trust through data sharing.",
            "Blockchain improves patient safety and public accountability through interoperability.",
            "Artificial intelligence improves health equity and government transparency through digital literacy.",
            "Health governance requires consent, coordination, privacy and public-service access.",
        ],
    }
    tokens = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]
    for domain, sentences in examples.items():
        folder = source / ("mixed" if domain == "Mixed" else f"{domain}/Words")
        prefix = "SDGM" if domain == "Mixed" else domain
        for number, token in enumerate(tokens, 1):
            paragraphs = [f"Synthetic {domain} {token} study", "Abstract", *sentences,
                          "Discussion", f"Invented example {token} has a distinct context and is not empirical evidence.",
                          "References", "No real publications are used in this synthetic test."]
            write_docx(folder / f"{prefix}-{number:02d}.docx", paragraphs)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()
    work = args.work_dir.expanduser().resolve()
    if work.exists() and (not work.is_dir() or any(work.iterdir())):
        parser.error("Use a new or empty work directory; existing files are preserved.")
    source, output = work / "source", work / "run"
    create_corpus(source)
    result = subprocess.run([sys.executable, str(ROOT / "main.py"), "--source", str(source),
                             "--output", str(output)], check=False)
    if result.returncode:
        return result.returncode
    manifest = json.loads((output / "run_manifest.json").read_text())
    assert manifest["status"] == "complete" and len(manifest["stages"]) == 5
    assert all(stage["status"] == "complete" for stage in manifest["stages"])
    assert len(list((output / "manuscript_figures").glob("*.png"))) == 10
    assert (output / "master/DT_SDG3_16_Technology_Outcome_Master.xlsx").is_file()
    print("Synthetic end-to-end check passed: all five stages, XLSX, ten manuscript PNGs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
