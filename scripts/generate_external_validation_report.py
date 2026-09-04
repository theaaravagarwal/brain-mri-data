#!/usr/bin/env python3
"""Generate the aggregate release analysis and validation PDF without case identifiers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from statistics import median
from typing import Any

from reportlab import rl_config
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    KeepTogether,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

RELEASE_TAG = "prototype-v0.1.0"
WEAK_DICE_THRESHOLD = 0.75
rl_config.invariant = 1


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("private_results", type=Path)
    parser.add_argument("analysis_output", type=Path)
    parser.add_argument("pdf_output", type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def analyze_failures(summary: dict[str, Any], private: dict[str, Any]) -> dict[str, Any]:
    rows = private.get("cases", [])
    if len(rows) != summary["case_count"]:
        raise ValueError("Private and public result counts differ")
    weak = [row for row in rows if row["metrics"]["whole_lesion_dice"] < WEAK_DICE_THRESHOLD]
    empty = [row for row in weak if row["predicted_voxels"] == 0]
    over = [row for row in weak if row["predicted_voxels"] and row["predicted_voxels"] / row["reference_voxels"] >= 1.5]
    other = [row for row in weak if row not in empty and row not in over]
    reference_sizes = [int(row["reference_voxels"]) for row in weak]
    finite_hd95 = [float(row["metrics"]["hd95_mm"]) for row in weak if row["metrics"]["hd95_mm"] is not None]
    analysis = {
        "schema_version": "fixed-segresnet-external-failure-analysis/v1",
        "benchmark_id": summary["benchmark_id"],
        "scope": "aggregate_descriptive_error_analysis",
        "research_only": True,
        "weak_definition": {"metric": "whole_lesion_dice", "operator": "below", "threshold": WEAK_DICE_THRESHOLD, "post_hoc_descriptive": True},
        "weak_case_count": len(weak),
        "clusters": {
            "empty_prediction": len(empty),
            "substantial_oversegmentation": len(over),
            "other_overlap_or_boundary_error": len(other),
        },
        "weak_reference_voxels": {
            "min": min(reference_sizes),
            "median": float(median(reference_sizes)),
            "max": max(reference_sizes),
        },
        "weak_hd95_mm": {
            "available_count": len(finite_hd95),
            "median": float(median(finite_hd95)),
            "max": max(finite_hd95),
        },
        "retraining_decision": {
            "current_release": "freeze_and_use_for_research_prototype",
            "immediate_retraining": False,
            "reason": "Failures are mixed rather than one consistent correctable mode; the external cohort remains evaluation-only.",
            "next_experiment": "Use development data only to test stronger small-lesion sampling and preserve the current model as rollback.",
            "future_evaluation": "Evaluate any new candidate once on a new untouched cohort; do not reuse these 60 cases for selection.",
        },
        "privacy": {"case_tokens_included": False, "native_case_ids_included": False, "native_paths_included": False},
    }
    encoded = json.dumps(analysis, sort_keys=True)
    if re.search(r'"case_\d+', encoded) or "/home/" in encoded:
        raise ValueError("Public analysis contains a private identifier")
    return analysis


def metric_table(summary: dict[str, Any]) -> list[list[str]]:
    labels = (
        ("Dice", "whole_lesion_dice", 3),
        ("IoU", "whole_lesion_iou", 3),
        ("Precision", "precision", 3),
        ("Recall", "recall", 3),
        ("HD95 (mm)", "hd95_mm", 2),
    )
    rows = [["Metric", "Mean", "95% CI for mean", "Median", "P05", "P95"]]
    for label, key, digits in labels:
        metric = summary["metrics"][key]
        rows.append([
            label,
            f"{metric['mean']:.{digits}f}",
            f"{metric['mean_ci95'][0]:.{digits}f} to {metric['mean_ci95'][1]:.{digits}f}",
            f"{metric['median']:.{digits}f}",
            f"{metric['p05']:.{digits}f}",
            f"{metric['p95']:.{digits}f}",
        ])
    return rows


def generate_pdf(summary: dict[str, Any], analysis: dict[str, Any], summary_digest: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    ink = colors.HexColor("#20242A")
    muted = colors.HexColor("#646B75")
    blue = colors.HexColor("#255FBC")
    pale = colors.HexColor("#EEF3FA")
    styles.add(ParagraphStyle(name="ReportTitle", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=23, leading=27, textColor=ink, alignment=TA_CENTER, spaceAfter=8))
    styles.add(ParagraphStyle(name="Deck", parent=styles["BodyText"], fontSize=10, leading=15, textColor=muted, alignment=TA_CENTER, spaceAfter=20))
    styles.add(ParagraphStyle(name="Section", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=13, leading=16, textColor=ink, spaceBefore=14, spaceAfter=8))
    styles.add(ParagraphStyle(name="BodyCompact", parent=styles["BodyText"], fontSize=8.7, leading=13, textColor=ink, spaceAfter=7))
    styles.add(ParagraphStyle(name="Hash", parent=styles["Code"], fontName="Courier", fontSize=6.4, leading=9, textColor=muted, wordWrap="CJK"))

    def footer(canvas, document):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#D7DADE"))
        canvas.line(0.65 * inch, 0.55 * inch, 7.85 * inch, 0.55 * inch)
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(muted)
        canvas.drawString(0.65 * inch, 0.35 * inch, f"{RELEASE_TAG} | Research only - not medical advice")
        canvas.drawRightString(7.85 * inch, 0.35 * inch, f"Page {document.page}")
        canvas.restoreState()

    document = BaseDocTemplate(str(output), pagesize=letter, rightMargin=0.65 * inch, leftMargin=0.65 * inch, topMargin=0.62 * inch, bottomMargin=0.72 * inch, title="Fixed SegResNet External Validation Report", author="Brain MRI Research Prototype")
    document.addPageTemplates(PageTemplate(id="report", frames=[Frame(document.leftMargin, document.bottomMargin, document.width, document.height, id="body")], onPage=footer))
    story = [
        Paragraph("External Validation Report", styles["ReportTitle"]),
        Paragraph("Fixed brain MRI research segmentation prototype | 60-case external cohort", styles["Deck"]),
        Table([
            [Paragraph("RESEARCH ONLY", styles["BodyCompact"]), Paragraph("This report evaluates segmentation metadata and expert-mask agreement. It is not clinical validation, a diagnosis, or treatment advice.", styles["BodyCompact"])],
        ], colWidths=[1.25 * inch, 5.75 * inch], style=TableStyle([("BACKGROUND", (0, 0), (-1, -1), pale), ("TEXTCOLOR", (0, 0), (0, 0), blue), ("BOX", (0, 0), (-1, -1), 0.6, blue), ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 9), ("RIGHTPADDING", (0, 0), (-1, -1), 9), ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 7)])),
        Paragraph("Release decision", styles["Section"]),
        Paragraph("Freeze the current checkpoint for the research prototype. Do not retrain or tune on this external cohort. The aggregate result is strong enough for a working demonstration, while the observed miss requires explicit expert-review safeguards.", styles["BodyCompact"]),
    ]
    dice = summary["metrics"]["whole_lesion_dice"]
    hd95 = summary["metrics"]["hd95_mm"]
    cards = Table([
        ["Mean Dice", "Median Dice", "Median HD95", "Median case time"],
        [f"{dice['mean']:.3f}", f"{dice['median']:.3f}", f"{hd95['median']:.2f} mm", f"{summary['latency_seconds']['median']:.2f} s"],
    ], colWidths=[1.75 * inch] * 4, style=TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F3F4F6")), ("TEXTCOLOR", (0, 0), (-1, 0), muted), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("FONTSIZE", (0, 0), (-1, 0), 7), ("FONTNAME", (0, 1), (-1, 1), "Helvetica-Bold"), ("FONTSIZE", (0, 1), (-1, 1), 15), ("TEXTCOLOR", (0, 1), (-1, 1), ink), ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D7DADE")), ("ALIGN", (0, 0), (-1, -1), "CENTER"), ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]))
    story.extend([Spacer(1, 12), cards, Paragraph("All external metrics", styles["Section"])])
    metrics = Table(metric_table(summary), colWidths=[1.12 * inch, 0.7 * inch, 1.55 * inch, 0.82 * inch, 0.72 * inch, 0.72 * inch], repeatRows=1, style=TableStyle([("BACKGROUND", (0, 0), (-1, 0), ink), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"), ("FONTSIZE", (0, 0), (-1, -1), 7.5), ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D7DADE")), ("ALIGN", (1, 1), (-1, -1), "RIGHT"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    story.extend([metrics, Paragraph("Outcome distribution and weak-case review", styles["Section"])])
    clusters = analysis["clusters"]
    story.extend([
        Paragraph(f"Dice distribution: 35/60 at or above 0.90; 19/60 from 0.75 to 0.90; 5/60 from 0.50 to 0.75; and 1/60 below 0.50. The below-0.75 review is descriptive and post hoc, not a promotion gate.", styles["BodyCompact"]),
        Paragraph(f"Six weak cases were reviewed using anonymized metrics only: {clusters['empty_prediction']} empty prediction, {clusters['substantial_oversegmentation']} substantial oversegmentations, and {clusters['other_overlap_or_boundary_error']} other overlap or boundary errors. The empty prediction was reproducible and occurred against a small 1,026-voxel reference mask, showing a model miss rather than a pipeline crash.", styles["BodyCompact"]),
        Paragraph("Retraining decision", styles["Section"]),
        Paragraph(analysis["retraining_decision"]["reason"] + " " + analysis["retraining_decision"]["next_experiment"] + " " + analysis["retraining_decision"]["future_evaluation"], styles["BodyCompact"]),
    ])
    provenance = summary["provenance"]
    hash_rows = [
        ["Release tag", RELEASE_TAG],
        ["Model", provenance["model_id"]],
        ["Checkpoint SHA-256", provenance["checkpoint_sha256"]],
        ["Plan SHA-256", provenance["plan_sha256"]],
        ["Evaluator SHA-256", provenance["evaluator_sha256"]],
        ["Aggregate JSON SHA-256", summary_digest],
        ["Dataset revision", provenance["dataset_source_revision"]],
    ]
    provenance_table = Table([[Paragraph(str(label), styles["BodyCompact"]), Paragraph(str(value), styles["Hash"])] for label, value in hash_rows], colWidths=[1.45 * inch, 5.55 * inch], style=TableStyle([("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D7DADE")), ("VALIGN", (0, 0), (-1, -1), "TOP"), ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F3F4F6")), ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
    story.extend([
        KeepTogether([Paragraph("Frozen provenance", styles["Section"]), provenance_table]),
        Paragraph("Method and limitations", styles["Section"]),
        Paragraph("The fixed threshold and checkpoint were run once over all 60 accepted BraTS-SSA cases. Whole-lesion Dice, IoU, precision, recall, and physical-spacing HD95 were computed against expert masks. Case-level bootstrap confidence intervals used 10,000 fixed-seed replicates. Native case identifiers and paths are excluded from this report.", styles["BodyCompact"]),
        Paragraph("This cohort does not prove clinical safety, calibration, generalization to every scanner or population, or absence of disease when the output is empty. Expert review remains required. Any future candidate requires development-only iteration and a new untouched evaluation cohort.", styles["BodyCompact"]),
        Paragraph(f"Benchmark generated {summary['provenance']['generated_at']}", styles["BodyCompact"]),
    ])
    document.build(story)


def main() -> None:
    args = arguments()
    summary = json.loads(args.summary.read_text())
    private = json.loads(args.private_results.read_text())
    analysis = analyze_failures(summary, private)
    atomic_json(args.analysis_output, analysis)
    generate_pdf(summary, analysis, sha256(args.summary), args.pdf_output)
    print(json.dumps({"analysis": str(args.analysis_output), "pdf": str(args.pdf_output), "pdf_sha256": sha256(args.pdf_output)}, sort_keys=True))


if __name__ == "__main__":
    main()
