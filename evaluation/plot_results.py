"""Create minimal paired reranking outputs from results.json."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "data" / "evaluation" / "results"


def number(value: Any) -> float | None:
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def chunk_summary(record: dict[str, Any]) -> str:
    chunks = record.get("retrieved_chunks") or []
    items = []
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        rank = chunk.get("rank", "?")
        chunk_id = chunk.get("chunk_id") or "unknown-id"
        title = chunk.get("title") or chunk.get("url") or chunk.get("source_id") or ""
        items.append(f"{rank}. {chunk_id}" + (f" — {title}" if title else ""))
    return " | ".join(items)


def load_records(results_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = results_dir / "results.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run the evaluation script first.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("results", [])
    if not records:
        raise ValueError(f"No result records found in {path}")
    required = {"case_id", "reranking", "question", "answer"}
    for record in records:
        missing = required - record.keys()
        if missing:
            raise ValueError(f"Record missing required fields: {sorted(missing)}")
    return payload, records


def paired_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_case: dict[str, dict[str, dict[str, Any]]] = {}
    for record in records:
        by_case.setdefault(str(record["case_id"]), {})[str(record["reranking"])] = record
    pairs = []
    for case_id, group in by_case.items():
        off, on = group.get("off"), group.get("on")
        base = off or on
        if base is None:
            continue
        off_latency = number(off.get("latency_ms")) if off else None
        on_latency = number(on.get("latency_ms")) if on else None
        pairs.append({
            "case_id": case_id,
            "question": base.get("question", ""),
            "reference_answer": base.get("reference_answer", ""),
            "answerable_expected": base.get("answerable_expected", ""),
            "success_off": off.get("success", False) if off else False,
            "success_on": on.get("success", False) if on else False,
            "error_off": off.get("error", "") if off else "configuration not run",
            "error_on": on.get("error", "") if on else "configuration not run",
            "answer_off": off.get("answer", "") if off else "",
            "answer_on": on.get("answer", "") if on else "",
            "latency_off_ms": off_latency if off else None,
            "latency_on_ms": on_latency if on else None,
            "latency_delta_ms": round(on_latency - off_latency, 2)
                if off_latency is not None and on_latency is not None else None,
            "evidence_off": chunk_summary(off) if off else "",
            "evidence_on": chunk_summary(on) if on else "",
            "quality_winner": "",
            "review_notes": "",
        })
    return pairs


def write_review_csv(path: Path, pairs: list[dict[str, Any]]) -> None:
    fields = [
        "case_id", "question", "reference_answer", "answerable_expected",
        "success_off", "success_on", "error_off", "error_on",
        "answer_off", "answer_on", "evidence_off", "evidence_on",
        "latency_off_ms", "latency_on_ms", "latency_delta_ms",
        "quality_winner", "review_notes",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(pairs)


def make_latency_chart(path: Path, pairs: list[dict[str, Any]]) -> int:
    # Only successful answers are eligible for latency comparisons.
    valid = [
        p for p in pairs
        if p["success_off"] and p["success_on"]
        and number(p["latency_off_ms"]) is not None
        and number(p["latency_on_ms"]) is not None
    ]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if not valid:
        ax.text(0.5, 0.5, "No successful paired answers to compare",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    else:
        for pair in valid:
            y1, y2 = pair["latency_off_ms"], pair["latency_on_ms"]
            ax.plot([0, 1], [y1, y2], marker="o", linewidth=1.8)
            ax.annotate(pair["case_id"], (1, y2), xytext=(7, 0),
                        textcoords="offset points", va="center", fontsize=9)
        ax.set_xticks([0, 1], ["Reranking off", "Reranking on"])
        ax.set_ylabel("End-to-end latency (ms)")
        ax.set_title("Successful paired-answer latency")
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return len(valid)


def write_findings(path: Path, payload: dict[str, Any], pairs: list[dict[str, Any]], paired_latency_count: int) -> None:
    successful_off = sum(bool(p["success_off"]) for p in pairs)
    successful_on = sum(bool(p["success_on"]) for p in pairs)
    both_success = [p for p in pairs if p["success_off"] and p["success_on"]]
    lines = [
        "# DocuPilot reranking evaluation",
        "",
        "## Run summary",
        "",
        f"- Questions represented: {len(pairs)}",
        f"- Successful answers, reranking off: {successful_off}/{len(pairs)}",
        f"- Successful answers, reranking on: {successful_on}/{len(pairs)}",
        f"- Valid paired latency comparisons: {paired_latency_count}",
        f"- Evaluation created (UTC): {payload.get('metadata', {}).get('created_at_utc', 'unknown')}",
        "",
        "## What this run can tell us",
        "",
        "- Compare each paired answer with the reference and the retrieved evidence in `manual_review.csv`.",
        "- Use `quality_winner` for a single manual verdict per question: `off`, `on`, `tie`, or `unclear`.",
        "- Latency chart includes only questions with successful answers in both configurations.",
        "- An error or missing answer is not a quality score and is excluded from paired-answer latency.",
        "",
        "## Limits",
        "",
        "- This is a five-question pilot, not statistical proof that reranking always improves quality.",
        "- Retrieved chunk IDs/previews show what was passed through the retrieval pipeline; they do not by themselves prove relevance.",
        "- Do not claim reranking improves quality until the answer/evidence pairs have been manually reviewed.",
        "",
        "## Paired cases",
        "",
    ]
    for p in pairs:
        lines.append(f"- **{p['case_id']}** — off success: {p['success_off']}; on success: {p['success_on']}; "
                     f"latency change (on − off): {p['latency_delta_ms']} ms.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS))
    args = parser.parse_args()
    results_dir = Path(args.results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    visuals = results_dir / "visuals"
    visuals.mkdir(parents=True, exist_ok=True)

    payload, records = load_records(results_dir)
    pairs = paired_records(records)
    review_path = results_dir / "manual_review.csv"
    write_review_csv(review_path, pairs)
    paired_latency_count = make_latency_chart(visuals / "latency_by_question.png", pairs)
    write_findings(results_dir / "findings.md", payload, pairs, paired_latency_count)

    print(f"Review table: {review_path}")
    print(f"Latency chart: {visuals / 'latency_by_question.png'}")
    print(f"Findings: {results_dir / 'findings.md'}")


if __name__ == "__main__":
    main()
