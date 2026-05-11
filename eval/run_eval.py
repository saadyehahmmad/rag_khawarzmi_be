"""
Offline eval loop for the LangGraph RAG pipeline (no HTTP server).

What this script does:
- Loads golden JSON cases (question + cheap string checks + optional routing/citation bars).
- Invokes the compiled graph once per case (same code path as /chat/sync pre-HTTP).
- Computes per-case latency, keyword-MRR score, routing accuracy, and citation metrics.
- Writes a JSON report under eval/reports/ for trend tracking in CI or local iteration.

Prerequisites:
- ANTHROPIC_API_KEY set, vector stores ingested, working .env as for production.

Usage (from repository root):
  python eval/run_eval.py
  python eval/run_eval.py --golden eval/golden.json --out eval/reports/run.json
  python eval/run_eval.py --fail-threshold 0.80   # exit 1 when pass_rate < 80%

Metrics added beyond v1:
  - duration_s:     wall-clock seconds for graph.invoke per case
  - keyword_mrr:    mean reciprocal rank for keyword presence (order of keyword hit in answer)
  - retrieval_dist: top-1 dense L2 distance (lower = more relevant)
  - routing_acc:    fraction of non-off-topic cases routed to the declared system
  - fallback_acc:   fraction of off-topic cases correctly returning irrelevant
  - avg_latency_s:  mean per-case duration
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Repository root on sys.path so `agent` imports resolve when run as a file.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv

load_dotenv(_ROOT / ".env")

from agent.graph import build_graph


def _load_golden(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "cases" not in data:
        raise ValueError("golden file must be a JSON object with a 'cases' array")
    cases = data["cases"]
    if not isinstance(cases, list):
        raise ValueError("'cases' must be a list")
    return data


def _keyword_mrr(answer: str, keywords: list[str]) -> float:
    """
    Mean Reciprocal Rank proxy: rank keywords by first occurrence position in answer.

    Returns a score in [0, 1].  A keyword found at position 0 scores 1.0; at position
    near the end scores close to 0.  Averaged across all keywords in the list.
    Missing = 0 contribution.  Empty keyword list = 1.0 (no constraint).
    """
    if not keywords:
        return 1.0
    ans_low = answer.lower()
    total_len = max(len(ans_low), 1)
    scores: list[float] = []
    for kw in keywords:
        idx = ans_low.find(str(kw).lower())
        if idx == -1:
            scores.append(0.0)
        else:
            # Reciprocal of the fractional position + 1 (position 0 → rank 1 → score 1.0)
            rank = (idx / total_len) * len(keywords) + 1
            scores.append(1.0 / rank)
    return round(sum(scores) / len(scores), 4)


def _score_case(case: dict[str, Any], result: dict[str, Any], duration_s: float) -> dict[str, Any]:
    """Heuristic checks + retrieval and latency metrics."""
    ans = str(result.get("answer", "")).lower()
    rel = str(result.get("relevance", ""))
    citations = result.get("retrieved_source_refs") or []
    n_cit = len(citations) if isinstance(citations, list) else 0

    kw_any: list[str] = case.get("keywords_any") or []
    kw_all: list[str] = case.get("keywords_all") or []
    pass_any = True
    if isinstance(kw_any, list) and kw_any:
        pass_any = any(str(k).lower() in ans for k in kw_any)
    pass_all = True
    if isinstance(kw_all, list) and kw_all:
        pass_all = all(str(k).lower() in ans for k in kw_all)

    expect_fb = bool(case.get("expect_fallback"))
    pass_fallback_rule = (rel == "irrelevant") if expect_fb else (rel == "relevant")

    exp_sys = case.get("expect_system")
    pass_system = True
    if isinstance(exp_sys, str) and exp_sys.strip():
        pass_system = str(result.get("system", "")).lower() == exp_sys.strip().lower()

    min_cit = int(case.get("min_citations", 0))
    pass_citations = n_cit >= min_cit

    passed = pass_any and pass_all and pass_fallback_rule and pass_system and pass_citations

    mrr_any = _keyword_mrr(ans, kw_any)
    mrr_all = _keyword_mrr(ans, kw_all)

    return {
        "passed": passed,
        "duration_s": round(duration_s, 3),
        "keyword_mrr_any": mrr_any,
        "keyword_mrr_all": mrr_all,
        "checks": {
            "keywords_any": pass_any,
            "keywords_all": pass_all,
            "fallback_rule": pass_fallback_rule,
            "expect_system": pass_system,
            "min_citations": pass_citations,
        },
        "relevance": rel,
        "system": result.get("system"),
        "retrieval_best_distance": result.get("retrieval_best_distance"),
        "citations_count": n_cit,
        "answer_chars": len(str(result.get("answer", ""))),
    }


def _compute_aggregate_metrics(rows: list[dict[str, Any]], cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute aggregate metrics across all evaluated cases."""
    total = len(rows)
    passed_n = sum(1 for r in rows if r.get("passed"))
    errors = sum(1 for r in rows if "error" in r)

    durations = [r["scores"]["duration_s"] for r in rows if "scores" in r]
    mrr_any_scores = [r["scores"]["keyword_mrr_any"] for r in rows if "scores" in r]
    mrr_all_scores = [r["scores"]["keyword_mrr_all"] for r in rows if "scores" in r]
    distances = [
        r["scores"]["retrieval_best_distance"]
        for r in rows
        if "scores" in r and r["scores"].get("retrieval_best_distance") is not None
    ]

    # Routing accuracy: cases with expect_system set (excluding "none").
    routing_cases = [
        (r, c)
        for r, c in zip(rows, cases)
        if "scores" in r and isinstance(c.get("expect_system"), str) and c["expect_system"] not in ("", "none")
    ]
    routing_acc = (
        sum(1 for r, _ in routing_cases if r["scores"]["checks"]["expect_system"]) / len(routing_cases)
        if routing_cases
        else None
    )

    # Fallback accuracy: cases with expect_fallback=True.
    fallback_cases = [
        r for r, c in zip(rows, cases) if "scores" in r and bool(c.get("expect_fallback"))
    ]
    fallback_acc = (
        sum(1 for r in fallback_cases if r["scores"]["checks"]["fallback_rule"]) / len(fallback_cases)
        if fallback_cases
        else None
    )

    def _mean(lst: list[float]) -> float | None:
        return round(sum(lst) / len(lst), 4) if lst else None

    return {
        "pass_rate": round(passed_n / total, 4) if total else 0.0,
        "passed": passed_n,
        "failed": total - passed_n - errors,
        "errors": errors,
        "total": total,
        "avg_duration_s": _mean(durations),
        "p95_duration_s": round(sorted(durations)[int(len(durations) * 0.95)], 3) if len(durations) > 1 else _mean(durations),
        "avg_keyword_mrr_any": _mean(mrr_any_scores),
        "avg_keyword_mrr_all": _mean(mrr_all_scores),
        "avg_retrieval_distance": _mean(distances),
        "routing_accuracy": round(routing_acc, 4) if routing_acc is not None else None,
        "fallback_accuracy": round(fallback_acc, 4) if fallback_acc is not None else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Run golden-set eval against the LangGraph RAG app.")
    ap.add_argument("--golden", type=Path, default=_ROOT / "eval" / "golden.json")
    ap.add_argument("--out", type=Path, default=_ROOT / "eval" / "reports" / "latest.json")
    ap.add_argument(
        "--fail-threshold",
        type=float,
        default=0.0,
        metavar="RATE",
        help="Exit 1 when pass_rate < RATE (e.g. 0.80 for 80%%). Default: always pass.",
    )
    args = ap.parse_args()

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set; eval would fail on LLM nodes.", file=sys.stderr)
        return 2

    golden_path: Path = args.golden
    if not golden_path.is_file():
        print(f"Golden file not found: {golden_path}", file=sys.stderr)
        return 2

    data = _load_golden(golden_path)
    cases: list[dict[str, Any]] = [c for c in data["cases"] if isinstance(c, dict)]

    print(f"Running eval on {len(cases)} golden cases…", flush=True)
    graph = build_graph()
    rows: list[dict[str, Any]] = []

    for i, case in enumerate(cases, start=1):
        cid = str(case.get("id", "unknown"))
        q = str(case.get("question", "")).strip()
        print(f"  [{i:02d}/{len(cases)}] {cid} … ", end="", flush=True)
        if not q:
            rows.append({"id": cid, "error": "empty_question", "passed": False})
            print("SKIP (empty)")
            continue
        initial = {
            "question": q,
            "language": "en",
            "rewritten_question": "",
            "system": None,
            "retrieved_chunks": [],
            "retrieved_source_refs": [],
            "relevance": "unknown",
            "answer": "",
            "thread_id": str(uuid.uuid4()),
            "conversation_history": [],
        }
        t0 = time.perf_counter()
        try:
            result = graph.invoke(initial)
        except Exception as exc:  # noqa: BLE001
            duration_s = time.perf_counter() - t0
            rows.append({"id": cid, "passed": False, "error": str(exc), "duration_s": round(duration_s, 3)})
            print(f"ERROR ({exc})")
            continue
        duration_s = time.perf_counter() - t0
        detail = _score_case(case, result, duration_s)
        ok = bool(detail.pop("passed"))
        rows.append({
            "id": cid,
            "passed": ok,
            "scores": detail,
            "answer_preview": str(result.get("answer", ""))[:240],
        })
        status = "PASS" if ok else "FAIL"
        checks = detail.get("checks", {})
        failed_checks = [k for k, v in checks.items() if not v]
        suffix = f" (failed: {', '.join(failed_checks)})" if failed_checks else ""
        print(f"{status} [{duration_s:.1f}s]{suffix}")

    summary = _compute_aggregate_metrics(rows, cases)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "golden": str(golden_path),
        "summary": summary,
        "rows": rows,
    }

    out_path: Path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nReport written → {out_path}")

    pass_rate = summary["pass_rate"]
    threshold = args.fail_threshold
    if threshold > 0 and pass_rate < threshold:
        print(f"\nFAIL: pass_rate {pass_rate:.0%} < threshold {threshold:.0%}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
