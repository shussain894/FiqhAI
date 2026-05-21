"""
run_eval.py

Runs the approved evaluation questions through the full RAG pipeline
and produces a scored report.

Metrics:
  source_hit        — pipeline retrieved at least one chunk from the correct book
  parse_success     — markdown answer parsed into a valid FiqhAnswer
  short_answer_ok   — short_answer field is filled (not empty / not template text)
  confidence        — High / Medium / Low as reported by the model
  high_risk_blocked — query was blocked by the safety filter

Run with:
    python -m src.evaluation.run_eval
"""

import json
import time
from collections import Counter, defaultdict
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

EVAL_FILE = Path("data/eval/candidate_questions.jsonl")
RESULTS_FILE = Path("data/eval/eval_results.jsonl")

_TEMPLATE_PHRASES = [
    "one or two sentence",
    "direct answer",
    "a one or two",
]


def load_approved(eval_file: Path) -> list[dict]:
    questions = []
    for line in eval_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            q = json.loads(line)
            if q.get("approved", True):
                questions.append(q)
        except json.JSONDecodeError:
            pass
    return questions


def source_hit(result: dict, source_hint: str) -> bool:
    """
    Returns True if the pipeline retrieved at least one chunk from the
    same book named in source_hint (e.g. "Al Hidayah p.76" → "Al Hidayah").
    """
    book = source_hint.split(" p.")[0].strip().lower()
    return any(book in s.get("source_title", "").lower() for s in result.get("sources", []))


def short_answer_ok(structured_answer) -> bool:
    if structured_answer is None:
        return False
    sa = structured_answer.short_answer.strip().lower()
    if not sa:
        return False
    return not any(phrase in sa for phrase in _TEMPLATE_PHRASES)


def score_result(question: dict, result: dict) -> dict:
    sa = result.get("structured_answer")
    sources = result.get("sources", [])
    return {
        "id":               question["id"],
        "topic":            question["topic"],
        "question":         question["question"],
        "source_hint":      question["source_hint"],
        "source_hit":       source_hit(result, question["source_hint"]),
        "parse_success":    sa is not None,
        "short_answer_ok":  short_answer_ok(sa),
        "confidence":       sa.confidence if sa else None,
        "high_risk_blocked": result.get("high_risk", False),
        "rewritten_query":  result.get("rewritten_query"),
        "topic_detected":   result.get("topic_detected"),
        "latency_s":        result.get("latency_s"),
        "n_sources":        len(sources),
        # Full answer fields for frontend display
        "short_answer":     sa.short_answer if sa else None,
        "ruling":           sa.ruling if sa else None,
        "conditions":       sa.conditions if sa else None,
        "explanation":      sa.explanation if sa else None,
        "citations":        sa.citations if sa else [],
        "note":             sa.note if sa else None,
        "sources":          [
            {
                "source_title": s.get("source_title"),
                "page":         s.get("page"),
                "topic":        s.get("topic"),
                "score":        s.get("score"),
                "rerank_score": s.get("rerank_score"),
            }
            for s in sources
        ],
    }


def print_report(scores: list[dict]):
    total = len(scores)
    if total == 0:
        print("No results to report.")
        return

    not_blocked = [s for s in scores if not s["high_risk_blocked"]]
    n = len(not_blocked)

    source_hits    = sum(1 for s in not_blocked if s["source_hit"])
    parse_ok       = sum(1 for s in not_blocked if s["parse_success"])
    sa_ok          = sum(1 for s in not_blocked if s["short_answer_ok"])
    blocked        = total - n
    conf_counts    = Counter(s["confidence"] for s in not_blocked if s["confidence"])
    avg_latency    = sum(s["latency_s"] for s in scores if s["latency_s"]) / total

    print("\n" + "=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)
    print(f"  Questions run:       {total}")
    print(f"  High-risk blocked:   {blocked}")
    print(f"  Answered:            {n}")
    print()
    print(f"  Source hit rate:     {source_hits}/{n}  ({100*source_hits//n if n else 0}%)")
    print(f"  Parse success:       {parse_ok}/{n}  ({100*parse_ok//n if n else 0}%)")
    print(f"  Short answer filled: {sa_ok}/{n}  ({100*sa_ok//n if n else 0}%)")
    print(f"  Avg latency:         {avg_latency:.1f}s")
    print()
    print("  Confidence breakdown:")
    for level in ("High", "Medium", "Low"):
        count = conf_counts.get(level, 0)
        print(f"    {level:<8} {count}  ({100*count//n if n else 0}%)")

    print()
    print("  Source hit rate by topic:")
    by_topic: dict[str, list] = defaultdict(list)
    for s in not_blocked:
        by_topic[s["topic"]].append(s)
    for topic in ["Taharah", "Salah", "Sawm", "Zakah", "Usul"]:
        qs = by_topic.get(topic, [])
        if not qs:
            continue
        hits = sum(1 for s in qs if s["source_hit"])
        print(f"    {topic:<12} {hits}/{len(qs)}  ({100*hits//len(qs)}%)")

    print("=" * 60)


def run():
    print("Loading eval questions...")
    questions = load_approved(EVAL_FILE)
    print(f"Loaded {len(questions)} approved questions.\n")

    print("Loading pipeline...")
    from src.retrieval.retrieve import load_retriever
    from src.retrieval.rerank import load_reranker
    from src.generation.generate import run_rag_query

    model, collection = load_retriever()
    reranker = load_reranker()
    print(f"Ready. Index contains {collection.count()} chunks.\n")

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    scores = []

    with open(RESULTS_FILE, "w", encoding="utf-8") as out:
        for question in tqdm(questions, desc="Evaluating"):
            t0 = time.time()
            result = run_rag_query(
                question["question"],
                model,
                collection,
                reranker=reranker,
            )
            result["latency_s"] = round(time.time() - t0, 2)

            scored = score_result(question, result)
            scores.append(scored)
            out.write(json.dumps(scored, ensure_ascii=False) + "\n")

    print_report(scores)
    print(f"\nFull results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    run()
