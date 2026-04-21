"""
eval.py — Sprint 4: Evaluation & Scorecard
==========================================
Mục tiêu Sprint 4 (60 phút):
  - Chạy 10 test questions qua pipeline
  - Chấm điểm theo 4 metrics: Faithfulness, Relevance, Context Recall, Completeness
  - So sánh baseline vs variant
  - Ghi kết quả ra scorecard

Definition of Done Sprint 4:
  ✓ Demo chạy end-to-end (index → retrieve → answer → score)
  ✓ Scorecard trước và sau tuning
  ✓ A/B comparison: baseline vs variant với giải thích vì sao variant tốt hơn

A/B Rule: Chỉ đổi MỘT biến mỗi lần.
"""

import os
import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from rag_answer import rag_answer, call_llm

# =============================================================================
# CẤU HÌNH
# =============================================================================

TEST_QUESTIONS_PATH = Path(__file__).parent / "data" / "test_questions.json"
RESULTS_DIR         = Path(__file__).parent / "results"

BASELINE_CONFIG = {
    "retrieval_mode": "dense",
    "top_k_search":   10,
    "top_k_select":   3,
    "use_rerank":     False,
    "label":          "baseline_dense",
}

VARIANT_CONFIG = {
    "retrieval_mode": "hybrid",
    "top_k_search":   10,
    "top_k_select":   3,
    "use_rerank":     True,
    "label":          "variant_hybrid_rerank",
}


# =============================================================================
# LLM-AS-JUDGE HELPER
# Gọi LLM để chấm điểm, parse JSON an toàn
# =============================================================================

def _llm_judge(prompt: str, fallback_score: int = 3) -> Dict[str, Any]:
    """
    Gọi call_llm() với prompt chấm điểm, parse kết quả JSON.
    Trả về dict có "score" (int 1-5) và "reason" (str).
    Nếu parse lỗi thì fallback về fallback_score với ghi chú.
    """
    try:
        raw   = call_llm(prompt)
        clean = raw.strip().strip("```json").strip("```").strip()
        data  = json.loads(clean)
        score = int(data.get("score", fallback_score))
        score = max(1, min(5, score))   # clamp 1-5
        return {
            "score": score,
            "notes": str(data.get("reason", "")).strip(),
        }
    except Exception as e:
        return {
            "score": fallback_score,
            "notes": f"LLM-as-Judge parse lỗi ({e}) — dùng fallback score={fallback_score}",
        }


# =============================================================================
# SCORING FUNCTIONS
# =============================================================================

def score_faithfulness(
    answer: str,
    chunks_used: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Faithfulness (1-5): Answer có bám đúng retrieved context không?
    Model có tự bịa thêm thông tin ngoài context không?

    5 = mọi thông tin đều có trong chunks
    1 = phần lớn câu trả lời là hallucination
    """
    if not chunks_used:
        return {"score": 1, "notes": "Không có chunks — không thể grounded"}

    context_texts = "\n\n".join(
        f"[{i+1}] {c.get('text', '')[:400]}"
        for i, c in enumerate(chunks_used)
    )

    prompt = f"""You are an evaluation judge. Rate the FAITHFULNESS of the answer below.

FAITHFULNESS means: every claim in the answer is supported by the retrieved context.
A score of 5 means the answer is completely grounded in the context.
A score of 1 means the answer contains significant hallucination not found in the context.

Retrieved context:
{context_texts}

Answer to evaluate:
{answer}

Rate faithfulness on a scale of 1 to 5.
Output ONLY valid JSON with no explanation outside JSON, like:
{{"score": <integer 1-5>, "reason": "<one sentence>"}}"""

    return _llm_judge(prompt)


def score_answer_relevance(
    query: str,
    answer: str,
) -> Dict[str, Any]:
    """
    Answer Relevance (1-5): Answer có trả lời đúng câu hỏi không?
    Model có bị lạc đề hay trả lời đúng trọng tâm không?

    5 = trả lời trực tiếp và đầy đủ câu hỏi
    1 = không trả lời câu hỏi
    """
    prompt = f"""You are an evaluation judge. Rate the RELEVANCE of the answer to the question.

RELEVANCE means: does the answer directly address what was asked?
A score of 5 means the answer perfectly addresses the question.
A score of 1 means the answer is off-topic or does not address the question at all.

Question: {query}

Answer: {answer}

Rate relevance on a scale of 1 to 5.
Output ONLY valid JSON with no explanation outside JSON, like:
{{"score": <integer 1-5>, "reason": "<one sentence>"}}"""

    return _llm_judge(prompt)


def score_context_recall(
    chunks_used: List[Dict[str, Any]],
    expected_sources: List[str],
) -> Dict[str, Any]:
    """
    Context Recall (0-1 → scaled to 1-5): Retriever có mang về đủ evidence không?
    Expected source có nằm trong retrieved chunks không?

    recall = số expected sources được retrieve / tổng expected sources
    """
    if not expected_sources:
        return {"score": None, "recall": None, "notes": "No expected sources defined"}

    retrieved_sources = {
        c.get("metadata", {}).get("source", "")
        for c in chunks_used
    }

    found   = 0
    missing = []
    for expected in expected_sources:
        # Partial match: so sánh tên file không phân biệt hoa thường và extension
        expected_stem = Path(expected).stem.lower()
        matched = any(expected_stem in Path(r).stem.lower() for r in retrieved_sources)
        if matched:
            found += 1
        else:
            missing.append(expected)

    recall     = found / len(expected_sources)
    score_1_5  = max(1, round(recall * 5))   # 0→1, 0.2→1, 0.4→2, 0.6→3, 0.8→4, 1.0→5

    return {
        "score":   score_1_5,
        "recall":  round(recall, 3),
        "found":   found,
        "missing": missing,
        "notes":   (
            f"Retrieved {found}/{len(expected_sources)} expected sources"
            + (f". Missing: {missing}" if missing else " — all found ✓")
        ),
    }


def score_completeness(
    query: str,
    answer: str,
    expected_answer: str,
) -> Dict[str, Any]:
    """
    Completeness (1-5): Answer có thiếu điểm quan trọng so với expected_answer không?

    5 = bao phủ đầy đủ tất cả điểm quan trọng
    1 = thiếu phần lớn nội dung cốt lõi
    """
    if not expected_answer:
        return {"score": None, "notes": "No expected_answer defined"}

    prompt = f"""You are an evaluation judge. Rate the COMPLETENESS of the model answer
compared to the expected answer.

COMPLETENESS means: does the model answer cover all key points in the expected answer?
A score of 5 means all important points are present.
A score of 1 means most key points are missing.

Question: {query}

Expected answer (reference):
{expected_answer}

Model answer to evaluate:
{answer}

Rate completeness on a scale of 1 to 5.
Output ONLY valid JSON with no explanation outside JSON, like:
{{"score": <integer 1-5>, "reason": "<one sentence>", "missing_points": ["<point1>", "<point2>"]}}"""

    result = _llm_judge(prompt)
    # Thử parse missing_points nếu có
    try:
        raw   = call_llm(prompt)
        clean = raw.strip().strip("```json").strip("```").strip()
        data  = json.loads(clean)
        result["missing_points"] = data.get("missing_points", [])
    except Exception:
        result["missing_points"] = []
    return result


# =============================================================================
# SCORECARD RUNNER
# =============================================================================

def run_scorecard(
    config: Dict[str, Any],
    test_questions: Optional[List[Dict]] = None,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Chạy toàn bộ test questions qua pipeline và chấm 4 metrics.

    Returns:
        List scorecard rows — mỗi row là một câu hỏi với đầy đủ scores.
    """
    if test_questions is None:
        with open(TEST_QUESTIONS_PATH, "r", encoding="utf-8") as f:
            test_questions = json.load(f)

    results = []
    label   = config.get("label", "unnamed")

    print(f"\n{'='*70}")
    print(f"Chạy scorecard: {label}")
    print(f"Config: { {k: v for k, v in config.items() if k != 'label'} }")
    print('='*70)

    for q in test_questions:
        question_id      = q["id"]
        query            = q["question"]
        expected_answer  = q.get("expected_answer", "")
        expected_sources = q.get("expected_sources", [])
        category         = q.get("category", "")

        if verbose:
            print(f"\n[{question_id}] {query}")

        # --- Gọi pipeline ---
        try:
            result = rag_answer(
                query          = query,
                retrieval_mode = config.get("retrieval_mode", "dense"),
                top_k_search   = config.get("top_k_search", 10),
                top_k_select   = config.get("top_k_select", 3),
                use_rerank     = config.get("use_rerank", False),
                verbose        = False,
            )
            answer      = result["answer"]
            chunks_used = result["chunks_used"]

        except Exception as e:
            answer      = f"PIPELINE_ERROR: {e}"
            chunks_used = []
            if verbose:
                print(f"  ⚠ Pipeline lỗi: {e}")

        # --- Chấm 4 metrics ---
        faith     = score_faithfulness(answer, chunks_used)
        relevance = score_answer_relevance(query, answer)
        recall    = score_context_recall(chunks_used, expected_sources)
        complete  = score_completeness(query, answer, expected_answer)

        row = {
            "id":                    question_id,
            "category":              category,
            "query":                 query,
            "answer":                answer,
            "expected_answer":       expected_answer,
            # scores
            "faithfulness":          faith["score"],
            "faithfulness_notes":    faith["notes"],
            "relevance":             relevance["score"],
            "relevance_notes":       relevance["notes"],
            "context_recall":        recall["score"],
            "context_recall_recall": recall.get("recall"),
            "context_recall_notes":  recall["notes"],
            "completeness":          complete["score"],
            "completeness_notes":    complete["notes"],
            "config_label":          label,
        }
        results.append(row)

        if verbose:
            f_s  = faith["score"]
            r_s  = relevance["score"]
            rc_s = recall["score"]
            c_s  = complete["score"]
            print(f"  Answer  : {answer[:120]}{'...' if len(answer) > 120 else ''}")
            print(f"  Scores  : Faithful={f_s} | Relevant={r_s} | Recall={rc_s} | Complete={c_s}")
            if recall.get("missing"):
                print(f"  Missing : {recall['missing']}")

    # --- Print averages ---
    print(f"\n{'─'*50}")
    print(f"Kết quả tổng hợp — {label}")
    print(f"{'─'*50}")
    metrics_display = [
        ("faithfulness",   "Faithfulness   "),
        ("relevance",      "Answer Relevance"),
        ("context_recall", "Context Recall  "),
        ("completeness",   "Completeness    "),
    ]
    for key, display in metrics_display:
        scores = [r[key] for r in results if r[key] is not None]
        if scores:
            avg = sum(scores) / len(scores)
            bar = "█" * round(avg) + "░" * (5 - round(avg))
            print(f"  {display}: {avg:.2f}/5  [{bar}]  (n={len(scores)})")
        else:
            print(f"  {display}: N/A")

    return results


# =============================================================================
# A/B COMPARISON
# =============================================================================

def compare_ab(
    baseline_results: List[Dict],
    variant_results: List[Dict],
    output_csv: Optional[str] = "ab_comparison.csv",
) -> None:
    """
    So sánh baseline vs variant theo metric tổng hợp và per-question.
    """
    metrics = [
        ("faithfulness",   "Faithfulness   "),
        ("relevance",      "Answer Relevance"),
        ("context_recall", "Context Recall  "),
        ("completeness",   "Completeness    "),
    ]

    print(f"\n{'='*70}")
    print("A/B Comparison: Baseline vs Variant")
    print('='*70)
    print(f"{'Metric':<22} {'Baseline':>10} {'Variant':>10} {'Delta':>10}  {'Winner'}")
    print("─" * 62)

    for key, display in metrics:
        b_scores = [r[key] for r in baseline_results if r[key] is not None]
        v_scores = [r[key] for r in variant_results  if r[key] is not None]
        b_avg    = sum(b_scores) / len(b_scores) if b_scores else None
        v_avg    = sum(v_scores) / len(v_scores) if v_scores else None
        delta    = (v_avg - b_avg) if (b_avg is not None and v_avg is not None) else None

        b_str    = f"{b_avg:.2f}" if b_avg is not None else "N/A"
        v_str    = f"{v_avg:.2f}" if v_avg is not None else "N/A"
        d_str    = f"{delta:+.2f}" if delta is not None else "N/A"
        winner   = ("✓ Variant" if delta and delta > 0.1
                    else "✓ Baseline" if delta and delta < -0.1
                    else "≈ Tie")

        print(f"{display:<22} {b_str:>10} {v_str:>10} {d_str:>10}  {winner}")

    # --- Per-question comparison ---
    print(f"\n{'─'*70}")
    print("Per-question breakdown:")
    print(f"{'ID':<6} {'Query (30c)':<32} {'B:F/R/Rc/C':>12} {'V:F/R/Rc/C':>12} {'Winner'}")
    print("─" * 70)

    b_by_id = {r["id"]: r for r in baseline_results}
    for v_row in variant_results:
        qid   = v_row["id"]
        b_row = b_by_id.get(qid, {})
        query_short = v_row["query"][:30]

        def fmt(row):
            return "/".join(str(row.get(k, "?")) for k in
                            ["faithfulness", "relevance", "context_recall", "completeness"])

        b_str    = fmt(b_row)
        v_str    = fmt(v_row)
        b_total  = sum(b_row.get(k, 0) or 0 for k in
                       ["faithfulness", "relevance", "context_recall", "completeness"])
        v_total  = sum(v_row.get(k, 0) or 0 for k in
                       ["faithfulness", "relevance", "context_recall", "completeness"])
        better   = ("Variant ↑" if v_total > b_total
                    else "Baseline ↑" if b_total > v_total
                    else "Tie")

        print(f"{qid:<6} {query_short:<32} {b_str:>12} {v_str:>12} {better}")

    # --- Export CSV ---
    if output_csv:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        csv_path = RESULTS_DIR / output_csv
        combined = baseline_results + variant_results
        if combined:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=combined[0].keys())
                writer.writeheader()
                writer.writerows(combined)
            print(f"\nKết quả đã lưu vào: {csv_path}")


# =============================================================================
# REPORT GENERATOR — Markdown scorecard
# =============================================================================

def generate_scorecard_summary(results: List[Dict], label: str) -> str:
    """Tạo báo cáo tóm tắt scorecard dạng markdown."""
    metrics = ["faithfulness", "relevance", "context_recall", "completeness"]
    averages = {}
    for m in metrics:
        scores         = [r[m] for r in results if r[m] is not None]
        averages[m]    = sum(scores) / len(scores) if scores else None

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    md  = f"# Scorecard: {label}\n"
    md += f"Generated: {timestamp}\n\n"
    md += "## Summary\n\n"
    md += "| Metric | Average Score | Bar |\n"
    md += "|--------|:-------------:|-----|\n"

    for m in metrics:
        avg     = averages[m]
        avg_str = f"{avg:.2f}/5" if avg is not None else "N/A"
        bar     = ("█" * round(avg) + "░" * (5 - round(avg))) if avg is not None else "─────"
        md += f"| {m.replace('_', ' ').title()} | {avg_str} | {bar} |\n"

    md += "\n## Per-Question Results\n\n"
    md += "| ID | Category | Q (short) | Faithful | Relevant | Recall | Complete | Notes |\n"
    md += "|----|----------|-----------|:--------:|:--------:|:------:|:--------:|-------|\n"

    for r in results:
        notes = (r.get("faithfulness_notes") or "")[:60]
        md += (
            f"| {r['id']} | {r['category']} | {r['query'][:35]} "
            f"| {r.get('faithfulness', '?')} "
            f"| {r.get('relevance', '?')} "
            f"| {r.get('context_recall', '?')} "
            f"| {r.get('completeness', '?')} "
            f"| {notes} |\n"
        )

    md += "\n## Full Answers\n\n"
    for r in results:
        md += f"### [{r['id']}] {r['query']}\n"
        md += f"**Answer:** {r['answer']}\n\n"
        if r.get("expected_answer"):
            md += f"**Expected:** {r['expected_answer']}\n\n"
        md += "---\n\n"

    return md


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Sprint 4: Evaluation & Scorecard")
    print("=" * 60)

    # --- Load test questions ---
    print(f"\nLoading test questions từ: {TEST_QUESTIONS_PATH}")
    try:
        with open(TEST_QUESTIONS_PATH, "r", encoding="utf-8") as f:
            test_questions = json.load(f)
        print(f"Tìm thấy {len(test_questions)} câu hỏi:")
        for q in test_questions[:3]:
            print(f"  [{q['id']}] {q['question']} ({q['category']})")
        if len(test_questions) > 3:
            print(f"  ... và {len(test_questions) - 3} câu nữa")
    except FileNotFoundError:
        print("Không tìm thấy test_questions.json — tạo sample questions.")
        test_questions = [
            {"id": "Q01", "category": "SLA",    "question": "SLA xử lý ticket P1 là bao lâu?",
             "expected_answer": "...", "expected_sources": ["sla_p1_2026"]},
            {"id": "Q02", "category": "Refund", "question": "Khách hàng có thể yêu cầu hoàn tiền trong bao nhiêu ngày?",
             "expected_answer": "...", "expected_sources": ["policy_refund_v4"]},
            {"id": "Q03", "category": "Access", "question": "Ai phải phê duyệt để cấp quyền Level 3?",
             "expected_answer": "...", "expected_sources": ["access_control_sop"]},
            {"id": "Q04", "category": "Abstain","question": "ERR-403-AUTH là lỗi gì?",
             "expected_answer": "Không đủ dữ liệu", "expected_sources": []},
        ]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Chạy Baseline ---
    print("\n--- Chạy Baseline ---")
    baseline_results = run_scorecard(
        config         = BASELINE_CONFIG,
        test_questions = test_questions,
        verbose        = True,
    )

    # Lưu scorecard baseline
    baseline_md   = generate_scorecard_summary(baseline_results, BASELINE_CONFIG["label"])
    baseline_path = RESULTS_DIR / "scorecard_baseline.md"
    baseline_path.write_text(baseline_md, encoding="utf-8")
    print(f"\nScorecard baseline lưu tại: {baseline_path}")

    # --- Chạy Variant ---
    print("\n--- Chạy Variant ---")
    variant_results = run_scorecard(
        config         = VARIANT_CONFIG,
        test_questions = test_questions,
        verbose        = True,
    )

    # Lưu scorecard variant
    variant_md   = generate_scorecard_summary(variant_results, VARIANT_CONFIG["label"])
    variant_path = RESULTS_DIR / "scorecard_variant.md"
    variant_path.write_text(variant_md, encoding="utf-8")
    print(f"\nScorecard variant lưu tại: {variant_path}")

    # --- A/B Comparison ---
    print("\n--- A/B Comparison ---")
    compare_ab(
        baseline_results = baseline_results,
        variant_results  = variant_results,
        output_csv       = "ab_comparison.csv",
    )

    print("\n✓ Sprint 4 hoàn thành!")
    print(f"  Kết quả trong: {RESULTS_DIR}/")