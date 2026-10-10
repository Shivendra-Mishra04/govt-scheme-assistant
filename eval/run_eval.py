"""Evaluation script for PM-USP CSSS RAG pipeline.
Measures:
- Retrieval: Hit@k and MRR for answerable questions
- Generation: Correctness (must_include strings)
- Refusal accuracy: Correctly refuses unanswerable questions
- Baseline comparison: RAG vs the same model without retrieval
  (correctness on answerable, LLM-judged refusal/fabrication on unanswerable)
- Faithfulness (LLM-as-judge, optional)

Run: python -m eval.run_eval [--models m1,m2] [--judge] [--k 4] [--retries 2] [--sleep 1]
"""
import sys
import json
import time
import argparse
import re
from pathlib import Path
from openai import OpenAI

sys.stdout.reconfigure(encoding="utf-8")

from src.config import EVAL_LLM_MODEL, LLM_BASE_URL, LLM_API_KEY, TOP_K
from src.rag import retrieve, answer, answer_no_rag, REFUSAL_ANSWER


def normalize_text(text):
    """Normalize text for matching (remove commas, extra spaces, lowercase)."""
    text = text.lower()
    text = re.sub(r'[,₹\s]+', ' ', text)
    return text.strip()


def check_must_include(answer_text, must_include_list):
    """Return True if ANY string in must_include_list appears in the answer.

    Note: the list is treated as alternatives (e.g. ["20,000", "20000"]),
    so one match is enough.
    """
    if not must_include_list:
        return True

    norm_answer = normalize_text(answer_text)

    for item in must_include_list:
        norm_item = normalize_text(item)
        if norm_item in norm_answer:
            return True

    return False


def is_refusal(answer_text, grounded):
    """Check if the answer is a refusal."""
    if not grounded:
        return True

    refusal_phrases = [
        "i don't know",
        "i do not know",
        "not mentioned",
        "not provided",
        "not available in the guidelines",
        "guidelines do not contain",
        REFUSAL_ANSWER.lower()
    ]

    norm_answer = answer_text.lower().replace("\u2019", "'")
    return any(phrase in norm_answer for phrase in refusal_phrases)


def judge_faithfulness(question, context, answer_text, client, model):
    """Use LLM-as-judge to score faithfulness.

    Returns 1 (faithful), 0 (not faithful) or None (judge failed / unclear reply).
    """
    prompt = f"""You are an evaluator. Determine if the answer is completely faithful to the provided context.
Answer 1 if every claim in the answer is supported by the context, or 0 if it contains claims not supported by the context.

Context:
{context}

Question:
{question}

Answer:
{answer_text}

Reply with ONLY the single digit 1 or 0."""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
            temperature=0
        )
        text = (response.choices[0].message.content or "").strip()
        if not text:
            return None
        match = re.search(r'\b([01])\b', text)
        if not match:
            return None
        return int(match.group(1))
    except Exception:
        return None


def judge_no_rag_baseline(answer_text, client, model):
    """Classify a no-RAG answer as refused_or_clarified or fabricated_specific_answer.

    Returns "error" if the judge fails or gives an empty / unclear reply.
    """
    prompt = f"""You are evaluating an AI assistant's answer to a question it had no documents for.

Classify the answer into exactly ONE category:

1. refused_or_clarified: the assistant says it does not know, asks for clarification, or says it needs more information, and does not give specific figures or facts as if they were true.
2. fabricated_specific_answer: the assistant gives specific figures, amounts, dates, percentages or procedures as if they were facts.

Answer text:
{answer_text}

Reply with ONLY one of these two labels: refused_or_clarified or fabricated_specific_answer"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
            temperature=0
        )
        text = (response.choices[0].message.content or "").strip().lower()
        if not text:
            return "error"
        if "fabricated" in text:
            return "fabricated_specific_answer"
        if "refused" in text or "clarified" in text:
            return "refused_or_clarified"
        return "error"
    except Exception:
        return "error"


def evaluate_model(model_name, golden_set, k=TOP_K, judge=False, retries=2, sleep_time=1.0):
    """Run full evaluation on a specific model."""
    client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)

    results = []
    models_used = set()

    # Metrics counters
    hits_at_k = 0
    reciprocal_ranks = []
    correct_answers = 0
    correct_refusals = 0
    faithfulness_scores = []

    # Baseline counters
    no_rag_refused_or_clarified = 0
    no_rag_fabricated = 0
    no_rag_correct_on_answerable = 0
    no_rag_call_errors = 0

    # Judge failure counters
    baseline_judge_errors = 0
    faithfulness_judge_errors = 0

    # Track items with/without errors
    answerable_count = sum(1 for item in golden_set if item["type"] == "answerable")
    unanswerable_count = sum(1 for item in golden_set if item["type"] == "unanswerable")

    answerable_errors = 0
    unanswerable_errors = 0

    print(f"\nEvaluating model: {model_name}")
    print(f"Total test items: {len(golden_set)} ({answerable_count} answerable, {unanswerable_count} unanswerable)")
    print("-" * 60)

    for idx, item in enumerate(golden_set, 1):
        q_id = item["id"]
        question = item["question"]
        q_type = item["type"]
        expected_sections = [str(s) for s in item["expected_sections"]]
        must_include = item["must_include"]

        print(f"[{idx}/{len(golden_set)}] Q{q_id} ({q_type}): {question[:50]}...")

        # 1. Retrieval evaluation (not affected by errors)
        retrieval_results = retrieve(question, k=k)
        retrieved_sections = [str(r["chunk"]["section_number"]) for r in retrieval_results]

        # Calculate Hit@k and MRR for answerable items
        is_hit = False
        rr = 0.0

        if q_type == "answerable":
            for rank, sec in enumerate(retrieved_sections, 1):
                if sec in expected_sections:
                    if not is_hit:
                        is_hit = True
                        rr = 1.0 / rank

            if is_hit:
                hits_at_k += 1
            reciprocal_ranks.append(rr)

        # 2. RAG Generation with retry
        rag_res = None
        for attempt in range(retries + 1):
            try:
                rag_res = answer(question, model=model_name)
                break
            except Exception as e:
                if attempt == retries:
                    print(f"   ERROR calling RAG: {e}")
                    if q_type == "answerable":
                        answerable_errors += 1
                    else:
                        unanswerable_errors += 1
                else:
                    time.sleep(sleep_time)

        if rag_res is None:
            results.append({
                "id": q_id,
                "error": True,
                "question": question,
                "type": q_type
            })
            continue

        rag_answer = rag_res["answer"]
        grounded = rag_res["grounded"]
        model_used = rag_res.get("model_used", model_name)
        top_score = rag_res.get("top_score")
        if model_used:
            models_used.add(model_used)

        # 3. Check correctness / refusal (only for non-error items)
        is_correct = None
        is_refused = None

        if q_type == "answerable":
            is_correct = check_must_include(rag_answer, must_include)
            if is_correct:
                correct_answers += 1
        else:  # unanswerable
            is_refused = is_refusal(rag_answer, grounded)
            if is_refused:
                correct_refusals += 1

        # 4. Baseline No-RAG comparison (one call per item)
        no_rag_res = None
        for attempt in range(retries + 1):
            try:
                no_rag_res = answer_no_rag(question, model=model_name)
                break
            except Exception:
                if attempt < retries:
                    time.sleep(sleep_time)

        classification = None
        if no_rag_res is None:
            no_rag_call_errors += 1
        else:
            no_rag_ans = no_rag_res["answer"]

            if q_type == "unanswerable":
                classification = judge_no_rag_baseline(no_rag_ans, client, model_name)
                if classification == "fabricated_specific_answer":
                    no_rag_fabricated += 1
                elif classification == "refused_or_clarified":
                    no_rag_refused_or_clarified += 1
                else:
                    baseline_judge_errors += 1
            else:  # answerable
                if check_must_include(no_rag_ans, must_include):
                    no_rag_correct_on_answerable += 1

        # 5. Faithfulness (LLM-as-judge, only for answerable grounded items)
        faithfulness = None
        if judge and q_type == "answerable" and grounded:
            context = "\n\n".join([r["chunk"]["text"] for r in retrieval_results])
            faithfulness = judge_faithfulness(question, context, rag_answer, client, model_name)
            if faithfulness is None:
                faithfulness_judge_errors += 1
            else:
                faithfulness_scores.append(faithfulness)

        results.append({
            "id": q_id,
            "question": question,
            "type": q_type,
            "retrieved_sections": retrieved_sections,
            "hit": is_hit if q_type == "answerable" else None,
            "rr": rr if q_type == "answerable" else None,
            "rag_answer": rag_answer,
            "model_used": model_used,
            "grounded": grounded,
            "top_score": top_score,
            "correct": is_correct,
            "refused": is_refused,
            "no_rag_answer": no_rag_res["answer"] if no_rag_res else None,
            "no_rag_classification": classification,
            "faithfulness": faithfulness
        })

        time.sleep(sleep_time)

    # Calculate final scores - exclude errored items from generation metrics
    total_errors = answerable_errors + unanswerable_errors
    valid_answerable = answerable_count - answerable_errors
    valid_unanswerable = unanswerable_count - unanswerable_errors

    # Retrieval metrics use all answerable items (errors don't affect retrieval)
    hit_rate = (hits_at_k / answerable_count) * 100 if answerable_count > 0 else 0
    mrr = (sum(reciprocal_ranks) / len(reciprocal_ranks)) if reciprocal_ranks else 0

    # Generation metrics exclude errored items
    correctness = (correct_answers / valid_answerable) * 100 if valid_answerable > 0 else 0
    refusal_acc = (correct_refusals / valid_unanswerable) * 100 if valid_unanswerable > 0 else 0
    faith_rate = (sum(faithfulness_scores) / len(faithfulness_scores) * 100) if faithfulness_scores else None

    # No-RAG baseline metrics
    no_rag_correctness = (no_rag_correct_on_answerable / valid_answerable) * 100 if valid_answerable > 0 else 0

    judge_errors = baseline_judge_errors + faithfulness_judge_errors

    summary = {
        "model": model_name,
        "models_used": list(models_used),
        "hit_at_k": f"{hit_rate:.1f}% ({hits_at_k}/{answerable_count})",
        "mrr": f"{mrr:.3f}",
        "correctness": f"{correctness:.1f}% ({correct_answers}/{valid_answerable})" if valid_answerable > 0 else "N/A (all errored)",
        "refusal_accuracy": f"{refusal_acc:.1f}% ({correct_refusals}/{valid_unanswerable})" if valid_unanswerable > 0 else "N/A (all errored)",
        "no_rag_baseline_correctness": f"{no_rag_correctness:.1f}% ({no_rag_correct_on_answerable}/{valid_answerable})" if valid_answerable > 0 else "N/A",
        "no_rag_refused": no_rag_refused_or_clarified,
        "no_rag_fabricated": no_rag_fabricated,
        "no_rag_call_errors": no_rag_call_errors,
        "faithfulness": f"{faith_rate:.1f}% ({sum(faithfulness_scores)}/{len(faithfulness_scores)})" if faith_rate is not None else "N/A",
        "judge_errors": judge_errors,
        "errors": total_errors,
        "error_rate": (total_errors / len(golden_set)) * 100 if len(golden_set) > 0 else 0
    }

    return summary, results


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG pipeline")
    parser.add_argument("--models", type=str, help="Comma-separated model names")
    parser.add_argument("--judge", action="store_true", help="Run LLM faithfulness judge")
    parser.add_argument("--k", type=int, default=TOP_K, help="Top-K for retrieval")
    parser.add_argument("--retries", type=int, default=2, help="Max retries for LLM calls")
    parser.add_argument("--sleep", type=float, default=1.0, help="Sleep between calls in seconds")

    args = parser.parse_args()

    # Load golden set
    golden_set_path = Path(__file__).parent / "golden_set.json"
    with open(golden_set_path, "r", encoding="utf-8") as f:
        golden_set = json.load(f)

    models_to_test = [m.strip() for m in args.models.split(",")] if args.models else [EVAL_LLM_MODEL]

    all_summaries = []
    all_results = {}

    print("=" * 80)
    print("PM-USP CSSS RAG Evaluation Suite")
    print("=" * 80)

    for model in models_to_test:
        summary, results = evaluate_model(
            model,
            golden_set,
            k=args.k,
            judge=args.judge,
            retries=args.retries,
            sleep_time=args.sleep
        )
        all_summaries.append(summary)
        all_results[model] = results

        # Check if fallback happened
        if len(summary["models_used"]) > 1:
            print(f"\nWARNING: Model combo fell back across multiple models: {summary['models_used']}")

    # Print summary, one metric per line
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)

    for s in all_summaries:
        print(f"Model (combo):           {s['model']}")
        print(f"Models actually used:    {', '.join(s['models_used'])}")
        print(f"Hit@k:                   {s['hit_at_k']}")
        print(f"MRR:                     {s['mrr']}")
        print(f"RAG correctness:         {s['correctness']}")
        print(f"RAG refusal accuracy:    {s['refusal_accuracy']}")
        print(f"No-RAG correctness:      {s['no_rag_baseline_correctness']}")
        print(f"No-RAG refused/clarified:{s['no_rag_refused']}")
        print(f"No-RAG fabricated:       {s['no_rag_fabricated']}")
        print(f"No-RAG call errors:      {s['no_rag_call_errors']}")
        print(f"Faithfulness (judge):    {s['faithfulness']}")
        print(f"Judge errors:            {s['judge_errors']}")
        print(f"RAG errors:              {s['errors']}")
        print("-" * 80)

    # Print error summary
    for s in all_summaries:
        if s["errors"] > 0:
            print(f"\n{s['errors']} items excluded due to errors")
            if s["error_rate"] > 25:
                print(f"WARNING: {s['error_rate']:.1f}% of items errored - numbers are not reliable")
        if s["judge_errors"] > 0 or s["no_rag_call_errors"] > 0:
            print(f"\nWARNING: {s['judge_errors']} judge errors and {s['no_rag_call_errors']} no-RAG call errors - "
                  f"baseline and faithfulness numbers are incomplete")

    # Save results to file
    out_path = Path(__file__).parent / "results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "summaries": all_summaries,
            "detailed_results": all_results
        }, f, ensure_ascii=False, indent=2)

    print(f"\nDetailed results saved to: {out_path}\n")


if __name__ == "__main__":
    main()