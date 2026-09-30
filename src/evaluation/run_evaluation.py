import time
import json
import re
from pathlib import Path

from src.agent.orchestrator import run_agent


# Paths are resolved from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEST_CASES_PATH = PROJECT_ROOT / "src" / "evaluation" / "test_cases.json"


def get_available_sources(paper_evidence):
    """Return the paper/page pairs present in retrieved evidence."""
    return {
        (item.get("paper_id"), item.get("page"))
        for item in paper_evidence
        if item.get("paper_id") is not None
        and item.get("page") is not None
    }


def validate_citations(answer, paper_evidence):
    """
    Validate in-text citations against the exact retrieved evidence item.

    Expected format:
    [P1: paper_id, p. 8]
    """

    citation_pattern = r"\[P(\d+):\s*([^,\]]+),\s*p\.\s*(\d+)\]"

    citations = re.findall(citation_pattern, answer)

    results = []

    for citation_id, paper_id, page in citations:
        citation_index = int(citation_id)
        page = int(page)
        paper_id = paper_id.strip()

        evidence_index = citation_index - 1

        valid_mapping = False

        if 0 <= evidence_index < len(paper_evidence):
            evidence = paper_evidence[evidence_index]

            expected_paper = evidence.get("paper_id")
            expected_page = evidence.get("page")

            valid_mapping = (
                paper_id == expected_paper
                and page == expected_page
            )

        results.append({
            "citation": f"P{citation_id}",
            "paper_id": paper_id,
            "page": page,
            "valid": valid_mapping,
        })

    total = len(results)
    valid_count = sum(item["valid"] for item in results)

    return {
        "total_citations": total,
        "valid_citations": valid_count,
        "invalid_citations": total - valid_count,
        "citation_validity_rate": (
            valid_count / total if total else None
        ),
        "details": results,
    }
def load_test_cases():
    with open(TEST_CASES_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def evaluate_case(test_case):
    query = test_case["query"]
    expected_route = test_case["expected_route"]
    expected_type = test_case["expected_type"]

    print(f"\nRunning: {test_case['id']}")
    print(f"Query: {query}")

    start_time = time.perf_counter()

    result = run_agent(query)

    elapsed_time = time.perf_counter() - start_time

    actual_route = result.get("route")
    llm_calls = expected_llm_calls(actual_route)
    answer = result.get("answer", "")

    paper_evidence = result.get("paper_evidence", [])

    data_evidence = result.get("data_evidence", {})

    data_evaluation = evaluate_data_answer(
        answer,
        data_evidence
    )

    relevant_papers = test_case.get("relevant_papers", [])

    retrieval_evaluation = evaluate_retrieval(
        paper_evidence,
        relevant_papers
    )

    paper_evidence = result.get("paper_evidence", [])

    citation_validation = validate_citations(
        answer,
        paper_evidence,
    )

    # 1. Route correctness
    route_correct = actual_route == expected_route

    # 2. Basic citation presence check
    # This only checks whether citations appear in the answer.
    citation_pattern = r"\[P\d+\]"
    citations_present = bool(re.search(citation_pattern, answer))

    # 3. Basic refusal check
    # This is a rough signal, not a semantic correctness judge.
    refusal_phrases = [
        "insufficient evidence",
        "cannot be established",
        "does not contain",
        "don't have enough evidence",
        "not enough information",
        "cannot answer",
    ]

    answer_lower = answer.lower()
    refusal_detected = any(
        phrase in answer_lower for phrase in refusal_phrases
    )

    if expected_type == "unsupported":
        answer_behavior_correct = refusal_detected
    else:
        answer_behavior_correct = bool(answer.strip())

    return {
        "id": test_case["id"],
        "expected_route": expected_route,
        "actual_route": actual_route,
        "route_correct": route_correct,
        "expected_type": expected_type,
        "citations_present": citations_present,
        "refusal_detected": refusal_detected,
        "answer_behavior_correct": answer_behavior_correct,
        "answer": answer,
        "paper_evidence": result.get("paper_evidence", []),
        "data_evidence": result.get("data_evidence", []),
        "citation_validation": citation_validation,
        "retrieval_evaluation": retrieval_evaluation,
        "data_evaluation": data_evaluation,
        "latency_seconds": round(elapsed_time, 3),
        "expected_llm_calls": llm_calls,
    }

def evaluate_retrieval(paper_evidence, relevant_papers):
    """
    Evaluate whether relevant papers appear in the retrieved evidence.

    Metrics:
    - Hit@1
    - Hit@3
    - Hit@5
    - MRR
    """

    if not relevant_papers:
        return {
            "evaluated": False,
            "hit_at_1": None,
            "hit_at_3": None,
            "hit_at_5": None,
            "mrr": None,
        }

    retrieved_papers = [
        item.get("paper_id")
        for item in paper_evidence
    ]

    relevant_papers = set(relevant_papers)

    # Hit@K
    hit_at_1 = any(
        paper in relevant_papers
        for paper in retrieved_papers[:1]
    )

    hit_at_3 = any(
        paper in relevant_papers
        for paper in retrieved_papers[:3]
    )

    hit_at_5 = any(
        paper in relevant_papers
        for paper in retrieved_papers[:5]
    )

    # Mean Reciprocal Rank
    reciprocal_rank = 0.0

    for rank, paper in enumerate(retrieved_papers, start=1):
        if paper in relevant_papers:
            reciprocal_rank = 1.0 / rank
            break

    return {
        "evaluated": True,
        "hit_at_1": hit_at_1,
        "hit_at_3": hit_at_3,
        "hit_at_5": hit_at_5,
        "mrr": reciprocal_rank,
    }

def evaluate_data_answer(answer, data_evidence):
    """
    Verify that a generated answer contains the numerical
    result returned by the deterministic data tool.
    """

    if not data_evidence:
        return {
            "evaluated": False,
            "correct": None,
            "expected_value": None,
            "found_value": None,
        }

    result = data_evidence.get("result")

    if not isinstance(result, (int, float)):
        return {
            "evaluated": False,
            "correct": None,
            "expected_value": result,
            "found_value": None,
        }

    # Look for numbers in the generated answer.
    numbers = re.findall(r"\d+(?:\.\d+)?", answer)

    found_numbers = [float(number) for number in numbers]

    tolerance = 0.01

    correct = any(
        abs(number - result) <= tolerance
        for number in found_numbers
    )

    return {
        "evaluated": True,
        "correct": correct,
        "expected_value": result,
        "found_value": next(
            (
                number
                for number in found_numbers
                if abs(number - result) <= tolerance
            ),
            None,
        ),
    }

def expected_llm_calls(route):
    """
    Expected external LLM calls for the current architecture.

    PAPER = router + generator
    DATA  = router + data agent + generator
    BOTH  = router + data agent + generator
    """

    if route == "PAPER":
        return 2

    if route in {"DATA", "BOTH"}:
        return 3

    return 0

def main():
    test_cases = load_test_cases()
    results = []

    for test_case in test_cases:
        try:
            results.append(evaluate_case(test_case))
        except Exception as error:
            results.append({
                "id": test_case.get("id", "unknown"),
                "error": str(error),
            })

    total = len(results)
    completed = sum("error" not in result for result in results)

    route_correct_count = sum(
        result.get("route_correct", False)
        for result in results
    )

    behavior_correct_count = sum(
        result.get("answer_behavior_correct", False)
        for result in results
    )

    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Cases: {total}")
    print(f"Completed: {completed}")

    if completed:
        print(
            "Route accuracy: "
            f"{route_correct_count}/{completed} "
            f"({route_correct_count / completed:.1%})"
        )
        print(
            "Basic answer/refusal behavior: "
            f"{behavior_correct_count}/{completed} "
            f"({behavior_correct_count / completed:.1%})"
        )

    # Save detailed results for later inspection.
    output_path = PROJECT_ROOT / "src" / "evaluation" / "evaluation_results.json"

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)

    print(f"\nDetailed results saved to: {output_path}")

    citation_cases = [
    result["citation_validation"]
    for result in results
    if "citation_validation" in result
    and result["citation_validation"]["total_citations"] > 0
    ]

    total_citations = sum(
        item["total_citations"]
        for item in citation_cases
    )

    valid_citations = sum(
        item["valid_citations"]
        for item in citation_cases
    )

    print(f"Total in-text citations: {total_citations}")
    print(f"Valid citations: {valid_citations}")

    if total_citations:
        print(
            "Citation validity: "
            f"{valid_citations}/{total_citations} "
            f"({valid_citations / total_citations:.1%})"
        )
    else:
        print("Citation validity: N/A (no citations found)")

    retrieval_cases = [
    result["retrieval_evaluation"]
    for result in results
    if result.get("retrieval_evaluation", {}).get("evaluated")
]

    if retrieval_cases:
        hit_at_1 = sum(
            item["hit_at_1"]
            for item in retrieval_cases
        )

        hit_at_3 = sum(
            item["hit_at_3"]
            for item in retrieval_cases
        )

        hit_at_5 = sum(
            item["hit_at_5"]
            for item in retrieval_cases
        )

        mean_mrr = sum(
            item["mrr"]
            for item in retrieval_cases
        ) / len(retrieval_cases)

        total = len(retrieval_cases)

        print(f"Retrieval cases: {total}")
        print(
            f"Hit@1: {hit_at_1}/{total} "
            f"({hit_at_1 / total:.1%})"
        )
        print(
            f"Hit@3: {hit_at_3}/{total} "
            f"({hit_at_3 / total:.1%})"
        )
        print(
            f"Hit@5: {hit_at_5}/{total} "
            f"({hit_at_5 / total:.1%})"
        )
        print(f"MRR: {mean_mrr:.3f}")
    else:
        print("Retrieval evaluation: N/A")

    data_cases = [
    result["data_evaluation"]
        for result in results
        if result.get("data_evaluation", {}).get("evaluated")
    ]

    if data_cases:
        correct_data_answers = sum(
            item["correct"]
            for item in data_cases
        )

        total_data_cases = len(data_cases)

        print(
            f"Data answer correctness: "
            f"{correct_data_answers}/{total_data_cases} "
            f"({correct_data_answers / total_data_cases:.1%})"
        )
    else:
        print("Data answer correctness: N/A")

    latencies = [
        result["latency_seconds"]
        for result in results
        if "latency_seconds" in result
    ]

    if latencies:
        average_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)
        min_latency = min(latencies)

        print(
            f"Average latency: {average_latency:.2f}s"
        )
        print(
            f"Min latency: {min_latency:.2f}s"
        )
        print(
            f"Max latency: {max_latency:.2f}s"
        )

    total_llm_calls = sum(
        result.get("expected_llm_calls", 0)
        for result in results
    )

    print(f"Expected LLM calls: {total_llm_calls}")

if __name__ == "__main__":
    main()