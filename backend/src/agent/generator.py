
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(dotenv_path=BASE_DIR / ".env")

MODEL_NAME = "qwen/qwen3.8-27b"

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


SYSTEM_PROMPT = """
You are a scientific research assistant.

Answer the user's question using ONLY the supplied evidence.

Paper evidence is provided with IDs such as [P1], [P2].
These IDs map to actual retrieved paper chunks.

STRICT RULES:

- Do not use outside knowledge.
- Do not invent facts, numbers, authors, paper titles, years, or pages.
- Support scientific claims with the provided paper evidence IDs.
- Cite paper evidence using exactly the supplied IDs, e.g. [P1].
- Never write your own author-year-page citations.
- Only use an ID if its evidence supports the claim.
- Do not cite evidence that is irrelevant to the claim.
- Use data evidence only for the dataset values it provides.
- Clearly distinguish literature findings from dataset results.
- First determine whether the supplied evidence actually answers the question.
- If evidence is insufficient, clearly state what cannot be established.
- Answer only the parts of the question supported by evidence.
- Do not treat a retrieved passage as proof unless it supports the claim.
- If paper evidence is absent, do not make paper-based claims.
- If data evidence is absent, do not invent dataset values.
- If evidence is relevant but incomplete, provide the supported information and identify the gap.
- Do not make unsupported claims.
- Keep the answer concise and scientifically precise.

Return only the answer text, with evidence IDs where appropriate.
"""


def prepare_paper_evidence(paper_evidence):
    """
    Assign stable IDs to the paper chunks for this answer.
    Return both the ID-tagged evidence and an ID-to-source map.
    """

    tagged_evidence = []
    source_map = {}

    for i, result in enumerate(paper_evidence or [], start=1):
        evidence_id = f"P{i}"

        tagged_evidence.append({
            "evidence_id": evidence_id,
            "paper_id": result["paper_id"],
            "page": result["page"],
            "text": result["text"]
        })

        source_map[evidence_id] = {
            "paper_id": result["paper_id"],
            "page": result["page"]
        }

    return tagged_evidence, source_map


def format_citations(answer, source_map):
    """
    Replace model-generated evidence IDs with verified source details.
    Remove unknown IDs rather than treating them as valid citations.
    """

    used_ids = []

    def replace_id(match):
        evidence_id = match.group(1)

        if evidence_id not in source_map:
            return ""

        if evidence_id not in used_ids:
            used_ids.append(evidence_id)

        source = source_map[evidence_id]

        return (
            f"[{evidence_id}: "
            f"{source['paper_id']}, "
            f"p. {source['page']}]"
        )

    formatted_answer = re.sub(
        r"\[(P\d+)\]",
        replace_id,
        answer or ""
    )

    # Append a source list for traceability.
    if used_ids:
        formatted_answer += "\n\n**Sources**\n"

        for evidence_id in used_ids:
            source = source_map[evidence_id]

            formatted_answer += (
                f"\n- [{evidence_id}] "
                f"{source['paper_id']}, "
                f"page {source['page']}"
            )

    return formatted_answer

def has_usable_evidence(paper_evidence, data_evidence):
    """
    Check whether any evidence was supplied to the generator.
    """
    has_papers = bool(paper_evidence)
    has_data = bool(data_evidence)

    return has_papers or has_data

def generate_answer(
    query,
    paper_evidence=None,
    data_evidence=None
):

    paper_evidence = paper_evidence or []
    data_evidence = data_evidence or {}

    if not has_usable_evidence(
        paper_evidence,
        data_evidence
    ):
        return (
            "I could not find relevant evidence in the "
            "available papers or dataset to answer this question."
        )

    tagged_papers, source_map = prepare_paper_evidence(
        paper_evidence
    )

    evidence = {
        "paper_evidence": tagged_papers,
        "data_evidence": data_evidence
    }

    user_prompt = f"""
USER QUESTION:
{query}

EVIDENCE:
{json.dumps(evidence, indent=2, ensure_ascii=False)}

Answer using ONLY this evidence.
Use the supplied paper evidence IDs when citing paper-based claims.
Do not create author-year-page citations.
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        temperature=0,
        reasoning_effort="none",
        max_completion_tokens=1400
    )

    raw_answer = response.choices[0].message.content or ""

    return format_citations(
        raw_answer,
        source_map
    )


if __name__ == "__main__":

    print("\n===== EVIDENCE CHECK TESTS =====")

    print(
        "No evidence:",
        has_usable_evidence([], {})
    )

    print(
        "Paper evidence:",
        has_usable_evidence(
            [{"paper_id": "test", "page": 1, "text": "test"}],
            {}
        )
    )

    print(
        "Data evidence:",
        has_usable_evidence(
            [],
            {"operation": "average_temperature", "result": 27.8}
        )
    )

    test_query = "What does the evidence say about ENSO?"

    test_paper_evidence = [
        {
            "paper_id": "enso_changes_global_warming",
            "page": 3,
            "text": (
                "Example evidence: projections of ENSO "
                "under global warming remain uncertain."
            )
        }
    ]

    answer = generate_answer(
        query=test_query,
        paper_evidence=test_paper_evidence,
        data_evidence={}
    )

    print("\n===== GENERATED ANSWER =====")
    print(answer)