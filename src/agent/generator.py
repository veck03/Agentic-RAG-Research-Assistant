import json
import os

from dotenv import load_dotenv
from groq import Groq



load_dotenv()

MODEL_NAME = "qwen/qwen3.8-27b"

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


SYSTEM_PROMPT = """
You are a scientific research assistant.

Your job is to answer the user's question using ONLY the evidence
provided to you.

You have two possible evidence sources:

1. PAPER EVIDENCE
   - Extracted from scientific papers.
   - Each result contains paper_id, page, and text.

2. DATA EVIDENCE
   - Results calculated from a structured ocean dataset.
   - These values are authoritative for the dataset.

STRICT RULES:

- Do not use outside knowledge.
- Do not invent facts, numbers, papers, pages, or citations.
- Do not make claims that are unsupported by the provided evidence.
- If the evidence is insufficient, explicitly say that the available
  evidence is insufficient to answer the question.
- When using paper evidence, mention the paper and page naturally.
- When using data evidence, report the calculated values accurately.
- If both paper and data evidence are provided, clearly distinguish
  literature findings from dataset results.
- Prefer concise, scientifically precise answers.

Answer only the user's question.
"""


def generate_answer(query, paper_evidence=None, data_evidence=None):

    paper_evidence = paper_evidence or []
    data_evidence = data_evidence or {}

    evidence = {
        "paper_evidence": paper_evidence,
        "data_evidence": data_evidence
    }

    user_prompt = f"""
USER QUESTION:
{query}

EVIDENCE:
{json.dumps(evidence, indent=2)}

Using ONLY the evidence above, answer the user's question.
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
        max_completion_tokens=500
    )

    return response.choices[0].message.content

if __name__ == "__main__":

    test_query = "How does ENSO change under global warming?"

    test_paper_evidence = [
        {
            "paper_id": "enso_changes_global_warming",
            "page": 1,
            "text": (
                "Example evidence from the paper discussing "
                "changes in ENSO under global warming."
            )
        },
        {
            "paper_id": "enso_changes_global_warming",
            "page": 3,
            "text": (
                "Example evidence describing projected changes "
                "in ENSO characteristics."
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