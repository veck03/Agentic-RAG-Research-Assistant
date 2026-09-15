import json
import os

from dotenv import load_dotenv
from groq import Groq


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "qwen/qwen3.8-27b"


# ============================================================
# LOAD ENVIRONMENT
# ============================================================

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError(
        "GROQ_API_KEY was not found.\n"
        "Make sure your .env file contains:\n"
        "GROQ_API_KEY=your_api_key_here"
    )


client = Groq(api_key=api_key)


# ============================================================
# ROUTER PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are a query router for a scientific research assistant.

The assistant has two information sources:

1. PAPER
   Scientific papers containing:
   - explanations
   - mechanisms
   - findings
   - methods
   - theories
   - qualitative scientific information

2. DATA
   Structured oceanographic datasets containing:
   - temperature
   - salinity
   - depth
   - latitude
   - longitude
   - dates
   - numerical measurements
   - statistics and aggregations

Choose exactly ONE route:

PAPER
DATA
BOTH

Rules:

Use PAPER when the question can be answered using scientific
literature or requires scientific explanations.

Use DATA when the question requires numerical measurements,
filtering, aggregation, averages, minimums, maximums, or
statistics from the structured dataset.

Use BOTH when the question requires information from scientific
papers AND numerical information from the structured dataset.

Examples:

Question:
"What mechanisms drive ENSO?"

Route:
PAPER

Question:
"What was the average surface temperature in the dataset?"

Route:
DATA

Question:
"What does the literature say about ENSO under global warming?"

Route:
PAPER

Question:
"What does the literature say about ENSO under global warming,
and what was the average temperature in our dataset?"

Route:
BOTH

Return ONLY valid JSON.

The JSON must have exactly this format:

{
    "route": "PAPER"
}

or

{
    "route": "DATA"
}

or

{
    "route": "BOTH"
}
"""


# ============================================================
# ROUTE QUERY
# ============================================================

def route_query(query: str) -> str:

    response = client.chat.completions.create(
        model=MODEL_NAME,

        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": query
            }
        ],

        # Router does not need randomness
        temperature=0,

        # Qwen 3.8 supports disabling reasoning
        reasoning_effort="none",

        # Give the model enough room for the JSON response
        max_completion_tokens=50,

        # Force JSON output
        response_format={
            "type": "json_object"
        }
    )

    # --------------------------------------------------------
    # Get model response
    # --------------------------------------------------------

    content = response.choices[0].message.content

    print("\nDEBUG RESPONSE:")
    print(repr(content))

    # --------------------------------------------------------
    # Validate response
    # --------------------------------------------------------

    if not content:
        print("\nFULL MESSAGE OBJECT:")
        print(response.choices[0].message)

        raise ValueError(
            "Groq returned an empty response."
        )

    content = content.strip()

    # --------------------------------------------------------
    # Parse JSON
    # --------------------------------------------------------

    try:
        result = json.loads(content)

    except json.JSONDecodeError:
        raise ValueError(
            f"Router returned invalid JSON:\n{content}"
        )

    # --------------------------------------------------------
    # Validate route
    # --------------------------------------------------------

    route = result.get("route")

    if route not in {"PAPER", "DATA", "BOTH"}:
        raise ValueError(
            f"Invalid route returned by router: {route}"
        )

    return route


# ============================================================
# TEST PROGRAM
# ============================================================

if __name__ == "__main__":

    print("Scientific Research Assistant Router")
    print(f"Model: {MODEL_NAME}")

    while True:

        query = input(
            "\nEnter your question (or type 'exit'): "
        )

        if query.lower().strip() == "exit":
            break

        if not query.strip():
            continue

        try:

            route = route_query(query)

            print(f"\nSelected route: {route}")

        except Exception as e:

            print(f"\nRouter error: {e}")