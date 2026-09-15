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
        "GROQ_API_KEY was not found in the .env file."
    )


client = Groq(api_key=api_key)


# ============================================================
# DATA QUERY PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are a query parser for a scientific ocean data tool.

The tool contains structured oceanographic data with:

- date
- latitude
- longitude
- depth_m
- temperature_c
- salinity_psu

Your job is NOT to answer the user's question.

Your job is to convert the user's natural-language question
into a structured JSON operation that a Python data tool can
execute.

Supported operations:

1. dataset_info

Use when the user asks about the dataset itself, such as:
- how many rows
- available dates
- available depths
- dataset information

JSON:

{
    "operation": "dataset_info"
}


2. average_temperature

Use when the user asks for average temperature.

Optional parameter:
- depth_m

For surface temperature, use:

"depth_m": 0

JSON:

{
    "operation": "average_temperature",
    "depth_m": 0
}


3. maximum_temperature

Use when the user asks for the maximum temperature.

JSON:

{
    "operation": "maximum_temperature"
}


4. minimum_temperature

Use when the user asks for the minimum temperature.

JSON:

{
    "operation": "minimum_temperature"
}


5. yearly_average_temperature

Use when the user asks for temperature averaged by year.

JSON:

{
    "operation": "yearly_average_temperature"
}


Important rules:

- Do not calculate any values yourself.
- Do not answer the question.
- Only determine the correct operation.
- Surface temperature means depth_m = 0.
- Return ONLY valid JSON.
- Do not include explanations.
- Do not use operations that are not listed above.

Examples:

Question:
"What was the average surface temperature?"

Output:
{
    "operation": "average_temperature",
    "depth_m": 0
}

Question:
"What was the average temperature at 100 meters?"

Output:
{
    "operation": "average_temperature",
    "depth_m": 100
}

Question:
"What was the highest temperature?"

Output:
{
    "operation": "maximum_temperature"
}

Question:
"What was the lowest temperature?"

Output:
{
    "operation": "minimum_temperature"
}

Question:
"Show me the average temperature for each year."

Output:
{
    "operation": "yearly_average_temperature"
}

Question:
"Tell me about the dataset."

Output:
{
    "operation": "dataset_info"
}
"""


# ============================================================
# PARSE DATA QUERY
# ============================================================

def parse_data_query(query: str) -> dict:

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

        temperature=0,

        reasoning_effort="none",

        max_completion_tokens=100,

        response_format={
            "type": "json_object"
        }
    )

    content = response.choices[0].message.content

    print("\nDEBUG DATA QUERY:")
    print(repr(content))

    if not content:
        raise ValueError(
            "Data query parser returned an empty response."
        )

    content = content.strip()

    # --------------------------------------------------------
    # Parse JSON
    # --------------------------------------------------------

    try:

        result = json.loads(content)

    except json.JSONDecodeError:

        raise ValueError(
            f"Data query parser returned invalid JSON:\n{content}"
        )

    # --------------------------------------------------------
    # Validate operation
    # --------------------------------------------------------

    allowed_operations = {
        "dataset_info",
        "average_temperature",
        "maximum_temperature",
        "minimum_temperature",
        "yearly_average_temperature"
    }

    operation = result.get("operation")

    if operation not in allowed_operations:

        raise ValueError(
            f"Invalid data operation: {operation}"
        )

    return result


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print(
        "Ocean Data Query Parser"
    )

    while True:

        query = input(
            "\nEnter your question "
            "(or type 'exit'): "
        )

        if query.lower().strip() == "exit":
            break

        if not query.strip():
            continue

        try:

            result = parse_data_query(
                query
            )

            print(
                "\nParsed operation:"
            )

            print(result)

        except Exception as e:

            print(
                f"\nParser error: {e}"
            )