from src.agent.router import route_query
from src.retrieval.hybrid_search import hybrid_search
from src.data_tools.ocean_data import OceanDataTool
from src.agent.data_agent import parse_data_query


# ============================================================
# INITIALIZE DATA TOOL
# ============================================================

ocean_data_tool = OceanDataTool(
    "data/ocean/sample_ocean_data.csv"
)


# ============================================================
# PAPER TOOL
# ============================================================

def run_paper_tool(query):

    print("\n[AGENT] Running paper retrieval...")

    results = hybrid_search(
        query,
        final_k=5
    )

    return results


# ============================================================
# DATA TOOL
# ============================================================

def run_data_tool(query):

    print("\n[AGENT] Running ocean data tool...")

    # --------------------------------------------------------
    # STEP 1 — Interpret the question
    # --------------------------------------------------------

    operation = parse_data_query(
        query
    )

    print(
        f"\n[AGENT] Data operation: "
        f"{operation}"
    )

    operation_name = operation["operation"]

    # --------------------------------------------------------
    # STEP 2 — Execute the operation
    # --------------------------------------------------------

    if operation_name == "dataset_info":

        result = ocean_data_tool.dataset_info()

    elif operation_name == "average_temperature":

        depth_m = operation.get(
            "depth_m"
        )

        result = ocean_data_tool.average_temperature(
            depth_m=depth_m
        )

    elif operation_name == "maximum_temperature":

        result = ocean_data_tool.maximum_temperature()

    elif operation_name == "minimum_temperature":

        result = ocean_data_tool.minimum_temperature()

    elif operation_name == "yearly_average_temperature":

        result = ocean_data_tool.yearly_average_temperature()

    else:

        raise ValueError(
            f"Unsupported operation: "
            f"{operation_name}"
        )

    return {
        "operation": operation,
        "result": result
    }

# ============================================================
# ORCHESTRATOR
# ============================================================

def run_agent(query):

    print("\n========================================")
    print("AGENT")
    print("========================================")

    # --------------------------------------------------------
    # STEP 1 — ROUTE QUERY
    # --------------------------------------------------------

    route = route_query(query)

    print(
        f"\n[AGENT] Selected route: {route}"
    )

    # --------------------------------------------------------
    # STEP 2 — PAPER
    # --------------------------------------------------------

    if route == "PAPER":

        paper_results = run_paper_tool(
            query
        )

        return {
            "query": query,
            "route": route,
            "paper_evidence": paper_results,
            "data_evidence": None
        }

    # --------------------------------------------------------
    # STEP 3 — DATA
    # --------------------------------------------------------

    elif route == "DATA":

        data_results = run_data_tool(
            query
        )

        return {
            "query": query,
            "route": route,
            "paper_evidence": None,
            "data_evidence": data_results
        }

    # --------------------------------------------------------
    # STEP 4 — BOTH
    # --------------------------------------------------------

    elif route == "BOTH":

        paper_results = run_paper_tool(
            query
        )

        data_results = run_data_tool(
            query
        )

        return {
            "query": query,
            "route": route,
            "paper_evidence": paper_results,
            "data_evidence": data_results
        }

    # --------------------------------------------------------
    # SAFETY CHECK
    # --------------------------------------------------------

    else:

        raise ValueError(
            f"Unknown route: {route}"
        )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print(
        "Agentic Scientific Research Assistant"
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

            evidence = run_agent(
                query
            )

            print(
                "\n========================================"
            )

            print(
                "EVIDENCE BUNDLE"
            )

            print(
                "========================================"
            )

            print(
                f"\nRoute: {evidence['route']}"
            )

            # ------------------------------------------------
            # Paper evidence
            # ------------------------------------------------

            if evidence["paper_evidence"]:

                print(
                    "\n----- PAPER EVIDENCE -----"
                )

                for i, result in enumerate(
                    evidence["paper_evidence"],
                    start=1
                ):

                    print(
                        f"\nResult {i}"
                    )

                    print(
                        f"Paper: "
                        f"{result['paper_id']}"
                    )

                    print(
                        f"Page: "
                        f"{result['page']}"
                    )

                    print(
                        f"Reranker score: "
                        f"{result['reranker_score']:.4f}"
                    )

                    print(
                        result["text"][:500]
                    )

            # ------------------------------------------------
            # Data evidence
            # ------------------------------------------------

            if evidence["data_evidence"]:

                print(
                    "\n----- DATA EVIDENCE -----"
                )

                print(
                    evidence["data_evidence"]
                )

        except Exception as e:

            print(
                f"\nAgent error: {e}"
            )