# responsible for evaluating semantic retrieval and grounded rag answers
"""
1. load golden dataset
2. evaluate retrieval quality
3. evaluate rag answer quality
4. run evaluation over all cases
5. print evaluation report
"""
import json
from pathlib import Path
from source.semantic import semantic_search
from source.rag import REFUSAL, rag_search

DATASET_PATH = Path("evaluation/golden_dataset.json")

# -----------------------------------------------------------------------------------------------------------------------------
# 1. load golden dataset
def load_dataset() -> list[dict]:
    """load evaluation cases from the golden dataset json file"""
    # read json file as text
    dataset_text = DATASET_PATH.read_text(encoding="utf-8")
    # convert json text into python objects
    cases = json.loads(dataset_text)
    return cases

# -----------------------------------------------------------------------------------------------------------------------------
# 2. evaluate semantic retrieval
def evaluate_retrieval(case: dict, k: int = 3) -> dict:
    """
    check whether expected source files appear in semantic search results
    k: number of top retrieved chunks to inspect
    """
    # run semantic search using the case question
    results = semantic_search(case["question"], k)
    # collect source filename from each retrieved result
    sources = []
    for result in results:
        source = result["metadata"]["source"]
        sources.append(source)

    # convert expected sources into a set for easy membership checks
    expected = set(case["expected_sources"])
    # find rank of first retrieved source that matches an expected source
    first_rank = None
    for rank, source in enumerate(sources, start=1):
        if source in expected:
            first_rank = rank
            break

    # true if first result is from an expected source
    hit_at_1 = bool(sources and sources[0] in expected)
    # true if any result within top-k is from an expected source
    hit_at_k = False

    for source in sources:
        if source in expected:
            hit_at_k = True
            break

    # reciprocal rank rewards expected sources appearing earlier
    if first_rank:
        reciprocal_rank = 1 / first_rank
    else:
        reciprocal_rank = 0.0

    return {
        "id": case["id"],
        "retrieved_sources": sources,
        "hit_at_1": hit_at_1,
        "hit_at_k": hit_at_k,
        "reciprocal_rank": reciprocal_rank,
    }

# -----------------------------------------------------------------------------------------------------------------------------
# 3. evaluate rag answer
def evaluate_rag(case: dict, k: int = 3) -> dict:
    """
    check expected facts, refusal behavior and returned source files
    """

    # generate rag response for case question
    response = rag_search(case["question"], k)

    # extract answer text
    answer = response["answer"]

    # check whether answer contains the refusal message
    refused = REFUSAL.lower() in answer.lower()

    # tells us whether this case is expected to be answerable
    should_answer = case["should_answer"]

    # check whether every expected fact appears in answer
    expected_facts_found = True

    for term in case["expected_answer_contains"]:
        if term.lower() not in answer.lower():
            expected_facts_found = False
            break

    # check refusal behavior
    if should_answer:
        refusal_correct = not refused
    else:
        refusal_correct = refused

    # check whether answerable cases return one of their expected source files
    if should_answer:
        source_behavior_correct = bool(
            set(response["sources"]) & set(case["expected_sources"])
        )
    else:
        source_behavior_correct = not response["sources"]

    return {
        "id": case["id"],
        "answer": answer,
        "expected_facts_found": expected_facts_found,
        "refusal_correct": refusal_correct,
        "source_behavior_correct": source_behavior_correct,
    }

# -----------------------------------------------------------------------------------------------------------------------------
# 4. run full evaluation
def run_evaluation(mode: str, k: int = 3) -> list[dict]:
    """
    run either retrieval or rag evaluation across applicable cases
    """

    # load all evaluation cases
    cases = load_dataset()

    # holds evaluation results
    evaluation_results = []

    # run retrieval evaluation
    if mode == "retrieval":
        for case in cases:

            # only evaluate retrieval when expected sources are provided
            if case["expected_sources"]:
                result = evaluate_retrieval(case, k)
                evaluation_results.append(result)

    # run rag evaluation
    else:
        for case in cases:
            result = evaluate_rag(case, k)
            evaluation_results.append(result)

    return evaluation_results

# -----------------------------------------------------------------------------------------------------------------------------
# 5. command line entry point
def main() -> None:
    """
    run either:

    python -m source.evaluation retrieval

    or:

    python -m source.evaluation rag

    optional:
    --k 5
    """
    import argparse

    # create command line parser
    parser = argparse.ArgumentParser(description="Evaluate Fetchdex")

    # choose evaluation type
    parser.add_argument("mode", choices=["retrieval", "rag"])

    # choose how many top results to use
    parser.add_argument("--k", type=int, default=3)

    # read command line arguments
    args = parser.parse_args()

    # run evaluation
    results = run_evaluation(args.mode, args.k)

    # convert results into formatted json and print
    report = json.dumps(results, indent=2)
    print(report)

if __name__ == "__main__":
    main()
