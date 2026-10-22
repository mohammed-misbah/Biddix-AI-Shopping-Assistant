
"""
Biddix Gemini-first evaluation: 100 grounded, multi-turn test messages.

Run from the project root:
    python tests/run_evaluation.py --limit 10
    python tests/run_evaluation.py --limit 100 --delay 1

This is a rule-based evaluation. Subjective response quality and
every possible hallucination cannot be proven automatically.
Inspect the detailed JSON report.
"""

import argparse
import csv
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx


ROOT = Path(__file__).resolve().parents[1]

DATASET = ROOT / "tests" / "evaluation_dataset.json"

REPORT_DIR = ROOT / "tests" / "evaluation_reports"

API = "http://127.0.0.1:8000/api/v1/chat"


def normalized_price(value):
    try:
        return round(
            float(str(value).replace(",", "")),
            2,
        )
    except (TypeError, ValueError):
        return None


def price_mentions(answer):
    patterns = [
        r"\bAED\s*([\d,]+(?:\.\d{1,2})?)\b",
        r"\b([\d,]+(?:\.\d{1,2})?)\s*AED\b",
    ]

    found = set()

    for pattern in patterns:
        for match in re.finditer(
            pattern,
            answer,
            flags=re.IGNORECASE,
        ):
            price = normalized_price(match.group(1))

            if price is not None:
                found.add(price)

    return found


def basic_quality(answer):
    lower_answer = answer.lower()

    if len(answer.strip()) < 12:
        return False

    technical_errors = [
        "as an ai language model",
        "traceback (most recent call last)",
    ]

    return not any(
        error in lower_answer
        for error in technical_errors
    )


def grade(case, answer, products, previous):
    expected = case["expected"]
    problems = []

    # Check 1: Basic answer quality
    if not basic_quality(answer):
        problems.append(
            "Response is empty, too short, "
            "or contains a technical error"
        )

    # Check 2: Number of products
    if "count" in expected:
        expected_count = expected["count"]

        if len(products) != expected_count:
            problems.append(
                f"Expected {expected_count} product cards, "
                f"received {len(products)}"
            )

    # Check 3: Product material
    if "material" in expected:
        if not products:
            problems.append(
                "No products returned for material filter"
            )

        for product in products:
            actual_material = str(
                product.get("material", "")
            ).casefold()

            if actual_material != expected["material"]:
                problems.append(
                    "Incorrect material: "
                    f"{product.get('name')}"
                )

    # Check 4: Maximum price
    if "max_price_exclusive" in expected:
        if not products:
            problems.append(
                "No products returned for price filter"
            )

        for product in products:
            price = normalized_price(
                product.get("price")
            )

            if (
                price is None
                or price >= expected["max_price_exclusive"]
            ):
                problems.append(
                    "Incorrect price filter: "
                    f"{product.get('name')}"
                )

    # Check 5: Product year
    if "year" in expected:
        if not products:
            problems.append(
                "No products returned for year filter"
            )

        for product in products:
            actual_year = str(
                product.get("year")
            )

            if actual_year != str(expected["year"]):
                problems.append(
                    "Incorrect year: "
                    f"{product.get('name')}"
                )

    # Check 6: Named product retrieval
    if "name_contains" in expected:
        name_found = any(
            expected["name_contains"]
            in str(product.get("name", "")).lower()
            for product in products
        )

        if not name_found:
            problems.append(
                "Requested named product missing from cards"
            )

    # Check 7: Conversation memory
    if "reference_index" in expected:
        index = expected["reference_index"]

        if len(previous) <= index:
            problems.append(
                "Previous turn did not return enough "
                "products to check the reference"
            )

        elif (
            len(products) != 1
            or str(products[0].get("product_id"))
            != str(previous[index].get("product_id"))
        ):
            problems.append(
                "Follow-up referred to the wrong product ID"
            )

    # Check 8: Unconfirmed availability
    if expected.get("availability_unconfirmed"):
        unsupported_stock_claim = re.search(
            r"\b(?:in stock|available now|"
            r"currently available|sold out|"
            r"out of stock)\b",
            answer,
            flags=re.IGNORECASE,
        )

        if unsupported_stock_claim:
            problems.append(
                "Potentially unsupported availability "
                "claim; inspect manually"
            )

    # Check 9: Price grounding heuristic
    #
    # Every explicitly stated AED amount should
    # correspond to a returned product card.
    #
    # This DOES NOT catch every hallucination
    # or a price swapped between real products.
    mentioned_prices = price_mentions(answer)

    known_prices = {
        normalized_price(product.get("price"))
        for product in products
    }

    user_mentioned_prices = price_mentions(
        case["message"]
    )

    unknown_prices = sorted(
        mentioned_prices
        - known_prices
        - user_mentioned_prices
    )

    if unknown_prices and products:
        problems.append(
            "Unmatched AED amounts in response: "
            f"{unknown_prices}"
        )

    return problems


def run(args):
    cases = json.loads(
        DATASET.read_text(encoding="utf-8")
    )

    if not isinstance(cases, list):
        raise ValueError(
            "Evaluation dataset must be a JSON list"
        )

    if len(cases) != 100:
        raise ValueError(
            "Expected exactly 100 JSON test cases"
        )

    selected = cases[:args.limit]

    # Every conversation group gets its own session ID.
    sessions = {}

    # Remember the last product cards for follow-up tests.
    previous_products = {}

    rows = []
    started = time.monotonic()

    with httpx.Client(
        timeout=httpx.Timeout(args.timeout),
        trust_env=False,
    ) as client:

        for index, case in enumerate(selected, 1):
            group = case["conversation"]

            payload = {
                "message": case["message"],
            }

            if group in sessions:
                payload["session_id"] = sessions[group]

            problems = []
            response = {}
            status = None

            request_started = time.monotonic()

            try:
                result = client.post(
                    args.url,
                    json=payload,
                )

                status = result.status_code

                if result.status_code != 200:
                    problems.append(
                        f"HTTP {result.status_code}: "
                        f"{result.text[:180]}"
                    )

                else:
                    response = result.json()

                    answer = str(
                        response.get("answer") or ""
                    )

                    products = response.get(
                        "products",
                        [],
                    )

                    if not isinstance(products, list):
                        products = []
                        problems.append(
                            "products is not a list"
                        )

                    problems.extend(
                        grade(
                            case,
                            answer,
                            products,
                            previous_products.get(
                                group,
                                [],
                            ),
                        )
                    )

                    session_id = response.get(
                        "session_id"
                    )

                    if not session_id:
                        problems.append(
                            "Missing session_id"
                        )

                    elif (
                        group in sessions
                        and sessions[group] != session_id
                    ):
                        problems.append(
                            "Session ID changed "
                            "during conversation"
                        )

                    else:
                        sessions[group] = session_id

                    previous_products[group] = products

            except (
                httpx.RequestError,
                ValueError,
            ) as exc:
                problems.append(
                    "Request failed: "
                    f"{type(exc).__name__}: "
                    f"{str(exc)[:140]}"
                )

            duration = round(
                time.monotonic() - request_started,
                2,
            )

            row = {
                "id": case["id"],
                "category": case["category"],
                "conversation": group,
                "message": case["message"],
                "http_status": status,
                "passed": not problems,
                "issues": problems,
                "seconds": duration,
                "answer": response.get("answer", ""),
                "products": response.get("products", []),
            }

            rows.append(row)

            outcome = (
                "PASS"
                if not problems
                else "FAIL"
            )

            print(
                f"[{index:03d}/{len(selected):03d}] "
                f"{outcome} "
                f"{duration:5.2f}s "
                f"{case['id']} | "
                f"{case['message'][:52]}",
                flush=True,
            )

            if problems:
                for problem in problems:
                    print(
                        f"    -> {problem}",
                        flush=True,
                    )

            if (
                index < len(selected)
                and args.delay > 0
            ):
                time.sleep(args.delay)

    total = len(rows)

    successes = sum(
        row["passed"]
        for row in rows
    )

    categories = sorted(
        set(
            row["category"]
            for row in rows
        )
    )

    score_by_category = {}

    for category in categories:
        subset = [
            row
            for row in rows
            if row["category"] == category
        ]

        category_passes = sum(
            row["passed"]
            for row in subset
        )

        score_by_category[category] = {
            "passed": category_passes,
            "total": len(subset),
            "pass_rate_percent": round(
                100 * category_passes / len(subset),
                1,
            ),
        }

    summary = {
        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "total_questions": total,
        "passed": successes,
        "failed": total - successes,
        "automated_pass_rate_percent": (
            round(
                100 * successes / total,
                1,
            )
            if total
            else 0
        ),
        "elapsed_seconds": round(
            time.monotonic() - started,
            2,
        ),
        "by_category": score_by_category,
        "limitations": [
            (
                "Response quality is assessed with "
                "minimal structural checks, "
                "not human judgments."
            ),
            (
                "Price grounding checks only "
                "explicit AED amounts and "
                "returned product cards."
            ),
            (
                "This is NOT a measured full "
                "hallucination rate; "
                "unsupported claims can be missed."
            ),
            (
                "Retrieval checks are constraint-based, "
                "not recall or ranked relevance "
                "against labeled ground truth."
            ),
        ],
    }

    # Save both report formats.
    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    stamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%d_%H%M%S")

    report_path = (
        REPORT_DIR
        / f"evaluation_{stamp}.json"
    )

    csv_path = (
        REPORT_DIR
        / f"evaluation_{stamp}.csv"
    )

    report_path.write_text(
        json.dumps(
            {
                "summary": summary,
                "results": rows,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    with csv_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as handle:

        fields = [
            "id",
            "category",
            "message",
            "http_status",
            "passed",
            "seconds",
            "issues",
        ]

        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(
                            row[key],
                            ensure_ascii=False,
                        )
                        if key == "issues"
                        else row[key]
                    )
                    for key in fields
                }
            )

    # Print the final summary.
    print(
        "\n====== BIDDIX EVALUATION RESULTS ======"
    )

    print(
        f"Questions: {total} | "
        f"Passed: {successes} | "
        f"Failed: {total - successes}"
    )

    print(
        "Automated pass rate: "
        f"{summary['automated_pass_rate_percent']}%"
    )

    for name, score in score_by_category.items():
        print(
            f"{name:14} "
            f"{score['passed']}/{score['total']} "
            f"({score['pass_rate_percent']}%)"
        )

    print(
        f"JSON report: {report_path}"
    )

    print(
        f"CSV report:  {csv_path}"
    )

    print(
        "NOTE: These checks DO NOT establish a "
        "full hallucination or response-quality score."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Biddix 100-question evaluation"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Run the first N cases (1 to 100)",
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Seconds between requests",
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=90.0,
        help="Timeout per request",
    )

    parser.add_argument(
        "--url",
        default=API,
        help="Biddix chat endpoint",
    )

    args = parser.parse_args()

    if not 1 <= args.limit <= 100:
        parser.error(
            "--limit must be between 1 and 100"
        )

    if args.delay < 0:
        parser.error(
            "--delay must be >= 0"
        )

    if args.timeout <= 0:
        parser.error(
            "--timeout must be > 0"
        )

    if not DATASET.exists():
        parser.error(
            "Dataset missing: "
            "tests/evaluation_dataset.json"
        )

    run(args)


if __name__ == "__main__":
    main()
