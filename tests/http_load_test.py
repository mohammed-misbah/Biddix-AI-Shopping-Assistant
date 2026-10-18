import asyncio
import statistics
import time

import httpx


URL = "http://127.0.0.1:8000/api/v1/chat"

USERS = 100


async def one_user(
    user_number: int,
):

    started = time.perf_counter()

    async with httpx.AsyncClient(
        timeout=30.0,
    ) as client:

        response = await client.post(
            URL,
            json={
                "message": (
                    "Find silver under AED 500"
                )
            },
        )

    elapsed = (
        time.perf_counter()
        - started
    )

    return {
        "user": user_number,
        "status": response.status_code,
        "seconds": elapsed,
    }


async def main():

    print(
        f"\nStarting {USERS} "
        f"simultaneous HTTP users...\n"
    )

    test_started = time.perf_counter()

    results = await asyncio.gather(
        *[
            one_user(user_number)
            for user_number
            in range(1, USERS + 1)
        ],
        return_exceptions=True,
    )

    total_time = (
        time.perf_counter()
        - test_started
    )

    successful = []
    failed = []

    for result in results:

        if isinstance(
            result,
            Exception,
        ):
            failed.append(
                str(result)
            )
            continue

        if result["status"] == 200:
            successful.append(
                result
            )
        else:
            failed.append(
                result
            )

    latencies = [
        result["seconds"]
        for result
        in successful
    ]

    print(
        "=============================="
    )

    print(
        "HTTP LOAD TEST RESULT"
    )

    print(
        "=============================="
    )

    print(
        f"Users: {USERS}"
    )

    print(
        f"Successful: {len(successful)}"
    )

    print(
        f"Failed: {len(failed)}"
    )

    print(
        f"Total test time: "
        f"{total_time:.3f}s"
    )

    if latencies:

        print(
            f"Fastest response: "
            f"{min(latencies):.3f}s"
        )

        print(
            f"Slowest response: "
            f"{max(latencies):.3f}s"
        )

        print(
            f"Average response: "
            f"{statistics.mean(latencies):.3f}s"
        )

        print(
            f"Median response: "
            f"{statistics.median(latencies):.3f}s"
        )

    if failed:

        print(
            "\nFailed requests:"
        )

        for failure in failed:
            print(
                failure
            )


if __name__ == "__main__":
    asyncio.run(
        main()
    )