import asyncio
import json
import logging

from collections.abc import (
    Awaitable,
    Callable,
)

from typing import (
    Any,
    TypeVar,
)

from redis.asyncio import Redis

from redis.exceptions import (
    ConnectionError as RedisConnectionError,
    RedisError,
    TimeoutError as RedisTimeoutError,
)

from app.core.config import settings

from app.core.exceptions import (
    RedisServiceError,
)


logger = logging.getLogger(__name__)


T = TypeVar("T")


class RedisService:

    def __init__(
        self,
    ) -> None:

        # =================================================
        # REDIS CLIENT
        #
        # IMPORTANT:
        #
        # socket_connect_timeout:
        # How long we wait to CONNECT to Redis.
        #
        # socket_timeout:
        # How long we wait for Redis to ANSWER.
        #
        # retry_on_timeout=False:
        # Redis library itself must NOT secretly retry.
        #
        # We control retries ourselves below.
        # =================================================

        self.client = Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=settings.REDIS_DB,
            decode_responses=True,

            socket_connect_timeout=(
                settings
                .REDIS_CONNECT_TIMEOUT_SECONDS
            ),

            socket_timeout=(
                settings
                .REDIS_SOCKET_TIMEOUT_SECONDS
            ),

            retry_on_timeout=False,
        )

    # SAFE RETRY HELPER

    async def _execute_with_retry(
        self,
        operation_name: str,
        operation: Callable[
            [],
            Awaitable[T],
        ],
    ) -> T:
        """
        Execute one SAFE Redis operation.

        Safe transient failures:
        - timeout
        - connection problem

        Maximum:
        initial attempt + configured retries.

        Example with REDIS_RETRY_ATTEMPTS=1:

            attempt 1
                ↓
            temporary connection failure
                ↓
            wait 0.2 seconds
                ↓
            attempt 2
                ↓
            success OR clean failure

        There is never an infinite retry loop.
        """

        retry_count = max(
            settings.REDIS_RETRY_ATTEMPTS,
            0,
        )

        total_attempts = (
            retry_count
            + 1
        )

        for attempt in range(
            1,
            total_attempts + 1,
        ):

            try:

                return await operation()

            # ---------------------------------------------
            # TEMPORARY / TRANSIENT PROBLEMS
            #
            # These are safe to retry for commands
            # passed through this helper.
            # ---------------------------------------------

            except (
                RedisTimeoutError,
                RedisConnectionError,
            ) as exc:

                # No attempts left.
                if attempt >= total_attempts:

                    logger.error(
                        "Redis operation failed | "
                        "operation=%s | "
                        "attempt=%s/%s | "
                        "error_type=%s",
                        operation_name,
                        attempt,
                        total_attempts,
                        type(exc).__name__,
                    )

                    raise RedisServiceError(
                        (
                            "Redis operation "
                            f"{operation_name} failed."
                        )
                    ) from exc

                # One controlled retry.
                logger.warning(
                    "Redis transient failure | "
                    "operation=%s | "
                    "attempt=%s/%s | "
                    "retrying=true | "
                    "error_type=%s",
                    operation_name,
                    attempt,
                    total_attempts,
                    type(exc).__name__,
                )

                await asyncio.sleep(
                    settings
                    .REDIS_RETRY_DELAY_SECONDS
                )

            # ---------------------------------------------
            # OTHER REDIS ERRORS
            #
            # Do NOT retry unknown Redis errors.
            # ---------------------------------------------

            except RedisError as exc:

                logger.exception(
                    "Redis operation failed | "
                    "operation=%s | "
                    "retrying=false | "
                    "error_type=%s",
                    operation_name,
                    type(exc).__name__,
                )

                raise RedisServiceError(
                    (
                        "Redis operation "
                        f"{operation_name} failed."
                    )
                ) from exc

        # This should never normally be reached.
        raise RedisServiceError(
            (
                "Redis operation "
                f"{operation_name} failed."
            )
        )

    # HEALTH CHECK

    async def ping(
        self,
    ) -> bool:
        """
        Check whether Redis / Memurai is reachable.

        PING is safe to retry because it does not
        modify Redis data.
        """

        try:

            result = (
                await self._execute_with_retry(
                    operation_name="PING",
                    operation=(
                        lambda:
                            self.client.ping()
                    ),
                )
            )

            return bool(
                result
            )

        except RedisServiceError:

            return False

    # GET JSON

    async def get_json(
        self,
        key: str,
    ) -> dict[str, Any] | None:
        """
        Read one JSON object from Redis.

        GET is safe to retry because it only reads data.
        """

        try:

            value = (
                await self._execute_with_retry(
                    operation_name="GET",
                    operation=(
                        lambda:
                            self.client.get(
                                key
                            )
                    ),
                )
            )

            if value is None:
                return None

            data = json.loads(
                value
            )

            if not isinstance(
                data,
                dict,
            ):

                logger.warning(
                    "Redis key %s did not contain "
                    "a JSON object.",
                    key,
                )

                return None

            return data

        except json.JSONDecodeError as exc:

            logger.error(
                "Invalid JSON stored in Redis "
                "key %s: %s",
                key,
                exc,
            )

            return None

    # SET JSON WITH TTL

    async def set_json(
        self,
        key: str,
        value: dict[str, Any],
        ttl_seconds: int | None = None,
    ) -> None:
        """
        Store a JSON object in Redis.

        SET is safe for this use case because retrying
        writes the same value to the same key again.
        """

        try:

            payload = json.dumps(
                value,
                ensure_ascii=False,
                separators=(",", ":"),
            )

        except (
            TypeError,
            ValueError,
        ) as exc:

            logger.exception(
                "Failed to serialize Redis "
                "value for key %s: %s",
                key,
                exc,
            )

            raise

        if ttl_seconds is not None:

            await self._execute_with_retry(
                operation_name="SET",
                operation=(
                    lambda:
                        self.client.set(
                            key,
                            payload,
                            ex=ttl_seconds,
                        )
                ),
            )

        else:

            await self._execute_with_retry(
                operation_name="SET",
                operation=(
                    lambda:
                        self.client.set(
                            key,
                            payload,
                        )
                ),
            )

    # DELETE

    async def delete(
        self,
        key: str,
    ) -> bool:
        """
        Delete a Redis key.

        DELETE is safe to retry.

        If the first DELETE worked but the reply was lost,
        the second DELETE simply finds no key.
        """

        deleted = (
            await self._execute_with_retry(
                operation_name="DELETE",
                operation=(
                    lambda:
                        self.client.delete(
                            key
                        )
                ),
            )
        )

        return deleted > 0

    # EXISTS

    async def exists(
        self,
        key: str,
    ) -> bool:
        """
        Check whether a key exists.

        Read-only operation, so retry is safe.
        """

        result = (
            await self._execute_with_retry(
                operation_name="EXISTS",
                operation=(
                    lambda:
                        self.client.exists(
                            key
                        )
                ),
            )
        )

        return result > 0

    # TTL

    async def ttl(
        self,
        key: str,
    ) -> int:
        """
        Return remaining TTL in seconds.

        Redis can return:

        -1 -> key exists but has no expiry
        -2 -> key does not exist

        Read-only operation, so retry is safe.
        """

        result = (
            await self._execute_with_retry(
                operation_name="TTL",
                operation=(
                    lambda:
                        self.client.ttl(
                            key
                        )
                ),
            )
        )

        return int(
            result
        )

    # REFRESH TTL

    async def expire(
        self,
        key: str,
        ttl_seconds: int,
    ) -> bool:
        """
        Refresh expiry for an existing key.

        Retrying EXPIRE with the same TTL is safe
        for this application.
        """

        result = (
            await self._execute_with_retry(
                operation_name="EXPIRE",
                operation=(
                    lambda:
                        self.client.expire(
                            key,
                            ttl_seconds,
                        )
                ),
            )
        )

        return bool(
            result
        )

    # RATE LIMIT

    async def check_rate_limit(
        self,
        identifier: str,
        max_requests: int,
        window_seconds: int,
    ) -> tuple[
        bool,
        int,
        int,
    ]:
        """
        Check whether one identifier can make another
        request.

        IMPORTANT:

        This operation is intentionally NOT retried.

        Why?

        The Lua script performs INCR.

        Example:

            request count = 5

            INCR succeeds
            count becomes 6

            network reply is lost

            if we automatically retry:
            INCR runs again
            count becomes 7

        One customer request would then be counted twice.

        So rate-limit mutation receives Redis timeout
        protection, but NO automatic retry.
        """

        key = (
            "biddix:rate_limit:"
            f"{identifier}"
        )

        script = """
        local current = redis.call('INCR', KEYS[1])

        if current == 1 then
            redis.call(
                'EXPIRE',
                KEYS[1],
                ARGV[1]
            )
        end

        local ttl = redis.call(
            'TTL',
            KEYS[1]
        )

        return {
            current,
            ttl
        }
        """

        try:

            result = (
                await self.client.eval(
                    script,
                    1,
                    key,
                    window_seconds,
                )
            )

            current_count = int(
                result[0]
            )

            ttl = int(
                result[1]
            )

            if ttl < 0:

                ttl = (
                    window_seconds
                )

            allowed = (
                current_count
                <= max_requests
            )

            return (
                allowed,
                current_count,
                ttl,
            )

        except (
            RedisTimeoutError,
            RedisConnectionError,
        ) as exc:

            logger.error(
                "Redis rate-limit operation failed | "
                "identifier=%s | "
                "retrying=false | "
                "error_type=%s",
                identifier,
                type(exc).__name__,
            )

            raise RedisServiceError(
                "Redis rate-limit operation failed."
            ) from exc

        except RedisError as exc:

            logger.exception(
                "Redis rate-limit operation failed | "
                "identifier=%s | "
                "retrying=false | "
                "error_type=%s",
                identifier,
                type(exc).__name__,
            )

            raise RedisServiceError(
                "Redis rate-limit operation failed."
            ) from exc

    # CLOSE CONNECTION

    async def close(
        self,
    ) -> None:
        """
        Close Redis connection cleanly when FastAPI
        shuts down.

        Shutdown does not need retry behaviour.
        """

        try:

            await self.client.aclose()

        except RedisError as exc:

            logger.warning(
                "Redis close failed: %s",
                exc,
            )


redis_service = RedisService()
