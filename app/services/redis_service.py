import json
import logging
from typing import Any

from redis.asyncio import Redis
from redis.exceptions import RedisError

from app.core.config import settings


logger = logging.getLogger(__name__)


class RedisService:

    def __init__(self) -> None:

        self.client = Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=settings.REDIS_DB,
            decode_responses=True,
        )

    # HEALTH CHECK

    async def ping(self) -> bool:
        """
        Check whether Redis / Memurai is reachable.
        """

        try:

            result = await self.client.ping()

            return bool(result)

        except RedisError as exc:

            logger.exception(
                "Redis ping failed: %s",
                exc,
            )

            return False

    # GET JSON

    async def get_json(
        self,
        key: str,
    ) -> dict[str, Any] | None:
        """
        Read one JSON object from Redis.

        Returns:
            dict -> when key exists
            None -> when key does not exist
        """

        try:

            value = await self.client.get(
                key
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
                    "Redis key %s did not contain a JSON object.",
                    key,
                )

                return None

            return data

        except json.JSONDecodeError as exc:

            logger.error(
                "Invalid JSON stored in Redis key %s: %s",
                key,
                exc,
            )

            return None

        except RedisError as exc:

            logger.exception(
                "Redis GET failed for key %s: %s",
                key,
                exc,
            )

            raise

    # SET JSON WITH TTL

    async def set_json(
        self,
        key: str,
        value: dict[str, Any],
        ttl_seconds: int | None = None,
    ) -> None:
        """
        Store a JSON object in Redis.

        If ttl_seconds is provided,
        Redis automatically expires the key.
        """

        try:

            payload = json.dumps(
                value,
                ensure_ascii=False,
                separators=(",", ":"),
            )

            if ttl_seconds is not None:

                await self.client.set(
                    key,
                    payload,
                    ex=ttl_seconds,
                )

            else:

                await self.client.set(
                    key,
                    payload,
                )

        except (
            TypeError,
            ValueError,
        ) as exc:

            logger.exception(
                "Failed to serialize Redis value for key %s: %s",
                key,
                exc,
            )

            raise

        except RedisError as exc:

            logger.exception(
                "Redis SET failed for key %s: %s",
                key,
                exc,
            )

            raise

    # DELETE

    async def delete(
        self,
        key: str,
    ) -> bool:
        """
        Delete a Redis key.

        Returns True if a key was deleted.
        """

        try:

            deleted = await self.client.delete(
                key
            )

            return deleted > 0

        except RedisError as exc:

            logger.exception(
                "Redis DELETE failed for key %s: %s",
                key,
                exc,
            )

            raise

    # EXISTS

    async def exists(
        self,
        key: str,
    ) -> bool:
        """
        Check whether a key exists.
        """

        try:

            result = await self.client.exists(
                key
            )

            return result > 0

        except RedisError as exc:

            logger.exception(
                "Redis EXISTS failed for key %s: %s",
                key,
                exc,
            )

            raise

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
        """

        try:

            return await self.client.ttl(
                key
            )

        except RedisError as exc:

            logger.exception(
                "Redis TTL failed for key %s: %s",
                key,
                exc,
            )

            raise

    # REFRESH TTL

    async def expire(
        self,
        key: str,
        ttl_seconds: int,
    ) -> bool:
        """
        Refresh expiry for an existing key.
        """

        try:

            result = await self.client.expire(
                key,
                ttl_seconds,
            )

            return bool(result)

        except RedisError as exc:

            logger.exception(
                "Redis EXPIRE failed for key %s: %s",
                key,
                exc,
            )

            raise

    # CLOSE CONNECTION

    async def close(self) -> None:
        """
        Close Redis connection cleanly
        when FastAPI shuts down.
        """

        try:

            await self.client.aclose()

        except RedisError as exc:

            logger.warning(
                "Redis close failed: %s",
                exc,
            )


redis_service = RedisService()