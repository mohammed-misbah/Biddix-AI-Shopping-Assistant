import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Literal, TypeVar

import redis

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


QueryMode = Literal[
    "catalog",
    "knowledge",
    "mixed",
]


@dataclass
class CatalogFilters:
    """
    Persistent structured shopping filters.

    These survive follow-up messages such as:

    "Show me coins above 500 AED."
    "Only silver ones."

    The second message changes material,
    but keeps min_price=500.
    """

    year: int | None = None

    min_price: float | None = None
    min_price_inclusive: bool = False

    max_price: float | None = None
    max_price_inclusive: bool = True

    material: str | None = None

    in_stock: bool = False

    # Generic future catalog facets:
    #
    # country
    # region
    # theme
    # era
    # category
    # denomination
    #
    # No specific country is hard-coded here.
    facets: dict[str, set[str]] = field(
        default_factory=dict
    )


@dataclass
class ConversationTurn:

    role: Literal[
        "user",
        "assistant",
    ]

    content: str


@dataclass
class ConversationSession:

    turns: list[
        ConversationTurn
    ] = field(
        default_factory=list
    )

    last_mode: QueryMode | None = None

    # -------------------------------------------------
    # IMPORTANT PRODUCT MEMORY
    # -------------------------------------------------

    # Last FULL result set returned by retrieval.
    #
    # This remains separate from focus/selection.
    last_result_product_ids: list[
        str
    ] = field(
        default_factory=list
    )

    # Products intentionally selected from the
    # previous result set.
    selected_product_ids: list[
        str
    ] = field(
        default_factory=list
    )

    # One currently focused product.
    focused_product_id: str | None = None

    # -------------------------------------------------
    # SEARCH STATE
    # -------------------------------------------------

    active_filters: CatalogFilters = field(
        default_factory=CatalogFilters
    )

    semantic_query: str = ""

    updated_at: datetime = field(
        default_factory=lambda: datetime.now(
            timezone.utc
        )
    )


class ConversationService:

    def __init__(
        self,
    ) -> None:

        # =================================================
        # REDIS CLIENT
        #
        # This remains synchronous because the existing
        # conversation service is synchronous.
        #
        # We are NOT changing conversation behaviour.
        #
        # We only add:
        #
        # - connect timeout
        # - socket timeout
        # - controlled retry
        # - RedisServiceError
        # =================================================

        self.redis = redis.Redis(
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

            # We control retries ourselves.
            retry_on_timeout=False,
        )

        self.session_ttl_seconds = (
            settings.SESSION_TTL_SECONDS
        )

    # =====================================================
    # SAFE REDIS RETRY
    # =====================================================

    def _execute_with_retry(
        self,
        operation_name: str,
        operation: Callable[
            [],
            T,
        ],
    ) -> T:
        """
        Execute one safe Redis conversation operation.

        Retry only:
        - Redis timeout
        - Redis connection problem

        Maximum attempts:

            first attempt
            +
            configured retry count

        With REDIS_RETRY_ATTEMPTS=1:

            attempt 1
            attempt 2

        Then stop.

        No infinite retries.
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

                return operation()

            # -------------------------------------------------
            # TEMPORARY REDIS PROBLEM
            # -------------------------------------------------

            except (
                RedisTimeoutError,
                RedisConnectionError,
            ) as exc:

                if attempt >= total_attempts:

                    logger.error(
                        "Redis session operation failed | "
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
                            "Redis session operation "
                            f"{operation_name} failed."
                        )
                    ) from exc

                logger.warning(
                    "Redis session transient failure | "
                    "operation=%s | "
                    "attempt=%s/%s | "
                    "retrying=true | "
                    "error_type=%s",
                    operation_name,
                    attempt,
                    total_attempts,
                    type(exc).__name__,
                )

                time.sleep(
                    settings
                    .REDIS_RETRY_DELAY_SECONDS
                )

            # -------------------------------------------------
            # OTHER REDIS ERROR
            #
            # Unknown Redis errors are NOT retried.
            # -------------------------------------------------

            except RedisError as exc:

                logger.exception(
                    "Redis session operation failed | "
                    "operation=%s | "
                    "retrying=false | "
                    "error_type=%s",
                    operation_name,
                    type(exc).__name__,
                )

                raise RedisServiceError(
                    (
                        "Redis session operation "
                        f"{operation_name} failed."
                    )
                ) from exc

        # Safety fallback.
        # This should not normally be reached.

        raise RedisServiceError(
            (
                "Redis session operation "
                f"{operation_name} failed."
            )
        )

    # =====================================================
    # SESSION
    # =====================================================

    def get_session(
        self,
        session_id: str,
    ) -> ConversationSession:

        key = self._session_key(
            session_id
        )

        # GET is read-only.
        # Safe to retry once.

        raw = self._execute_with_retry(
            operation_name="GET",
            operation=lambda:
                self.redis.get(
                    key
                ),
        )

        if raw is None:

            return ConversationSession()

        try:

            data = json.loads(
                raw
            )

            return self._deserialize_session(
                data
            )

        except (
            json.JSONDecodeError,
            TypeError,
            ValueError,
            KeyError,
        ):

            # Keep your original behaviour:
            #
            # Bad/corrupt session data should not crash
            # the customer's chat.
            #
            # Start with a clean session instead.

            return ConversationSession()

    def _save_session(
        self,
        session_id: str,
        session: ConversationSession,
    ) -> None:

        key = self._session_key(
            session_id
        )

        data = self._serialize_session(
            session
        )

        payload = json.dumps(
            data,
            ensure_ascii=False,
        )

        # SET is safe for this session use case.
        #
        # Retrying writes the same session JSON
        # back to the same Redis key.

        self._execute_with_retry(
            operation_name="SET",
            operation=lambda:
                self.redis.set(
                    key,
                    payload,
                    ex=(
                        self
                        .session_ttl_seconds
                    ),
                ),
        )

    # =====================================================
    # HISTORY
    # =====================================================

    def get_history(
        self,
        session_id: str,
    ) -> list[
        ConversationTurn
    ]:

        session = self.get_session(
            session_id
        )

        return session.turns[
            -settings.MAX_TURNS:
        ]

    def get_last_mode(
        self,
        session_id: str,
    ) -> QueryMode | None:

        return self.get_session(
            session_id
        ).last_mode

    # =====================================================
    # PRODUCT STATE
    # =====================================================

    def get_last_result_product_ids(
        self,
        session_id: str,
    ) -> list[str]:

        return list(
            self.get_session(
                session_id
            ).last_result_product_ids
        )

    def get_selected_product_ids(
        self,
        session_id: str,
    ) -> list[str]:

        return list(
            self.get_session(
                session_id
            ).selected_product_ids
        )

    def get_focused_product_id(
        self,
        session_id: str,
    ) -> str | None:

        return self.get_session(
            session_id
        ).focused_product_id

    def set_last_results(
        self,
        session_id: str,
        products: list[dict],
    ) -> None:
        """
        Save a new FULL search result set.

        A fresh/refined search resets any old
        selection/focused-product state.
        """

        session = self.get_session(
            session_id
        )

        session.last_result_product_ids = [
            str(
                product["product_id"]
            )
            for product
            in products
            if product.get(
                "product_id"
            )
        ]

        # New search = old selection
        # is no longer active.

        session.selected_product_ids = []

        session.focused_product_id = None

        self._touch(
            session
        )

        self._save_session(
            session_id,
            session,
        )

    def set_selected_products(
        self,
        session_id: str,
        products: list[dict],
    ) -> None:

        session = self.get_session(
            session_id
        )

        session.selected_product_ids = [
            str(
                product["product_id"]
            )
            for product
            in products
            if product.get(
                "product_id"
            )
        ]

        # Keep your original behaviour:
        #
        # exactly one selection naturally
        # becomes focused.

        if len(
            session.selected_product_ids
        ) == 1:

            session.focused_product_id = (
                session
                .selected_product_ids[0]
            )

        self._touch(
            session
        )

        self._save_session(
            session_id,
            session,
        )

    def set_focused_product(
        self,
        session_id: str,
        product: dict | None,
    ) -> None:

        session = self.get_session(
            session_id
        )

        if (
            product is not None
            and product.get(
                "product_id"
            )
        ):

            product_id = str(
                product["product_id"]
            )

            session.focused_product_id = (
                product_id
            )

            # Keep your original behaviour:
            #
            # focusing one product also makes that
            # current selected product list contain
            # that one product.
            #
            # IMPORTANT:
            # last_result_product_ids is NOT touched.

            session.selected_product_ids = [
                product_id
            ]

        else:

            session.focused_product_id = None

        self._touch(
            session
        )

        self._save_session(
            session_id,
            session,
        )

    def clear_selection(
        self,
        session_id: str,
    ) -> None:

        session = self.get_session(
            session_id
        )

        session.selected_product_ids = []

        session.focused_product_id = None

        self._touch(
            session
        )

        self._save_session(
            session_id,
            session,
        )

    # =====================================================
    # SEARCH STATE
    # =====================================================

    def get_active_filters(
        self,
        session_id: str,
    ) -> CatalogFilters:

        source = self.get_session(
            session_id
        ).active_filters

        # Return a copy so the caller cannot
        # accidentally mutate stored state.

        return CatalogFilters(
            year=source.year,

            min_price=(
                source.min_price
            ),

            min_price_inclusive=(
                source
                .min_price_inclusive
            ),

            max_price=(
                source.max_price
            ),

            max_price_inclusive=(
                source
                .max_price_inclusive
            ),

            material=(
                source.material
            ),

            in_stock=(
                source.in_stock
            ),

            facets={
                key: set(values)
                for key, values
                in source.facets.items()
            },
        )

    def get_semantic_query(
        self,
        session_id: str,
    ) -> str:

        return self.get_session(
            session_id
        ).semantic_query

    def set_search_state(
        self,
        session_id: str,
        products: list[dict],
        filters: CatalogFilters,
        semantic_query: str,
    ) -> None:

        session = self.get_session(
            session_id
        )

        session.active_filters = filters

        session.semantic_query = (
            semantic_query.strip()
        )

        # New retrieval results become the
        # complete previous-result set.

        session.last_result_product_ids = [
            str(
                product["product_id"]
            )
            for product
            in products
            if product.get(
                "product_id"
            )
        ]

        # Fresh/refined search invalidates
        # old focus/selection.

        session.selected_product_ids = []

        session.focused_product_id = None

        self._touch(
            session
        )

        self._save_session(
            session_id,
            session,
        )

    # =====================================================
    # MESSAGES
    # =====================================================

    def add_user_message(
        self,
        session_id: str,
        message: str,
        mode: QueryMode,
    ) -> None:

        session = self.get_session(
            session_id
        )

        session.turns.append(
            ConversationTurn(
                role="user",
                content=message,
            )
        )

        session.last_mode = mode

        self._trim(
            session
        )

        self._touch(
            session
        )

        self._save_session(
            session_id,
            session,
        )

    def add_assistant_message(
        self,
        session_id: str,
        message: str,
    ) -> None:

        session = self.get_session(
            session_id
        )

        session.turns.append(
            ConversationTurn(
                role="assistant",
                content=message,
            )
        )

        self._trim(
            session
        )

        self._touch(
            session
        )

        self._save_session(
            session_id,
            session,
        )

    # =====================================================
    # REDIS SERIALIZATION
    # =====================================================

    @staticmethod
    def _serialize_session(
        session: ConversationSession,
    ) -> dict:

        return {
            "turns": [
                {
                    "role":
                        turn.role,

                    "content":
                        turn.content,
                }
                for turn
                in session.turns
            ],

            "last_mode":
                session.last_mode,

            "last_result_product_ids":
                session
                .last_result_product_ids,

            "selected_product_ids":
                session
                .selected_product_ids,

            "focused_product_id":
                session
                .focused_product_id,

            "active_filters": {

                "year":
                    session
                    .active_filters
                    .year,

                "min_price":
                    session
                    .active_filters
                    .min_price,

                "min_price_inclusive":
                    (
                        session
                        .active_filters
                        .min_price_inclusive
                    ),

                "max_price":
                    session
                    .active_filters
                    .max_price,

                "max_price_inclusive":
                    (
                        session
                        .active_filters
                        .max_price_inclusive
                    ),

                "material":
                    session
                    .active_filters
                    .material,

                "in_stock":
                    session
                    .active_filters
                    .in_stock,

                "facets": {
                    key: list(values)
                    for key, values
                    in (
                        session
                        .active_filters
                        .facets
                        .items()
                    )
                },
            },

            "semantic_query":
                session.semantic_query,

            "updated_at":
                session
                .updated_at
                .isoformat(),
        }

    @staticmethod
    def _deserialize_session(
        data: dict,
    ) -> ConversationSession:

        raw_filters = data.get(
            "active_filters",
            {},
        )

        filters = CatalogFilters(

            year=raw_filters.get(
                "year"
            ),

            min_price=raw_filters.get(
                "min_price"
            ),

            min_price_inclusive=(
                raw_filters.get(
                    "min_price_inclusive",
                    False,
                )
            ),

            max_price=raw_filters.get(
                "max_price"
            ),

            max_price_inclusive=(
                raw_filters.get(
                    "max_price_inclusive",
                    True,
                )
            ),

            material=raw_filters.get(
                "material"
            ),

            in_stock=raw_filters.get(
                "in_stock",
                False,
            ),

            facets={
                key: set(values)
                for key, values
                in raw_filters.get(
                    "facets",
                    {},
                ).items()
            },
        )

        turns = [
            ConversationTurn(
                role=turn["role"],
                content=turn["content"],
            )

            for turn
            in data.get(
                "turns",
                [],
            )

            if (
                isinstance(
                    turn,
                    dict,
                )
                and turn.get(
                    "role"
                )
                in {
                    "user",
                    "assistant",
                }
                and isinstance(
                    turn.get(
                        "content"
                    ),
                    str,
                )
            )
        ]

        updated_at_raw = data.get(
            "updated_at"
        )

        if isinstance(
            updated_at_raw,
            str,
        ):

            try:

                updated_at = (
                    datetime
                    .fromisoformat(
                        updated_at_raw
                    )
                )

            except ValueError:

                updated_at = (
                    datetime.now(
                        timezone.utc
                    )
                )

        else:

            updated_at = (
                datetime.now(
                    timezone.utc
                )
            )

        return ConversationSession(

            turns=turns,

            last_mode=data.get(
                "last_mode"
            ),

            last_result_product_ids=list(
                data.get(
                    "last_result_product_ids",
                    [],
                )
            ),

            selected_product_ids=list(
                data.get(
                    "selected_product_ids",
                    [],
                )
            ),

            focused_product_id=data.get(
                "focused_product_id"
            ),

            active_filters=filters,

            semantic_query=data.get(
                "semantic_query",
                "",
            ),

            updated_at=updated_at,
        )

    # =====================================================
    # REDIS HELPERS
    # =====================================================

    @staticmethod
    def _session_key(
        session_id: str,
    ) -> str:

        return (
            f"biddix:session:"
            f"{session_id}"
        )

    def session_exists(
        self,
        session_id: str,
    ) -> bool:

        result = self._execute_with_retry(
            operation_name="EXISTS",
            operation=lambda:
                self.redis.exists(
                    self._session_key(
                        session_id
                    )
                ),
        )

        return bool(
            result
        )

    def get_session_ttl(
        self,
        session_id: str,
    ) -> int:

        result = self._execute_with_retry(
            operation_name="TTL",
            operation=lambda:
                self.redis.ttl(
                    self._session_key(
                        session_id
                    )
                ),
        )

        return int(
            result
        )

    def delete_session(
        self,
        session_id: str,
    ) -> bool:

        # DELETE is safe to retry.
        #
        # If attempt 1 deleted the key but its reply
        # was lost, attempt 2 simply sees no key.

        result = self._execute_with_retry(
            operation_name="DELETE",
            operation=lambda:
                self.redis.delete(
                    self._session_key(
                        session_id
                    )
                ),
        )

        return bool(
            result
        )

    def ping(
        self,
    ) -> bool:
        """
        Health check.

        Startup code expects a simple True/False result,
        so keep that behaviour.
        """

        try:

            result = self._execute_with_retry(
                operation_name="PING",
                operation=lambda:
                    self.redis.ping(),
            )

            return bool(
                result
            )

        except RedisServiceError:

            return False

    def close(
        self,
    ) -> None:
        """
        Close Redis cleanly.

        Shutdown should not fail just because Redis
        disappeared before the application closed.
        """

        try:

            self.redis.close()

        except RedisError as exc:

            logger.warning(
                "Redis session close failed: %s",
                exc,
            )

    # =====================================================
    # INTERNAL
    # =====================================================

    @staticmethod
    def _trim(
        session: ConversationSession,
    ) -> None:

        max_turns = max(
            settings.MAX_TURNS,
            2,
        )

        if len(
            session.turns
        ) > max_turns:

            session.turns = (
                session.turns[
                    -max_turns:
                ]
            )

    @staticmethod
    def _touch(
        session: ConversationSession,
    ) -> None:

        session.updated_at = (
            datetime.now(
                timezone.utc
            )
        )


conversation_service = ConversationService()