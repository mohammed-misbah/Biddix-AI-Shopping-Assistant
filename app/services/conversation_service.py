from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

from app.core.config import settings


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

    turns: list[ConversationTurn] = field(
        default_factory=list
    )

    last_mode: QueryMode | None = None

    # -------------------------------------------------
    # IMPORTANT PRODUCT MEMORY
    # -------------------------------------------------

    # Last FULL result set returned by retrieval.
    #
    # Example:
    #
    # Sheikh Zayed
    # Expo Dubai
    # Arabian Leopard
    #
    # This is preserved even if later the customer
    # focuses on Arabian Leopard.
    last_result_product_ids: list[str] = field(
        default_factory=list
    )

    # Products intentionally selected from the
    # last result set.
    #
    # Example:
    #
    # "Choose two"
    #
    # -> [Heritage, Emirates Map]
    selected_product_ids: list[str] = field(
        default_factory=list
    )

    # One currently focused product.
    #
    # Example:
    #
    # "Which is cheapest?"
    # -> Arabian Leopard
    #
    # "Tell me more about that coin"
    # -> use this product.
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

    def __init__(self) -> None:

        self.sessions: dict[
            str,
            ConversationSession,
        ] = {}

    # =================================================
    # SESSION
    # =================================================

    def get_session(
        self,
        session_id: str,
    ) -> ConversationSession:

        session = self.sessions.get(
            session_id
        )

        if session is None:

            session = ConversationSession()

            self.sessions[
                session_id
            ] = session

        return session

    # =================================================
    # HISTORY
    # =================================================

    def get_history(
        self,
        session_id: str,
    ) -> list[ConversationTurn]:

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

    # =================================================
    # PRODUCT STATE
    # =================================================

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
            for product in products
            if product.get(
                "product_id"
            )
        ]

        # New search = old selection is no longer active.
        session.selected_product_ids = []
        session.focused_product_id = None

        self._touch(
            session
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
            for product in products
            if product.get(
                "product_id"
            )
        ]

        # If exactly one product is selected,
        # it naturally becomes focused.
        if len(
            session.selected_product_ids
        ) == 1:

            session.focused_product_id = (
                session.selected_product_ids[0]
            )

        self._touch(
            session
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

            session.selected_product_ids = [
                product_id
            ]

        else:

            session.focused_product_id = None

        self._touch(
            session
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

    # =================================================
    # SEARCH STATE
    # =================================================

    def get_active_filters(
        self,
        session_id: str,
    ) -> CatalogFilters:

        source = self.get_session(
            session_id
        ).active_filters

        return CatalogFilters(
            year=source.year,

            min_price=source.min_price,
            min_price_inclusive=(
                source.min_price_inclusive
            ),

            max_price=source.max_price,
            max_price_inclusive=(
                source.max_price_inclusive
            ),

            material=source.material,

            in_stock=source.in_stock,

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

        # New retrieval results.
        session.last_result_product_ids = [
            str(
                product["product_id"]
            )
            for product in products
            if product.get(
                "product_id"
            )
        ]

        # Fresh/refined search invalidates old
        # focused/selected products.
        session.selected_product_ids = []
        session.focused_product_id = None

        self._touch(
            session
        )

    # =================================================
    # MESSAGES
    # =================================================

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

    # =================================================
    # INTERNAL
    # =================================================

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