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

    year: int | None = None

    min_price: float | None = None
    min_price_inclusive: bool = False

    max_price: float | None = None
    max_price_inclusive: bool = True

    material: str | None = None

    in_stock: bool = False

    # Generic catalog facets.
    #
    # Possible examples:
    # country
    # region
    # category
    # theme
    # era
    # period
    # denomination
    #
    # Nothing is country-specific.
    facets: dict[
        str,
        set[str],
    ] = field(
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

    # Products that the customer is
    # currently talking about.
    active_product_ids: list[str] = field(
        default_factory=list
    )

    # Search constraints persist across
    # refinement turns.
    active_filters: CatalogFilters = field(
        default_factory=CatalogFilters
    )

    # Semantic meaning of the current search.
    #
    # Example:
    # "meaningful historical collector coin"
    #
    # Geography/themes remain semantic unless
    # the catalog supplies structured facets.
    semantic_query: str = ""

    updated_at: datetime = field(
        default_factory=lambda: (
            datetime.now(
                timezone.utc
            )
        )
    )


class ConversationService:

    def __init__(self):

        self.sessions: dict[
            str,
            ConversationSession,
        ] = {}

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

    def get_active_product_ids(
        self,
        session_id: str,
    ) -> list[str]:

        return list(
            self.get_session(
                session_id
            ).active_product_ids
        )

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
                field: set(values)
                for field, values
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

        session.active_product_ids = [
            str(
                product["product_id"]
            )
            for product in products
            if product.get(
                "product_id"
            )
        ]

        session.active_filters = filters

        session.semantic_query = (
            semantic_query.strip()
        )

        self._touch(
            session
        )

    def set_active_products(
        self,
        session_id: str,
        products: list[dict],
    ) -> None:

        session = self.get_session(
            session_id
        )

        session.active_product_ids = [
            str(
                product["product_id"]
            )
            for product in products
            if product.get(
                "product_id"
            )
        ]

        self._touch(
            session
        )

    def clear_active_products(
        self,
        session_id: str,
    ) -> None:

        session = self.get_session(
            session_id
        )

        session.active_product_ids = []

        self._touch(
            session
        )

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

        self._trim(session)
        self._touch(session)

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

        self._trim(session)
        self._touch(session)

    @staticmethod
    def _trim(
        session: ConversationSession,
    ) -> None:

        max_turns = max(
            settings.MAX_TURNS,
            2,
        )

        if len(session.turns) > max_turns:

            session.turns = (
                session.turns[
                    -max_turns:
                ]
            )

    @staticmethod
    def _touch(
        session: ConversationSession,
    ) -> None:

        session.updated_at = datetime.now(
            timezone.utc
        )


conversation_service = ConversationService()