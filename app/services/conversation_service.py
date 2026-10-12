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

    # Product set currently being discussed.
    last_product_ids: list[str] = field(
        default_factory=list
    )

    # Last real catalog search.
    last_catalog_query: str | None = None

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

            session = (
                ConversationSession()
            )

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

    def get_last_user_message(
        self,
        session_id: str,
    ) -> str | None:

        session = self.get_session(
            session_id
        )

        for turn in reversed(
            session.turns
        ):

            if turn.role == "user":
                return turn.content

        return None

    def get_last_product_ids(
        self,
        session_id: str,
    ) -> list[str]:

        return list(
            self.get_session(
                session_id
            ).last_product_ids
        )

    def get_last_catalog_query(
        self,
        session_id: str,
    ) -> str | None:

        return self.get_session(
            session_id
        ).last_catalog_query

    def set_product_context(
        self,
        session_id: str,
        products: list[dict],
        catalog_query: str,
    ) -> None:

        session = self.get_session(
            session_id
        )

        session.last_product_ids = [
            str(
                product.get(
                    "product_id"
                )
            )
            for product in products
            if product.get(
                "product_id"
            )
        ]

        session.last_catalog_query = (
            catalog_query
        )

        session.updated_at = (
            datetime.now(
                timezone.utc
            )
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

        session.updated_at = (
            datetime.now(
                timezone.utc
            )
        )

        self._trim(
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

        session.updated_at = (
            datetime.now(
                timezone.utc
            )
        )

        self._trim(
            session
        )

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


conversation_service = (
    ConversationService()
)