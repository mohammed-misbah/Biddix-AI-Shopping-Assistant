import logging
import uuid
from typing import Any

from app.services.conversation_service import (
    conversation_service,
)
from app.services.llm_service import (
    llm_service,
)
from app.services.query_router import (
    query_router,
)
from app.services.retrieval_service import (
    retrieval_service,
)


logger = logging.getLogger(__name__)


class ChatService:

    async def chat(
        self,
        message: str,
        session_id: str | None = None,
    ):

        if not session_id:

            session_id = str(
                uuid.uuid4()
            )

        # =================================================
        # Current session state
        # =================================================

        previous_mode = (
            conversation_service
            .get_last_mode(
                session_id
            )
        )

        history = (
            conversation_service
            .get_history(
                session_id
            )
        )

        active_product_ids = (
            conversation_service
            .get_active_product_ids(
                session_id
            )
        )

        active_filters = (
            conversation_service
            .get_active_filters(
                session_id
            )
        )

        previous_semantic_query = (
            conversation_service
            .get_semantic_query(
                session_id
            )
        )

        # =================================================
        # Route
        # =================================================

        route = query_router.route(
            message=message,
            previous_mode=previous_mode,
        )

        logger.info(
            "route mode=%s "
            "follow_up=%s "
            "type=%s "
            "selection=%s",
            route.mode,
            route.is_follow_up,
            route.follow_up_type,
            route.selection_limit,
        )

        products: list[
            dict[str, Any]
        ] = []

        # =================================================
        # Knowledge
        # =================================================

        if route.mode == "knowledge":

            products = []

        # =================================================
        # Catalog / Mixed
        # =================================================

        else:

            # ---------------------------------------------
            # CASE A
            #
            # Existing-product reference.
            #
            # NEVER call FAISS.
            # ---------------------------------------------

            if (
                route.follow_up_type
                == "reuse_products"
            ):

                if active_product_ids:

                    products = (
                        retrieval_service
                        .get_products_by_ids(
                            active_product_ids
                        )
                    )

                else:

                    products = []

            # ---------------------------------------------
            # CASE B
            #
            # Search refinement.
            #
            # Keep old filters.
            # Add/change only what the new message says.
            # ---------------------------------------------

            elif (
                route.follow_up_type
                == "refine_search"
            ):

                filters = (
                    retrieval_service
                    .extract_filters(
                        query=message,
                        base_filters=(
                            active_filters
                        ),
                    )
                )

                semantic_query = (
                    self._merge_semantic_query(
                        previous_semantic_query,
                        message,
                    )
                )

                products = (
                    retrieval_service
                    .retrieve(
                        query=semantic_query,
                        filters=filters,
                        limit=5,
                        strict_match=(
                            route
                            .strict_catalog_match
                        ),
                    )
                )

                conversation_service.set_search_state(
                    session_id=session_id,
                    products=products,
                    filters=filters,
                    semantic_query=(
                        semantic_query
                    ),
                )

            # ---------------------------------------------
            # CASE C
            #
            # Completely new search.
            # ---------------------------------------------

            else:

                filters = (
                    retrieval_service
                    .extract_filters(
                        query=message
                    )
                )

                products = (
                    retrieval_service
                    .retrieve(
                        query=message,
                        filters=filters,
                        limit=5,
                        strict_match=(
                            route
                            .strict_catalog_match
                        ),
                    )
                )

                conversation_service.set_search_state(
                    session_id=session_id,
                    products=products,
                    filters=filters,
                    semantic_query=message,
                )

        # =================================================
        # Conversation history
        # =================================================

        conversation_history = (
            self._format_history(
                history
            )
        )

        conversation_service.add_user_message(
            session_id=session_id,
            message=message,
            mode=route.mode,
        )

        # =================================================
        # Gemini
        # =================================================

        answer = await llm_service.generate_response(
            user_message=message,
            products=products,
            mode=route.mode,
            conversation_history=(
                conversation_history
            ),
            response_style=(
                route.response_style
            ),
        )

        conversation_service.add_assistant_message(
            session_id=session_id,
            message=answer,
        )

        # =================================================
        # Knowledge mode never has cards
        # =================================================

        if route.mode == "knowledge":

            return {
                "answer": answer,
                "session_id": session_id,
                "products": [],
            }

        # =================================================
        # Match cards to recommendation order
        # =================================================

        ordered_products = (
            self._order_products_by_answer(
                products,
                answer,
            )
        )

        # =================================================
        # Narrow active product set after selection
        # =================================================

        if (
            route.selection_limit
            is not None
        ):

            selected_products = (
                self._products_mentioned_in_answer(
                    products=ordered_products,
                    answer=answer,
                    limit=route.selection_limit,
                )
            )

            # If Gemini clearly mentioned fewer than
            # requested products, do not invent another.
            if selected_products:

                conversation_service.set_active_products(
                    session_id=session_id,
                    products=selected_products,
                )

                ordered_products = (
                    selected_products
                )

            else:

                # No valid selected product.
                conversation_service.clear_active_products(
                    session_id
                )

                ordered_products = []

        return {
            "answer": answer,

            "session_id":
                session_id,

            "products": [
                self._serialize_product(
                    product
                )

                for product
                in ordered_products
            ],
        }

    # =====================================================
    # Semantic refinement
    # =====================================================

    @staticmethod
    def _merge_semantic_query(
        previous_query: str,
        new_message: str,
    ) -> str:

        previous_query = (
            previous_query.strip()
        )

        new_message = (
            new_message.strip()
        )

        if not previous_query:
            return new_message

        if not new_message:
            return previous_query

        return (
            f"{previous_query}. "
            f"{new_message}"
        )

    # =====================================================
    # Conversation history
    # =====================================================

    @staticmethod
    def _format_history(
        history,
    ) -> str:

        if not history:
            return ""

        lines = []

        for turn in history:

            role = (
                "Customer"
                if turn.role == "user"
                else "Assistant"
            )

            lines.append(
                f"{role}: "
                f"{turn.content}"
            )

        return "\n".join(
            lines
        )

    # =====================================================
    # Detect products Gemini actually selected
    # =====================================================

    @staticmethod
    def _products_mentioned_in_answer(
        products: list[
            dict[str, Any]
        ],
        answer: str,
        limit: int,
    ) -> list[
        dict[str, Any]
    ]:

        answer_lower = (
            answer.lower()
        )

        mentioned = []

        for product in products:

            name = str(
                product.get(
                    "name",
                    "",
                )
            ).strip()

            if (
                name
                and name.lower()
                in answer_lower
            ):

                position = (
                    answer_lower.find(
                        name.lower()
                    )
                )

                mentioned.append(
                    (
                        position,
                        product,
                    )
                )

        mentioned.sort(
            key=lambda item:
                item[0]
        )

        return [
            product

            for _, product
            in mentioned[:limit]
        ]

    # =====================================================
    # Recommendation card ordering
    # =====================================================

    @staticmethod
    def _order_products_by_answer(
        products: list[
            dict[str, Any]
        ],
        answer: str,
    ) -> list[
        dict[str, Any]
    ]:

        answer_lower = (
            answer.lower()
        )

        ranked = []

        for (
            original_index,
            product,
        ) in enumerate(
            products
        ):

            name = str(
                product.get(
                    "name",
                    "",
                )
            ).strip()

            position = (
                answer_lower.find(
                    name.lower()
                )

                if name

                else -1
            )

            if position >= 0:

                key = (
                    0,
                    position,
                    original_index,
                )

            else:

                key = (
                    1,
                    original_index,
                    original_index,
                )

            ranked.append(
                (
                    key,
                    product,
                )
            )

        ranked.sort(
            key=lambda item:
                item[0]
        )

        return [
            product

            for _, product
            in ranked
        ]

    # =====================================================
    # API product card
    # =====================================================

    @staticmethod
    def _serialize_product(
        product: dict[
            str,
            Any,
        ],
    ) -> dict[str, Any]:

        return {
            "product_id":
                product.get(
                    "product_id"
                ),

            "name":
                product.get(
                    "name"
                ),

            "category":
                product.get(
                    "category"
                ),

            "year":
                product.get(
                    "year"
                ),

            "material":
                product.get(
                    "material"
                ),

            "weight":
                product.get(
                    "weight"
                ),

            "purity":
                product.get(
                    "purity"
                ),

            "price":
                product.get(
                    "price"
                ),

            "currency":
                product.get(
                    "currency"
                ),

            "description":
                product.get(
                    "description"
                ),

            "product_url":
                product.get(
                    "product_url"
                ),
        }


chat_service = ChatService()