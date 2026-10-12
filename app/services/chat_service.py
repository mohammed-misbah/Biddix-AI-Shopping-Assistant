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

        # =================================================
        # 1. Session
        # =================================================

        if not session_id:

            session_id = str(
                uuid.uuid4()
            )

        # =================================================
        # 2. Current conversation state
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

        active_catalog_query = (
            conversation_service
            .get_active_catalog_query(
                session_id
            )
        )

        # =================================================
        # 3. Understand message
        # =================================================

        route = query_router.route(
            message=message,
            previous_mode=previous_mode,
        )

        logger.info(
            "Chat route "
            "session=%s "
            "mode=%s "
            "follow_up=%s "
            "refresh=%s "
            "style=%s "
            "selection_limit=%s",
            session_id,
            route.mode,
            route.is_follow_up,
            route.refresh_products,
            route.response_style,
            route.selection_limit,
        )

        products: list[
            dict[str, Any]
        ] = []

        effective_catalog_query: (
            str | None
        ) = None

        # =================================================
        # 4. Knowledge mode
        # =================================================

        if route.mode == "knowledge":

            products = []

        # =================================================
        # 5. Catalog / mixed
        # =================================================

        elif route.mode in (
            "catalog",
            "mixed",
        ):

            # ---------------------------------------------
            # A. Product-selection follow-up
            #
            # "Which one?"
            # "Choose two."
            # "Compare those two."
            #
            # DO NOT RUN RAG AGAIN.
            # ---------------------------------------------

            if (
                route.is_follow_up
                and not route.refresh_products
                and active_product_ids
            ):

                products = (
                    retrieval_service
                    .get_products_by_ids(
                        active_product_ids
                    )
                )

                effective_catalog_query = (
                    active_catalog_query
                )

                logger.info(
                    "Reusing active products: %s",
                    active_product_ids,
                )

            # ---------------------------------------------
            # B. Search refinement
            #
            # Previous:
            # UAE coins above 500
            #
            # New:
            # Only silver ones
            #
            # Combined:
            # UAE coins above 500. Only silver ones.
            # ---------------------------------------------

            elif (
                route.is_follow_up
                and route.refresh_products
                and active_catalog_query
            ):

                effective_catalog_query = (
                    f"{active_catalog_query}. "
                    f"{message}"
                )

                products = (
                    retrieval_service
                    .retrieve(
                        query=(
                            effective_catalog_query
                        ),
                        limit=5,
                        strict_match=(
                            route
                            .strict_catalog_match
                        ),
                    )
                )

                conversation_service.set_product_context(
                    session_id=session_id,
                    products=products,
                    catalog_query=(
                        effective_catalog_query
                    ),
                )

                logger.info(
                    "Refined catalog query: %s",
                    effective_catalog_query,
                )

            # ---------------------------------------------
            # C. Fresh catalog search
            # ---------------------------------------------

            else:

                effective_catalog_query = (
                    message
                )

                products = (
                    retrieval_service
                    .retrieve(
                        query=message,
                        limit=5,
                        strict_match=(
                            route
                            .strict_catalog_match
                        ),
                    )
                )

                conversation_service.set_product_context(
                    session_id=session_id,
                    products=products,
                    catalog_query=message,
                )

        # Absolute safety.
        if route.mode == "knowledge":

            products = []

        # =================================================
        # 6. Conversation history
        # =================================================

        conversation_history = (
            self._format_history(
                history
            )
        )

        # =================================================
        # 7. Save user message
        # =================================================

        conversation_service.add_user_message(
            session_id=session_id,
            message=message,
            mode=route.mode,
        )

        # =================================================
        # 8. Gemini
        # =================================================

        answer = (
            await llm_service
            .generate_response(
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
        )

        # =================================================
        # 9. Save Gemini answer
        # =================================================

        conversation_service.add_assistant_message(
            session_id=session_id,
            message=answer,
        )

        # =================================================
        # 10. Knowledge has no cards
        # =================================================

        if route.mode == "knowledge":

            response_products = []

        else:

            # ---------------------------------------------
            # Put products Gemini actually mentioned first.
            # ---------------------------------------------

            ordered_products = (
                self._order_products_by_answer(
                    products=products,
                    answer=answer,
                )
            )

            # ---------------------------------------------
            # Selection follow-up.
            #
            # Example:
            # "Choose two"
            #
            # After this turn, those two become
            # the new active product set.
            # ---------------------------------------------

            if (
                route.selection_limit
                is not None
            ):

                selected_products = (
                    ordered_products[
                        :route.selection_limit
                    ]
                )

                # CRITICAL FIX:
                #
                # Old active set:
                # A B C D E
                #
                # "Choose two"
                #
                # New active set:
                # A C
                conversation_service.set_active_products(
                    session_id=session_id,
                    products=selected_products,
                )

                ordered_products = (
                    selected_products
                )

                logger.info(
                    "Narrowed active products to: %s",
                    [
                        product.get(
                            "product_id"
                        )
                        for product
                        in selected_products
                    ],
                )

            response_products = [
                self._serialize_product(
                    product
                )
                for product
                in ordered_products
            ]

        # =================================================
        # 11. Response
        # =================================================

        return {
            "answer": answer,
            "session_id": session_id,
            "products": response_products,
        }

    # =====================================================
    # Format conversation
    # =====================================================

    @staticmethod
    def _format_history(
        history,
    ) -> str:

        if not history:
            return ""

        lines = []

        for turn in history:

            label = (
                "Customer"
                if turn.role == "user"
                else "Assistant"
            )

            lines.append(
                f"{label}: "
                f"{turn.content}"
            )

        return "\n".join(
            lines
        )

    # =====================================================
    # Recommended products first
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
            original_position,
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

            if name:

                answer_position = (
                    answer_lower.find(
                        name.lower()
                    )
                )

            else:

                answer_position = -1

            if answer_position >= 0:

                key = (
                    0,
                    answer_position,
                    original_position,
                )

            else:

                key = (
                    1,
                    original_position,
                    original_position,
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
    # Product response card
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