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
        # 2. Read conversation state
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

        previous_product_ids = (
            conversation_service
            .get_last_product_ids(
                session_id
            )
        )

        previous_catalog_query = (
            conversation_service
            .get_last_catalog_query(
                session_id
            )
        )

        # =================================================
        # 3. Route current message
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
            "refresh_products=%s "
            "style=%s "
            "selection_limit=%s "
            "strict=%s",
            session_id,
            route.mode,
            route.is_follow_up,
            route.refresh_products,
            route.response_style,
            route.selection_limit,
            route.strict_catalog_match,
        )

        # =================================================
        # 4. Resolve products
        # =================================================

        products: list[
            dict[str, Any]
        ] = []

        # -------------------------------------------------
        # Knowledge:
        # absolutely no cards
        # -------------------------------------------------

        if route.mode == "knowledge":

            products = []

        # -------------------------------------------------
        # Catalog / mixed follow-up:
        # reuse previous exact product set
        # -------------------------------------------------

        elif (
            route.mode in (
                "catalog",
                "mixed",
            )
            and route.is_follow_up
            and not route.refresh_products
            and previous_product_ids
        ):

            products = (
                retrieval_service
                .get_products_by_ids(
                    previous_product_ids
                )
            )

            logger.info(
                "Reusing previous product set: %s",
                previous_product_ids,
            )

        # -------------------------------------------------
        # Fresh search or changed requirements
        # -------------------------------------------------

        elif route.mode in (
            "catalog",
            "mixed",
        ):

            retrieval_query = message

            products = (
                retrieval_service.retrieve(
                    query=retrieval_query,
                    limit=5,
                    strict_match=(
                        route
                        .strict_catalog_match
                    ),
                )
            )

            # Save EXACT retrieved product set.
            conversation_service.set_product_context(
                session_id=session_id,
                products=products,
                catalog_query=(
                    retrieval_query
                ),
            )

        # Final safety.
        if route.mode == "knowledge":
            products = []

        # =================================================
        # 5. Conversation history for Gemini
        # =================================================

        conversation_history = (
            self._format_history(
                history
            )
        )

        # =================================================
        # 6. Save current user message
        # =================================================

        conversation_service.add_user_message(
            session_id=session_id,
            message=message,
            mode=route.mode,
        )

        # =================================================
        # 7. Gemini
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
        # 8. Remember answer
        # =================================================

        conversation_service.add_assistant_message(
            session_id=session_id,
            message=answer,
        )

        # =================================================
        # 9. Knowledge mode never renders products
        # =================================================

        if route.mode == "knowledge":

            response_products = []

        else:

            # ---------------------------------------------
            # Put products mentioned by Gemini first.
            #
            # Example:
            #
            # Gemini:
            # "I'd choose the Emirates Map Silver Coin."
            #
            # Emirates Map card becomes card #1.
            # ---------------------------------------------

            ordered_products = (
                self._order_products_by_answer(
                    products=products,
                    answer=answer,
                )
            )

            # ---------------------------------------------
            # If customer explicitly asked:
            #
            # "Which one?"
            # → show 1 card
            #
            # "Choose one or two"
            # → show max 2 cards
            # ---------------------------------------------

            if (
                route.selection_limit
                is not None
            ):

                ordered_products = (
                    ordered_products[
                        :route.selection_limit
                    ]
                )

            response_products = [
                self._serialize_product(
                    product
                )
                for product
                in ordered_products
            ]

        # =================================================
        # 10. API response
        # =================================================

        return {
            "answer": answer,
            "session_id":
                session_id,
            "products":
                response_products,
        }

    # =====================================================
    # Format history
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
    ) -> list[dict[str, Any]]:

        answer_lower = (
            answer.lower()
        )

        ranked = []

        for original_position, product in enumerate(
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

                sort_key = (
                    0,
                    answer_position,
                    original_position,
                )

            else:

                sort_key = (
                    1,
                    original_position,
                    original_position,
                )

            ranked.append(
                (
                    sort_key,
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
    # Product card data
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