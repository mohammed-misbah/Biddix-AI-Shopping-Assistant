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
        # 1. SESSION
        # =================================================

        if not session_id:

            session_id = str(
                uuid.uuid4()
            )

        # =================================================
        # 2. READ CURRENT STATE
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

        last_result_ids = (
            conversation_service
            .get_last_result_product_ids(
                session_id
            )
        )

        selected_ids = (
            conversation_service
            .get_selected_product_ids(
                session_id
            )
        )

        focused_id = (
            conversation_service
            .get_focused_product_id(
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
        # 3. ROUTE MESSAGE
        # =================================================

        route = query_router.route(
            message=message,
            previous_mode=previous_mode,
        )

        logger.info(
            "Chat route mode=%s "
            "follow_up=%s "
            "type=%s "
            "action=%s "
            "selection_limit=%s",
            route.mode,
            route.is_follow_up,
            route.follow_up_type,
            route.product_action,
            route.selection_limit,
        )

        products: list[
            dict[str, Any]
        ] = []

        # =================================================
        # 4. KNOWLEDGE
        # =================================================

        if route.mode == "knowledge":

            products = []

        # =================================================
        # 5. CATALOG / MIXED
        # =================================================

        else:

            # ---------------------------------------------
            # A. FOCUSED PRODUCT
            #
            # "Tell me more about that coin."
            #
            # Never run FAISS.
            # ---------------------------------------------

            if (
                route.follow_up_type
                == "reuse_focus"
            ):

                if focused_id:

                    products = (
                        retrieval_service
                        .get_products_by_ids(
                            [focused_id]
                        )
                    )

                else:

                    products = []

            # ---------------------------------------------
            # B. PREVIOUS SELECTED PRODUCTS
            #
            # "Compare those two."
            #
            # Never run FAISS.
            # ---------------------------------------------

            elif (
                route.follow_up_type
                == "reuse_selection"
            ):

                if selected_ids:

                    products = (
                        retrieval_service
                        .get_products_by_ids(
                            selected_ids
                        )
                    )

                else:

                    products = []

            # ---------------------------------------------
            # C. LAST FULL SEARCH RESULTS
            #
            # "Which is cheapest?"
            # "Choose two."
            #
            # Never run FAISS.
            # ---------------------------------------------

            elif (
                route.follow_up_type
                == "reuse_results"
            ):

                if last_result_ids:

                    products = (
                        retrieval_service
                        .get_products_by_ids(
                            last_result_ids
                        )
                    )

                else:

                    products = []

            # ---------------------------------------------
            # D. SEARCH REFINEMENT
            #
            # Existing:
            # > 500
            #
            # New:
            # only silver
            #
            # Result:
            # >500 + silver
            # ---------------------------------------------

            elif (
                route.follow_up_type
                == "refine_search"
            ):

                new_filters = (
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
                        filters=new_filters,
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
                    filters=new_filters,
                    semantic_query=semantic_query,
                )

            # ---------------------------------------------
            # E. FRESH SEARCH
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
        # 6. DETERMINISTIC PRODUCT ACTIONS
        # =================================================
        #
        # Do things Python can determine safely.
        #
        # Cheapest should not depend on Gemini.
        # =================================================

        if (
            route.product_action
            == "cheapest"
        ):

            products = (
                self._choose_cheapest(
                    products
                )
            )

            if products:

                conversation_service.set_focused_product(
                    session_id=session_id,
                    product=products[0],
                )

        # =================================================
        # 7. HISTORY
        # =================================================

        conversation_history = (
            self._format_history(
                history
            )
        )

        # Save user message AFTER reading old history.
        conversation_service.add_user_message(
            session_id=session_id,
            message=message,
            mode=route.mode,
        )

        # =================================================
        # 8. GEMINI
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
        # 9. KNOWLEDGE HAS NO PRODUCT CARDS
        # =================================================

        if route.mode == "knowledge":

            return {
                "answer": answer,
                "session_id": session_id,
                "products": [],
            }

        # =================================================
        # 10. ORDER PRODUCTS BY ANSWER
        # =================================================

        ordered_products = (
            self._order_products_by_answer(
                products=products,
                answer=answer,
            )
        )

        # =================================================
        # 11. CHOOSE ACTION
        # =================================================
        #
        # If customer asks:
        #
        # "Choose two"
        #
        # Gemini decides based on the supplied products.
        #
        # We detect which exact product names Gemini used.
        # =================================================

        if (
            route.product_action
            == "choose"
            and route.selection_limit
            is not None
        ):

            selected_products = (
                self._products_mentioned_in_answer(
                    products=ordered_products,
                    answer=answer,
                    limit=route.selection_limit,
                )
            )

            # Safety fallback:
            #
            # if Gemini did not explicitly mention
            # enough names, keep only names it did
            # mention rather than inventing another.
            conversation_service.set_selected_products(
                session_id=session_id,
                products=selected_products,
            )

            if (
                len(selected_products)
                == 1
            ):

                conversation_service.set_focused_product(
                    session_id=session_id,
                    product=selected_products[0],
                )

            ordered_products = (
                selected_products
            )

        # =================================================
        # 12. CHEAPEST ACTION
        # =================================================

        elif (
            route.product_action
            == "cheapest"
        ):

            ordered_products = products

        # =================================================
        # 13. DETAIL ACTION
        # =================================================

        elif (
            route.product_action
            == "detail"
        ):

            ordered_products = products

        # =================================================
        # 14. COMPARE ACTION
        # =================================================

        elif (
            route.product_action
            == "compare"
        ):

            # Exactly the selected set.
            ordered_products = products

        # =================================================
        # 15. RESPONSE
        # =================================================

        response_products = [
            self._serialize_product(
                product
            )
            for product
            in ordered_products
        ]

        return {
            "answer": answer,
            "session_id": session_id,
            "products": response_products,
        }

    # =====================================================
    # CHEAPEST
    # =====================================================

    @staticmethod
    def _choose_cheapest(
        products: list[
            dict[str, Any]
        ],
    ) -> list[
        dict[str, Any]
    ]:

        valid = []

        for product in products:

            try:

                price = float(
                    product.get(
                        "price"
                    )
                )

            except (
                TypeError,
                ValueError,
            ):

                continue

            valid.append(
                (
                    price,
                    product,
                )
            )

        if not valid:
            return []

        valid.sort(
            key=lambda item:
                item[0]
        )

        return [
            valid[0][1]
        ]

    # =====================================================
    # SEMANTIC QUERY
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
    # HISTORY
    # =====================================================

    @staticmethod
    def _format_history(
        history,
    ) -> str:

        if not history:
            return ""

        parts = []

        for turn in history:

            role = (
                "Customer"
                if turn.role == "user"
                else "Assistant"
            )

            parts.append(
                f"{role}: "
                f"{turn.content}"
            )

        return "\n".join(
            parts
        )

    # =====================================================
    # PRODUCT MENTION DETECTION
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

        found = []

        for product in products:

            name = str(
                product.get(
                    "name",
                    "",
                )
            ).strip()

            if not name:
                continue

            position = (
                answer_lower.find(
                    name.lower()
                )
            )

            if position >= 0:

                found.append(
                    (
                        position,
                        product,
                    )
                )

        found.sort(
            key=lambda item:
                item[0]
        )

        return [
            product
            for _, product
            in found[:limit]
        ]

    # =====================================================
    # CARD ORDER
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

            if name:

                position = (
                    answer_lower.find(
                        name.lower()
                    )
                )

            else:

                position = -1

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
    # API CARD
    # =====================================================

    @staticmethod
    def _serialize_product(
        product: dict[
            str,
            Any,
        ],
    ) -> dict[
        str,
        Any,
    ]:

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