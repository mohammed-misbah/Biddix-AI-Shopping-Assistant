import logging
import re
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
        # Existing state
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
        # Route current message
        # =================================================

        route = query_router.route(
            message=message,
            previous_mode=previous_mode,
        )

        logger.info(
            "Chat route mode=%s "
            "follow_up=%s "
            "type=%s "
            "action=%s",
            route.mode,
            route.is_follow_up,
            route.follow_up_type,
            route.product_action,
        )

        products: list[
            dict[str, Any]
        ] = []

        # =================================================
        # Knowledge mode
        # =================================================

        if route.mode == "knowledge":

            products = []

        # =================================================
        # Catalog / mixed
        # =================================================

        else:

            # ---------------------------------------------
            # Reuse focused product
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
            # Reuse selected products
            # ---------------------------------------------

            elif (
                route.follow_up_type
                == "reuse_selection"
            ):

                products = (
                    retrieval_service
                    .get_products_by_ids(
                        selected_ids
                    )
                    if selected_ids
                    else []
                )

            # ---------------------------------------------
            # Reuse last full result set
            # ---------------------------------------------

            elif (
                route.follow_up_type
                == "reuse_results"
            ):

                products = (
                    retrieval_service
                    .get_products_by_ids(
                        last_result_ids
                    )
                    if last_result_ids
                    else []
                )

            # ---------------------------------------------
            # Refine existing search
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
                    semantic_query=(
                        semantic_query
                    ),
                )

            # ---------------------------------------------
            # Fresh search
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
        # Deterministic product-reference resolution
        # =================================================

        if (
            route.product_action
            == "availability"
        ):

            products = (
                self._resolve_referenced_products(
                    message=message,
                    products=products,
                )
            )

            if len(products) == 1:

                conversation_service.set_focused_product(
                    session_id=session_id,
                    product=products[0],
                )

        elif (
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
        # Generate response
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
        # Knowledge responses have no cards
        # =================================================

        if route.mode == "knowledge":

            return {
                "answer": answer,
                "session_id": session_id,
                "products": [],
            }

        ordered_products = (
            self._order_products_by_answer(
                products=products,
                answer=answer,
            )
        )

        # =================================================
        # Choose one/two
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

        return {
            "answer": answer,
            "session_id": session_id,
            "products": [
                self._serialize_product(
                    product
                )
                for product
                in ordered_products
            ],
        }

    # =====================================================
    # Resolve references
    # =====================================================

    def _resolve_referenced_products(
        self,
        message: str,
        products: list[
            dict[str, Any]
        ],
    ) -> list[
        dict[str, Any]
    ]:

        if not products:
            return []

        text = message.lower()

        # ---------------------------------------------
        # Cheaper / cheapest one
        # ---------------------------------------------

        if re.search(
            r"\b("
            r"cheaper one"
            r"|cheapest one"
            r")\b",
            text,
        ):

            cheapest = (
                self._choose_cheapest(
                    products
                )
            )

            return cheapest

        # ---------------------------------------------
        # More expensive / expensive one
        # ---------------------------------------------

        if re.search(
            r"\b("
            r"more expensive one"
            r"|expensive one"
            r"|most expensive one"
            r")\b",
            text,
        ):

            return self._choose_most_expensive(
                products
            )

        # ---------------------------------------------
        # First / second / third
        # ---------------------------------------------

        if re.search(
            r"\bfirst one\b",
            text,
        ):

            return products[:1]

        if re.search(
            r"\bsecond one\b",
            text,
        ):

            return (
                products[1:2]
                if len(products) >= 2
                else []
            )

        if re.search(
            r"\bthird one\b",
            text,
        ):

            return (
                products[2:3]
                if len(products) >= 3
                else []
            )

        # If there is exactly one supplied product,
        # a singular reference naturally points to it.
        if len(products) == 1:
            return products

        return products

    # =====================================================
    # Cheapest
    # =====================================================

    @staticmethod
    def _choose_cheapest(
        products: list[
            dict[str, Any]
        ],
    ) -> list[
        dict[str, Any]
    ]:

        priced = []

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

            priced.append(
                (
                    price,
                    product,
                )
            )

        if not priced:
            return []

        priced.sort(
            key=lambda item:
                item[0]
        )

        return [
            priced[0][1]
        ]

    # =====================================================
    # Most expensive
    # =====================================================

    @staticmethod
    def _choose_most_expensive(
        products: list[
            dict[str, Any]
        ],
    ) -> list[
        dict[str, Any]
    ]:

        priced = []

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

            priced.append(
                (
                    price,
                    product,
                )
            )

        if not priced:
            return []

        priced.sort(
            key=lambda item:
                item[0],
            reverse=True,
        )

        return [
            priced[0][1]
        ]

    # =====================================================
    # Semantic refinement query
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
    # Product names selected by Gemini
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
    # Card ordering
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