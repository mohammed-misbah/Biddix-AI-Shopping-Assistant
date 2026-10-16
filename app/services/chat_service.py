import logging
import re
import time
import uuid
from typing import Any

from app.core.exceptions import (
    RetrievalServiceError,
)

from app.services.conversation_service import (
    conversation_service,
)

from app.services.llm_service import (
    llm_service,
)

from app.services.query_router import (
    QueryRoute,
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
        request_id: str | None = None,
    ):

        if not session_id:
            session_id = str(
                uuid.uuid4()
            )

        request_id = (
            request_id
            or "unknown"
        )

        chat_start_time = (
            time.perf_counter()
        )

        retrieval_ms = 0.0
        gemini_ms = 0.0

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

        route = query_router.route(
            message=message,
            previous_mode=previous_mode,
        )

        named_previous_products: list[
            dict[str, Any]
        ] = []

        if (
            previous_mode
            in (
                "catalog",
                "mixed",
            )
            and last_result_ids
            and self._looks_like_named_product_follow_up(
                message
            )
        ):

            previous_products = (
                retrieval_service
                .get_products_by_ids(
                    last_result_ids
                )
            )

            named_previous_products = (
                self._match_named_product_reference(
                    message=message,
                    products=previous_products,
                )
            )

        if named_previous_products:

            route = QueryRoute(
                mode=(
                    previous_mode
                    or "catalog"
                ),
                is_follow_up=True,
                follow_up_type="reuse_results",
                product_action=(
                    "availability"
                    if self._contains_availability_request(
                        message
                    )
                    else "detail"
                ),
                response_style=(
                    route.response_style
                ),
                selection_limit=1,
            )

        logger.info(
            "CHAT ROUTE | "
            "request_id=%s | "
            "session_id=%s | "
            "mode=%s | "
            "follow_up=%s | "
            "follow_up_type=%s | "
            "action=%s",
            request_id,
            session_id,
            route.mode,
            route.is_follow_up,
            route.follow_up_type,
            route.product_action,
        )

        products: list[
            dict[str, Any]
        ] = []

        if route.mode == "knowledge":

            products = []

        elif named_previous_products:

            products = (
                named_previous_products
            )

            conversation_service.set_focused_product(
                session_id=session_id,
                product=products[0],
            )

        elif (
            route.follow_up_type
            == "reuse_focus"
        ):

            products = (
                retrieval_service
                .get_products_by_ids(
                    [focused_id]
                )
                if focused_id
                else []
            )

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

        elif (
            route.follow_up_type
            == "refine_search"
        ):

            new_filters = (
                retrieval_service
                .extract_filters(
                    query=message,
                    base_filters=active_filters,
                )
            )

            semantic_query = (
                self._merge_semantic_query(
                    previous_semantic_query,
                    message,
                )
            )

            retrieval_start = (
                time.perf_counter()
            )

            products = self._retrieve_products(
                query=semantic_query,
                filters=new_filters,
                limit=5,
                strict_match=(
                    route.strict_catalog_match
                ),
            )

            retrieval_ms = (
                (
                    time.perf_counter()
                    - retrieval_start
                )
                * 1000
            )

            conversation_service.set_search_state(
                session_id=session_id,
                products=products,
                filters=new_filters,
                semantic_query=semantic_query,
            )

        else:

            filters = (
                retrieval_service
                .extract_filters(
                    query=message
                )
            )

            retrieval_start = (
                time.perf_counter()
            )

            products = self._retrieve_products(
                query=message,
                filters=filters,
                limit=5,
                strict_match=(
                    route.strict_catalog_match
                ),
            )

            retrieval_ms = (
                (
                    time.perf_counter()
                    - retrieval_start
                )
                * 1000
            )

            conversation_service.set_search_state(
                session_id=session_id,
                products=products,
                filters=filters,
                semantic_query=message,
            )

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

        gemini_start = (
            time.perf_counter()
        )

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

        gemini_ms = (
            (
                time.perf_counter()
                - gemini_start
            )
            * 1000
        )

        chat_total_ms = (
            (
                time.perf_counter()
                - chat_start_time
            )
            * 1000
        )

        logger.info(
            "CHAT METRICS | "
            "request_id=%s | "
            "session_id=%s | "
            "mode=%s | "
            "follow_up=%s | "
            "follow_up_type=%s | "
            "action=%s | "
            "products=%s | "
            "retrieval_ms=%.2f | "
            "gemini_ms=%.2f | "
            "chat_total_ms=%.2f",
            request_id,
            session_id,
            route.mode,
            route.is_follow_up,
            route.follow_up_type,
            route.product_action,
            len(products),
            retrieval_ms,
            gemini_ms,
            chat_total_ms,
        )

        conversation_service.add_assistant_message(
            session_id=session_id,
            message=answer,
        )

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
                    limit=(
                        route.selection_limit
                    ),
                )
            )

            if (
                len(selected_products)
                < route.selection_limit
            ):

                already_selected = {
                    str(
                        product.get(
                            "product_id"
                        )
                    )
                    for product
                    in selected_products
                }

                for product in ordered_products:

                    product_id = str(
                        product.get(
                            "product_id"
                        )
                    )

                    if (
                        product_id
                        in already_selected
                    ):
                        continue

                    selected_products.append(
                        product
                    )

                    already_selected.add(
                        product_id
                    )

                    if (
                        len(selected_products)
                        >= route.selection_limit
                    ):
                        break

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

    def _retrieve_products(
        self,
        query: str,
        filters,
        limit: int,
        strict_match: bool,
    ) -> list[
        dict[str, Any]
    ]:

        try:

            return retrieval_service.retrieve(
                query=query,
                filters=filters,
                limit=limit,
                strict_match=strict_match,
            )

        except RetrievalServiceError:

            raise

        except Exception as exc:

            logger.exception(
                "PRODUCT RETRIEVAL ERROR | "
                "error_type=%s",
                type(exc).__name__,
            )

            raise RetrievalServiceError(
                "Product retrieval failed."
            ) from exc

    @staticmethod
    def _looks_like_named_product_follow_up(
        message: str,
    ) -> bool:

        text = " ".join(
            re.findall(
                r"[a-z0-9]+",
                message.lower(),
            )
        )

        if not text:
            return False

        follow_up_phrases = (
            "learn more",
            "tell me more",
            "tell me about",
            "more details",
            "more detail",
            "more information",
            "details about",
            "details on",
            "information about",
            "information on",
            "what about",
            "how about",
            "price",
            "cost",
            "purity",
            "weight",
            "material",
            "year",
            "in stock",
            "out of stock",
            "available",
            "availability",
        )

        if any(
            phrase in text
            for phrase
            in follow_up_phrases
        ):
            return True

        return (
            len(
                text.split()
            )
            <= 6
        )

    @classmethod
    def _match_named_product_reference(
        cls,
        message: str,
        products: list[
            dict[str, Any]
        ],
    ) -> list[
        dict[str, Any]
    ]:

        if not products:
            return []

        message_tokens = (
            cls._reference_tokens(
                message
            )
        )

        if not message_tokens:
            return []

        message_set = set(
            message_tokens
        )

        normalized_message = (
            " ".join(
                message_tokens
            )
        )

        generic_name_tokens = {
            "coin",
            "coins",
            "product",
            "products",
            "collectible",
            "collectibles",
            "edition",
            "piece",
        }

        prepared: list[
            tuple[
                dict[str, Any],
                list[str],
                list[str],
            ]
        ] = []

        token_frequency: dict[
            str,
            int,
        ] = {}

        for product in products:

            name = str(
                product.get(
                    "name",
                    "",
                )
            ).strip()

            if not name:
                continue

            name_tokens = (
                cls._reference_tokens(
                    name
                )
            )

            significant = [
                token
                for token
                in name_tokens
                if (
                    token
                    not in generic_name_tokens
                    and not token.isdigit()
                )
            ]

            if not significant:
                continue

            prepared.append(
                (
                    product,
                    name_tokens,
                    significant,
                )
            )

            for token in set(
                significant
            ):

                token_frequency[
                    token
                ] = (
                    token_frequency.get(
                        token,
                        0,
                    )
                    + 1
                )

        candidates: list[
            tuple[
                float,
                dict[str, Any],
            ]
        ] = []

        for (
            product,
            name_tokens,
            significant,
        ) in prepared:

            normalized_name = (
                " ".join(
                    name_tokens
                )
            )

            if (
                normalized_name
                and normalized_name
                in normalized_message
            ):

                candidates.append(
                    (
                        10000.0,
                        product,
                    )
                )

                continue

            overlap = (
                set(
                    significant
                )
                & message_set
            )

            overlap_count = len(
                overlap
            )

            if overlap_count == 0:
                continue

            strong_match = (
                overlap_count >= 2
            )

            single_unique_match = False

            if overlap_count == 1:

                token = next(
                    iter(
                        overlap
                    )
                )

                single_unique_match = (
                    len(token) >= 5
                    and token_frequency.get(
                        token,
                        0,
                    )
                    == 1
                )

            if not (
                strong_match
                or single_unique_match
            ):
                continue

            coverage = (
                overlap_count
                / max(
                    len(
                        set(
                            significant
                        )
                    ),
                    1,
                )
            )

            ordered_bonus = 0.0

            for size in range(
                min(
                    4,
                    len(
                        significant
                    ),
                ),
                1,
                -1,
            ):

                phrase_found = any(
                    " ".join(
                        significant[
                            start:
                            start + size
                        ]
                    )
                    in normalized_message
                    for start in range(
                        len(significant)
                        - size
                        + 1
                    )
                )

                if phrase_found:

                    ordered_bonus = (
                        size
                        * 25.0
                    )

                    break

            score = (
                overlap_count
                * 100.0
                + coverage
                * 20.0
                + ordered_bonus
            )

            candidates.append(
                (
                    score,
                    product,
                )
            )

        if not candidates:
            return []

        candidates.sort(
            key=lambda item:
                item[0],
            reverse=True,
        )

        best_score = (
            candidates[0][0]
        )

        best_products = [
            product
            for score, product
            in candidates
            if abs(
                score
                - best_score
            )
            < 0.0001
        ]

        if len(best_products) != 1:
            return []

        return [
            best_products[0]
        ]

    @staticmethod
    def _contains_availability_request(
        message: str,
    ) -> bool:

        text = message.lower()

        return any(
            phrase in text
            for phrase in (
                "in stock",
                "out of stock",
                "available",
                "availability",
            )
        )

    @staticmethod
    def _reference_tokens(
        text: str,
    ) -> list[str]:

        return re.findall(
            r"[a-z0-9]+",
            text.lower(),
        )

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

        if re.search(
            r"\b("
            r"cheaper one"
            r"|cheapest one"
            r")\b",
            text,
        ):

            return (
                self._choose_cheapest(
                    products
                )
            )

        if re.search(
            r"\b("
            r"more expensive one"
            r"|expensive one"
            r"|most expensive one"
            r")\b",
            text,
        ):

            return (
                self._choose_most_expensive(
                    products
                )
            )

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

        if len(products) == 1:
            return products

        return products

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

    @staticmethod
    def _format_history(
        history,
    ) -> str:

        if not history:
            return ""

        return "\n".join(
            (
                "Customer"
                if turn.role
                == "user"
                else "Assistant"
            )
            + ": "
            + turn.content

            for turn
            in history
        )

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