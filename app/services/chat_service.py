
import logging
import os
import time
import uuid
from typing import Any
from app.services.grounding_service import grounding_service
from app.core.exceptions import RetrievalServiceError
from app.services.conversation_service import (
    CatalogFilters,
    conversation_service,
)
from app.services.llm_service import llm_service
from app.services.product_service import product_service
from app.services.query_router import query_router
from app.services.retrieval_service import retrieval_service


logger = logging.getLogger(__name__)


class ChatService:

    async def chat(
        self,
        message: str,
        session_id: str | None = None,
        request_id: str | None = None,
        current_product_id: str | None = None,
    ) -> dict[str, Any]:

        session_id = session_id or str(uuid.uuid4())
        request_id = request_id or "unknown"

        started = time.perf_counter()

        history = conversation_service.get_history(
            session_id
        )
        history_text = self._format_history(history)

        previous_mode = (
            conversation_service.get_last_mode(
                session_id
            )
        )

        result_ids = (
            conversation_service
            .get_last_result_product_ids(session_id)
        )
        selected_ids = (
            conversation_service
            .get_selected_product_ids(session_id)
        )
        focused_id = (
            conversation_service
            .get_focused_product_id(session_id)
        )

        previous_products = (
            retrieval_service.get_products_by_ids(
                result_ids
            )
        )
        selected_products = (
            retrieval_service.get_products_by_ids(
                selected_ids
            )
        )
        focused_products = (
            retrieval_service.get_products_by_ids(
                [focused_id]
            )
            if focused_id
            else []
        )
        focused_product = (
            focused_products[0]
            if focused_products
            else None
        )

        if current_product_id:
            requested = (
                retrieval_service.get_products_by_ids(
                    [current_product_id]
                )
            )
            if requested:
                focused_product = requested[0]

        gemini_first = (
            os.getenv(
                "BIDDIX_GEMINI_FIRST", "0"
            ).strip().lower()
            in {"1", "true", "yes"}
        )

        retrieval_ms = 0.0
        intent_ms = 0.0
        gemini_ms = 0.0

        if gemini_first:
            intent_started = time.perf_counter()

            intent = await llm_service.interpret_request(
                user_message=message,
                conversation_history=history_text,
                previous_products=previous_products,
                selected_products=selected_products,
                focused_product=focused_product,
            )

            intent_ms = (
                time.perf_counter() - intent_started
            ) * 1000

            retrieval_started = time.perf_counter()

            products, mode, action, limit = (
                self._gemini_products(
                    message=message,
                    intent=intent,
                    previous_products=previous_products,
                    selected_products=selected_products,
                    focused_product=focused_product,
                    session_id=session_id,
                )
            )

            retrieval_ms = (
                time.perf_counter() - retrieval_started
            ) * 1000

            response_style = intent.get(
                "response_style", "normal"
            )

            route_name = intent.get(
                "intent", "unknown"
            )

        else:
            route = query_router.route(
                message=message,
                previous_mode=previous_mode,
            )

            retrieval_started = time.perf_counter()

            products = self._legacy_products(
                message=message,
                session_id=session_id,
                route=route,
                previous_products=previous_products,
                selected_products=selected_products,
                focused_product=focused_product,
            )

            retrieval_ms = (
                time.perf_counter() - retrieval_started
            ) * 1000

            mode = route.mode
            action = route.product_action
            limit = route.selection_limit
            response_style = route.response_style
            route_name = (
                route.follow_up_type
                if route.is_follow_up
                else "new"
            )

        logger.info(
            "CHAT ROUTE | request_id=%s | "
            "session_id=%s | mode=%s | "
            "action=%s | route=%s | "
            "gemini_first=%s",
            request_id,
            session_id,
            mode,
            action,
            route_name,
            gemini_first,
        )

        conversation_service.add_user_message(
            session_id=session_id,
            message=message,
            mode=mode,
        )

        gemini_started = time.perf_counter()

        answer = await llm_service.generate_response(
            user_message=message,
            products=products,
            mode=mode,
            conversation_history=history_text,
            response_style=response_style,
        )

        gemini_ms = (
            time.perf_counter() - gemini_started
        ) * 1000

        answer = grounding_service.ground_response(
            answer=answer,
            products=products,
            mode=mode,
            request_id=request_id,
        )

        conversation_service.add_assistant_message(
            session_id=session_id,
            message=answer,
        )

        if mode == "knowledge":
            visible_products = []
        else:
            visible_products = (
                self._order_products_by_answer(
                    products, answer
                )
            )

            if action == "choose" and limit:
                mentioned = (
                    self._products_mentioned_in_answer(
                        visible_products,
                        answer,
                        limit,
                    )
                )

                chosen_ids = {
                    str(p.get("product_id"))
                    for p in mentioned
                }

                for product in visible_products:
                    if len(mentioned) >= limit:
                        break

                    product_id = str(
                        product.get("product_id")
                    )

                    if product_id not in chosen_ids:
                        mentioned.append(product)
                        chosen_ids.add(product_id)

                visible_products = mentioned

                conversation_service.set_selected_products(
                    session_id=session_id,
                    products=visible_products,
                )

                if len(visible_products) == 1:
                    conversation_service.set_focused_product(
                        session_id=session_id,
                        product=visible_products[0],
                    )

        total_ms = (
            time.perf_counter() - started
        ) * 1000

        logger.info(
            "CHAT METRICS | request_id=%s | "
            "session_id=%s | mode=%s | "
            "products=%s | intent_ms=%.2f | "
            "retrieval_ms=%.2f | "
            "gemini_ms=%.2f | "
            "chat_total_ms=%.2f",
            request_id,
            session_id,
            mode,
            len(visible_products),
            intent_ms,
            retrieval_ms,
            gemini_ms,
            total_ms,
        )

        return {
            "answer": answer,
            "session_id": session_id,
            "products": [
                self._serialize_product(product)
                for product in visible_products
            ],
        }

    # =====================================================
    # NEW: GEMINI-FIRST PRODUCT SELECTION
    # =====================================================

    def _gemini_products(
        self,
        message: str,
        intent: dict[str, Any],
        previous_products: list[dict[str, Any]],
        selected_products: list[dict[str, Any]],
        focused_product: dict[str, Any] | None,
        session_id: str,
    ) -> tuple[list[dict[str, Any]], str, str, int | None]:

        kind = intent.get(
            "intent", "product_search"
        )
        scope = intent.get(
            "reference_scope", "none"
        )
        search_query = intent.get(
            "search_query", ""
        ).strip()
        count = intent.get("count")
        comparison_count = intent.get(
            "comparison_count"
        )

        requested_ids = intent.get(
            "referenced_product_ids", []
        )

        limit = count or 5
        limit = max(1, min(limit, 20))

        mode = (
            "knowledge"
            if kind == "knowledge"
            else "catalog"
        )

        if mode == "knowledge":
            return [], mode, "none", None

        products: list[dict[str, Any]] = []
        used_previous = False

        if requested_ids:
            allowed = {
                str(p.get("product_id"))
                for p in (
                    previous_products
                    + selected_products
                    + (
                        [focused_product]
                        if focused_product
                        else []
                    )
                )
            }

            safe_ids = [
                product_id
                for product_id in requested_ids
                if product_id in allowed
            ]

            if safe_ids:
                products = (
                    retrieval_service
                    .get_products_by_ids(safe_ids)
                )
                used_previous = True

        if not products:
            if scope == "focused_product":
                products = (
                    [focused_product]
                    if focused_product
                    else []
                )
                used_previous = True

            elif scope == "selected_products":
                products = list(selected_products)
                used_previous = True

            elif scope == "previous_results":
                products = list(previous_products)
                used_previous = True

        if (
            not products
            and kind in {
                "follow_up",
                "comparison",
                "cheapest",
                "availability",
                "product_detail",
            }
        ):
            products = (
                list(selected_products)
                or (
                    [focused_product]
                    if focused_product
                    else []
                )
                or list(previous_products)
            )
            used_previous = bool(products)

        if not products:
            if (
                kind == "catalog_list"
                and not search_query
            ):
                products = self._catalog_list(
                    limit=limit
                )
            else:
                filters = self._intent_filters(
                    intent.get("filters", {})
                )

                query = (
                    search_query
                    or message
                )

                products = self._retrieve_products(
                    query=query,
                    filters=filters,
                    limit=limit,
                    strict_match=False,
                )

                if (
                    not products
                    and kind == "catalog_list"
                ):
                    products = self._catalog_list(
                        limit=limit,
                        filters=filters,
                    )

        action = "none"
        selection_limit = None

        if kind == "cheapest":
            products = self._choose_cheapest(
                products
            )
            if products:
                conversation_service.set_focused_product(
                    session_id=session_id,
                    product=products[0],
                )

        elif kind == "comparison":
            selection_limit = (
                comparison_count or 2
            )

            if len(products) > selection_limit:
                products = products[
                    :selection_limit
                ]

            if products:
                conversation_service.set_selected_products(
                    session_id=session_id,
                    products=products,
                )

        elif kind == "recommendation":
            action = "choose"
            selection_limit = count or 1

        elif kind in {
            "availability",
            "product_detail",
        }:
            if len(products) == 1:
                conversation_service.set_focused_product(
                    session_id=session_id,
                    product=products[0],
                )

        if not used_previous and products:
            filters = self._intent_filters(
                intent.get("filters", {})
            )

            conversation_service.set_search_state(
                session_id=session_id,
                products=products,
                filters=filters,
                semantic_query=(
                    search_query or message
                ),
            )

        return (
            products,
            mode,
            action,
            selection_limit,
        )

    def _catalog_list(
        self,
        limit: int,
        filters: CatalogFilters | None = None,
    ) -> list[dict[str, Any]]:

        catalog = product_service.products

        if filters is None:
            return [
                dict(product)
                for product in catalog[:limit]
            ]

        return [
            dict(product)
            for product in catalog
            if retrieval_service._passes_filters(
                product, filters
            )
        ][:limit]

    @staticmethod
    def _intent_filters(
        raw: dict[str, Any],
    ) -> CatalogFilters:

        if not isinstance(raw, dict):
            raw = {}

        filters = CatalogFilters()

        filters.material = raw.get(
            "material"
        )
        filters.year = raw.get("year")
        filters.min_price = raw.get(
            "min_price"
        )
        filters.max_price = raw.get(
            "max_price"
        )

        if raw.get("in_stock") is True:
            filters.in_stock = True

        return filters

    # =====================================================
    # EXISTING: ROUTER COMPATIBILITY
    # =====================================================

    def _legacy_products(
        self,
        message: str,
        session_id: str,
        route,
        previous_products: list[dict[str, Any]],
        selected_products: list[dict[str, Any]],
        focused_product: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:

        if route.mode == "knowledge":
            return []

        # Recognize named products from previous results.
        matched = self._match_named_product_reference(
            message,
            previous_products,
        )

        if matched and route.follow_up_type != "refine_search":
            products = matched
            conversation_service.set_focused_product(
                session_id=session_id,
                product=products[0],
            )

        elif route.follow_up_type == "reuse_focus":
            products = (
                [focused_product]
                if focused_product
                else []
            )

        elif route.follow_up_type == "reuse_selection":
            products = list(selected_products)

        elif route.follow_up_type == "reuse_results":
            products = list(previous_products)

        elif route.follow_up_type == "refine_search":
            old_filters = (
                conversation_service
                .get_active_filters(session_id)
            )

            filters = retrieval_service.extract_filters(
                query=message,
                base_filters=old_filters,
            )

            previous_query = (
                conversation_service
                .get_semantic_query(session_id)
            )

            query = self._merge_semantic_query(
                previous_query,
                message,
            )

            products = self._retrieve_products(
                query=query,
                filters=filters,
                limit=5,
                strict_match=(
                    route.strict_catalog_match
                ),
            )

            conversation_service.set_search_state(
                session_id=session_id,
                products=products,
                filters=filters,
                semantic_query=query,
            )

        else:
            filters = retrieval_service.extract_filters(
                query=message
            )

            products = self._retrieve_products(
                query=message,
                filters=filters,
                limit=5,
                strict_match=(
                    route.strict_catalog_match
                ),
            )

            conversation_service.set_search_state(
                session_id=session_id,
                products=products,
                filters=filters,
                semantic_query=message,
            )

        if route.product_action == "cheapest":
            products = self._choose_cheapest(
                products
            )

            if products:
                conversation_service.set_focused_product(
                    session_id=session_id,
                    product=products[0],
                )

        elif route.product_action == "availability":
            products = (
                self._resolve_referenced_products(
                    message, products
                )
            )

            if len(products) == 1:
                conversation_service.set_focused_product(
                    session_id=session_id,
                    product=products[0],
                )

        return products

    # =====================================================
    # SHARED HELPERS
    # =====================================================

    @staticmethod
    def _retrieve_products(
        query: str,
        filters: CatalogFilters,
        limit: int,
        strict_match: bool,
    ) -> list[dict[str, Any]]:

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
    def _format_history(history) -> str:
        if not history:
            return ""

        return "\n".join(
            (
                "Customer"
                if turn.role == "user"
                else "Assistant"
            )
            + ": "
            + turn.content
            for turn in history
        )

    @staticmethod
    def _choose_cheapest(
        products: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:

        priced = []

        for product in products:
            try:
                price = float(
                    product.get("price")
                )
            except (TypeError, ValueError):
                continue

            priced.append(
                (price, product)
            )

        if not priced:
            return []

        return [
            min(priced, key=lambda item: item[0])[1]
        ]

    @staticmethod
    def _merge_semantic_query(
        previous_query: str,
        message: str,
    ) -> str:

        if not previous_query:
            return message

        return (
            previous_query.strip()
            + " "
            + message.strip()
        )

    @staticmethod
    def _match_named_product_reference(
        message: str,
        products: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:

        import re

        ignored = {
            "coin", "coins", "product",
            "products", "piece", "pieces",
            "silver", "gold", "the", "and",
            "2023", "2024", "2025", "2026",
        }

        words = set(
            re.findall(
                r"[a-z0-9]+",
                message.lower(),
            )
        )

        matches = []

        for product in products:
            name = str(
                product.get("name", "")
            ).lower()

            tokens = set(
                re.findall(
                    r"[a-z0-9]+", name
                )
            ) - ignored

            if (
                tokens
                and len(tokens & words) >= 1
            ):
                matches.append(product)

        return (
            matches
            if len(matches) == 1
            else []
        )

    def _resolve_referenced_products(
        self,
        message: str,
        products: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:

        text = message.lower()

        if (
            "cheaper one" in text
            or "cheapest one" in text
        ):
            return self._choose_cheapest(
                products
            )

        if (
            "expensive one" in text
            or "most expensive" in text
        ):
            try:
                return [
                    max(
                        products,
                        key=lambda p: float(
                            p["price"]
                        ),
                    )
                ]
            except (ValueError, KeyError, TypeError):
                return []

        if "first one" in text:
            return products[:1]

        if "second one" in text:
            return products[1:2]

        return products

    @staticmethod
    def _products_mentioned_in_answer(
        products: list[dict[str, Any]],
        answer: str,
        limit: int,
    ) -> list[dict[str, Any]]:

        lower = answer.lower()

        return [
            product
            for product in products
            if str(
                product.get("name", "")
            ).lower() in lower
        ][:limit]

    @staticmethod
    def _order_products_by_answer(
        products: list[dict[str, Any]],
        answer: str,
    ) -> list[dict[str, Any]]:

        lower = answer.lower()

        indexed = list(
            enumerate(products)
        )

        indexed.sort(
            key=lambda item: (
                0
                if str(
                    item[1].get("name", "")
                ).lower() in lower
                else 1,
                lower.find(
                    str(
                        item[1].get("name", "")
                    ).lower()
                )
                if str(
                    item[1].get("name", "")
                ).lower() in lower
                else item[0],
                item[0],
            )
        )

        return [
            product
            for _, product in indexed
        ]

    @staticmethod
    def _serialize_product(
        product: dict[str, Any],
    ) -> dict[str, Any]:

        fields = (
            "product_id",
            "name",
            "category",
            "year",
            "material",
            "weight",
            "purity",
            "price",
            "currency",
            "description",
            "product_url",
        )

        return {
            field: product.get(field)
            for field in fields
        }


chat_service = ChatService()
