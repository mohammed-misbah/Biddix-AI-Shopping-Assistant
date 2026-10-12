import re
from dataclasses import dataclass
from typing import Literal


QueryMode = Literal[
    "catalog",
    "knowledge",
    "mixed",
]

ResponseStyle = Literal[
    "normal",
    "brief",
    "very_brief",
    "detailed",
]

FollowUpType = Literal[
    "none",
    "refine_search",
    "reuse_results",
    "reuse_selection",
    "reuse_focus",
]

ProductAction = Literal[
    "none",
    "choose",
    "cheapest",
    "compare",
    "detail",
    "availability",
]


@dataclass(frozen=True)
class QueryRoute:
    mode: QueryMode

    strict_catalog_match: bool = False

    is_follow_up: bool = False

    follow_up_type: FollowUpType = "none"

    product_action: ProductAction = "none"

    response_style: ResponseStyle = "normal"

    selection_limit: int | None = None


class QueryRouter:

    KNOWLEDGE_PATTERNS = (
        r"\blearn\b",
        r"\bteach me\b",
        r"\bexplain\b",
        r"\btell me about\b",
        r"\bhistory\b",
        r"\bhistorical\b",
        r"\bwhat is\b",
        r"\bwhat are\b",
        r"\bwhy\b",
        r"\borigin\b",
        r"\borigins\b",
        r"\bmeaning\b",
        r"\bspecial\b",
        r"\bspecialty\b",
        r"\bspeciality\b",
        r"\bfamous\b",
        r"\boverview\b",
    )

    CATALOG_PATTERNS = (
        r"\bdo you have\b",
        r"\bdo you sell\b",
        r"\bdo you carry\b",
        r"\bavailable\b",
        r"\bin stock\b",
        r"\bprice\b",
        r"\bcost\b",
        r"\bbuy\b",
        r"\bpurchase\b",
        r"\brecommend\b",
        r"\bsuggest\b",
        r"\bshow me\b",
        r"\bfind me\b",
        r"\blooking for\b",
        r"\bi need\b",
        r"\bi want\b",
        r"\bbudget\b",
        r"\bgift\b",
        r"\bexplore\b",
    )

    STRICT_CATALOG_PATTERNS = (
        r"\bdo you have\b",
        r"\bdo you sell\b",
        r"\bdo you carry\b",
        r"\bavailable\b",
        r"\bin stock\b",
        r"\bstock\b",
    )

    REFINEMENT_PATTERNS = (
        r"\bonly\b",
        r"\binstead\b",
        r"\bunder\s*(?:aed\s*)?\d+",
        r"\bbelow\s*(?:aed\s*)?\d+",
        r"\bless than\s*(?:aed\s*)?\d+",
        r"\bover\s*(?:aed\s*)?\d+",
        r"\babove\s*(?:aed\s*)?\d+",
        r"\bgreater than\s*(?:aed\s*)?\d+",
        r"\bmore than\s*(?:aed\s*)?\d+",
        r"\bat least\s*(?:aed\s*)?\d+",
        r"\bup to\s*(?:aed\s*)?\d+",
        r"\bmaximum\b",
        r"\bminimum\b",
        r"\bbudget\b",
        r"\b(19|20)\d{2}\b",
    )

    VERY_BRIEF_PATTERNS = (
        r"\beven shorter\b",
        r"\beven short\b",
        r"\bvery short\b",
        r"\bvery briefly\b",
        r"\bone sentence\b",
        r"\b1 sentence\b",
    )

    BRIEF_PATTERNS = (
        r"\bshort\b",
        r"\bshorter\b",
        r"\bbrief\b",
        r"\bbriefly\b",
        r"\bconcise\b",
        r"\bsimple explanation\b",
        r"\bsmall explanation\b",
    )

    DETAILED_PATTERNS = (
        r"\bmore detail\b",
        r"\bmore details\b",
        r"\bmore information\b",
        r"\bexplain more\b",
        r"\bdetailed\b",
        r"\bin depth\b",
        r"\belaborate\b",
    )

    def route(
        self,
        message: str,
        previous_mode: QueryMode | None = None,
    ) -> QueryRoute:

        text = message.strip().lower()

        style = self._detect_response_style(
            text
        )

        if not text:
            return QueryRoute(
                mode=previous_mode or "catalog",
                response_style=style,
            )

        # =================================================
        # IMPORTANT:
        # Product references MUST be checked BEFORE
        # search-refinement patterns such as "in stock".
        # =================================================

        if previous_mode in ("catalog", "mixed"):

            # ---------------------------------------------
            # Availability of one of the currently
            # selected/focused products.
            #
            # Examples:
            #
            # Is the cheaper one in stock?
            # Is that one available?
            # Is that coin in stock?
            # Is the expensive one available?
            # ---------------------------------------------

            if (
                self._contains_availability_question(text)
                and self._contains_product_reference(text)
            ):
                return QueryRoute(
                    mode=previous_mode,
                    is_follow_up=True,
                    follow_up_type=self._reference_source(
                        text
                    ),
                    product_action="availability",
                    response_style=style,
                    selection_limit=1,
                )

            # ---------------------------------------------
            # Details about focused product
            # ---------------------------------------------

            if re.search(
                r"\b(tell me more|more details?|more information)"
                r"\s+(about\s+)?(that|this)"
                r"\s+(coin|one|product)?\b",
                text,
            ):
                return QueryRoute(
                    mode=previous_mode,
                    is_follow_up=True,
                    follow_up_type="reuse_focus",
                    product_action="detail",
                    response_style=style,
                    selection_limit=1,
                )

            # ---------------------------------------------
            # Compare selected products
            # ---------------------------------------------

            if re.search(
                r"\bcompare\b",
                text,
            ):
                return QueryRoute(
                    mode=previous_mode,
                    is_follow_up=True,
                    follow_up_type="reuse_selection",
                    product_action="compare",
                    response_style=style,
                )

            # ---------------------------------------------
            # Cheapest from last result set
            # ---------------------------------------------

            if re.search(
                r"\bcheapest\b",
                text,
            ):
                return QueryRoute(
                    mode=previous_mode,
                    is_follow_up=True,
                    follow_up_type="reuse_results",
                    product_action="cheapest",
                    response_style=style,
                    selection_limit=1,
                )

            # ---------------------------------------------
            # Choose two from previous results
            # ---------------------------------------------

            if re.search(
                r"\b("
                r"choose two"
                r"|pick two"
                r"|choose one or two"
                r"|one or two"
                r")\b",
                text,
            ):
                return QueryRoute(
                    mode=previous_mode,
                    is_follow_up=True,
                    follow_up_type="reuse_results",
                    product_action="choose",
                    response_style=style,
                    selection_limit=2,
                )

            # ---------------------------------------------
            # Choose one from previous results
            # ---------------------------------------------

            if re.search(
                r"\b("
                r"which one"
                r"|choose one"
                r"|pick one"
                r"|which is better"
                r"|which would you choose"
                r")\b",
                text,
            ):
                return QueryRoute(
                    mode=previous_mode,
                    is_follow_up=True,
                    follow_up_type="reuse_results",
                    product_action="choose",
                    response_style=style,
                    selection_limit=1,
                )

            # ---------------------------------------------
            # Generic references
            # ---------------------------------------------

            if self._contains_product_reference(
                text
            ):
                return QueryRoute(
                    mode=previous_mode,
                    is_follow_up=True,
                    follow_up_type=self._reference_source(
                        text
                    ),
                    response_style=style,
                )

            # ---------------------------------------------
            # Search refinement
            #
            # IMPORTANT:
            # This happens only AFTER product-reference
            # handling.
            # ---------------------------------------------

            if self._matches_any(
                text,
                self.REFINEMENT_PATTERNS,
            ):
                return QueryRoute(
                    mode=previous_mode,
                    is_follow_up=True,
                    follow_up_type="refine_search",
                    response_style=style,
                )

        # =================================================
        # Normal new-intent classification
        # =================================================

        knowledge = self._matches_any(
            text,
            self.KNOWLEDGE_PATTERNS,
        )

        catalog = self._matches_any(
            text,
            self.CATALOG_PATTERNS,
        )

        strict_catalog = self._matches_any(
            text,
            self.STRICT_CATALOG_PATTERNS,
        )

        if knowledge and catalog:
            return QueryRoute(
                mode="mixed",
                strict_catalog_match=strict_catalog,
                response_style=style,
            )

        if knowledge:
            return QueryRoute(
                mode="knowledge",
                response_style=style,
            )

        if catalog:
            return QueryRoute(
                mode="catalog",
                strict_catalog_match=strict_catalog,
                response_style=style,
            )

        # =================================================
        # Generic continuation
        # =================================================

        if previous_mode is not None:

            return QueryRoute(
                mode=previous_mode,
                is_follow_up=True,
                follow_up_type=(
                    "reuse_selection"
                    if previous_mode
                    in ("catalog", "mixed")
                    else "none"
                ),
                response_style=style,
            )

        return QueryRoute(
            mode="catalog",
            response_style=style,
        )

    # =====================================================
    # Product-reference detection
    # =====================================================

    @staticmethod
    def _contains_product_reference(
        text: str,
    ) -> bool:

        patterns = (
            r"\bthat coin\b",
            r"\bthis coin\b",
            r"\bthat one\b",
            r"\bthis one\b",
            r"\bthe cheaper one\b",
            r"\bcheaper one\b",
            r"\bthe cheapest one\b",
            r"\bcheapest one\b",
            r"\bthe more expensive one\b",
            r"\bmore expensive one\b",
            r"\bthe expensive one\b",
            r"\bexpensive one\b",
            r"\bthe first one\b",
            r"\bfirst one\b",
            r"\bthe second one\b",
            r"\bsecond one\b",
            r"\bthe third one\b",
            r"\bthird one\b",
            r"\bthose two\b",
            r"\bthese two\b",
            r"\bthem\b",
        )

        return any(
            re.search(
                pattern,
                text,
                re.IGNORECASE,
            )
            for pattern in patterns
        )

    @staticmethod
    def _contains_availability_question(
        text: str,
    ) -> bool:

        return any(
            phrase in text
            for phrase in (
                "in stock",
                "available",
                "availability",
                "out of stock",
            )
        )

    @staticmethod
    def _reference_source(
        text: str,
    ) -> FollowUpType:

        # Comparative phrases refer naturally
        # to a previously selected group.
        if re.search(
            r"\b("
            r"cheaper one"
            r"|cheapest one"
            r"|more expensive one"
            r"|expensive one"
            r"|those two"
            r"|these two"
            r"|them"
            r")\b",
            text,
        ):
            return "reuse_selection"

        # Direct singular references use focus.
        return "reuse_focus"

    # =====================================================
    # Style
    # =====================================================

    def _detect_response_style(
        self,
        text: str,
    ) -> ResponseStyle:

        if self._matches_any(
            text,
            self.VERY_BRIEF_PATTERNS,
        ):
            return "very_brief"

        if self._matches_any(
            text,
            self.BRIEF_PATTERNS,
        ):
            return "brief"

        if self._matches_any(
            text,
            self.DETAILED_PATTERNS,
        ):
            return "detailed"

        return "normal"

    # =====================================================
    # Helpers
    # =====================================================

    @staticmethod
    def _matches_any(
        text: str,
        patterns: tuple[str, ...],
    ) -> bool:

        return any(
            re.search(
                pattern,
                text,
                flags=re.IGNORECASE,
            )
            for pattern in patterns
        )


query_router = QueryRouter()