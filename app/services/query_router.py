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

    # =====================================================
    # KNOWLEDGE INTENT
    # =====================================================

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

    # =====================================================
    # GENERAL CATALOG INTENT
    # =====================================================

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
        r"\bfind\b",
        r"\bsearch for\b",
        r"\blook for\b",
        r"\blooking for\b",
        r"\bi need\b",
        r"\bi want\b",
        r"\bbudget\b",
        r"\bgift\b",
        r"\bexplore\b",
    )

    # =====================================================
    # STRONG SHOPPING SIGNALS
    #
    # These are used to distinguish:
    #
    # "I need to learn about German coins"
    # -> knowledge
    #
    # from:
    #
    # "I need to find German coins under AED 500"
    # -> catalog / mixed
    # =====================================================

    STRONG_CATALOG_PATTERNS = (
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
        r"\bfind\b",
        r"\bsearch for\b",
        r"\blook for\b",
        r"\bbudget\b",
        r"\bgift\b",
        r"\bexplore\b",
    )

    # =====================================================
    # STRICT CATALOG
    #
    # These requests are specifically asking whether a
    # product exists / is available in the catalog.
    # =====================================================

    STRICT_CATALOG_PATTERNS = (
        r"\bdo you have\b",
        r"\bdo you sell\b",
        r"\bdo you carry\b",
        r"\bavailable\b",
        r"\bin stock\b",
        r"\bstock\b",
    )

    # =====================================================
    # EXPLICIT NEW SEARCH
    #
    # VERY IMPORTANT:
    #
    # These MUST be detected globally, before previous-mode
    # handling.
    #
    # That means:
    #
    # German knowledge conversation
    # -> "Find silver under AED 500"
    #
    # becomes a clean new catalog search.
    #
    # Also:
    #
    # old gold search
    # -> "Find silver under AED 500"
    #
    # resets old filters.
    # =====================================================

    FRESH_SEARCH_PATTERNS = (
        r"^\s*find\b",
        r"^\s*find me\b",
        r"^\s*show me\b",
        r"^\s*search\b",
        r"^\s*search for\b",
        r"^\s*look for\b",
        r"^\s*explore\b",
        r"^\s*do you have\b",
        r"^\s*do you sell\b",
        r"^\s*do you carry\b",
    )

    # =====================================================
    # SEARCH REFINEMENT
    #
    # These keep the previous search context when they are
    # short follow-ups such as:
    #
    # "Only silver ones"
    # "Under AED 500 instead"
    # =====================================================

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

    # =====================================================
    # RESPONSE STYLE
    # =====================================================

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

    # =====================================================
    # MAIN ROUTER
    # =====================================================

    def route(
        self,
        message: str,
        previous_mode: QueryMode | None = None,
    ) -> QueryRoute:

        text = message.strip().lower()

        style = self._detect_response_style(
            text
        )

        # -------------------------------------------------
        # Empty input
        # -------------------------------------------------

        if not text:
            return QueryRoute(
                mode=previous_mode or "catalog",
                response_style=style,
            )

        # =================================================
        # 1. GLOBAL EXPLICIT FRESH SEARCH
        #
        # This MUST happen before checking previous_mode.
        #
        # It fixes:
        #
        # knowledge
        # -> "Find silver under AED 500"
        #
        # and:
        #
        # old filtered catalog search
        # -> "Find silver under AED 500"
        # =================================================

        if self._matches_any(
            text,
            self.FRESH_SEARCH_PATTERNS,
        ):
            return QueryRoute(
                mode="catalog",
                strict_catalog_match=(
                    self._matches_any(
                        text,
                        self.STRICT_CATALOG_PATTERNS,
                    )
                ),
                is_follow_up=False,
                follow_up_type="none",
                product_action="none",
                response_style=style,
            )

        # =================================================
        # 2. PRODUCT / CATALOG FOLLOW-UPS
        # =================================================

        if previous_mode in (
            "catalog",
            "mixed",
        ):

            # ---------------------------------------------
            # Availability of referenced product
            #
            # Example:
            # "Is the cheaper one in stock?"
            #
            # This must beat refinement logic.
            # ---------------------------------------------

            if (
                self._contains_availability_question(
                    text
                )
                and self._contains_product_reference(
                    text
                )
            ):
                return QueryRoute(
                    mode=previous_mode,
                    is_follow_up=True,
                    follow_up_type=(
                        self._reference_source(
                            text
                        )
                    ),
                    product_action="availability",
                    response_style=style,
                    selection_limit=1,
                )

            # ---------------------------------------------
            # Focused-product factual/details follow-up
            #
            # Examples:
            # "Tell me more about that coin"
            # "What's its purity?"
            # ---------------------------------------------

            if self._is_focused_detail_reference(
                text
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
            # Cheapest from last full result set
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
            # Choose two from previous full results
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
            # Choose one from previous full results
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
            #
            # Examples:
            # that coin
            # those two
            # second one
            # ---------------------------------------------

            if self._contains_product_reference(
                text
            ):
                return QueryRoute(
                    mode=previous_mode,
                    is_follow_up=True,
                    follow_up_type=(
                        self._reference_source(
                            text
                        )
                    ),
                    product_action="none",
                    response_style=style,
                )

            # ---------------------------------------------
            # Search refinement
            #
            # Examples:
            # "Only silver ones"
            # "Under AED 500 instead"
            #
            # Because fresh search commands were already
            # caught globally above, these are now safe.
            # ---------------------------------------------

            if self._matches_any(
                text,
                self.REFINEMENT_PATTERNS,
            ):
                return QueryRoute(
                    mode=previous_mode,
                    is_follow_up=True,
                    follow_up_type="refine_search",
                    product_action="none",
                    response_style=style,
                )

        # =================================================
        # 3. NEW INTENT CLASSIFICATION
        # =================================================

        knowledge = self._matches_any(
            text,
            self.KNOWLEDGE_PATTERNS,
        )

        catalog = self._matches_any(
            text,
            self.CATALOG_PATTERNS,
        )

        strict_catalog = (
            self._matches_any(
                text,
                self.STRICT_CATALOG_PATTERNS,
            )
        )

        # ---------------------------------------------
        # Knowledge + weak shopping wording
        #
        # "I need to learn about German coins"
        #
        # should remain knowledge.
        # ---------------------------------------------

        if knowledge and catalog:

            strong_catalog = (
                self._matches_any(
                    text,
                    self.STRONG_CATALOG_PATTERNS,
                )
            )

            if not strong_catalog:
                return QueryRoute(
                    mode="knowledge",
                    response_style=style,
                )

            return QueryRoute(
                mode="mixed",
                strict_catalog_match=(
                    strict_catalog
                ),
                response_style=style,
            )

        # ---------------------------------------------
        # Pure knowledge
        # ---------------------------------------------

        if knowledge:
            return QueryRoute(
                mode="knowledge",
                response_style=style,
            )

        # ---------------------------------------------
        # Pure catalog
        # ---------------------------------------------

        if catalog:
            return QueryRoute(
                mode="catalog",
                strict_catalog_match=(
                    strict_catalog
                ),
                response_style=style,
            )

        # =================================================
        # 4. GENERIC CONTINUATION
        #
        # This is only used after explicit fresh search,
        # product references, refinement, and new-intent
        # classification have all failed.
        # =================================================

        if previous_mode is not None:

            return QueryRoute(
                mode=previous_mode,
                is_follow_up=True,
                follow_up_type=(
                    "reuse_selection"
                    if previous_mode
                    in (
                        "catalog",
                        "mixed",
                    )
                    else "none"
                ),
                response_style=style,
            )

        # -------------------------------------------------
        # Default completely new conversation
        # -------------------------------------------------

        return QueryRoute(
            mode="catalog",
            response_style=style,
        )

    # =====================================================
    # FOCUSED DETAIL REFERENCE
    # =====================================================

    @staticmethod
    def _is_focused_detail_reference(
        text: str,
    ) -> bool:

        patterns = (
            # Tell me more about that coin
            (
                r"\b("
                r"tell me more"
                r"|more details?"
                r"|more information"
                r")"
                r"\s+(about\s+)?"
                r"(that|this)"
                r"\s+(coin|one|product)?\b"
            ),

            # What's its purity?
            (
                r"\bwhat(?:'s| is)\s+its\s+"
                r"("
                r"purity"
                r"|price"
                r"|cost"
                r"|weight"
                r"|year"
                r"|material"
                r"|denomination"
                r")\b"
            ),

            # How much is it?
            r"\bhow much is (it|that|this)\b",
        )

        return any(
            re.search(
                pattern,
                text,
            )
            for pattern in patterns
        )

    # =====================================================
    # PRODUCT REFERENCE DETECTION
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

    # =====================================================
    # AVAILABILITY DETECTION
    # =====================================================

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

    # =====================================================
    # CHOOSE PRODUCT MEMORY SOURCE
    # =====================================================

    @staticmethod
    def _reference_source(
        text: str,
    ) -> FollowUpType:

        # Comparative / plural references belong to
        # previously selected products.

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

        # Direct singular reference uses focused product.
        return "reuse_focus"

    # =====================================================
    # RESPONSE STYLE
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
    # REGEX HELPER
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