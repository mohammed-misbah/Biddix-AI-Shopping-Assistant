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
    "reuse_products",
    "refine_search",
]


@dataclass(frozen=True)
class QueryRoute:
    mode: QueryMode

    strict_catalog_match: bool = False

    is_follow_up: bool = False

    follow_up_type: FollowUpType = "none"

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
        r"\bfamous\b",
        r"\bspecial\b",
        r"\bspeciality\b",
        r"\bspecialty\b",
        r"\binformation about\b",
        r"\bdetails about\b",
        r"\bdetail about\b",
        r"\boverview\b",
        r"\bsummarize\b",
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
        r"\bcollection\b",
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

    PRODUCT_REFERENCE_PATTERNS = (
        r"\bwhich one\b",
        r"\bwhich ones\b",
        r"\bwhich is cheaper\b",
        r"\bwhich is cheapest\b",
        r"\bwhich is the cheapest\b",
        r"\bwhich is better\b",
        r"\bchoose one\b",
        r"\bchoose two\b",
        r"\bchoose one or two\b",
        r"\bpick one\b",
        r"\bpick two\b",
        r"\bcompare them\b",
        r"\bcompare these\b",
        r"\bcompare those\b",
        r"\bthose two\b",
        r"\bthese two\b",
        r"\bthat coin\b",
        r"\bthis coin\b",
        r"\bthat one\b",
        r"\bthis one\b",
        r"\bfirst one\b",
        r"\bsecond one\b",
        r"\bthird one\b",
        r"\btell me more about that\b",
        r"\btell me more about this\b",
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
        r"\bin stock\b",
        r"\bavailable\b",
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

        if not text:
            return QueryRoute(
                mode=previous_mode or "catalog"
            )

        response_style = (
            self._detect_response_style(text)
        )

        selection_limit = (
            self._detect_selection_limit(text)
        )

        # -----------------------------------------
        # Product-reference follow-up
        # -----------------------------------------

        if (
            previous_mode in ("catalog", "mixed")
            and self._matches_any(
                text,
                self.PRODUCT_REFERENCE_PATTERNS,
            )
        ):
            return QueryRoute(
                mode=previous_mode,
                is_follow_up=True,
                follow_up_type="reuse_products",
                response_style=response_style,
                selection_limit=selection_limit,
            )

        # -----------------------------------------
        # Search refinement
        # -----------------------------------------

        if (
            previous_mode in ("catalog", "mixed")
            and self._looks_like_refinement(text)
        ):
            return QueryRoute(
                mode=previous_mode,
                is_follow_up=True,
                follow_up_type="refine_search",
                response_style=response_style,
                selection_limit=selection_limit,
            )

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
                response_style=response_style,
            )

        if knowledge:
            return QueryRoute(
                mode="knowledge",
                response_style=response_style,
            )

        if catalog:
            return QueryRoute(
                mode="catalog",
                strict_catalog_match=strict_catalog,
                response_style=response_style,
            )

        # -----------------------------------------
        # Normal conversation continuation
        # -----------------------------------------

        if previous_mode is not None:

            return QueryRoute(
                mode=previous_mode,
                is_follow_up=True,
                follow_up_type=(
                    "reuse_products"
                    if previous_mode
                    in ("catalog", "mixed")
                    else "none"
                ),
                response_style=response_style,
                selection_limit=selection_limit,
            )

        return QueryRoute(
            mode="catalog",
            response_style=response_style,
        )

    def _looks_like_refinement(
        self,
        text: str,
    ) -> bool:

        if self._matches_any(
            text,
            self.REFINEMENT_PATTERNS,
        ):
            return True

        # Short fragments such as:
        #
        # "silver ones"
        # "gold only"
        # "2024"
        #
        # are commonly refinements.
        words = text.split()

        if len(words) <= 5:
            return True

        return False

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

    @staticmethod
    def _detect_selection_limit(
        text: str,
    ) -> int | None:

        if re.search(
            r"\b(which one|pick one|choose one)\b",
            text,
        ):
            return 1

        if re.search(
            r"\bcheapest\b",
            text,
        ):
            return 1

        if re.search(
            r"\btell me more about (that|this)\b",
            text,
        ):
            return 1

        if re.search(
            r"\b(one or two|choose two|pick two|those two|these two)\b",
            text,
        ):
            return 2

        return None

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