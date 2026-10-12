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


@dataclass(frozen=True)
class QueryRoute:
    mode: QueryMode

    strict_catalog_match: bool = False

    # Is this message continuing the previous turn?
    is_follow_up: bool = False

    # Does the user add/change search requirements?
    # Example:
    # "Only silver ones."
    refresh_products: bool = False

    response_style: ResponseStyle = "normal"

    # Example:
    # "Which one?" -> 1
    # "Choose two" -> 2
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
        r"\bspeciality\b",
        r"\bspecialty\b",
        r"\bmeaning\b",
        r"\bfamous\b",
        r"\bminted\b",
        r"\binformation about\b",
        r"\bdetails about\b",
        r"\bdetail about\b",
        r"\bmore about\b",
        r"\boverview of\b",
        r"\bsummary of\b",
        r"\bknowledge about\b",
        r"\bhow were\b",
        r"\bhow was\b",
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
        r"\brecommendation\b",
        r"\bsuggest\b",
        r"\bshow me\b",
        r"\bfind me\b",
        r"\blooking for\b",
        r"\bi need\b",
        r"\bi want\b",
        r"\bbudget\b",
        r"\bunder\s+\d+",
        r"\bbelow\s+\d+",
        r"\bless than\s+\d+",
        r"\bgreater than\s+\d+",
        r"\bmore than\s+\d+",
        r"\bover\s+\d+",
        r"\babove\s+\d+",
        r"\bat least\s+\d+",
        r"\bup to\s+\d+",
        r"\bcheapest\b",
        r"\bpremium\b",
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
        r"\bcarry\b",
    )

    FOLLOW_UP_PATTERNS = (
        r"^\s*make (it )?(even )?(more )?short(er)?",
        r"^\s*(even )?(more )?short(er)?",
        r"^\s*simple explanation",
        r"^\s*small explanation",
        r"^\s*brief explanation",
        r"^\s*explain (a little )?more",
        r"^\s*tell me more",
        r"^\s*more details?",
        r"^\s*more information",
        r"^\s*why\s*\??\s*$",
        r"^\s*how\s*\??\s*$",
        r"^\s*what about",
        r"^\s*which one",
        r"^\s*which is",
        r"^\s*choose one",
        r"^\s*choose two",
        r"^\s*choose one or two",
        r"^\s*choose .* from .* listed",
        r"^\s*compare them",
        r"^\s*compare these",
        r"^\s*compare only",
        r"^\s*the first one",
        r"^\s*the second one",
        r"^\s*the third one",
        r"^\s*this one",
        r"^\s*that one",
        r"^\s*tell me more about that",
        r"^\s*tell me more about this",
    )

    # If one of these appears in a follow-up,
    # we must run retrieval again.
    #
    # Example:
    #
    # Previous:
    # "UAE coins above 500"
    #
    # New:
    # "Only silver ones"
    #
    # We need a NEW search while keeping the
    # previous search requirements.
    SEARCH_REFRESH_PATTERNS = (
        r"\b(19|20)\d{2}\b",
        r"\bunder\s+\d+",
        r"\bbelow\s+\d+",
        r"\bless than\s+\d+",
        r"\bgreater than\s+\d+",
        r"\bmore than\s+\d+",
        r"\bover\s+\d+",
        r"\babove\s+\d+",
        r"\bat least\s+\d+",
        r"\bup to\s+\d+",
        r"\bbudget\b",
        r"\bsilver\b",
        r"\bgold\b",
        r"\bcopper\b",
        r"\bbronze\b",
        r"\bplatinum\b",
        r"\bin stock\b",
        r"\bavailable\b",
        r"\bonly\b",
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
        r"\bsimple explanation\b",
        r"\bsmall explanation\b",
        r"\bin short\b",
        r"\bsummarize\b",
        r"\bsummary\b",
        r"\bconcise\b",
        r"\bshort paragraph\b",
    )

    DETAILED_PATTERNS = (
        r"\bmore detail\b",
        r"\bmore details\b",
        r"\bexplain more\b",
        r"\bexplain a little more\b",
        r"\bmore information\b",
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
                mode=previous_mode or "catalog",
            )

        response_style = self._detect_response_style(
            text
        )

        selection_limit = self._detect_selection_limit(
            text
        )

        # -------------------------------------------------
        # Clear conversational follow-up
        # -------------------------------------------------

        if (
            previous_mode is not None
            and self._matches_any(
                text,
                self.FOLLOW_UP_PATTERNS,
            )
        ):
            refresh_products = (
                previous_mode in (
                    "catalog",
                    "mixed",
                )
                and self._matches_any(
                    text,
                    self.SEARCH_REFRESH_PATTERNS,
                )
            )

            return QueryRoute(
                mode=previous_mode,
                is_follow_up=True,
                refresh_products=refresh_products,
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

        # -------------------------------------------------
        # New mixed question
        # -------------------------------------------------

        if knowledge and catalog:
            return QueryRoute(
                mode="mixed",
                strict_catalog_match=strict_catalog,
                response_style=response_style,
                selection_limit=selection_limit,
            )

        # -------------------------------------------------
        # New knowledge question
        # -------------------------------------------------

        if knowledge:
            return QueryRoute(
                mode="knowledge",
                response_style=response_style,
            )

        # -------------------------------------------------
        # New catalog question
        # -------------------------------------------------

        if catalog:
            return QueryRoute(
                mode="catalog",
                strict_catalog_match=strict_catalog,
                response_style=response_style,
                selection_limit=selection_limit,
            )

        # -------------------------------------------------
        # Ambiguous message but we already have context
        #
        # Example:
        # "Only silver ones."
        #
        # No normal catalog phrase is present,
        # but it is clearly modifying the previous query.
        # -------------------------------------------------

        if previous_mode is not None:

            refresh_products = (
                previous_mode in (
                    "catalog",
                    "mixed",
                )
                and self._matches_any(
                    text,
                    self.SEARCH_REFRESH_PATTERNS,
                )
            )

            return QueryRoute(
                mode=previous_mode,
                is_follow_up=True,
                refresh_products=refresh_products,
                response_style=response_style,
                selection_limit=selection_limit,
            )

        return QueryRoute(
            mode="catalog",
            response_style=response_style,
        )

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

        # One explicit choice
        if re.search(
            r"\bwhich one\b",
            text,
            re.IGNORECASE,
        ):
            return 1

        if re.search(
            r"\bwhich is the cheapest\b",
            text,
            re.IGNORECASE,
        ):
            return 1

        if re.search(
            r"\bwhich is cheapest\b",
            text,
            re.IGNORECASE,
        ):
            return 1

        if re.search(
            r"\btell me more about (that|this) coin\b",
            text,
            re.IGNORECASE,
        ):
            return 1

        if re.search(
            r"\bchoose one or two\b",
            text,
            re.IGNORECASE,
        ):
            return 2

        if re.search(
            r"\bone or two\b",
            text,
            re.IGNORECASE,
        ):
            return 2

        if re.search(
            r"\bchoose two\b",
            text,
            re.IGNORECASE,
        ):
            return 2

        if re.search(
            r"\bthose two\b",
            text,
            re.IGNORECASE,
        ):
            return 2

        if re.search(
            r"\bchoose one\b",
            text,
            re.IGNORECASE,
        ):
            return 1

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