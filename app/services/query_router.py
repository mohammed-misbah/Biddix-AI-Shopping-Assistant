import re
from dataclasses import dataclass
from typing import Literal


QueryMode = Literal[
    "catalog",
    "knowledge",
    "mixed",
]


@dataclass(frozen=True)
class QueryRoute:
    mode: QueryMode
    strict_catalog_match: bool = False


class QueryRouter:

    KNOWLEDGE_PATTERNS = (
        r"\blearn\b",
        r"\bexplain\b",
        r"\bhistory\b",
        r"\bhistorical\b",
        r"\bwhat is\b",
        r"\bwhat are\b",
        r"\btell me about\b",
        r"\bteach me\b",
        r"\bwhy\b",
        r"\borigin\b",
        r"\borigins\b",
        r"\bspeciality\b",
        r"\bspecialty\b",
        r"\bspecial\b",
        r"\bmeaning\b",
        r"\bfamous\b",
        r"\bminted\b",
        r"\bmade\b",
        r"\bused for\b",
        r"\bdifference between\b",
        r"\bhow were\b",
        r"\bhow was\b",
    )

    CATALOG_PATTERNS = (
        r"\bdo you have\b",
        r"\bdo you sell\b",
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
        r"\bunder\s+\d+",
        r"\bbelow\s+\d+",
        r"\bup to\s+\d+",
        r"\bcheapest\b",
        r"\bpremium\b",
        r"\bgift\b",
        r"\bexplore\b",
        r"\bcollection\b",
    )

    STRICT_CATALOG_PATTERNS = (
        r"\bdo you have\b",
        r"\bdo you sell\b",
        r"\bavailable\b",
        r"\bin stock\b",
        r"\bcarry\b",
        r"\bstock\b",
    )

    def route(
        self,
        message: str,
    ) -> QueryRoute:

        text = message.strip().lower()

        if not text:
            return QueryRoute(
                mode="catalog",
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
            )

        if knowledge:
            return QueryRoute(
                mode="knowledge",
            )

        if catalog:
            return QueryRoute(
                mode="catalog",
                strict_catalog_match=strict_catalog,
            )

        # Biddix is primarily a shopping assistant.
        return QueryRoute(
            mode="catalog",
        )

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