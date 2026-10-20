
import logging
import re
from typing import Any

from app.services.product_service import product_service


logger = logging.getLogger(__name__)


class GroundingService:
    """
    Check customer-facing product claims against
    real catalog data.

    If an answer contains a detected contradiction,
    replace it with a safe catalog-based answer.
    """

    @staticmethod
    def _money_values(text: str) -> set[float]:
        patterns = (
            r"\bAED\s*([\d,]+(?:\.\d+)?)\b",
            r"\b([\d,]+(?:\.\d+)?)\s*AED\b",
        )

        amounts = set()

        for pattern in patterns:
            for match in re.finditer(
                pattern,
                text,
                re.IGNORECASE,
            ):
                try:
                    amounts.add(
                        float(
                            match.group(1).replace(",", "")
                        )
                    )
                except ValueError:
                    continue

        return amounts

    @staticmethod
    def _product_id(
        product: dict[str, Any],
    ) -> str:
        return str(
            product.get("product_id", "")
        ).strip()

    def _check(
        self,
        answer: str,
        products: list[dict[str, Any]],
    ) -> list[str]:
        issues = []

        allowed_ids = {
            self._product_id(product)
            for product in products
        }

        # -----------------------------------------
        # 1. Check known product names.
        #
        # An answer cannot introduce a different
        # catalog product without product context.
        # -----------------------------------------

        for catalog_product in product_service.products:
            name = str(
                catalog_product.get("name", "")
            ).strip()

            if not name:
                continue

            if re.search(
                rf"(?<!\w){re.escape(name)}(?!\w)",
                answer,
                re.IGNORECASE,
            ):
                catalog_id = self._product_id(
                    catalog_product
                )

                if catalog_id not in allowed_ids:
                    issues.append(
                        "Product outside supplied context"
                    )
                    break

        # -----------------------------------------
        # 2. Check explicit AED amounts.
        #
        # Every amount must match a supplied
        # product's actual catalog price.
        # -----------------------------------------

        real_prices = set()

        for product in products:
            try:
                real_prices.add(
                    round(
                        float(product["price"]),
                        2,
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue

        stated_prices = self._money_values(answer)

        for amount in stated_prices:
            if round(amount, 2) not in real_prices:
                issues.append(
                    "Price not found in supplied products"
                )
                break

        # -----------------------------------------
        # 3. Check product links.
        #
        # A URL must belong to a supplied product.
        # -----------------------------------------

        allowed_urls = {
            str(product.get("product_url", "")).strip()
            for product in products
            if product.get("product_url")
        }

        mentioned_paths = re.findall(
            r"/products/[A-Za-z0-9_-]+",
            answer,
        )

        for path in mentioned_paths:
            if not any(
                path == url
                or url.endswith(path)
                for url in allowed_urls
            ):
                issues.append(
                    "Unsupported product link"
                )
                break

        # -----------------------------------------
        # 4. Check availability claims.
        #
        # When availability is unconfirmed,
        # Gemini must not claim confirmed stock.
        # -----------------------------------------

        claims_in_stock = bool(
            re.search(
                r"\b(?:in stock|available now|"
                r"currently available)\b",
                answer,
                re.IGNORECASE,
            )
        )

        claims_out_of_stock = bool(
            re.search(
                r"\b(?:out of stock|sold out)\b",
                answer,
                re.IGNORECASE,
            )
        )

        if claims_in_stock or claims_out_of_stock:
            all_confirmed = bool(products) and all(
                product.get(
                    "availability_confirmed"
                ) is True
                for product in products
            )

            if not all_confirmed:
                issues.append(
                    "Unconfirmed availability claim"
                )
            else:
                try:
                    stock_values = [
                        int(product.get("stock", 0))
                        for product in products
                    ]

                    if (
                        claims_in_stock
                        and not all(
                            stock > 0
                            for stock in stock_values
                        )
                    ):
                        issues.append(
                            "Incorrect in-stock claim"
                        )

                    if (
                        claims_out_of_stock
                        and not all(
                            stock <= 0
                            for stock in stock_values
                        )
                    ):
                        issues.append(
                            "Incorrect out-of-stock claim"
                        )

                except (TypeError, ValueError):
                    issues.append(
                        "Invalid stock data"
                    )

        return issues

    @staticmethod
    def _safe_fallback(
        products: list[dict[str, Any]],
    ) -> str:
        if not products:
            return (
                "I couldn't confirm a matching "
                "product for that request. "
                "Please tell me what you are "
                "looking for."
            )

        lines = [
            "Here are the product details "
            "I can confirm:"
        ]

        for index, product in enumerate(
            products,
            start=1,
        ):
            name = str(
                product.get("name", "Product")
            )

            details = [name]

            if product.get("price") is not None:
                currency = str(
                    product.get("currency") or "AED"
                )

                details.append(
                    f"{currency} {product['price']}"
                )

            if product.get("material"):
                details.append(
                    str(product["material"])
                )

            if product.get("weight"):
                details.append(
                    str(product["weight"])
                )

            if product.get("purity"):
                details.append(
                    str(product["purity"])
                    + " purity"
                )

            confirmed = (
                product.get(
                    "availability_confirmed"
                ) is True
            )

            if confirmed:
                try:
                    stock = int(
                        product.get("stock", 0)
                    )
                    details.append(
                        "In stock"
                        if stock > 0
                        else "Out of stock"
                    )
                except (TypeError, ValueError):
                    details.append(
                        "Availability unconfirmed"
                    )
            else:
                details.append(
                    "Availability unconfirmed"
                )

            lines.append(
                f"{index}. " + " - ".join(details)
            )

        return "\n".join(lines)

    def ground_response(
        self,
        answer: str,
        products: list[dict[str, Any]],
        mode: str,
        request_id: str = "unknown",
    ) -> str:
        """
        Return a checked answer.

        Knowledge-only answers are not verified
        against the product catalog here.
        """

        if mode == "knowledge":
            return answer

        issues = self._check(
            answer=answer,
            products=products,
        )

        if not issues:
            return answer

        logger.warning(
            "GROUNDING CHECK FAILED | "
            "request_id=%s | issues=%s",
            request_id,
            issues,
        )

        return self._safe_fallback(products)


grounding_service = GroundingService()
