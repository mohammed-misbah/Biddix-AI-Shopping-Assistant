import uuid

from app.services.llm_service import (
    llm_service,
)
from app.services.query_router import (
    query_router,
)
from app.services.retrieval_service import (
    retrieval_service,
)


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

        # ==========================================
        # 1. Understand what customer wants
        # ==========================================

        route = query_router.route(
            message
        )

        products = []

        # ==========================================
        # 2. Catalog / mixed questions use RAG
        # ==========================================

        if route.mode in (
            "catalog",
            "mixed",
        ):

            products = (
                retrieval_service.retrieve(
                    query=message,
                    limit=5,
                    strict_match=(
                        route.strict_catalog_match
                    ),
                )
            )

        # ==========================================
        # 3. Knowledge questions do NOT use RAG
        # ==========================================

        elif route.mode == "knowledge":

            products = []

        # ==========================================
        # 4. Generate final response
        # ==========================================

        answer = await llm_service.generate_response(
            user_message=message,
            products=products,
            mode=route.mode,
        )

        # ==========================================
        # 5. Product cards
        # ==========================================

        response_products = [
            {
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

                "stock":
                    product.get(
                        "stock"
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
            for product in products
        ]

        return {
            "answer": answer,
            "session_id": session_id,
            "products": response_products,
        }


chat_service = ChatService()