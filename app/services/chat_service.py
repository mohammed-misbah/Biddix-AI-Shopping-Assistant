import uuid

from app.services.llm_service import llm_service
from app.services.product_service import product_service


class ChatService:

    async def chat(
        self,
        message: str,
        session_id: str | None = None,
    ):

        if not session_id:
            session_id = str(uuid.uuid4())

        products = product_service.search_products(
            query=message,
            limit=10,
        )

        answer = await llm_service.generate_response(
            user_message=message,
            products=products,
        )

        response_products = [
            {
                "product_id": product.get("product_id"),
                "name": product.get("name"),
                "category": product.get("category"),
                "year": product.get("year"),
                "material": product.get("material"),
                "weight": product.get("weight"),
                "purity": product.get("purity"),
                "price": product.get("price"),
                "currency": product.get("currency"),
                "description": product.get("description"),
                "product_url": product.get("product_url"),
            }
            for product in products
        ]

        return {
            "answer": answer,
            "session_id": session_id,
            "products": response_products,
        }


chat_service = ChatService()