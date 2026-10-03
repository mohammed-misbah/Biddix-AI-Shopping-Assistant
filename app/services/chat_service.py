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

        # 1. Search products locally
        products = product_service.search_products(
            query=message,
            limit=5,
        )

        # 2. Send only matched products to Gemini
        answer = await llm_service.generate_response(
            user_message=message,
            products=products,
        )

        # 3. Return response
        return {
            "answer": answer,
            "session_id": session_id,
            "products": products,
        }


chat_service = ChatService()