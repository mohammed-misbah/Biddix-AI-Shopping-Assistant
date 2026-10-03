from typing import Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000,)

    session_id: Optional[str] = None

    current_product_id: Optional[str] = None


class ProductReference(BaseModel):
    product_id: str
    name: str
    price: float
    currency: str
    product_url: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    products: list[ProductReference] = []