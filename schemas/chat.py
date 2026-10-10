from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
    )

    session_id: str | None = None
    current_product_id: str | None = None


class ProductReference(BaseModel):
    product_id: str
    name: str

    category: str | None = None
    year: int | None = None
    material: str | None = None
    weight: str | None = None
    purity: str | None = None

    price: float | None = None
    currency: str | None = None

    description: str | None = None
    product_url: str | None = None


class ChatResponse(BaseModel):
    answer: str
    session_id: str

    products: list[ProductReference] = Field(
        default_factory=list
    )