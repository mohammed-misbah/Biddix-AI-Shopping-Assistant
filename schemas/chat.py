from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)


class ChatRequest(BaseModel):

    # -------------------------------------------------
    # REQUEST SECURITY / VALIDATION
    #
    # str_strip_whitespace=True
    # removes spaces from the beginning/end.
    #
    # extra="forbid"
    # rejects JSON fields that our API does not expect.
    # -------------------------------------------------

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
    )

    message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
    )

    session_id: str | None = Field(
        default=None,
        max_length=100,
    )

    current_product_id: str | None = Field(
        default=None,
        max_length=100,
    )
     

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