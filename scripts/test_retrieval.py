from app.services.retrieval_service import retrieval_service


query = "I want something meaningful for a UAE collector"

results = retrieval_service.retrieve(
    query=query,
    limit=5,
)

for product in results:
    print(
        product["product_id"],
        "-",
        product["name"],
        "-",
        product["price"],
        product["currency"],
    )