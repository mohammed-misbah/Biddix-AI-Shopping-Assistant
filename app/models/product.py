from sqlalchemy import Column, Integer, String, Float
from app.db.database import Base


class Product(Base):
    __tablename__ = "products"

    id = Column(Integer, primary_key=True)
    product_id = Column(String, unique=True, index=True)
    name = Column(String)
    category = Column(String)
    year = Column(Integer)
    material = Column(String)
    weight = Column(String)
    purity = Column(String)
    price = Column(Float)
    currency = Column(String)
    stock = Column(Integer)
    description = Column(String)
    product_url = Column(String)