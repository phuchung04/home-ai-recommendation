"""
seed_db.py  —  populate MongoDB with sample furniture products
Run: python seed_db.py
"""

import asyncio
from motor.motor_asyncio import AsyncIOMotorClient

MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "interior_db"
COLLECTION = "products"

PRODUCTS = [
    # Sofa
    {"name": "Nordic Sofa L-Shape", "category": "Sofa", "styles": ["Scandinavian", "Minimalist"],
     "price": 12500000, "dimensions": {"width": 240, "depth": 160, "height": 85},
     "colors": ["#F5F5F5", "#E0D5C5"], "image_url": "https://example.com/sofa1.jpg"},
    {"name": "Modern Fabric Sofa 3-Seat", "category": "Sofa", "styles": ["Modern"],
     "price": 9800000, "dimensions": {"width": 210, "depth": 85, "height": 80},
     "colors": ["#808080", "#A0A0A0"], "image_url": "https://example.com/sofa2.jpg"},
    {"name": "Classic Velvet Chesterfield", "category": "Sofa", "styles": ["Classic"],
     "price": 18000000, "dimensions": {"width": 195, "depth": 90, "height": 78},
     "colors": ["#4A235A", "#2C1A4E"], "image_url": "https://example.com/sofa3.jpg"},

    # Bàn
    {"name": "Minimalist Oak Coffee Table", "category": "Bàn", "styles": ["Minimalist", "Scandinavian"],
     "price": 3200000, "dimensions": {"width": 100, "depth": 50, "height": 45},
     "colors": ["#D4A96A", "#8B6914"], "image_url": "https://example.com/table1.jpg"},
    {"name": "Industrial Pipe Dining Table", "category": "Bàn", "styles": ["Industrial"],
     "price": 6500000, "dimensions": {"width": 160, "depth": 80, "height": 75},
     "colors": ["#2C2C2C", "#8B7355"], "image_url": "https://example.com/table2.jpg"},
    {"name": "White Marble Side Table", "category": "Bàn", "styles": ["Modern", "Classic"],
     "price": 2800000, "dimensions": {"width": 55, "depth": 55, "height": 50},
     "colors": ["#FFFFFF", "#E8E8E8"], "image_url": "https://example.com/table3.jpg"},

    # Ghế
    {"name": "Modern Armchair Wool", "category": "Ghế", "styles": ["Modern", "Scandinavian"],
     "price": 4800000, "dimensions": {"width": 80, "depth": 75, "height": 90},
     "colors": ["#B8860B", "#8B6914"], "image_url": "https://example.com/chair1.jpg"},
    {"name": "Bohemian Rattan Chair", "category": "Ghế", "styles": ["Bohemian", "Rustic"],
     "price": 2200000, "dimensions": {"width": 65, "depth": 65, "height": 150},
     "colors": ["#C8A96E", "#8B6914"], "image_url": "https://example.com/chair2.jpg"},
    {"name": "Eames Style Office Chair", "category": "Ghế", "styles": ["Modern", "Minimalist"],
     "price": 5500000, "dimensions": {"width": 60, "depth": 60, "height": 85},
     "colors": ["#2C2C2C", "#FFFFFF"], "image_url": "https://example.com/chair3.jpg"},

    # Giường
    {"name": "Platform Bed Queen Walnut", "category": "Giường", "styles": ["Modern", "Minimalist"],
     "price": 14000000, "dimensions": {"width": 160, "depth": 210, "height": 35},
     "colors": ["#5C4033", "#3E2723"], "image_url": "https://example.com/bed1.jpg"},
    {"name": "Scandinavian Linen Bed King", "category": "Giường", "styles": ["Scandinavian"],
     "price": 16500000, "dimensions": {"width": 180, "depth": 215, "height": 100},
     "colors": ["#F5F5F5", "#DCDCDC"], "image_url": "https://example.com/bed2.jpg"},

    # Tủ
    {"name": "6-Door Wardrobe White", "category": "Tủ", "styles": ["Modern", "Minimalist"],
     "price": 22000000, "dimensions": {"width": 240, "depth": 60, "height": 220},
     "colors": ["#FFFFFF", "#F0F0F0"], "image_url": "https://example.com/wardrobe1.jpg"},
    {"name": "Rustic Pine Wood Cabinet", "category": "Tủ", "styles": ["Rustic", "Bohemian"],
     "price": 8500000, "dimensions": {"width": 90, "depth": 45, "height": 160},
     "colors": ["#C8A96E", "#A0785A"], "image_url": "https://example.com/cabinet1.jpg"},

    # Kệ
    {"name": "Floating Wall Shelf Set", "category": "Kệ", "styles": ["Modern", "Minimalist"],
     "price": 1800000, "dimensions": {"width": 80, "depth": 20, "height": 15},
     "colors": ["#FFFFFF", "#F5F5F5"], "image_url": "https://example.com/shelf1.jpg"},
    {"name": "Industrial Metal Bookshelf", "category": "Kệ", "styles": ["Industrial"],
     "price": 4200000, "dimensions": {"width": 90, "depth": 35, "height": 180},
     "colors": ["#2C2C2C", "#5C5C5C"], "image_url": "https://example.com/shelf2.jpg"},

    # Đèn
    {"name": "Pendant Rattan Lamp", "category": "Đèn", "styles": ["Bohemian", "Rustic"],
     "price": 980000, "dimensions": {"width": 45, "depth": 45, "height": 35},
     "colors": ["#C8A96E", "#8B6914"], "image_url": "https://example.com/lamp1.jpg"},
    {"name": "Arc Floor Lamp Marble Base", "category": "Đèn", "styles": ["Modern", "Classic"],
     "price": 3200000, "dimensions": {"width": 40, "depth": 40, "height": 170},
     "colors": ["#FFFFFF", "#C0C0C0"], "image_url": "https://example.com/lamp2.jpg"},
]


async def seed():
    client = AsyncIOMotorClient(MONGO_URI)
    col = client[DB_NAME][COLLECTION]

    await col.drop()
    result = await col.insert_many(PRODUCTS)
    print(f"✅ Inserted {len(result.inserted_ids)} products into '{DB_NAME}.{COLLECTION}'")

    # Create indexes
    await col.create_index("styles")
    await col.create_index("category")
    await col.create_index([("dimensions.width", 1)])
    await col.create_index([("dimensions.depth", 1)])
    print("✅ Indexes created")

    client.close()


if __name__ == "__main__":
    asyncio.run(seed())