"""
mongo_service.py
Hard-filter candidates from MongoDB, then rank by color distance.
"""

import os
import math
from typing import List, Optional

from motor.motor_asyncio import AsyncIOMotorClient

from app.models.schemas import GeminiAnalysisResult, Product, ProductDimensions

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("MONGO_DB", "interior_db")
COLLECTION = os.getenv("MONGO_COLLECTION", "products")

_client: Optional[AsyncIOMotorClient] = None


def get_client() -> AsyncIOMotorClient:
    global _client
    if _client is None:
        _client = AsyncIOMotorClient(MONGO_URI)
    return _client


# ── Color utilities ──────────────────────────────────────────────────────────

def _hex_to_rgb(hex_color: str):
    h = hex_color.lstrip("#")
    if len(h) != 6:
        return (128, 128, 128)
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def _color_distance(hex1: str, hex2: str) -> float:
    """Euclidean distance in RGB space (0–441.67)."""
    r1, g1, b1 = _hex_to_rgb(hex1)
    r2, g2, b2 = _hex_to_rgb(hex2)
    return math.sqrt((r1 - r2) ** 2 + (g1 - g2) ** 2 + (b1 - b2) ** 2)


def _min_color_distance(product_colors: List[str], target_colors: List[str]) -> float:
    """Minimum distance from any product color to any target color."""
    if not product_colors or not target_colors:
        return 999.0
    distances = [
        _color_distance(pc, tc)
        for pc in product_colors
        for tc in target_colors
    ]
    return min(distances)


# ── Hard filter ──────────────────────────────────────────────────────────────

def _build_mongo_query(analysis: GeminiAnalysisResult) -> dict:
    rf = analysis.recommendedFilter
    query = {
        "styles": {"$in": rf.styles},
        "category": {"$in": rf.categories},
        "dimensions.width": {"$lte": rf.maxProductWidth},
        "dimensions.depth": {"$lte": rf.maxProductDepth},
    }
    return query


# ── Main function ────────────────────────────────────────────────────────────

async def get_recommendations(
    analysis: GeminiAnalysisResult,
    top_n: int = 15,
    candidate_limit: int = 50,
) -> tuple[List[Product], int]:
    """
    1. Hard-filter up to `candidate_limit` products from MongoDB.
    2. Rank by color distance to recommendedFilter.colorHexRange.
    3. Return top_n products + total candidates count.
    """
    db = get_client()[DB_NAME]
    col = db[COLLECTION]

    query = _build_mongo_query(analysis)
    target_colors = analysis.recommendedFilter.colorHexRange

    cursor = col.find(query).limit(candidate_limit)
    candidates: List[Product] = []

    async for doc in cursor:
        prod_colors = doc.get("colors", [])
        dist = _min_color_distance(prod_colors, target_colors)

        dim_raw = doc.get("dimensions", {})
        dims = ProductDimensions(
            width=dim_raw.get("width"),
            depth=dim_raw.get("depth"),
            height=dim_raw.get("height"),
        ) if dim_raw else None

        candidates.append(Product(
            id=str(doc.get("_id", "")),
            name=doc.get("name", ""),
            category=doc.get("category", ""),
            styles=doc.get("styles", []),
            price=doc.get("price"),
            dimensions=dims,
            colors=prod_colors,
            imageUrl=doc.get("imageUrl"),
            color_distance=round(dist, 2),
        ))

    total_candidates = len(candidates)

    # Semantic ranking: lower distance = better color match
    ranked = sorted(candidates, key=lambda p: p.color_distance or 999)
    return ranked[:top_n], total_candidates


# ── Mock data for local development (no MongoDB needed) ─────────────────────

MOCK_PRODUCTS = [
    {
        "_id": "p001", "name": "Nordic Sofa L-Shape", "category": "Sofa",
        "styles": ["Scandinavian", "Modern"], "price": 12500000,
        "dimensions": {"width": 240, "depth": 160, "height": 85},
        "colors": ["#F5F5F5", "#E0D5C5"], "imageUrl": "https://example.com/sofa1.jpg",
    },
    {
        "_id": "p002", "name": "Minimalist Coffee Table", "category": "Bàn",
        "styles": ["Minimalist"], "price": 3200000,
        "dimensions": {"width": 100, "depth": 50, "height": 45},
        "colors": ["#8B7355", "#5C4A32"], "imageUrl": "https://example.com/table1.jpg",
    },
    {
        "_id": "p003", "name": "Modern Armchair", "category": "Ghế",
        "styles": ["Modern"], "price": 4800000,
        "dimensions": {"width": 80, "depth": 75, "height": 90},
        "colors": ["#2C2C2C", "#FFFFFF"], "imageUrl": "https://example.com/chair1.jpg",
    },
    {
        "_id": "p004", "name": "Oak Bookshelf", "category": "Kệ",
        "styles": ["Scandinavian"], "price": 5500000,
        "dimensions": {"width": 90, "depth": 30, "height": 180},
        "colors": ["#D4A96A", "#8B6914"], "imageUrl": "https://example.com/shelf1.jpg",
    },
    {
        "_id": "p005", "name": "Industrial Pendant Light", "category": "Đèn",
        "styles": ["Industrial"], "price": 1200000,
        "dimensions": {"width": 30, "depth": 30, "height": 40},
        "colors": ["#2C2C2C", "#B8860B"], "imageUrl": "https://example.com/lamp1.jpg",
    },
]


async def get_recommendations_mock(
    analysis: GeminiAnalysisResult,
    top_n: int = 15,
) -> tuple[List[Product], int]:
    """Mock version for development without MongoDB."""
    rf = analysis.recommendedFilter
    target_colors = rf.colorHexRange
    candidates = []

    for doc in MOCK_PRODUCTS:
        # Soft filter: check style and category overlap
        if not any(s in rf.styles for s in doc["styles"]) and doc["category"] not in rf.categories:
            continue
        if doc["dimensions"]["width"] > rf.maxProductWidth:
            continue
        if doc["dimensions"]["depth"] > rf.maxProductDepth:
            continue

        dist = _min_color_distance(doc["colors"], target_colors)
        dims = ProductDimensions(**doc["dimensions"])
        candidates.append(Product(
            id=doc["_id"],
            name=doc["name"],
            category=doc["category"],
            styles=doc["styles"],
            price=doc["price"],
            dimensions=dims,
            colors=doc["colors"],
            imageUrl=doc["imageUrl"],
            color_distance=round(dist, 2),
        ))

    ranked = sorted(candidates, key=lambda p: p.color_distance or 999)
    return ranked[:top_n], len(candidates)