"""
mongo_service.py
Hard-filter candidates from MongoDB, then rank by color distance.
"""

import os
import math
import re
from typing import List, Optional

from motor.motor_asyncio import AsyncIOMotorClient

from app.services.cf_model import get_cf_scores
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
    """Converts a hex color string to an RGB tuple.
    Returns gray for invalid formats.
    """
    if not isinstance(hex_color, str) or not re.match(r"^#[0-9a-fA-F]{6}$", hex_color):
        return (128, 128, 128)
    h = hex_color.lstrip("#")
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

    # Case-insensitive regex for styles
    style_patterns = [re.compile(f"^{s}$", re.IGNORECASE) for s in rf.styles]

    query = {
        "styles": {"$in": style_patterns},
        "category": {"$in": rf.categories},
        "dimensions.width": {"$lte": rf.maxProductWidth},
        "dimensions.depth": {"$lte": rf.maxProductDepth},
    }
    return query


# ── Main function ────────────────────────────────────────────────────────────

async def get_recommendations(
    analysis: GeminiAnalysisResult,
    user_id: str = None,
    top_n: int = 20,
    candidate_limit: int = 70,
) -> tuple[List[Product], int]:
    """
    1. Hard-filter up to `candidate_limit` products from MongoDB.
    2. Rank using a weighted score: style, color, availability, and collaborative filtering.
    3. Return top_n products + total candidates count.
    """
    db = get_client()[DB_NAME]
    col = db[COLLECTION]

    query = _build_mongo_query(analysis)
    rf = analysis.recommendedFilter
    target_colors = rf.colorHexRange
    target_styles_lower = {s.lower() for s in rf.styles}

    cursor = col.find(query).limit(candidate_limit)
    
    # Step 1: Gather candidates and their raw scores
    candidate_data = []
    product_ids_for_cf = []
    max_color_dist = 441.67  # Max possible Euclidean distance in RGB

    async for doc in cursor:
        prod_styles = doc.get("styles", [])
        color_obj = doc.get("color", {})
        prod_colors = [color_obj.get("hex")] if color_obj and color_obj.get("hex") else []
        is_in_stock = doc.get("inStock", True)

        # Individual scores
        prod_styles_lower = {s.lower() for s in prod_styles}
        matching_styles = len(target_styles_lower.intersection(prod_styles_lower))
        style_match_score = min(matching_styles / (len(target_styles_lower) or 1), 1.0)

        color_dist = _min_color_distance(prod_colors, target_colors)
        color_score = 1.0 - (color_dist / max_color_dist)

        in_stock_score = 1.0 if is_in_stock else 0.0

        product_id = str(doc.get("_id", ""))
        product_ids_for_cf.append(product_id)

        candidate_data.append({
            "doc": doc,
            "product_id": product_id,
            "scores": {
                "style": style_match_score,
                "color": color_score,
                "stock": in_stock_score
            },
            "color_dist": color_dist,
            "prod_colors": prod_colors,
            "prod_styles": prod_styles,
        })

    # Step 2: Get Collaborative Filtering scores
    cf_scores = {}
    if user_id:
        cf_scores = get_cf_scores(user_id, product_ids_for_cf)

    # Step 3: Calculate final ranking score and build Product objects
    candidates: List[Product] = []
    for data in candidate_data:
        doc = data["doc"]
        scores = data["scores"]
        cf_score = cf_scores.get(data["product_id"], 0.0)

        if cf_scores:  # New weights if CF is active
            ranking_score = (
                (scores["style"] * 0.40) +
                (scores["color"] * 0.25) +
                (scores["stock"] * 0.15) +
                (cf_score * 0.20)
            )
        else:  # Original weights
            ranking_score = (
                (scores["style"] * 0.5) +
                (scores["color"] * 0.3) +
                (scores["stock"] * 0.2)
            )

        dim_raw = doc.get("dimensions", {})
        dims = ProductDimensions(
            width=dim_raw.get("width"),
            depth=dim_raw.get("depth"),
            height=dim_raw.get("height"),
        ) if dim_raw else None

        candidates.append(Product(
            id=data["product_id"],
            name=doc.get("name", ""),
            category=doc.get("category", ""),
            styles=data["prod_styles"],
            price=doc.get("price"),
            dimensions=dims,
            colors=data["prod_colors"],
            imageUrl=doc.get("imageUrl"),
            color_distance=round(data["color_dist"], 2),
            ranking_score=round(ranking_score, 4),
        ))

    total_candidates = len(candidates)

    # Sort by descending ranking score
    ranked = sorted(candidates, key=lambda p: p.ranking_score or 0, reverse=True)
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
    top_n: int = 20,
) -> tuple[List[Product], int]:
    """Mock version for development using the same weighted ranking."""
    rf = analysis.recommendedFilter
    target_colors = rf.colorHexRange
    target_styles_lower = {s.lower() for s in rf.styles}
    candidates = []
    max_color_dist = 441.67

    for doc in MOCK_PRODUCTS:
        # Hard filter mock data
        prod_styles_lower = {s.lower() for s in doc["styles"]}
        if not target_styles_lower.intersection(prod_styles_lower):
            continue
        if doc["category"] not in rf.categories:
            continue
        if doc["dimensions"]["width"] > rf.maxProductWidth:
            continue
        if doc["dimensions"]["depth"] > rf.maxProductDepth:
            continue

        # --- Scoring ---
        style_match_score = min(
            len(target_styles_lower.intersection(prod_styles_lower)) /
            (len(target_styles_lower) or 1), 1.0
        )
        color_dist = _min_color_distance(doc["colors"], target_colors)
        color_score = 1.0 - (color_dist / max_color_dist)
        in_stock_score = 1.0 if doc.get("inStock", True) else 0.0

        ranking_score = (
            (style_match_score * 0.5) +
            (color_score * 0.3) +
            (in_stock_score * 0.2)
        )

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
            color_distance=round(color_dist, 2),
            ranking_score=round(ranking_score, 4),
        ))

    # Sort by descending ranking score
    ranked = sorted(candidates, key=lambda p: p.ranking_score or 0, reverse=True)
    return ranked[:top_n], len(candidates)