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
    
    # Extract category names from the new structure
    category_names = [c.category for c in rf.categories]

    query = {
        "$or": [
            {"styles": {"$in": style_patterns}},
            {"category": {"$in": category_names}},
        ],
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
    
    # Create a map from category name to reasoning
    category_reasoning_map = {c.category: c.reasoning for c in rf.categories}

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

        product_category = doc.get("category", "")
        reasoning = category_reasoning_map.get(product_category, "Sản phẩm này là một lựa chọn phù hợp.")

        # Handle image URL
        images_list = doc.get("images", [])
        image_url = images_list[0] if images_list else None

        candidates.append(Product(
            id=data["product_id"],
            name=doc.get("name", ""),
            category=product_category,
            styles=data["prod_styles"],
            price=doc.get("price"),
            dimensions=dims,
            colors=data["prod_colors"],
            imageUrl=image_url,
            color_distance=round(data["color_dist"], 2),
            ranking_score=round(ranking_score, 4),
            reasoning=reasoning,
        ))

    total_candidates = len(candidates)

    # Sort by descending ranking score
    ranked = sorted(candidates, key=lambda p: p.ranking_score or 0, reverse=True)
    return ranked[:top_n], total_candidates



# ── Metadata retrieval ─────────────────────────────────────────────────────

async def get_distinct_categories() -> List[str]:
    """Returns a sorted list of unique 'category' values from the collection."""
    db = get_client()[DB_NAME]
    col = db[COLLECTION]
    categories = await col.distinct("category")
    return sorted([c for c in categories if c])


async def get_distinct_styles() -> List[str]:
    """Returns a sorted list of unique 'styles' values from the collection."""
    db = get_client()[DB_NAME]
    col = db[COLLECTION]
    styles = await col.distinct("styles")
    return sorted([s for s in styles if s])

