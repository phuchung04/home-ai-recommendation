# -*- coding: utf-8 -*-
"""
mongo_service.py
Hard-filter candidates from MongoDB, then rank by color distance.
"""

import os
import math
import re
from typing import List, Optional

# pyrefly: ignore [missing-import]
from motor.motor_asyncio import AsyncIOMotorClient

from app.services.cf_model import get_cf_scores, get_user_behavior_count
from app.services.behavior_service import get_popular_product_scores
from app.models.schemas import GeminiAnalysisResult, Product, ProductDimensions

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("MONGO_DB", "cap2")
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


def _normalize_lookup_key(value: str) -> str:
    """Normalize category/style lookup keys so matching is case- and space-insensitive."""
    if not isinstance(value, str):
        return ""
    return re.sub(r"\s+", " ", value).strip().casefold()


def _normalize_room_type_key(room_type: str) -> str:
    if not isinstance(room_type, str):
        return "living room"

    normalized = _normalize_lookup_key(room_type)
    if "bed" in normalized:
        return "bedroom"
    return "living room"


def _adjust_density_for_area(area_m2: float, density: str) -> tuple[str, Optional[str]]:
    density_value = density if isinstance(density, str) else str(density)
    density_key = density_value.strip().casefold()

    if area_m2 < 6:
        applied = "sparse"
    elif area_m2 < 8:
        applied = "sparse" if density_key in {"medium", "dense"} else density_value
    else:
        applied = density_value

    warning = None
    if applied != density_value:
        warning = f"Diện tích phòng {area_m2}m² quá nhỏ, mật độ nội thất đã được điều chỉnh từ {density_value} sang {applied}."

    return applied, warning


def _merge_unique(values: List[str]) -> List[str]:
    seen = set()
    merged = []
    for value in values:
        normalized = _normalize_lookup_key(value)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        merged.append(value)
    return merged


def _category_alias(category: str) -> str:
    normalized = _normalize_lookup_key(category)
    alias_map = {
        "armchair": "Armchair",
        "ghế thư giãn": "Armchair",
        "couch": "Sofa",
        "sectional": "Sofa góc",
        "coffee table": "Bàn nước",
        "side table": "Bàn bên",
        "console table": "Bàn console",
        "table": "Bàn",
        "dining table": "Bàn ăn",
        "dining chair": "Ghế ăn",
        "ghế ăn": "Ghế ăn",
        "working desk": "Bàn làm việc",
        "desk": "Bàn làm việc",
        "tv stand": "Tủ tivi",
        "display cabinet": "Tủ ly",
        "built in cabinet": "Tủ âm tường",
        "floor lamp": "Đèn trang trí",
        "đèn": "Đèn trang trí",
        "lamp": "Đèn trang trí",
        "storage cabinet": "Tủ lưu trữ",
        "shelf": "Kệ lưu trữ",
        "bookshelf": "Kệ sách",
        "mirror": "Gương",
        "rug": "Thảm",
        "plant": "Hoa & Cây",
        "plant pot": "Chậu hoa",
        "ottoman": "Ghế dài & đôn",
        "chair": "Ghế",
        "nightstand": "Bàn đầu giường",
        "bedside lamp": "Đèn trang trí",
        "wardrobe": "Tủ lưu trữ",
        "dresser": "Bàn trang điểm",
        "bed": "Giường",
        "mattress": "Nệm",
        "pillow": "Gối",
        "storage box": "Hộp lưu trữ",
        "clothes rack": "Giá treo quần áo",
        "decor": "Hàng trang trí",
        "decor item": "Hàng trang trí khác",
        "ornament": "Tượng trang trí",
        "vase": "Bình trang trí",
        "furniture": "Đồ nội thất",
        "essential furniture": "Đồ nội thất",
        "home decor": "Hàng trang trí",
        "indoor plant": "Hoa & Cây",
        "plant pot": "Chậu hoa",
        "cart": "Xe đẩy",
    }
    return alias_map.get(normalized, category)


def _get_room_category_tiers(room_type: str, area_m2: Optional[float], density: str) -> tuple[List[List[str]], str, Optional[str]]:
    room_key = _normalize_room_type_key(room_type)
    effective_area = float(area_m2) if area_m2 and area_m2 > 0 else None
    density_applied = density if isinstance(density, str) else str(density)
    warning = None

    if effective_area is not None:
        density_applied, warning = _adjust_density_for_area(effective_area, density_applied)

    if room_key == "bedroom":
        tier1 = ["Giường", "Nệm", "Bàn đầu giường", "Đèn trang trí", "Tủ lưu trữ"]
        tier2 = ["Bàn trang điểm", "Ghế", "Gương", "Thảm", "Kệ lưu trữ"]
        tier3 = ["Chậu hoa", "Hoa & Cây", "Bình trang trí", "Tượng trang trí", "Hộp lưu trữ"]
        tier4 = ["Giỏ lưu trữ", "Giá treo quần áo", "Đệm ngồi", "Ghế lười", "Tinh dầu"]
        tier5 = ["Hàng trang trí", "Hàng trang trí khác", "Đồ nội thất"]
    else:
        tier1 = ["Sofa", "Sofa góc", "Armchair", "Bàn nước", "Tủ tivi"]
        tier2 = ["Bàn bên", "Bàn console", "Thảm", "Tủ lưu trữ", "Kệ phòng khách"]
        tier3 = ["Kệ lưu trữ", "Kệ sách", "Đèn trang trí", "Gương", "Ghế thư giãn"]
        tier4 = ["Ghế dài & đôn", "Ghế lười", "Ghế làm việc", "Ghế ăn", "Chậu hoa"]
        tier5 = ["Hoa & Cây", "Bình trang trí", "Tượng trang trí", "Hàng trang trí", "Hàng trang trí khác"]
        tier6 = ["Phụ kiện nội thất", "Xe đẩy", "Tinh dầu", "Đồ nội thất"]

    if effective_area is not None and effective_area < 8:
        if room_key == "bedroom":
            tier2 = [category for category in tier2 if category not in {"Ghế", "Kệ lưu trữ"}]
            tier3 = [category for category in tier3 if category not in {"Hộp lưu trữ"}]
            tier4 = [category for category in tier4 if category not in {"Ghế lười", "Giá treo quần áo"}]
        else:
            tier2 = [category for category in tier2 if category not in {"Tủ lưu trữ", "Kệ phòng khách"}]
            tier3 = [category for category in tier3 if category not in {"Kệ sách", "Ghế thư giãn"}]
            tier4 = [category for category in tier4 if category not in {"Ghế lười", "Ghế làm việc", "Ghế ăn"}]
            tier5 = [category for category in tier5 if category not in {"Hàng trang trí", "Hàng trang trí khác"}]

    if density_applied == "sparse":
        if effective_area is not None and effective_area >= 20:
            # Large room: suggest tier2 as well for balance
            tiers = [tier1, tier2]
            warning = (warning or "") + f" Phòng rộng {effective_area}m², đề xuất thêm một số nội thất tier 2 để cân bằng không gian."
        else:
            tiers = [tier1]
    elif density_applied == "medium":
        tiers = [tier1, tier2, tier3]
    else:
        if room_key == "bedroom":
            tiers = [tier1, tier2, tier3, tier4, tier5]
        else:
            tiers = [tier1, tier2, tier3, tier4, tier5, tier6]

    tiers = [tier for tier in tiers if tier]
    return tiers, density_applied, warning


# ── Hard filter ──────────────────────────────────────────────────────────────

def _build_mongo_query(analysis: GeminiAnalysisResult) -> dict:
    rf = analysis.recommendedFilter
    room_tiers, _, _ = _get_room_category_tiers(rf.roomType, rf.roomAreaM2, rf.furnitureDensityHint)
    tier_categories = [_category_alias(category) for tier in room_tiers for category in tier]

    # Case-insensitive regex for styles
    style_patterns = [re.compile(f"^{s}$", re.IGNORECASE) for s in rf.styles]
    
    # Extract category names from the new structure
    gemini_category_names = [c.category for c in rf.categories]
    category_names = _merge_unique(tier_categories + gemini_category_names)

    # Ensure required categories for the room type are present in the category list
    REQUIRED_CATEGORIES = {
        "bedroom": ["Giường", "Nệm", "Bàn đầu giường"],
        "living room": ["Sofa", "Sofa góc"],
    }
    room_key = _normalize_room_type_key(rf.roomType)
    req_cats = REQUIRED_CATEGORIES.get(room_key, [])
    # If none of the required categories are present, inject them at the front
    normalized_names = {c.casefold() for c in category_names}
    to_inject = [c for c in req_cats if c.casefold() not in normalized_names]
    if to_inject:
        category_names = to_inject + category_names
    
    # Store original names for later bed-specific query
    category_names_with_required = category_names

    max_product_area = getattr(rf, "maxProductArea", rf.maxProductWidth * rf.maxProductDepth)

    # Build base query with mandatory category filter
    room_key = _normalize_room_type_key(rf.roomType)
    excluded_categories = ["Ghế ăn", "Bàn ăn", "Sofa", "Sofa góc", "Tủ tivi", "Bàn nước"]
    
    # Filter category list: include required + remove excluded
    filtered_categories = [c for c in category_names_with_required if c not in excluded_categories]
    
    if room_key == "bedroom":
        BED_CATS = {"giường", "nệm"}
        WARDROBE_CATS = {"tủ lưu trữ", "tủ"}
        
        bed_cats = [c for c in filtered_categories if _normalize_lookup_key(c) in BED_CATS]
        wardrobe_cats = [c for c in filtered_categories if _normalize_lookup_key(c) in WARDROBE_CATS]
        non_large_cats = [c for c in filtered_categories if _normalize_lookup_key(c) not in BED_CATS and _normalize_lookup_key(c) not in WARDROBE_CATS]

        or_clauses = []
        
        # Beds/Mattresses: NO dimension filter at all (beds can be any size)
        if bed_cats:
            or_clauses.append({"category": {"$in": bed_cats}})
        
        # Wardrobes: only width filter
        if wardrobe_cats:
            or_clauses.append({
                "category": {"$in": wardrobe_cats},
                "$or": [
                    {"dimensions.width": {"$exists": False}},
                    {"dimensions.width": None},
                    {"dimensions.width": {"$lte": rf.maxProductWidth * 1.15}},
                ]
            })
        
        # Other furniture: full dimension filter
        if non_large_cats:
            or_clauses.append({
                "category": {"$in": non_large_cats},
                "$or": [
                    {"dimensions.width": {"$exists": False}},
                    {"dimensions.width": {"$lte": rf.maxProductWidth * 1.15}},
                ],
                "$or": [
                    {"dimensions.depth": {"$exists": False}},
                    {"dimensions.depth": {"$lte": rf.maxProductDepth * 1.15}},
                ],
            })
        
        query = {"$or": or_clauses} if or_clauses else {"category": {"$in": filtered_categories}}
    else:
        # Living room: standard strict query
        query = {
            "category": {"$in": filtered_categories},
            "$or": [
                {"styles": {"$in": style_patterns}},
                {"dimensions.width": {"$lte": rf.maxProductWidth}},
            ],
            "dimensions.width": {"$lte": rf.maxProductWidth},
            "dimensions.depth": {"$lte": rf.maxProductDepth},
            "$expr": {
                "$lte": [
                    {"$multiply": [{"$ifNull": ["$dimensions.width", 0]}, {"$ifNull": ["$dimensions.depth", 0]}]},
                    max_product_area,
                ]
            },
        }
    
    return query


# ── Main function ────────────────────────────────────────────────────────────

async def get_recommendations(
    analysis: GeminiAnalysisResult,
    user_id: str = None,
    top_n: int = 30,
    candidate_limit: int = 300,
) -> tuple[List[Product], int, Optional[str], Optional[str]]:
    """
    1. Hard-filter up to `candidate_limit` products from MongoDB.
    2. Rank using a weighted score: style, color, availability, and collaborative filtering.
    3. Return top_n products + total candidates count.
    """
    db = get_client()[DB_NAME]
    col = db[COLLECTION]

    query = _build_mongo_query(analysis)
    rf = analysis.recommendedFilter
    room_tiers, density_applied, warning = _get_room_category_tiers(rf.roomType, rf.roomAreaM2, rf.furnitureDensityHint)
    tier_lookup = {}
    for tier_index, tier in enumerate(room_tiers):
        for category in tier:
            tier_lookup.setdefault(_normalize_lookup_key(_category_alias(category)), tier_index)

    target_colors = rf.colorHexRange
    target_styles_lower = {s.lower() for s in rf.styles}
    
    # Create a map from normalized category name to reasoning
    category_reasoning_map = {
        _normalize_lookup_key(c.category): c.reasoning
        for c in rf.categories
        if c.category
    }

    if density_applied == "sparse":
        max_products_per_category = 6
    elif density_applied == "medium":
        max_products_per_category = 5
    else:
        max_products_per_category = 4

    cursor = col.find(query).limit(candidate_limit)
    
    # Step 1: Gather candidates and their raw scores
    candidate_data = []
    product_ids_for_cf = []
    max_color_dist = 441.67  # Max possible Euclidean distance in RGB

    async for doc in cursor:
        prod_styles = doc.get("styles", [])
        color_obj = doc.get("color")
        if isinstance(color_obj, dict):
            prod_colors = [color_obj.get("hex")] if color_obj.get("hex") else []
        elif isinstance(color_obj, list):
            prod_colors = [c.get("hex") for c in color_obj if isinstance(c, dict) and c.get("hex")]
        else:
            prod_colors = []
        is_in_stock = doc.get("inStock", True)

        # Individual scores
        prod_styles_lower = {s.lower() for s in prod_styles}
        matching_styles = len(target_styles_lower.intersection(prod_styles_lower))
        style_match_score = min(matching_styles / (len(target_styles_lower) or 1), 1.0)

        color_dist = _min_color_distance(prod_colors, target_colors)
        color_score = max(0.0, 1.0 - (color_dist / max_color_dist))  # Clamp to [0, 1]

        in_stock_score = 1.0 if is_in_stock else 0.0

        product_id = str(doc.get("_id", ""))
        product_ids_for_cf.append(product_id)
        category_rank = tier_lookup.get(_normalize_lookup_key(doc.get("category", "")), len(room_tiers))

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
            "category_rank": category_rank,
        })

    # If no bed/mattress candidates were found for a bedroom, run a relaxed fallback
    room_key = _normalize_room_type_key(rf.roomType)
    if room_key == "bedroom":
        has_bed = any(
            _normalize_lookup_key(d.get("doc", {}).get("category", "")) in {"giường", "nệm"}
            for d in candidate_data
        )
        if not has_bed:
            try:
                bed_query = {"category": {"$in": ["Giường", "Nệm"]}}
                # fetch a small number of bed candidates ignoring strict dimension filters
                bed_cursor = col.find(bed_query).limit(12)
                async for bdoc in bed_cursor:
                    b_id = str(bdoc.get("_id", ""))
                    if b_id in product_ids_for_cf:
                        continue
                    b_styles = bdoc.get("styles", [])
                    b_color_obj = bdoc.get("color")
                    if isinstance(b_color_obj, dict):
                        b_colors = [b_color_obj.get("hex")] if b_color_obj.get("hex") else []
                    elif isinstance(b_color_obj, list):
                        b_colors = [c.get("hex") for c in b_color_obj if isinstance(c, dict) and c.get("hex")]
                    else:
                        b_colors = []
                    # conservative scoring: style match and color distance computed later
                    prod_styles_lower = {s.lower() for s in b_styles}
                    matching_styles = len(target_styles_lower.intersection(prod_styles_lower))
                    style_match_score = min(matching_styles / (len(target_styles_lower) or 1), 1.0)
                    color_dist = _min_color_distance(b_colors, target_colors)
                    color_score = max(0.0, 1.0 - (color_dist / max_color_dist))
                    in_stock_score = 1.0 if bdoc.get("inStock", True) else 0.0

                    product_ids_for_cf.append(b_id)
                    category_rank = tier_lookup.get(_normalize_lookup_key(bdoc.get("category", "")), len(room_tiers))

                    candidate_data.append({
                        "doc": bdoc,
                        "product_id": b_id,
                        "scores": {
                            "style": style_match_score,
                            "color": color_score,
                            "stock": in_stock_score,
                        },
                        "color_dist": color_dist,
                        "prod_colors": b_colors,
                        "prod_styles": b_styles,
                        "category_rank": category_rank,
                    })
            except Exception:
                # fallback should be best-effort; ignore failures and continue
                pass

    # Step 2: Get Collaborative Filtering scores and adaptive weights
    behavior_count = 0
    if user_id:
        try:
            behavior_count = await get_user_behavior_count(user_id)
        except Exception:
            behavior_count = 0

    # Select weights based on behavior_count
    if behavior_count == 0:
        cf_weight = 0.0
        style_w = 0.50
        color_w = 0.30
        stock_w = 0.20
    elif 1 <= behavior_count <= 19:
        cf_weight = 0.10
        style_w = 0.45
        color_w = 0.30
        stock_w = 0.15
    elif 20 <= behavior_count <= 49:
        cf_weight = 0.15
        style_w = 0.42
        color_w = 0.28
        stock_w = 0.15
    else:
        cf_weight = 0.20
        style_w = 0.40
        color_w = 0.25
        stock_w = 0.15

    cf_scores = get_cf_scores(user_id, product_ids_for_cf) if user_id else {}

    # Popularity boost for cold-start users
    popularity_scores = {}
    popularity_weight = 0.0
    if behavior_count < 20 and product_ids_for_cf:
        try:
            popularity_scores = await get_popular_product_scores(product_ids_for_cf)
            popularity_weight = 0.05
        except Exception:
            popularity_scores = {}

    # Step 3: Calculate final ranking score and build Product objects
    candidates: List[Product] = []
    for data in candidate_data:
        doc = data["doc"]
        scores = data["scores"]
        cf_score = cf_scores.get(data["product_id"], 0.0)

        # Weighted ranking using adaptive CF weight and optional popularity boost
        ranking_score = (
            (scores["style"] * style_w)
            + (scores["color"] * color_w)
            + (scores["stock"] * stock_w)
            + (cf_score * cf_weight)
        )
        # Add small popularity boost for cold-start users
        if popularity_weight > 0:
            pop_count = popularity_scores.get(data["product_id"], 0)
            max_pop = max(popularity_scores.values()) if popularity_scores else 0
            pop_norm = (pop_count / max_pop) if max_pop > 0 else 0
            ranking_score += pop_norm * popularity_weight
        
        # Ensure ranking_score is never negative
        ranking_score = max(0.0, ranking_score)

        dim_raw = doc.get("dimensions", {})
        dims = ProductDimensions(
            width=dim_raw.get("width"),
            depth=dim_raw.get("depth"),
            height=dim_raw.get("height"),
        ) if dim_raw else None

        product_category = doc.get("category", "")
        normalized_category = _normalize_lookup_key(product_category)
        
        # Try to get reasoning from category map; if not found, generate dynamic reasoning
        if normalized_category in category_reasoning_map:
            reasoning = category_reasoning_map[normalized_category]
        else:
            # Fallback: generate reasoning based on matched styles and product category
            matched_styles = [s for s in rf.styles if s.casefold() in {st.casefold() for st in data["prod_styles"]}]
            style_hint = matched_styles[0] if matched_styles else (rf.styles[0] if rf.styles else "được chọn")
            reasoning = f"{product_category} với phong cách {style_hint} phù hợp với không gian của bạn, được chọn vì khớp với bảng màu đề xuất ({', '.join(rf.colorHexRange[:2])}) và kích thước phù hợp."

        # Handle image URL
        images_list = doc.get("images", [])

        # Normalize image entries into a list of URL strings (some docs store dicts)
        image_candidates: List[str] = []
        for img in images_list:
            if isinstance(img, str):
                image_candidates.append(img)
            elif isinstance(img, dict):
                for k in ("url", "imageUrl", "src", "image"):
                    if k in img and isinstance(img[k], str):
                        image_candidates.append(img[k])
                        break

        # Choose the best-matching image by scoring how many tokens from the
        # product name / category appear in the image URL path. This avoids
        # picking unrelated images (e.g., lamp image for a bedside table).
        def _image_score(url: str) -> float:
            if not url:
                return 0.0
            path = url.lower()
            # tokens from category + product name (skip short words)
            name_tokens = re.findall(r"\w+", (product_category + " " + doc.get("name", "")).lower())
            tokens = {t for t in name_tokens if len(t) > 2}
            score = 0.0
            for t in tokens:
                if t in path:
                    score += 2.0

            # prefer matching suggested styles
            for s in rf.styles:
                if isinstance(s, str) and s.lower() in path:
                    score += 1.0

            # small boost for common image extensions
            if path.endswith((".jpg", ".jpeg", ".png", ".webp")):
                score += 0.5

            return score

        image_url = None
        if image_candidates:
            # pick the highest scoring image; fall back to first if tie/zero
            scored = [(img, _image_score(img)) for img in image_candidates]
            scored.sort(key=lambda x: x[1], reverse=True)
            best_img, best_score = scored[0]
            if best_score > 0:
                image_url = best_img
            else:
                image_url = image_candidates[0]

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
            style_score=round(scores["style"], 4),
            color_score=round(scores["color"], 4),
            stock_score=round(scores["stock"], 4),
            cf_score=round(cf_score, 4),
            cf_active=bool(cf_scores),
            reasoning=reasoning,
        ))

    total_candidates = len(candidates)

    ranked = sorted(
        zip(candidates, candidate_data),
        key=lambda pair: (
            pair[1].get("category_rank", 999),
            -(pair[0].ranking_score or 0),
            pair[0].color_distance or 999,
        ),
    )

    category_counts = {}
    selected: List[Product] = []
    for product, _data in ranked:
        category_key = _normalize_lookup_key(product.category)
        category_counts.setdefault(category_key, 0)
        if category_counts[category_key] >= max_products_per_category:
            continue
        category_counts[category_key] += 1
        selected.append(product)
        if len(selected) >= top_n:
            break

    return selected, total_candidates, warning, density_applied
    



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

