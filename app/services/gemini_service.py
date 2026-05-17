"""
gemini_service.py
"""

import os
import json
import base64
import math
import httpx
import asyncio
import time
import hashlib
import random
import itertools
import threading
from typing import Optional, List, Dict, Tuple, Any
from fastapi import HTTPException
from pydantic import ValidationError

from app.models.schemas import RecommendRequest, GeminiAnalysisResult
from app.services.mongo_service import get_distinct_categories, get_distinct_styles


GEMINI_TIMEOUT_SECONDS = float(os.getenv("GEMINI_TIMEOUT_SECONDS", "90"))


def _normalize_gemini_response(data: dict) -> dict:
    """
    Normalize Gemini response to match schema enums.
    Handles case mismatches (e.g., 'living room' → 'Living Room').
    """
    if "room_type" in data and isinstance(data["room_type"], str):
        room = data["room_type"].lower().strip()
        if "living" in room:
            data["room_type"] = "Living Room"
        elif "bedroom" in room or "bed room" in room:
            data["room_type"] = "Bedroom"
    return data


def _extract_gemini_error_reason(resp: httpx.Response) -> str:
    """
    Extract normalized Gemini error reason from response body if present.
    Example values: RATE_LIMIT_EXCEEDED, RESOURCE_EXHAUSTED.
    """
    try:
        payload = resp.json()
    except Exception:
        return ""

    error_obj = payload.get("error", {})
    details = error_obj.get("details", [])
    for item in details:
        if isinstance(item, dict):
            # Gemini often returns reason in ErrorInfo payload.
            reason = item.get("reason")
            if isinstance(reason, str) and reason.strip():
                return reason.strip().upper()
    return ""


def _is_daily_quota_exhausted(resp: httpx.Response) -> bool:
    """
    Detect non-retryable quota exhaustion from Gemini error payload.
    """
    reason = _extract_gemini_error_reason(resp)
    if reason in {"RESOURCE_EXHAUSTED", "QUOTA_EXCEEDED"}:
        return True

    body_text = (resp.text or "").upper()
    return "RESOURCE_EXHAUSTED" in body_text or "QUOTA" in body_text


def _is_invalid_api_key(resp: httpx.Response) -> bool:
    """Detect Gemini API key invalidation or expiration."""
    reason = _extract_gemini_error_reason(resp)
    if reason == "API_KEY_INVALID":
        return True

    body_text = (resp.text or "").upper()
    return any(token in body_text for token in ["API KEY EXPIRED", "API_KEY_INVALID", "KEY INVALID"])


def _normalize_image_mime_type(image_bytes: Optional[bytes], declared_mime: str) -> str:
    """Normalize or recover the image MIME type before sending to Gemini."""
    supported = {"image/jpeg", "image/png", "image/webp", "image/bmp", "image/gif"}
    declared = declared_mime.strip().lower().split(";", 1)[0].strip() if isinstance(declared_mime, str) else ""
    if declared in supported:
        return declared

    if not image_bytes or len(image_bytes) < 12:
        return "image/jpeg"

    if image_bytes.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if image_bytes.startswith(b"GIF8"):
        return "image/gif"
    if image_bytes.startswith(b"BM"):
        return "image/bmp"
    if image_bytes[8:12] == b"WEBP":
        return "image/webp"

    # Fallback to a safe default when the declared type is unknown.
    return "image/jpeg"


def _extract_json_text(raw_text: str) -> str:
    """Best-effort extraction of a JSON object from model output."""
    if not raw_text:
        return raw_text

    text = raw_text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start : end + 1]
    return text


GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.5-flash:generateContent"
)

# ── Simple in-memory request cache (prompt_hash -> (timestamp, parsed_json))
_request_cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}
_CACHE_TTL_SECONDS = 60 * 10  # 10 minutes

# Limit concurrent outbound calls to Gemini to avoid bursts
_gemini_semaphore = asyncio.Semaphore(2)


def _load_api_keys() -> list[str]:
    """Load tất cả Gemini API keys từ environment variables."""
    keys = []
    # Key chính
    primary = os.getenv("GEMINI_API_KEY", "").strip()
    if primary:
        keys.append(primary)
    # Key phụ: GEMINI_API_KEY_2, GEMINI_API_KEY_3, ...
    i = 2
    while True:
        key = os.getenv(f"GEMINI_API_KEY_{i}", "").strip()
        if not key:
            break
        keys.append(key)
        i += 1
    return keys


class _GeminiKeyPool:
    """Thread-safe round-robin key pool với blacklist tạm thời khi quota hết."""

    def __init__(self, keys: list[str]):
        self._keys = keys
        self._cycle = itertools.cycle(keys) if keys else iter([])
        self._lock = threading.Lock()
        self._exhausted: set[str] = set()  # keys bị quota hết

    def next_key(self) -> str | None:
        with self._lock:
            available = [k for k in self._keys if k not in self._exhausted]
            if not available:
                # Reset blacklist — thử lại tất cả
                self._exhausted.clear()
                available = self._keys
            if not available:
                return None
            # Round-robin trên available keys
            key = next(k for k in self._cycle if k in available)
            return key

    def mark_exhausted(self, key: str):
        """Đánh dấu key bị quota hết — không dùng tạm thời."""
        with self._lock:
            self._exhausted.add(key)
            print(f"[KeyPool] Key ...{key[-6:]} marked as exhausted. "
                  f"Remaining: {len(self._keys) - len(self._exhausted)}")

    @property
    def has_keys(self) -> bool:
        return bool(self._keys)


_key_pool = _GeminiKeyPool(_load_api_keys())

# ── Metadata Caching ─────────────────────────────────────────────────────────

_cached_categories: List[str] = []
_cached_styles: List[str] = []


def _normalize_catalog_label(value: str) -> str:
    """Normalize catalog labels for stable prompt generation and matching."""
    if not isinstance(value, str):
        return ""
    return " ".join(value.split()).strip()


def _dedupe_normalized_labels(values: List[str]) -> List[str]:
    """Deduplicate labels while preserving first-seen display form."""
    seen = set()
    result: List[str] = []
    for value in values:
        normalized = _normalize_catalog_label(value)
        if not normalized:
            continue
        key = normalized.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(normalized)
    return result


def _normalize_room_type_key(room_type: str) -> str:
    if not isinstance(room_type, str):
        return "living room"

    normalized = " ".join(room_type.split()).strip().casefold()
    if "bed" in normalized:
        return "bedroom"
    return "living room"


def _get_room_area_m2(req: RecommendRequest) -> float:
    if req.area_m2 and req.area_m2 > 0:
        return round(float(req.area_m2), 2)
    return round(req.width * req.length, 2)


def _adjust_density_for_area(area_m2: float, density: str) -> tuple[str, Optional[str]]:
    density_value = density if isinstance(density, str) else str(density)
    density_key = density_value.strip().casefold()

    if area_m2 <= 10:
        applied = "sparse"
    else:
        applied = density_value

    warning = None
    if applied != density_value:
        warning = f"Diện tích phòng {area_m2}m² quá nhỏ, mật độ nội thất đã được điều chỉnh từ {density_value} sang {applied}."

    return applied, warning


def _pick_prompt_categories(req: RecommendRequest, limit: int = 15) -> List[str]:
    room_key = _normalize_room_type_key(req.room_type.value if hasattr(req.room_type, "value") else str(req.room_type))
    requested_style = _normalize_catalog_label(req.style)

    if room_key == "bedroom":
        preferred = [
            "Giường",
            "Bàn đầu giường",
            "Tủ",
            "Tủ lưu trữ",
            "Đèn trang trí",
            "Gương",
            "Ghế",
            "Kệ",
            "Thảm",
            "Đồ trang trí",
            "Hàng trang trí",
            "Hàng trang trí khác",
            "Chậu hoa",
            "Hoa & Cây",
            "Bình trang trí",
            "Tượng trang trí",
            "Bàn trang điểm",
            "Giỏ lưu trữ",
            "Hộp lưu trữ",
            "Giá treo quần áo",
            "Đệm ngồi",
            "Ghế lười",
            "Nệm",
            "Gối",
            "Tinh dầu",
            "Đồ nội thất",
        ]
    else:
        preferred = [
            "Sofa",
            "Sofa góc",
            "Bàn nước",
            "Bàn bên",
            "Bàn console",
            "Bàn",
            "Bàn ăn",
            "Bàn làm việc",
            "Tủ tivi",
            "Tủ lưu trữ",
            "Tủ ly",
            "Tủ âm tường",
            "Kệ phòng khách",
            "Kệ lưu trữ",
            "Kệ sách",
            "Đèn trang trí",
            "Gương",
            "Thảm",
            "Armchair",
            "Ghế thư giãn",
            "Ghế dài & đôn",
            "Ghế lười",
            "Ghế làm việc",
            "Ghế ăn",
            "Hoa & Cây",
            "Chậu hoa",
            "Bình trang trí",
            "Tượng trang trí",
            "Hàng trang trí",
            "Hàng trang trí khác",
            "Phụ kiện nội thất",
            "Xe đẩy",
            "Tinh dầu",
            "Đồ nội thất",
        ]

    category_aliases = {
        "armchair": "Armchair",
        "ghế thư giãn": "Armchair",
        "couch": "Sofa",
        "sectional": "Sofa góc",
        "coffee table": "Bàn nước",
        "side table": "Bàn bên",
        "console table": "Bàn console",
        "table": "Bàn",
        "dining table": "Bàn ăn",
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
        "essential": "Đồ nội thất",
        "furniture": "Đồ nội thất",
        "home decor": "Hàng trang trí",
        "indoor plant": "Hoa & Cây",
        "cart": "Xe đẩy",
    }

    cached_lookup = {category.casefold(): category for category in _cached_categories}
    selected: List[str] = []

    for category in preferred:
        alias = category_aliases.get(category.casefold(), category)
        cached_category = cached_lookup.get(alias.casefold()) or cached_lookup.get(category.casefold())
        selected.append(cached_category or alias)

    if requested_style and len(selected) < limit:
        selected.extend([style for style in _cached_styles if style.casefold() != requested_style.casefold()][:2])

    return _dedupe_normalized_labels(selected)[:limit]


def _build_response_schema() -> dict:
    return {
        "type": "object",
        "required": ["isRoom", "notRoomReason", "imageAnalysis", "recommendedFilter", "reasoning"],
        "properties": {
            "isRoom": {"type": "boolean"},
            "notRoomReason": {"type": ["string", "null"]},
            "roomVisibility": {"type": ["string", "null"]},
            "visibilityWarning": {"type": ["string", "null"]},
            "roomContentValid": {"type": "boolean"},
            "missingFurniture": {"type": ["array", "null"], "items": {"type": "string"}},
            "imageAnalysis": {
                "type": "object",
                "required": [
                    "dominantColors",
                    "colorTone",
                    "detectedStyle",
                    "lightingType",
                    "existingFurnitureCategories",
                ],
                "properties": {
                    "dominantColors": {"type": "array", "items": {"type": "string"}},
                    "colorTone": {"type": "string"},
                    "detectedStyle": {"type": "string"},
                    "lightingType": {"type": "string"},
                    "existingFurnitureCategories": {"type": "array", "items": {"type": "string"}},
                },
            },
            "recommendedFilter": {
                "type": "object",
                "required": [
                    "styles",
                    "colorHexRange",
                    "colorTone",
                    "categories",
                    "roomType",
                    "maxProductWidth",
                    "maxProductDepth",
                    "maxProductArea",
                    "furnitureDensityHint",
                ],
                "properties": {
                    "styles": {"type": "array", "items": {"type": "string"}},
                    "colorHexRange": {"type": "array", "items": {"type": "string"}},
                    "colorTone": {"type": "string"},
                    "categories": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["category", "reasoning", "styleAlignment", "suggestedColorHex", "materialHint"],
                            "properties": {
                                "category": {"type": "string"},
                                "reasoning": {"type": "string"},
                                "styleAlignment": {"type": "string"},
                                "suggestedColorHex": {"type": "string"},
                                "materialHint": {"type": "string"},
                            },
                        },
                    },
                    "roomType": {"type": "string"},
                    "roomAreaM2": {"type": ["number", "null"]},
                    "maxProductWidth": {"type": "number"},
                    "maxProductDepth": {"type": "number"},
                    "maxProductArea": {"type": "number"},
                    "furnitureDensityHint": {"type": "string"},
                },
            },
            "reasoning": {
                "type": "object",
                "required": ["styleJustification", "colorJustification", "densityJustification", "userProfileNote"],
                "properties": {
                    "styleJustification": {"type": "string"},
                    "colorJustification": {"type": "string"},
                    "densityJustification": {"type": "string"},
                    "userProfileNote": {"type": "string"},
                },
            },
            "warning": {"type": ["string", "null"]},
            "densityApplied": {"type": ["string", "null"]},
        },
    }


def _recover_truncated_json_text(raw_text: str) -> str:
    text = raw_text.strip()
    if not text:
        return text

    end = text.rfind("}")
    while end > 0:
        candidate = text[: end + 1]
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            end = candidate[:-1].rfind("}")

    return text


def _ensure_analysis_defaults(parsed: dict, req: RecommendRequest) -> dict:
    if not isinstance(parsed, dict):
        return parsed

    parsed.setdefault("isRoom", True)
    parsed.setdefault("notRoomReason", None)
    # Visibility and content defaults
    parsed.setdefault("roomVisibility", "FULL")
    parsed.setdefault("visibilityWarning", None)
    parsed.setdefault("roomContentValid", True)
    parsed.setdefault("missingFurniture", [])

    room_area_m2 = _get_room_area_m2(req)
    max_product_area = round(room_area_m2 * 10000 * 0.10, 1)

    recommended_filter = parsed.get("recommendedFilter")
    if isinstance(recommended_filter, dict):
        recommended_filter.setdefault("roomType", req.room_type.value if hasattr(req.room_type, "value") else str(req.room_type))
        recommended_filter.setdefault("roomAreaM2", room_area_m2)
        recommended_filter.setdefault("maxProductArea", max_product_area)

    return parsed


def _hydrate_non_room_payload(parsed: dict, req: RecommendRequest) -> dict:
    """
    Gemini may omit or null nested analysis fields when isRoom=false.
    Fill required nested objects from fallback so Pydantic validation can pass
    and router can return a clean 422 INVALID_IMAGE response.
    """
    if not isinstance(parsed, dict) or parsed.get("isRoom") is not False:
        return parsed

    fallback = _build_fallback_analysis(req).model_dump()

    if not isinstance(parsed.get("imageAnalysis"), dict):
        parsed["imageAnalysis"] = fallback["imageAnalysis"]
    else:
        for key, value in fallback["imageAnalysis"].items():
            if parsed["imageAnalysis"].get(key) is None:
                parsed["imageAnalysis"][key] = value

    if not isinstance(parsed.get("recommendedFilter"), dict):
        parsed["recommendedFilter"] = fallback["recommendedFilter"]
    else:
        for key, value in fallback["recommendedFilter"].items():
            if parsed["recommendedFilter"].get(key) is None:
                parsed["recommendedFilter"][key] = value

    if not isinstance(parsed.get("reasoning"), dict):
        parsed["reasoning"] = fallback["reasoning"]
    else:
        for key, value in fallback["reasoning"].items():
            if parsed["reasoning"].get(key) is None:
                parsed["reasoning"][key] = value

    parsed["isRoom"] = False
    parsed["notRoomReason"] = parsed.get("notRoomReason") or "Không nhận diện được phòng từ ảnh tải lên."
    # When not a room, visibility is minimal and content invalid
    parsed["roomVisibility"] = "MINIMAL"
    parsed["visibilityWarning"] = parsed.get("visibilityWarning") or "Ảnh không cho đủ góc nhìn phòng để phân tích."
    return parsed


def _pick_fallback_styles(req: RecommendRequest) -> List[str]:
    styles: List[str] = []
    requested_style = _normalize_catalog_label(req.style)
    if requested_style:
        styles.append(requested_style)

    if _cached_styles:
        for style in _cached_styles:
            if style.casefold() != requested_style.casefold():
                styles.append(style)
            if len(styles) >= 3:
                break

    if not styles:
        styles = ["Modern"]

    return _dedupe_normalized_labels(styles)


def _pick_fallback_categories(req: RecommendRequest) -> List[str]:
    room_type = req.room_type.value if hasattr(req.room_type, "value") else str(req.room_type)
    room_key = room_type.casefold()

    living_room_defaults = ["Sofa", "Coffee Table", "TV Stand", "Armchair", "Floor Lamp"]
    bedroom_defaults = ["Bed", "Nightstand", "Wardrobe", "Dresser", "Bedside Lamp"]

    if "living" in room_key:
        preferred = living_room_defaults
    elif "bed" in room_key:
        preferred = bedroom_defaults
    else:
        preferred = ["Chair", "Table", "Shelf", "Lamp", "Storage Cabinet"]

    categories: List[str] = []
    cached_lookup = {category.casefold(): category for category in _cached_categories}
    for category in preferred:
        cached_category = cached_lookup.get(category.casefold())
        categories.append(cached_category or category)

    return _dedupe_normalized_labels(categories)


def _build_fallback_analysis(req: RecommendRequest) -> GeminiAnalysisResult:
    width_cm = round(req.width * 100 * 0.4, 1)
    depth_cm = round(req.length * 100 * 0.35, 1)
    room_area_m2 = _get_room_area_m2(req)
    density_applied, warning = _adjust_density_for_area(room_area_m2, req.furniture_density.value)
    fallback_styles = _pick_fallback_styles(req)
    fallback_categories = _pick_prompt_categories(req)
    room_type_value = req.room_type.value if hasattr(req.room_type, "value") else str(req.room_type)
    max_area_cm2 = round(room_area_m2 * 10000 * 0.10, 1)

    if req.age is not None and req.age <= 25:
        user_profile_note = "Người dùng trẻ nên ưu tiên bố cục gọn, màu sáng và vật liệu dễ phối để giữ không gian năng động."
    elif req.age is not None and req.age <= 45:
        user_profile_note = "Người dùng trưởng thành phù hợp với bố cục cân bằng, màu trung tính và nội thất đa dụng cho sinh hoạt hằng ngày."
    elif req.age is not None:
        user_profile_note = "Người dùng lớn tuổi nên ưu tiên sản phẩm dễ sử dụng, màu dịu và công năng rõ ràng để tăng tiện nghi."
    else:
        user_profile_note = "Không có thông tin tuổi nên hệ thống ưu tiên lựa chọn an toàn, dễ phối và phù hợp nhiều ngữ cảnh sử dụng."

    return GeminiAnalysisResult(
        isRoom=True,
        notRoomReason=None,
        imageAnalysis={
            "dominantColors": ["#F3F0EA", "#D9D5CF", "#A8A29E"],
            "colorTone": "neutral warm",
            "detectedStyle": req.style,
            "lightingType": "balanced natural light",
            "existingFurnitureCategories": fallback_categories[:3],
        },
        recommendedFilter={
            "styles": fallback_styles,
            "colorHexRange": ["#F3F0EA", "#D9D5CF", "#A8A29E", "#6B7280"],
            "colorTone": "neutral warm",
            "categories": [
                {
                    "category": category,
                    "reasoning": f"Khi Gemini quá tải, hệ thống ưu tiên {category.lower()} để vẫn bám theo phong cách {req.style} và bố cục của {room_type_value}.",
                    "styleAlignment": f"Giữ tinh thần {req.style}",
                    "suggestedColorHex": "#D9D5CF",
                    "materialHint": "gỗ sáng / vải trung tính",
                }
                for category in fallback_categories
            ],
            "roomType": room_type_value,
            "roomAreaM2": room_area_m2,
            "maxProductWidth": width_cm,
            "maxProductDepth": depth_cm,
            "maxProductArea": max_area_cm2,
            "furnitureDensityHint": req.furniture_density.value,
        },
        reasoning={
            "styleJustification": f"Fallback sử dụng style '{req.style}' làm neo chính để giữ đúng ý định ban đầu của user, dù Gemini không phản hồi kịp.",
            "colorJustification": "Bảng màu trung tính sáng giúp query MongoDB vẫn có tín hiệu ổn định khi thiếu phân tích ảnh.",
            "densityJustification": f"Giới hạn kích thước được suy từ kích thước phòng và mật độ '{req.furniture_density.value}' để tránh đề xuất đồ quá lớn.",
            "userProfileNote": user_profile_note,
        },
        # New fields for visibility and content validation
        roomVisibility="FULL",
        visibilityWarning=None,
        roomContentValid=True,
        missingFurniture=[],
        warning=warning,
        densityApplied=density_applied,
    )

async def load_metadata():
    """
    Fetches distinct categories and styles from MongoDB and caches them.
    This should be called at application startup.
    """
    global _cached_categories, _cached_styles
    print("[Metadata] Loading distinct categories and styles from DB...")
    
    try:
        # Load categories
        categories = await get_distinct_categories()
        if categories:
            _cached_categories = _dedupe_normalized_labels(categories)
            print(f"[Metadata] Loaded {len(_cached_categories)} categories.")
        else:
            print("[Metadata] Warning: No categories found in DB, using empty list.")

        # Load styles and merge with the hardcoded list
        styles_from_db = await get_distinct_styles()
        if styles_from_db:
            print(f"[Metadata] Found {len(styles_from_db)} styles in DB.")
            
            # Merge and remove duplicates, then sort
            combined_styles = _dedupe_normalized_labels(_cached_styles + styles_from_db)
            _cached_styles = sorted(combined_styles)
        
        print(f"[Metadata] Final style list has {len(_cached_styles)} unique styles.")

    except Exception as e:
        print(f"[Metadata] Error loading metadata: {e}")
        print("[Metadata] Using hardcoded fallback styles.")


def _build_prompt(req: RecommendRequest) -> str:
    # Escape quotes to prevent prompt injection/corruption
    room_type = req.room_type.replace('"', '\\"')
    style = req.style.replace('"', '\\"')
    furniture_density = req.furniture_density.value.replace('"', '\\"')
    gender = req.gender.value.replace('"', '\\"')
    room_area_m2 = _get_room_area_m2(req)
    density_applied, warning = _adjust_density_for_area(room_area_m2, req.furniture_density.value)
    prompt_categories = _pick_prompt_categories(req)
    room_type_value = req.room_type.value if hasattr(req.room_type, "value") else str(req.room_type)

    user_input = f"""Room type: {room_type}
Desired style: {style}
Room dimensions: {req.width}m x {req.length}m x {req.height}m
Room area: {room_area_m2}m²
Furniture density: {furniture_density}
User gender: {gender}"""
    if req.age:
        user_input += f"\nUser age: {req.age}"

    max_w = round(req.width * 100 * 0.4, 1)
    max_d = round(req.length * 100 * 0.35, 1)
    max_area_cm2 = round(room_area_m2 * 10000 * 0.10, 1)

    # Use cached metadata
    available_styles = ", ".join(f'"{s}"' for s in _cached_styles)
    available_categories = ", ".join(f'"{c}"' for c in prompt_categories)

    return f"""You are an expert interior design AI assistant specializing in Vietnamese home decor.
Return ONLY a valid JSON object. No explanation, no markdown, no code fences.

## USER CONTEXT
{user_input}

## TASK
## STEP 0 -- IMAGE VALIDATION (MANDATORY)
Run this step before any other analysis. Do NOT skip.

### Task
Determine whether the uploaded image shows a real or realistically rendered
INDOOR ROOM SPACE suitable for interior design analysis.

### Classification Rules
Set "isRoom": true ONLY when the image clearly shows:
- A real-life interior room (living room, bedroom, kitchen, bathroom,
    dining room, home office, etc.)
- A photorealistic 3D render of an interior room
- A staged showroom or model home interior

Set "isRoom": false when the image shows ANY of the following:
- A person, animal, or human face
- An outdoor or semi-outdoor scene (garden, balcony, patio, street)
- Food, plants, or isolated objects on plain backgrounds
- Abstract, artistic, or cartoon illustrations
- A floor plan, blueprint, or top-down schematic
- A hand-drawn sketch or mood board
- A single furniture item with no room context
- An image too dark, blurry, or cropped to identify the space

### Confidence Handling
If the image is ambiguous (e.g., partially visible room, unusual angle):
- Set "isRoom": true only if you are >= 80% confident it is an interior room
- Otherwise set "isRoom": false and explain in "notRoomReason"

### Room Visibility Assessment (chỉ khi isRoom: true)
Set "roomVisibility":
- "FULL"    → ≥ 60% diện tích phòng rõ ràng
- "PARTIAL" → 30–60% phòng hiển thị (một góc, một phần)
- "MINIMAL" → < 30% (chỉ thấy 1 góc tường hoặc 1 vật thể)
Set "visibilityWarning": chuỗi giải thích nếu PARTIAL/MINIMAL, null nếu FULL

    ### Room Content Validation (chỉ khi isRoom: true)
    Assess whether the image shows sufficient contextual furniture for meaningful recommendations.
    - Set "roomContentValid": true when the image provides clear context for furnishing suggestions.
    - Set "missingFurniture": list any notably absent major items when relevant, or [] if not applicable.
    Note: Do NOT enforce hard-coded mandatory categories that would prevent returning reasonable suggestions.

    ### MANDATORY category inclusion (CRITICAL — ngôn ngữ mô hình)
    - For Bedroom: The "categories" array MUST include AT LEAST one of: "Giường" (bed) OR "Nệm" (mattress).
    - For Living Room: The "categories" array MUST include AT LEAST one of: "Sofa" OR "Sofa góc" (sectional sofa).
    
    This is non-negotiable. If Gemini detects the room type, these core categories MUST be in the output, even if image analysis suggests sparse density. Never omit core furniture from recommendations.


### Output Format
Respond ONLY with a valid JSON object. No explanation outside the JSON.

{{
    "isRoom": true,
    "confidence": 0.95,
    "notRoomReason": null,
    "roomType": "Living Room"
}}

### Examples
- Photo of a living room -> isRoom: true
- 3D render of a bedroom -> isRoom: true
- Photo of a cat on a sofa -> isRoom: false ("Image shows a cat, not a room")
- Floor plan drawing -> isRoom: false ("Image is a floor plan, not a room photo")
- Very dark blurry image -> isRoom: false ("Image too unclear to identify a room")

Analyze the room context and user profile, then recommend the most suitable furniture categories with detailed reasoning.

## OUTPUT SCHEMA
{{
    "isRoom": true,
    "notRoomReason": null,
    "roomVisibility": "FULL",
    "visibilityWarning": null,
    "roomContentValid": true,
    "missingFurniture": [],
  "imageAnalysis": {{
    "dominantColors": ["#hex1", "#hex2", "#hex3"],
    "colorTone": "warm|cool|neutral",
    "detectedStyle": "detected style name",
    "lightingType": "natural|artificial|mixed",
    "existingFurnitureCategories": []
  }},
  "recommendedFilter": {{
    "styles": ["Style1"],
    "colorHexRange": ["#hex1", "#hex2"],
    "colorTone": "warm|cool|neutral",
    "categories": [
      {{
        "category": "Sofa",
        "reasoning": "Sofa dạng modular phù hợp với phong cách Modern vì đường nét gọn gàng, không rườm rà. Tông màu xám trung tính (#6B7280) hoặc navy sẽ ăn khớp với bảng màu cool tone, đồng thời phù hợp với nam giới 26-45 tuổi ưa sự tối giản.",
        "styleAlignment": "Modern",
        "suggestedColorHex": "#6B7280",
        "materialHint": "vải chenille hoặc da tổng hợp"
      }}
    ],
        "roomType": "{room_type_value}",
        "roomAreaM2": {room_area_m2},
    "maxProductWidth": {max_w},
    "maxProductDepth": {max_d},
    "furnitureDensityHint": "{furniture_density}"
  }},
  "reasoning": {{
    "styleJustification": "Lý do chọn style dựa trên thông tin phòng và sở thích user",
    "colorJustification": "Lý do chọn bảng màu dựa trên ánh sáng tự nhiên và kích thước phòng",
    "densityJustification": "Lý do số lượng nội thất phù hợp với diện tích {req.width}x{req.length}m",
    "userProfileNote": "Ghi chú cá nhân hóa theo tuổi/giới tính nếu có thông tin"
  }}
}}

## STRICT RULES

### STEP 0 output behavior
- If the uploaded image is not a valid room image, set isRoom=false and provide notRoomReason.
- If isRoom=false, still return a valid JSON object and keep the output parseable.
- If isRoom=true, set notRoomReason=null.

### Schema constraints
- styles: chỉ dùng từ danh sách: [{available_styles}]
- categories[].category: chỉ dùng từ danh sách: [{available_categories}]
- colorTone: "warm" | "cool" | "neutral"
- lightingType: "natural" | "artificial" | "mixed"
- furnitureDensityHint: "sparse" | "medium" | "dense"
- styleAlignment: phải khớp với một style trong recommendedFilter.styles
- suggestedColorHex: phải nằm trong palette colorHexRange

### Furniture density
- "sparse" → 2-3 danh mục thiết yếu
- "medium" → 4-6 danh mục cốt lõi
- "dense" → 7-10 danh mục bao gồm phụ kiện trang trí

### Room-based category filtering
- Chỉ đưa vào prompt những category phù hợp với room type hiện tại.
- Bedroom chỉ ưu tiên các category liên quan đến ngủ, lưu trữ cá nhân và ánh sáng đầu giường.
- Living Room chỉ ưu tiên các category liên quan đến ngồi, bàn phụ, lưu trữ và chiếu sáng sinh hoạt.
- Nếu room area <= 8m², giữ prompt ngắn hơn và ưu tiên category thiết yếu.

### Small-room behavior
- Nếu room area <= 6m², ưu tiên sản phẩm compact, đa năng và tránh món chiếm nhiều mặt sàn.
- Món nào có footprint lớn hơn {max_area_cm2} cm² phải bị loại khỏi kết quả.

### Dimension constraints
- maxProductWidth = {max_w} cm
- maxProductDepth = {max_d} cm
- densityApplied = {density_applied}
- warning = {warning or "None"}
- maxProductArea = {max_area_cm2} cm²

### Reasoning quality (MANDATORY — minimum 2 sentences per category)
1. Tại sao item này phù hợp với style đã chọn, phải nhắc ít nhất 1 đặc điểm thiết kế cụ thể của chính category đó.
2. Màu sắc/chất liệu nào phù hợp với colorHexRange và tại sao, phải nêu 1 màu hoặc 1 chất liệu rõ ràng.
3. Liên hệ với tuổi/giới tính user nếu có thông tin, gắn trực tiếp vào công năng của item.
4. Không được dùng câu rập khuôn như "Sản phẩm này là một lựa chọn phù hợp." hoặc câu tương đương chung chung.

### Category reasoning quality
- Mỗi category phải có reasoning riêng, không được lặp lại nguyên văn giữa các category.
- Ưu tiên nhắc đến chi tiết thật của category: sofa/giường/tủ/kệ/bàn/ghế/đèn/thảm/cây trang trí.
- Nếu category có tên trong danh sách DB nhưng khác nhau chỉ bởi khoảng trắng hoặc hoa thường, hãy xem là cùng một category.
- Nếu category là nhóm rộng như "Đồ nội thất" hoặc "Hàng trang trí", hãy mô tả theo chức năng cụ thể của item thay vì câu mô tả mơ hồ.

### Age-based personalization
- 10-25 tuổi: style năng động (Bohemian, Modern, Tropical Modern), màu sắc tươi sáng, bão hòa cao
- 26-45 tuổi: style cân bằng (Scandinavian, Mid-Century, Japandi), màu trung tính linh hoạt
- 46+ tuổi: style cổ điển (Classic, Rustic, Haussmannian), màu trung tính/nhạt, nội thất tiện dụng

Return ONLY the JSON object, nothing else."""



async def analyze_room(
    req: RecommendRequest,
    image_bytes: Optional[bytes] = None,
    image_mime: str = "image/jpeg",
) -> GeminiAnalysisResult:

    if not _key_pool.has_keys:
        print("[Gemini] No API keys configured.")
        raise HTTPException(status_code=503, detail="AI service is not configured.")

    prompt_text = _build_prompt(req)

    parts = []
    normalized_mime = _normalize_image_mime_type(image_bytes, image_mime)
    if image_bytes:
        parts.append({
            "inline_data": {
                "mime_type": normalized_mime,
                "data": base64.b64encode(image_bytes).decode("utf-8"),
            }
        })
    parts.append({"text": prompt_text})

    payload = {
        "contents": [{"parts": parts}],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 8192,
            "responseMimeType": "application/json",
        },
    }

    print(f"[Gemini] Request URL: {GEMINI_URL}")
    print(f"[Gemini] Payload parts count: {len(parts)}")
    if image_bytes:
        print(f"[Gemini] Image included, size: {len(image_bytes)} bytes")

    # Compute a cache key for this request to avoid duplicate calls in quick succession
    hasher = hashlib.sha256()
    hasher.update(prompt_text.encode("utf-8"))
    if image_bytes:
        hasher.update(image_bytes)
    cache_key = hasher.hexdigest()

    # Return cached result if fresh
    cached = _request_cache.get(cache_key)
    if cached:
        ts, parsed_json = cached
        if time.time() - ts < _CACHE_TTL_SECONDS:
            print("[Gemini] Returning cached analysis result.")
            return GeminiAnalysisResult(**parsed_json)
        else:
            # expired
            del _request_cache[cache_key]

    # Try keys in round-robin with retries and fallback behavior
    max_attempts = 3 * len(_key_pool._keys) if _key_pool._keys else 3
    for attempt in range(max(max_attempts, 3)):
        current_key = _key_pool.next_key()
        if not current_key:
            print("[Gemini] All keys exhausted. Returning fallback.")
            return _build_fallback_analysis(req)

        try:
            # throttle concurrency to avoid bursts
            async with _gemini_semaphore:
                timeout = httpx.Timeout(
                    connect=10.0,
                    read=GEMINI_TIMEOUT_SECONDS,
                    write=GEMINI_TIMEOUT_SECONDS,
                    pool=10.0,
                )
                async with httpx.AsyncClient(timeout=timeout) as client:
                    resp = await client.post(
                        f"{GEMINI_URL}?key={current_key}",
                        json=payload,
                    )

            status = getattr(resp, "status_code", None)
            if status in (429, 503):
                if _is_daily_quota_exhausted(resp):
                    _key_pool.mark_exhausted(current_key)
                    print(f"[Gemini] Key exhausted, switching to next key...")
                    continue  # try next key immediately

                # Rate limit / transient upstream problem → retry with backoff
                if attempt == max_attempts - 1:
                    print("[Gemini] Final retry exhausted. Returning fallback analysis.")
                    return _build_fallback_analysis(req)

                wait = min(15, 2 ** (attempt % 3)) + random.uniform(0, 1)
                print(f"[Gemini] Upstream rate/availability issue (status={status}) attempt {attempt + 1}/{max_attempts}, waiting {wait:.1f}s...")
                await asyncio.sleep(wait)
                continue

            if status == 400:
                error_detail = resp.text[:300] if resp.text else "Bad Request"
                print(f"[Gemini] HTTP 400 error: {error_detail}")
                if _is_invalid_api_key(resp):
                    _key_pool.mark_exhausted(current_key)
                    print(f"[Gemini] Invalid/expired API key detected, switching to next key...")
                    continue
                if "UNSUPPORTED MIME TYPE" in error_detail.upper():
                    print("[Gemini] Unsupported inline image MIME type; using fallback analysis.")
                    return _build_fallback_analysis(req)
                raise HTTPException(status_code=400, detail=f"Invalid Gemini request: {error_detail[:100]}")

            resp.raise_for_status()
            data = resp.json()

            raw_text = data["candidates"][0]["content"]["parts"][0]["text"]
            print(f"[Gemini] Raw response: {raw_text[:200]}")

            # Clean JSON
            clean = _extract_json_text(raw_text)

            try:
                parsed = json.loads(clean)
            except json.JSONDecodeError as parse_error:
                print(f"[Gemini] JSON parse failed: {parse_error}")
                recovered = _recover_truncated_json_text(clean)
                if recovered != clean:
                    print(f"[Gemini] Retrying parse with recovered JSON tail, length={len(recovered)}")
                    parsed = json.loads(recovered)
                else:
                    raise
            parsed = _normalize_gemini_response(parsed)
            parsed = _ensure_analysis_defaults(parsed, req)
            parsed = _hydrate_non_room_payload(parsed, req)
            result = GeminiAnalysisResult(**parsed)

            # Cache only validated result
            _request_cache[cache_key] = (time.time(), result.model_dump())

            print(f"[Gemini] Success on attempt {attempt + 1}")
            return result

        except ValidationError as val_exc:
            print(f"[Gemini] Validation failed on attempt {attempt + 1}: {val_exc}")
            # Do not rotate all keys for schema-shape failures; return deterministic fallback.
            if isinstance(locals().get("parsed"), dict) and parsed.get("isRoom") is False:
                safe_parsed = _hydrate_non_room_payload(parsed, req)
                safe_result = GeminiAnalysisResult(**safe_parsed)
                _request_cache[cache_key] = (time.time(), safe_result.model_dump())
                print("[Gemini] Returned hydrated non-room response after validation mismatch.")
                return safe_result
            return _build_fallback_analysis(req)

        except httpx.HTTPStatusError as http_exc:
            # If response indicates daily quota exhaustion, blacklist this key and continue
            resp_obj = getattr(http_exc, "response", None)
            if resp_obj and _is_daily_quota_exhausted(resp_obj):
                _key_pool.mark_exhausted(current_key)
                print(f"[Gemini] Key marked exhausted from exception, trying next key...")
                continue

            print(f"[Gemini] HTTP status error attempt {attempt + 1}: {http_exc}")
            if attempt == max_attempts - 1:
                print("[Gemini] All retries failed. Returning fallback analysis.")
                return _build_fallback_analysis(req)
            await asyncio.sleep(min(15, 2 ** (attempt % 3)))

        except Exception as exc:
            print(f"[Gemini] Error with key ...{current_key[-6:]}: {exc}")
            if attempt == max_attempts - 1:
                print("[Gemini] All retries failed. Returning fallback analysis.")
                return _build_fallback_analysis(req)
            wait = min(10, 2 ** (attempt % 3))
            await asyncio.sleep(wait)

    return _build_fallback_analysis(req)