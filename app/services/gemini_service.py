"""
gemini_service.py
"""

import os
import json
import base64
import httpx
import asyncio
from typing import Optional, List
from fastapi import HTTPException

from app.models.schemas import RecommendRequest, GeminiAnalysisResult
from app.services.mongo_service import get_distinct_categories, get_distinct_styles

GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.5-flash:generateContent"
)

# ── Metadata Caching ─────────────────────────────────────────────────────────

_cached_categories: List[str] = []
_cached_styles: List[str] = []

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
            _cached_categories = categories
            print(f"[Metadata] Loaded {len(_cached_categories)} categories.")
        else:
            print("[Metadata] Warning: No categories found in DB, using empty list.")

        # Load styles and merge with the hardcoded list
        styles_from_db = await get_distinct_styles()
        if styles_from_db:
            print(f"[Metadata] Found {len(styles_from_db)} styles in DB.")
            
            # Merge and remove duplicates, then sort
            combined_styles = sorted(list(set(_cached_styles + styles_from_db)))
            _cached_styles = combined_styles
        
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

    user_input = f"""Room type: {room_type}
Desired style: {style}
Room dimensions: {req.width}m x {req.length}m x {req.height}m
Furniture density: {furniture_density}
User gender: {gender}"""
    if req.age:
        user_input += f"\nUser age: {req.age}"

    max_w = round(req.width * 100 * 0.4, 1)
    max_d = round(req.length * 100 * 0.35, 1)

    # Use cached metadata
    available_styles = ", ".join(f'"{s}"' for s in _cached_styles)
    available_categories = ", ".join(f'"{c}"' for c in _cached_categories)

    return f"""You are an interior design AI assistant.
Return ONLY a valid JSON object, no explanation, no markdown, no code fences.

USER INPUT:
{user_input}

Return this exact JSON structure:
{{
  "imageAnalysis": {{
    "dominantColors": ["#hex1", "#hex2", "#hex3"],
    "colorTone": "warm",
    "detectedStyle": "modern",
    "lightingType": "natural",
    "existingFurnitureCategories": []
  }},
  "recommendedFilter": {{
    "styles": ["Modern"],
    "colorHexRange": ["#hex1", "#hex2"],
    "colorTone": "warm",
    "categories": [
        {{ "category": "Sofa", "reasoning": "A sofa is essential for a living room."}},
        {{ "category": "Bàn", "reasoning": "A coffee table complements the sofa."}},
        {{ "category": "Ghế", "reasoning": "An armchair provides extra seating."}}
    ],
    "maxProductWidth": {max_w},
    "maxProductDepth": {max_d},
    "furnitureDensityHint": "{furniture_density}"
  }},
  "reasoning": "One sentence here"
}}

RULES:
- room_type must be one of: "Living Room", "Bedroom".
- styles must be from this dynamic list: [{available_styles}]
- categories must be from this dynamic list of Vietnamese names: [{available_categories}]
- colorTone must be one of: warm, cool, neutral
- lightingType must be one of: natural, artificial, mixed
- furnitureDensityHint must be one of: sparse, medium, dense
- Based on 'furnitureDensity', adjust the number of 'categories' recommended:
  - "sparse": Recommend 2-3 essential furniture categories.
  - "medium": Recommend 4-6 core furniture categories.
  - "dense": Recommend 7-10 furniture categories, including accessories and decor items.
- maxProductWidth = {max_w}
- maxProductDepth = {max_d}
- Each category in the list must have a brief 'reasoning' in Vietnamese.
- If 'User age' is provided, tailor recommendations:
  - Ages 10-25: Prefer vibrant, trendy styles (e.g., Bohemian, Modern, Tropical Modern) and brighter, more saturated color palettes.
  - Ages 26-45: Offer a balanced mix of contemporary and timeless styles (e.g., Scandinavian, Mid-Century, Japandi) with versatile color schemes.
  - Ages 46+: Suggest classic, comfortable, and elegant styles (e.g., Classic, Rustic, Haussmannian) with more neutral or muted color palettes and accessible furniture (e.g., lower beds, supportive chairs).
- Return ONLY the JSON, nothing else
"""


async def analyze_room(
    req: RecommendRequest,
    image_bytes: Optional[bytes] = None,
    image_mime: str = "image/jpeg",
) -> GeminiAnalysisResult:

    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    print(f"[Gemini] API KEY loaded: {'YES' if GEMINI_API_KEY else 'NO - MISSING!'}")

    if not GEMINI_API_KEY:
        print("[Gemini] No API key, cannot proceed with analysis.")
        raise HTTPException(status_code=503, detail="AI service is not configured.")

    prompt_text = _build_prompt(req)

    parts = []
    if image_bytes:
        parts.append({
            "inline_data": {
                "mime_type": image_mime,
                "data": base64.b64encode(image_bytes).decode("utf-8"),
            }
        })
    parts.append({"text": prompt_text})

    payload = {
        "contents": [{"parts": parts}],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 4096,
            "responseMimeType": "application/json",  # ✅ ép Gemini trả JSON thuần
        },
    }

    for attempt in range(3):
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(
                    f"{GEMINI_URL}?key={GEMINI_API_KEY}",
                    json=payload,
                )

                if resp.status_code == 429:
                    wait = (attempt + 1) * 10
                    print(f"[Gemini] Rate limited (attempt {attempt + 1}/3), waiting {wait}s...")
                    await asyncio.sleep(wait)
                    continue

                resp.raise_for_status()
                data = resp.json()

            raw_text = data["candidates"][0]["content"]["parts"][0]["text"]
            print(f"[Gemini] Raw response: {raw_text[:200]}")

            # Clean JSON
            clean = raw_text.strip()
            if clean.startswith("```"):
                clean = clean.split("```")[1]
                if clean.startswith("json"):
                    clean = clean[4:]
            clean = clean.strip()

            parsed = json.loads(clean)
            print(f"[Gemini] Success on attempt {attempt + 1}")
            return GeminiAnalysisResult(**parsed)

        except Exception as exc:
            print(f"[Gemini] Error attempt {attempt + 1}: {exc}")
            if attempt == 2:
                print("[Gemini] All retries failed.")
                raise HTTPException(status_code=503, detail="AI analysis service failed after multiple retries.")
            await asyncio.sleep(5)

    # This part should ideally not be reached if the loop logic is correct.
    raise HTTPException(status_code=500, detail="An unexpected error occurred in the AI analysis service.")