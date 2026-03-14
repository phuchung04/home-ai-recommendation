"""
gemini_service.py
"""

import os
import json
import base64
import httpx
import asyncio
from typing import Optional

from app.models.schemas import RecommendRequest, GeminiAnalysisResult

GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.5-flash:generateContent"
)


def _build_prompt(req: RecommendRequest) -> str:
    return f"""You are an interior design AI assistant.
Return ONLY a valid JSON object, no explanation, no markdown, no code fences.

USER INPUT:
Room type: {req.room_type}
Desired style: {req.style}
Room dimensions: {req.width}m x {req.length}m x {req.height}m
Furniture density: {req.furniture_density.value}

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
    "categories": ["Sofa", "Ban", "Ghe"],
    "maxProductWidth": {round(req.width * 100 * 0.4, 1)},
    "maxProductDepth": {round(req.length * 100 * 0.35, 1)},
    "furnitureDensityHint": "{req.furniture_density.value}"
  }},
  "reasoning": "One sentence here"
}}

RULES:
- styles must be from: ["Modern", "Minimalist", "Classic", "Scandinavian", "Industrial", "Bohemian", "Rustic"]
- colorTone must be one of: warm, cool, neutral
- lightingType must be one of: natural, artificial, mixed
- furnitureDensityHint must be one of: sparse, medium, dense
- maxProductWidth = {round(req.width * 100 * 0.4, 1)}
- maxProductDepth = {round(req.length * 100 * 0.35, 1)}
- categories use these Vietnamese names: Sofa, Ghế, Bàn, Giường, Tủ, Kệ, Đèn
- Return ONLY the JSON, nothing else
"""


def _fallback_analysis(req: RecommendRequest) -> GeminiAnalysisResult:
    style_map = {
        "modern": ["Modern", "Minimalist"],
        "minimalist": ["Minimalist", "Scandinavian"],
        "classic": ["Classic"],
        "scandinavian": ["Scandinavian", "Minimalist"],
        "industrial": ["Industrial", "Modern"],
        "bohemian": ["Bohemian", "Rustic"],
        "rustic": ["Rustic", "Bohemian"],
    }
    styles = style_map.get(req.style.strip().lower(), ["Modern"])

    room_category_map = {
        "living room": ["Sofa", "Bàn", "Ghế", "Kệ", "Đèn"],
        "bedroom": ["Giường", "Tủ", "Đèn", "Kệ"],
        "dining room": ["Bàn", "Ghế", "Đèn"],
        "office": ["Bàn", "Ghế", "Kệ", "Đèn"],
        "kitchen": ["Bàn", "Ghế", "Tủ"],
    }
    categories = room_category_map.get(req.room_type.lower(), ["Sofa", "Bàn", "Ghế"])

    return GeminiAnalysisResult(
        imageAnalysis={
            "dominantColors": ["#F5F5F5", "#E0D5C5", "#8B7355"],
            "colorTone": "neutral",
            "detectedStyle": req.style.strip().lower(),
            "lightingType": "natural",
            "existingFurnitureCategories": [],
        },
        recommendedFilter={
            "styles": styles,
            "colorHexRange": ["#F5F5F5", "#E0D5C5"],
            "colorTone": "neutral",
            "categories": categories,
            "maxProductWidth": round(req.width * 100 * 0.4, 1),
            "maxProductDepth": round(req.length * 100 * 0.35, 1),
            "furnitureDensityHint": req.furniture_density.value,
        },
        reasoning=f"Fallback analysis for {req.style.strip()} {req.room_type} "
                  f"({req.width}m x {req.length}m).",
    )


async def analyze_room(
    req: RecommendRequest,
    image_bytes: Optional[bytes] = None,
    image_mime: str = "image/jpeg",
) -> GeminiAnalysisResult:

    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    print(f"[Gemini] API KEY loaded: {'YES' if GEMINI_API_KEY else 'NO - MISSING!'}")

    if not GEMINI_API_KEY:
        print("[Gemini] No API key — using fallback")
        return _fallback_analysis(req)

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
            "maxOutputTokens": 1024,
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
                print("[Gemini] All retries failed — using fallback")
                return _fallback_analysis(req)
            await asyncio.sleep(5)

    return _fallback_analysis(req)