"""
routers/recommend.py
POST /api/v1/recommend  — accepts multipart/form-data (image optional)
"""

import os
from typing import Optional

from fastapi import APIRouter, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from app.models.schemas import RecommendRequest, RecommendResponse, FurnitureDensity, Gender
from app.services.gemini_service import analyze_room
from app.services.mongo_service import get_recommendations, get_recommendations_mock

router = APIRouter()

USE_MOCK_DB = os.getenv("USE_MOCK_DB", "true").lower() == "true"


@router.post("/recommend", response_model=RecommendResponse)
async def recommend_furniture(
    room_type: str = Form(..., example="living room"),
    style: str = Form(..., example="modern"),
    width: float = Form(..., example=4.5),
    length: float = Form(..., example=6.0),
    height: float = Form(..., example=2.8),
    furniture_density: FurnitureDensity = Form(..., example="medium"),
    gender: Gender = Form(..., example="female"),
    image: Optional[UploadFile] = File(None),
):
    """
    Analyze a room image (optional) + user preferences, then return
    furniture product recommendations ranked by color match.
    """
    # 1. Build request object
    req = RecommendRequest(
        room_type=room_type,
        style=style,
        width=width,
        length=length,
        height=height,
        furniture_density=furniture_density,
        gender=gender,
    )

    # 2. Read image bytes if provided
    image_bytes = None
    image_mime = "image/jpeg"
    if image and image.filename:
        image_bytes = await image.read()
        image_mime = image.content_type or "image/jpeg"

    # 3. Gemini analysis
    try:
        analysis = await analyze_room(req, image_bytes, image_mime)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Image analysis failed: {exc}")

    # 4. MongoDB query + semantic ranking
    try:
        if USE_MOCK_DB:
            products, total_candidates = await get_recommendations_mock(analysis)
        else:
            products, total_candidates = await get_recommendations(analysis)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Database query failed: {exc}")

    return RecommendResponse(
        analysis=analysis,
        products=products,
        total_candidates=total_candidates,
        total_returned=len(products),
    )


@router.post("/analyze-only", response_model=dict)
async def analyze_only(
    room_type: str = Form(...),
    style: str = Form(...),
    width: float = Form(...),
    length: float = Form(...),
    height: float = Form(...),
    furniture_density: FurnitureDensity = Form(...),
    gender: Gender = Form(...),
    image: Optional[UploadFile] = File(None),
):
    """Return only Gemini analysis JSON without product lookup."""
    req = RecommendRequest(
        room_type=room_type, style=style, width=width, length=length,
        height=height, furniture_density=furniture_density, gender=gender,
    )
    image_bytes = None
    image_mime = "image/jpeg"
    if image and image.filename:
        image_bytes = await image.read()
        image_mime = image.content_type or "image/jpeg"

    analysis = await analyze_room(req, image_bytes, image_mime)
    return analysis.dict()