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
from app.services.mongo_service import get_recommendations

router = APIRouter()


@router.post("/recommend", response_model=RecommendResponse)
async def recommend_furniture(
    room_type: str = Form(..., example="living room"),
    style: str = Form(..., example="modern"),
    width: float = Form(..., example=4.5),
    length: float = Form(..., example=6.0),
    height: float = Form(..., example=2.8),
    area_m2: Optional[float] = Form(None),
    furniture_density: FurnitureDensity = Form(..., example="medium"),
    gender: Gender = Form(..., example="female"),
    age: Optional[int] = Form(None, example=25),
    user_id: Optional[str] = Form(None),
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
        area_m2=area_m2,
        furniture_density=furniture_density,
        gender=gender,
        age=age,
        user_id=user_id,
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
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Image analysis failed: {exc}")

    if not analysis.isRoom:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "INVALID_IMAGE",
                "message": "Ảnh tải lên không phải là hình ảnh căn phòng.",
                "reason": analysis.notRoomReason or "Không nhận diện được phòng.",
            },
        )

    # If room visibility is too limited, block and ask for a wider shot
    if getattr(analysis, "roomVisibility", None) == "MINIMAL":
        raise HTTPException(status_code=422, detail={
            "error": "INSUFFICIENT_ROOM_VIEW",
            "message": "Góc chụp quá hẹp, không đủ thông tin để gợi ý sản phẩm.",
            "reason": analysis.visibilityWarning,
        })

    # Validate required furniture/content for the selected room type
    # NOTE: Do not block the recommendation flow when required furniture is missing.
    # Instead, attach a friendly warning message so the frontend can inform the user
    ROOM_CONTENT_MESSAGES = {
        "Bedroom": "Phòng ngủ nên có giường/nệm và bàn đầu giường.",
        "Living Room": "Phòng khách nên có sofa.",
    }
    content_warning = None
    if getattr(analysis, "roomContentValid", True) is False:
        missing = analysis.missingFurniture or []
        base = ROOM_CONTENT_MESSAGES.get(req.room_type, "Phòng có thể thiếu nội thất cần thiết.")
        if missing:
            content_warning = f"{base} Thiếu: {', '.join(missing)}."
        else:
            content_warning = base

    # 4. MongoDB query + semantic ranking
    try:
        products, total_candidates, warning, density_applied = await get_recommendations(
            analysis, user_id=req.user_id, top_n=30
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Database query failed: {exc}")

    # Merge warnings: gemini/mongo warning take precedence, but include content warning if present
    final_warning = warning
    if not final_warning and content_warning:
        final_warning = content_warning

    return RecommendResponse(
        analysis=analysis,
        products=products,
        total_candidates=total_candidates,
        total_returned=len(products),
        warning=final_warning,
        densityApplied=density_applied,
    )


@router.post("/analyze-only", response_model=dict)
async def analyze_only(
    room_type: str = Form(...),
    style: str = Form(...),
    width: float = Form(...),
    length: float = Form(...),
    height: float = Form(...),
    area_m2: Optional[float] = Form(None),
    furniture_density: FurnitureDensity = Form(...),
    gender: Gender = Form(...),
    age: Optional[int] = Form(None),
    image: Optional[UploadFile] = File(None),
):
    """Return only Gemini analysis JSON without product lookup."""
    req = RecommendRequest(
        room_type=room_type, style=style, width=width, length=length,
        height=height, area_m2=area_m2, furniture_density=furniture_density, gender=gender,
        age=age,
    )
    image_bytes = None
    image_mime = "image/jpeg"
    if image and image.filename:
        image_bytes = await image.read()
        image_mime = image.content_type or "image/jpeg"

    try:
        analysis = await analyze_room(req, image_bytes, image_mime)
    except HTTPException:
        raise
    return analysis.dict()