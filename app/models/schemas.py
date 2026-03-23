from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum


class FurnitureDensity(str, Enum):
    sparse = "sparse"
    medium = "medium"
    dense = "dense"


class Gender(str, Enum):
    male = "male"
    female = "female"
    other = "other"


class RoomType(str, Enum):
    LIVING_ROOM = "Living Room"
    BEDROOM = "Bedroom"


# ---------- Request ----------

class RecommendRequest(BaseModel):
    room_type: RoomType = Field(..., example="Living Room")
    style: str = Field(..., example="modern")
    width: float = Field(..., gt=0, description="Room width in meters", example=4.5)
    length: float = Field(..., gt=0, description="Room length in meters", example=6.0)
    height: float = Field(..., gt=0, description="Room height in meters", example=2.8)
    furniture_density: FurnitureDensity = Field(..., example="medium")
    gender: Gender = Field(..., example="female")
    age: Optional[int] = Field(None, gt=0, description="User's age", example=25)
    user_id: Optional[str] = None  # ← thêm để CF scoring


# ---------- Gemini Analysis Output ----------

class ImageAnalysis(BaseModel):
    dominantColors: List[str]
    colorTone: str
    detectedStyle: str
    lightingType: str
    existingFurnitureCategories: List[str]


class RecommendedFilter(BaseModel):
    styles: List[str]
    colorHexRange: List[str]
    colorTone: str
    categories: "List[CategoryRecommendation]"
    maxProductWidth: float
    maxProductDepth: float
    furnitureDensityHint: str

class CategoryRecommendation(BaseModel):
    category: str
    reasoning: str



class GeminiAnalysisResult(BaseModel):
    imageAnalysis: ImageAnalysis
    recommendedFilter: RecommendedFilter
    reasoning: str


# ---------- Product (from MongoDB) ----------

class ProductDimensions(BaseModel):
    width: Optional[float] = None
    depth: Optional[float] = None
    height: Optional[float] = None


class Product(BaseModel):
    id: str
    name: str
    category: str
    styles: Optional[List[str]] = []
    price: Optional[float] = None
    dimensions: Optional[ProductDimensions] = None
    colors: Optional[List[str]] = []
    imageUrl: Optional[str] = None
    color_distance: Optional[float] = None  # semantic ranking score
    ranking_score: Optional[float] = None  # weighted ranking score
    reasoning: Optional[str] = None


# ---------- Response ----------

class RecommendResponse(BaseModel):
    analysis: GeminiAnalysisResult
    products: List[Product]
    total_candidates: int
    total_returned: int