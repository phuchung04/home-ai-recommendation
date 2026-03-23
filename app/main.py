import os
from dotenv import load_dotenv
load_dotenv()

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import recommend, training
from app.services.cf_model import load_model
from app.services.gemini_service import load_metadata

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model
    print("Loading CF model...")
    load_model()
    await load_metadata()
    yield
    # Clean up the model and release the resources
    print("Successfully shut down")

app = FastAPI(
    title="Interior Design Furniture Recommendation API",
    description="AI-powered furniture recommendation using room image analysis",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(recommend.router, prefix="/api/v1", tags=["Recommendations"])
app.include_router(training.router, prefix="/api/v1/admin", tags=["Admin"])


@app.get("/")
def root():
    return {"status": "ok", "message": "Interior Design Recommendation API is running"}