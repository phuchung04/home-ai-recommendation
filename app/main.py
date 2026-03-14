import os
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import recommend

app = FastAPI(
    title="Interior Design Furniture Recommendation API",
    description="AI-powered furniture recommendation using room image analysis",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(recommend.router, prefix="/api/v1", tags=["Recommendations"])

@app.get("/")
def root():
    return {"status": "ok", "message": "Interior Design Recommendation API is running"}