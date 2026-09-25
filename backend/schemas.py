from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    status: str
    message: str


class UserCreate(BaseModel):
    name: str = Field(min_length=2, max_length=120)
    email: str = Field(min_length=3, max_length=255)
    password: str = Field(min_length=1, max_length=255)
    role: str = Field(default="user", max_length=50)


class LoginRequest(BaseModel):
    email: str
    password: str


class UserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    email: str
    role: str


class VendorResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    category: str
    description: str
    location: str
    distance: float
    rating: float
    review_count: int
    price: str
    available: bool
    phone: str


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)


class SearchResult(VendorResponse):
    recommendation_score: float
    matched_category: Optional[str] = None


class SentimentRequest(BaseModel):
    review: str = Field(min_length=1, max_length=2000)


class SentimentResponse(BaseModel):
    success: bool
    sentiment: str
    confidence: float
    explanation: str


class BookingCreate(BaseModel):
    user_id: int
    vendor_id: int
    service: str = Field(min_length=1, max_length=150)
    booking_date: datetime
    status: str = Field(default="pending", max_length=30)


class BookingUpdate(BaseModel):
    status: Optional[str] = Field(default=None, max_length=30)
    booking_date: Optional[datetime] = None
    service: Optional[str] = Field(default=None, min_length=1, max_length=150)


class BookingResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: int
    vendor_id: int
    service: str
    booking_date: datetime
    status: str


class ReviewCreate(BaseModel):
    user_id: int
    vendor_id: int
    text: str = Field(min_length=1, max_length=2000)
    rating: int = Field(ge=1, le=5)


class ReviewResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: int
    vendor_id: int
    text: str
    rating: int
    sentiment: str
    confidence: float
