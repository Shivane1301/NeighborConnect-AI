from datetime import datetime
from typing import Literal
from typing import Optional

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, computed_field, model_validator


ServiceCategory = Literal[
    "Plumber",
    "Electrician",
    "Cleaner",
    "Carpenter",
    "Food Vendor",
    "Appliance Repair",
]


class HealthResponse(BaseModel):
    status: str
    message: str


class UserCreate(BaseModel):
    name: str = Field(min_length=2, max_length=120)
    email: str = Field(min_length=3, max_length=255)
    password: str = Field(min_length=8, max_length=255)
    role: Literal["customer", "vendor"] = "customer"
    phone: str | None = Field(default=None, max_length=30)
    location: str | None = Field(default=None, max_length=150)
    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)
    preferences: dict = Field(default_factory=dict)
    category: ServiceCategory | None = None
    description: str | None = Field(default=None, max_length=3000)
    service_details: str | None = Field(default=None, max_length=3000)
    price: str | None = Field(default=None, max_length=80)
    available: bool = True

    @model_validator(mode="after")
    def validate_role_details(self):
        if (self.latitude is None) != (self.longitude is None):
            raise ValueError("Latitude and longitude must be provided together.")
        if self.role == "vendor":
            missing = [
                field
                for field in ("phone", "location", "category", "description", "price")
                if not getattr(self, field)
            ]
            if missing:
                raise ValueError(f"Vendor registration requires: {', '.join(missing)}.")
        elif not self.phone or not self.location:
            raise ValueError("Customer registration requires a phone number and location.")
        return self


class LoginRequest(BaseModel):
    email: str
    password: str


class UserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    email: str
    role: str
    phone: str | None = None
    location: str | None = None
    preferences: dict = Field(default_factory=dict)
    vendor_id: int | None = None
    access_token: str | None = None


class VendorResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    category: str
    description: str
    location: str
    distance: float | None = None
    distance_is_estimate: bool = True
    rating: float
    review_count: int
    price: str
    available: bool
    phone: str
    latitude: float | None = None
    longitude: float | None = None
    service_details: str = ""


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    location: str | None = Field(default=None, max_length=150)
    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)
    available_only: bool = False
    nearby: bool = False
    highly_rated: bool = False
    max_price: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_coordinates(self):
        if (self.latitude is None) != (self.longitude is None):
            raise ValueError("Latitude and longitude must be provided together.")
        return self


class SearchResult(VendorResponse):
    recommendation_score: float
    matched_category: Optional[str] = None
    recommendation_reasons: list[str]
    recommendation_summary: str


class SentimentRequest(BaseModel):
    review: str = Field(min_length=1, max_length=2000)


class SentimentResponse(BaseModel):
    success: bool
    sentiment: str
    confidence: Optional[float]
    explanation: str


class BookingCreate(BaseModel):
    user_id: int | None = None
    vendor_id: int
    service: str = Field(min_length=1, max_length=150)
    booking_date: datetime
    message: str = Field(default="", max_length=2000)


class BookingUpdate(BaseModel):
    status: Optional[Literal["pending", "accepted", "completed", "cancelled"]] = None
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
    message: str = ""
    reviewed: bool = False
    vendor_name: str | None = None
    customer_name: str | None = None


class ReviewCreate(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    user_id: int | None = None
    vendor_id: int | None = None
    booking_id: int
    review: str = Field(
        default="",
        min_length=1,
        max_length=2000,
        validation_alias=AliasChoices("review", "text"),
    )
    rating: int = Field(ge=1, le=5)

    @property
    def text(self) -> str:
        return self.review


class ReviewResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    id: int
    user_id: int
    vendor_id: int
    booking_id: int | None = None
    text: str = ""
    rating: int
    sentiment: str
    confidence: float
    created_at: datetime | None = None

    @computed_field
    @property
    def review(self) -> str:
        return self.text


class ReviewSubmissionResponse(BaseModel):
    success: bool = True
    id: int | None = None
    user_id: int | None = None
    vendor_id: int | None = None
    booking_id: int | None = None
    rating: int | None = None
    review: ReviewResponse
    sentiment: str | None = None
    confidence: float | None = None
    updated_vendor: VendorResponse


class VendorUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=2, max_length=150)
    category: ServiceCategory | None = None
    description: str | None = Field(default=None, max_length=3000)
    service_details: str | None = Field(default=None, max_length=3000)
    location: str | None = Field(default=None, max_length=150)
    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)
    price: str | None = Field(default=None, max_length=80)
    available: bool | None = None
    phone: str | None = Field(default=None, max_length=30)

    @model_validator(mode="after")
    def validate_coordinates(self):
        if (self.latitude is None) != (self.longitude is None):
            raise ValueError("Latitude and longitude must be updated together.")
        return self
