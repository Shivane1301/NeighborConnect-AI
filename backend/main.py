from contextlib import asynccontextmanager
from fastapi import Depends, FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from database import SessionLocal, engine, get_db, upgrade_schema
from models import Booking, Review, User, Vendor
from schemas import (
    BookingCreate,
    BookingResponse,
    BookingUpdate,
    HealthResponse,
    LoginRequest,
    ReviewCreate,
    ReviewResponse,
    ReviewSubmissionResponse,
    SearchRequest,
    SearchResult,
    SentimentRequest,
    SentimentResponse,
    UserCreate,
    UserResponse,
    VendorUpdate,
    VendorResponse,
)
from seed import seed_vendors
from services.auth import create_access_token, get_current_user, hash_password, verify_password
from services.location import coordinates_for_location, distance_in_km, resolve_coordinates
from services.recommendation import detect_category, rank_vendors
from services.sentiment import analyze_sentiment, load_sentiment_model


@asynccontextmanager
async def lifespan(_app: FastAPI):
    upgrade_schema(engine)
    load_sentiment_model()
    db = SessionLocal()
    try:
        seed_vendors(db)
    finally:
        db.close()
    yield


app = FastAPI(title="NeighborConnect AI API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:5175",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
        "http://127.0.0.1:5175",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def user_response(user: User) -> dict:
    return {
        "id": user.id,
        "name": user.name,
        "email": user.email,
        "role": user.role,
        "phone": user.phone,
        "location": user.location,
        "preferences": user.preferences or {},
        "vendor_id": user.vendor.id if user.vendor else None,
        "access_token": create_access_token(user.id),
    }


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return {"status": "ok", "message": "NeighborConnect AI backend is running"}


@app.post("/api/auth/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def register(payload: UserCreate, db: Session = Depends(get_db)):
    normalized_email = payload.email.strip().casefold()
    if db.query(User).filter(User.email == normalized_email).first():
        raise HTTPException(status_code=409, detail="An account with this email already exists")
    try:
        latitude, longitude = resolve_coordinates(
            payload.location, payload.latitude, payload.longitude
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    encoded_password = hash_password(payload.password)
    user = User(
        name=payload.name.strip(),
        email=normalized_email,
        password=encoded_password,
        password_hash=encoded_password,
        role=payload.role,
        phone=payload.phone,
        location=payload.location,
        latitude=latitude,
        longitude=longitude,
        preferences=payload.preferences,
    )
    db.add(user)
    db.flush()
    if payload.role == "vendor":
        city_center = coordinates_for_location("Puducherry")
        estimated_distance = 0.0
        if city_center and latitude is not None and longitude is not None:
            estimated_distance = round(
                distance_in_km(city_center[0], city_center[1], latitude, longitude), 2
            )
        user.vendor = Vendor(
            name=payload.name.strip(),
            category=payload.category,
            description=payload.description.strip(),
            service_details=(payload.service_details or "").strip(),
            location=payload.location.strip(),
            latitude=latitude,
            longitude=longitude,
            distance=estimated_distance,
            rating=0,
            review_count=0,
            price=payload.price.strip(),
            available=payload.available,
            phone=payload.phone.strip(),
            owner_id=user.id,
        )
    db.commit()
    db.refresh(user)
    return user_response(user)


@app.post("/api/auth/login", response_model=UserResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == payload.email.strip().casefold()).first()
    if not user or not verify_password(payload.password, user.password_hash or user.password):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    if not user.password_hash:
        user.password_hash = hash_password(payload.password)
        user.password = user.password_hash
        db.commit()
        db.refresh(user)
    return user_response(user)


@app.get("/api/vendors", response_model=list[VendorResponse])
def list_vendors(
    category: str | None = Query(default=None),
    available: bool | None = Query(default=None),
    latitude: float | None = Query(default=None, ge=-90, le=90),
    longitude: float | None = Query(default=None, ge=-180, le=180),
    location: str | None = Query(default=None),
    db: Session = Depends(get_db),
):
    query = db.query(Vendor)
    if category:
        query = query.filter(Vendor.category.ilike(category))
    if available is not None:
        query = query.filter(Vendor.available == available)
    vendors = query.order_by(Vendor.rating.desc()).all()
    try:
        customer_latitude, customer_longitude = resolve_coordinates(location, latitude, longitude)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    if customer_latitude is None:
        if location:
            return [
                {
                    **VendorResponse.model_validate(vendor).model_dump(),
                    "distance": None,
                    "distance_is_estimate": True,
                }
                for vendor in vendors
            ]
        return vendors
    results = []
    for vendor in vendors:
        result = VendorResponse.model_validate(vendor).model_dump()
        if vendor.latitude is not None and vendor.longitude is not None:
            from services.location import distance_in_km

            result["distance"] = round(
                distance_in_km(customer_latitude, customer_longitude, vendor.latitude, vendor.longitude),
                2,
            )
            result["distance_is_estimate"] = latitude is None
        else:
            result["distance"] = None
            result["distance_is_estimate"] = True
        results.append(result)
    return results


@app.patch("/api/vendors/me", response_model=VendorResponse)
def update_my_vendor(
    payload: VendorUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if current_user.role != "vendor" or not current_user.vendor:
        raise HTTPException(status_code=403, detail="Only vendor accounts can update a business profile.")
    updates = payload.model_dump(exclude_unset=True)
    vendor = current_user.vendor
    if "location" in updates or "latitude" in updates or "longitude" in updates:
        location = updates.get("location", vendor.location)
        latitude = updates.get("latitude")
        longitude = updates.get("longitude")
        try:
            latitude, longitude = resolve_coordinates(location, latitude, longitude)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        updates["latitude"] = latitude
        updates["longitude"] = longitude
    for field, value in updates.items():
        if isinstance(value, str):
            value = value.strip()
        setattr(vendor, field, value)
    db.commit()
    db.refresh(vendor)
    return vendor


@app.get("/api/vendors/{vendor_id}", response_model=VendorResponse)
def get_vendor(vendor_id: int, db: Session = Depends(get_db)):
    vendor = db.get(Vendor, vendor_id)
    if not vendor:
        raise HTTPException(status_code=404, detail="Vendor not found")
    return vendor


@app.get("/api/vendor/dashboard")
def get_vendor_dashboard(
    db: Session = Depends(get_db), current_user: User = Depends(get_current_user)
):
    if current_user.role != "vendor" or not current_user.vendor:
        raise HTTPException(status_code=403, detail="A vendor account is required.")
    requests = (
        db.query(Booking)
        .filter(Booking.vendor_id == current_user.vendor.id)
        .order_by(Booking.created_at.desc())
        .all()
    )
    return {
        "vendor": VendorResponse.model_validate(current_user.vendor).model_dump(),
        "requests": [
            {
                **BookingResponse.model_validate(item).model_dump(),
                "reviewed": item.review is not None,
                "customer_name": item.user.name,
            }
            for item in requests
        ],
    }


@app.post("/api/search", response_model=list[SearchResult])
def search(payload: SearchRequest, db: Session = Depends(get_db)):
    if detect_category(payload.query) is None:
        raise HTTPException(
            status_code=422,
            detail="Please specify the service you need, such as cleaning, plumbing, or electrical work.",
        )
    try:
        latitude, longitude = resolve_coordinates(
            payload.location, payload.latitude, payload.longitude
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return rank_vendors(
        db.query(Vendor).all(),
        payload.query,
        customer_latitude=latitude,
        customer_longitude=longitude,
        customer_location_is_estimate=payload.latitude is None,
        customer_location_provided=bool(payload.location),
        available_only=payload.available_only,
        nearby=payload.nearby,
        highly_rated=payload.highly_rated,
        max_price=payload.max_price,
    )


@app.post("/api/sentiment", response_model=SentimentResponse)
def sentiment(payload: SentimentRequest):
    try:
        return {"success": True, **analyze_sentiment(payload.review)}
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.post("/api/bookings", response_model=BookingResponse, status_code=status.HTTP_201_CREATED)
def create_booking(
    payload: BookingCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if current_user.role != "customer":
        raise HTTPException(status_code=403, detail="Only customers can request a service.")
    if payload.user_id is not None and payload.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="A service request must belong to your account.")
    vendor = db.get(Vendor, payload.vendor_id)
    if not vendor:
        raise HTTPException(status_code=404, detail="Vendor not found")
    booking = Booking(
        user_id=current_user.id,
        vendor_id=vendor.id,
        service=payload.service.strip(),
        booking_date=payload.booking_date,
        message=payload.message.strip(),
        status="pending",
    )
    db.add(booking)
    db.commit()
    db.refresh(booking)
    return {**BookingResponse.model_validate(booking).model_dump(), "vendor_name": vendor.name}


@app.get("/api/bookings/{user_id}", response_model=list[BookingResponse])
def get_bookings(
    user_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if current_user.role != "customer" or current_user.id != user_id:
        raise HTTPException(status_code=403, detail="You can only view your own service requests.")
    bookings = db.query(Booking).filter(Booking.user_id == user_id).order_by(Booking.booking_date).all()
    return [
        {
            **BookingResponse.model_validate(booking).model_dump(),
            "reviewed": booking.review is not None,
            "vendor_name": booking.vendor.name,
        }
        for booking in bookings
    ]


@app.patch("/api/bookings/{booking_id}", response_model=BookingResponse)
def update_booking(
    booking_id: int,
    payload: BookingUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    booking = db.get(Booking, booking_id)
    if not booking:
        raise HTTPException(status_code=404, detail="Booking not found")
    if current_user.role == "vendor":
        if not current_user.vendor or booking.vendor_id != current_user.vendor.id:
            raise HTTPException(status_code=403, detail="This request belongs to another business.")
        transitions = {
            "pending": {"accepted", "cancelled"},
            "accepted": {"completed", "cancelled"},
        }
        if payload.status not in transitions.get(booking.status, set()):
            raise HTTPException(status_code=400, detail="Choose an allowed request status.")
        if booking.status in {"completed", "cancelled"}:
            raise HTTPException(status_code=409, detail="This request is already closed.")
    elif current_user.role == "customer":
        if current_user.id != booking.user_id:
            raise HTTPException(status_code=403, detail="This request belongs to another customer.")
        if payload.status not in {None, "cancelled"}:
            raise HTTPException(status_code=403, detail="Customers may only cancel their own request.")
        if booking.status != "pending" and payload.status == "cancelled":
            raise HTTPException(status_code=409, detail="Only pending requests can be cancelled.")
    else:
        raise HTTPException(status_code=403, detail="This account cannot update service requests.")
    for field, value in payload.model_dump(exclude_unset=True).items():
        setattr(booking, field, value)
    db.commit()
    db.refresh(booking)
    return booking


@app.post("/api/reviews", response_model=ReviewSubmissionResponse, status_code=status.HTTP_201_CREATED)
def create_review(
    payload: ReviewCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if current_user.role != "customer":
        raise HTTPException(status_code=403, detail="Only customers can submit reviews.")
    if payload.user_id is not None and payload.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="A review must belong to your account.")

    booking = db.get(Booking, payload.booking_id)
    if not booking:
        raise HTTPException(status_code=404, detail="Booking not found")
    if booking.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="This booking belongs to another customer.")
    if booking.status != "completed":
        raise HTTPException(
            status_code=400,
            detail="Reviews can be submitted only for your completed service request.",
        )

    vendor = db.get(Vendor, booking.vendor_id)
    if not vendor:
        raise HTTPException(status_code=404, detail="Vendor not found")

    if payload.vendor_id is not None and payload.vendor_id != booking.vendor_id:
        payload.vendor_id = booking.vendor_id

    if db.query(Review).filter(Review.booking_id == booking.id).first():
        raise HTTPException(status_code=409, detail="This completed service already has a review.")

    review_text = payload.review.strip()
    if not review_text:
        raise HTTPException(status_code=422, detail="Review text is required.")
    try:
        analysis = analyze_sentiment(review_text)
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    review = Review(
        user_id=current_user.id,
        vendor_id=vendor.id,
        booking_id=booking.id,
        text=review_text,
        rating=payload.rating,
        sentiment=analysis["sentiment"],
        confidence=analysis["confidence"],
    )
    db.add(review)
    try:
        db.flush()
        all_reviews = db.query(Review).filter(Review.vendor_id == vendor.id).all()
        vendor.review_count = len(all_reviews)
        vendor.rating = round(sum(item.rating for item in all_reviews) / len(all_reviews), 2)
        db.commit()
    except IntegrityError as error:
        db.rollback()
        raise HTTPException(status_code=409, detail="This completed service already has a review.") from error
    db.refresh(review)
    db.refresh(vendor)
    review_payload = ReviewResponse.model_validate(review).model_dump()
    return {
        "success": True,
        "id": review.id,
        "user_id": review.user_id,
        "vendor_id": review.vendor_id,
        "booking_id": review.booking_id,
        "rating": review.rating,
        "review": review_payload,
        "sentiment": review.sentiment,
        "confidence": review.confidence,
        "updated_vendor": VendorResponse.model_validate(vendor).model_dump(),
    }


@app.get("/api/reviews/{vendor_id}", response_model=list[ReviewResponse])
def get_reviews(vendor_id: int, db: Session = Depends(get_db)):
    if not db.get(Vendor, vendor_id):
        raise HTTPException(status_code=404, detail="Vendor not found")
    return db.query(Review).filter(Review.vendor_id == vendor_id).order_by(Review.id.desc()).all()
