from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from database import Base, SessionLocal, engine, get_db
from models import Booking, Review, User, Vendor
from schemas import (
    BookingCreate,
    BookingResponse,
    BookingUpdate,
    HealthResponse,
    LoginRequest,
    ReviewCreate,
    ReviewResponse,
    SearchRequest,
    SearchResult,
    SentimentRequest,
    SentimentResponse,
    UserCreate,
    UserResponse,
    VendorResponse,
)
from seed import seed_vendors
from services.recommendation import rank_vendors
from services.sentiment import analyze_sentiment, load_sentiment_model


@asynccontextmanager
async def lifespan(_app: FastAPI):
    Base.metadata.create_all(bind=engine)
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
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return {"status": "ok", "message": "NeighborConnect AI backend is running"}


@app.post("/api/auth/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def register(payload: UserCreate, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == payload.email).first():
        raise HTTPException(status_code=409, detail="An account with this email already exists")
    user = User(**payload.model_dump())
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@app.post("/api/auth/login", response_model=UserResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == payload.email).first()
    if not user or user.password != payload.password:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    return user


@app.get("/api/vendors", response_model=list[VendorResponse])
def list_vendors(
    category: str | None = Query(default=None),
    available: bool | None = Query(default=None),
    db: Session = Depends(get_db),
):
    query = db.query(Vendor)
    if category:
        query = query.filter(Vendor.category.ilike(category))
    if available is not None:
        query = query.filter(Vendor.available == available)
    return query.order_by(Vendor.rating.desc()).all()


@app.get("/api/vendors/{vendor_id}", response_model=VendorResponse)
def get_vendor(vendor_id: int, db: Session = Depends(get_db)):
    vendor = db.get(Vendor, vendor_id)
    if not vendor:
        raise HTTPException(status_code=404, detail="Vendor not found")
    return vendor


@app.post("/api/search", response_model=list[SearchResult])
def search(payload: SearchRequest, db: Session = Depends(get_db)):
    return rank_vendors(db.query(Vendor).all(), payload.query)


@app.post("/api/sentiment", response_model=SentimentResponse)
def sentiment(payload: SentimentRequest):
    try:
        return {"success": True, **analyze_sentiment(payload.review)}
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.post("/api/bookings", response_model=BookingResponse, status_code=status.HTTP_201_CREATED)
def create_booking(payload: BookingCreate, db: Session = Depends(get_db)):
    if not db.get(User, payload.user_id):
        raise HTTPException(status_code=404, detail="User not found")
    if not db.get(Vendor, payload.vendor_id):
        raise HTTPException(status_code=404, detail="Vendor not found")
    booking = Booking(**payload.model_dump())
    db.add(booking)
    db.commit()
    db.refresh(booking)
    return booking


@app.get("/api/bookings/{user_id}", response_model=list[BookingResponse])
def get_bookings(user_id: int, db: Session = Depends(get_db)):
    return db.query(Booking).filter(Booking.user_id == user_id).order_by(Booking.booking_date).all()


@app.patch("/api/bookings/{booking_id}", response_model=BookingResponse)
def update_booking(booking_id: int, payload: BookingUpdate, db: Session = Depends(get_db)):
    booking = db.get(Booking, booking_id)
    if not booking:
        raise HTTPException(status_code=404, detail="Booking not found")
    for field, value in payload.model_dump(exclude_unset=True).items():
        setattr(booking, field, value)
    db.commit()
    db.refresh(booking)
    return booking


@app.post("/api/reviews", response_model=ReviewResponse, status_code=status.HTTP_201_CREATED)
def create_review(payload: ReviewCreate, db: Session = Depends(get_db)):
    if not db.get(User, payload.user_id):
        raise HTTPException(status_code=404, detail="User not found")
    if not db.get(Vendor, payload.vendor_id):
        raise HTTPException(status_code=404, detail="Vendor not found")
    try:
        analysis = analyze_sentiment(payload.text)
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    review = Review(**payload.model_dump(), **analysis)
    db.add(review)
    db.commit()
    db.refresh(review)
    return review


@app.get("/api/reviews/{vendor_id}", response_model=list[ReviewResponse])
def get_reviews(vendor_id: int, db: Session = Depends(get_db)):
    if not db.get(Vendor, vendor_id):
        raise HTTPException(status_code=404, detail="Vendor not found")
    return db.query(Review).filter(Review.vendor_id == vendor_id).order_by(Review.id.desc()).all()
