from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(120), nullable=False)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password = Column(String(255), nullable=False)
    password_hash = Column(String(255), nullable=True)
    role = Column(String(50), nullable=False, default="customer")
    phone = Column(String(30), nullable=True)
    location = Column(String(150), nullable=True)
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    preferences = Column(JSON, nullable=False, default=dict)

    bookings = relationship("Booking", back_populates="user")
    reviews = relationship("Review", back_populates="user")
    vendor = relationship("Vendor", back_populates="owner", uselist=False)


class Vendor(Base):
    __tablename__ = "vendors"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(150), nullable=False)
    category = Column(String(100), nullable=False, index=True)
    description = Column(Text, nullable=False)
    location = Column(String(150), nullable=False)
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    distance = Column(Float, nullable=True, default=None)
    rating = Column(Float, nullable=False, default=0)
    review_count = Column(Integer, nullable=False, default=0)
    price = Column(String(80), nullable=False)
    available = Column(Boolean, nullable=False, default=True)
    phone = Column(String(30), nullable=False)
    service_details = Column(Text, nullable=False, default="")
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=True, unique=True)

    bookings = relationship("Booking", back_populates="vendor")
    reviews = relationship("Review", back_populates="vendor")
    owner = relationship("User", back_populates="vendor")


class Review(Base):
    __tablename__ = "reviews"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    vendor_id = Column(Integer, ForeignKey("vendors.id"), nullable=False)
    booking_id = Column(Integer, ForeignKey("bookings.id"), nullable=True, unique=True)
    text = Column(Text, nullable=False)
    rating = Column(Integer, nullable=False)
    sentiment = Column(String(20), nullable=False)
    confidence = Column(Float, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", back_populates="reviews")
    vendor = relationship("Vendor", back_populates="reviews")
    booking = relationship("Booking", back_populates="review")


class Booking(Base):
    __tablename__ = "bookings"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    vendor_id = Column(Integer, ForeignKey("vendors.id"), nullable=False)
    service = Column(String(150), nullable=False)
    booking_date = Column(DateTime, nullable=False)
    message = Column(Text, nullable=False, default="")
    status = Column(String(30), nullable=False, default="pending")
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", back_populates="bookings")
    vendor = relationship("Vendor", back_populates="bookings")
    review = relationship("Review", back_populates="booking", uselist=False)
