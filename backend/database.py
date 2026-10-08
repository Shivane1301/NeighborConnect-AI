from pathlib import Path

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = f"sqlite:///{Path(__file__).resolve().parent / 'neighborconnect.db'}"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def upgrade_schema(target_engine=engine) -> None:
    """Apply additive SQLite changes while preserving existing application data."""
    from services.auth import hash_password
    from services.location import coordinates_for_location
    from seed import SEED_VENDORS

    Base.metadata.create_all(bind=target_engine)
    additions = {
        "users": {
            "password_hash": "VARCHAR(255)",
            "phone": "VARCHAR(30)",
            "location": "VARCHAR(150)",
            "latitude": "FLOAT",
            "longitude": "FLOAT",
            "preferences": "JSON NOT NULL DEFAULT '{}'",
        },
        "vendors": {
            "latitude": "FLOAT",
            "longitude": "FLOAT",
            "service_details": "TEXT NOT NULL DEFAULT ''",
            "owner_id": "INTEGER REFERENCES users(id)",
        },
        "bookings": {"message": "TEXT NOT NULL DEFAULT ''"},
        "reviews": {"booking_id": "INTEGER REFERENCES bookings(id)"},
    }

    with target_engine.begin() as connection:
        inspector = inspect(connection)
        for table, columns in additions.items():
            existing = {column["name"] for column in inspector.get_columns(table)}
            for column, definition in columns.items():
                if column not in existing:
                    connection.exec_driver_sql(
                        f"ALTER TABLE {table} ADD COLUMN {column} {definition}"
                    )

        users = connection.execute(
            text("SELECT id, password, password_hash FROM users WHERE password_hash IS NULL")
        ).mappings()
        for user in users:
            legacy_password = user["password"] or ""
            encoded = (
                legacy_password
                if legacy_password.startswith("pbkdf2_sha256$")
                else hash_password(legacy_password)
            )
            connection.execute(
                text("UPDATE users SET password_hash = :password_hash, password = :password WHERE id = :id"),
                {"password_hash": encoded, "password": encoded, "id": user["id"]},
            )

        connection.execute(text("UPDATE users SET role = 'customer' WHERE role = 'user' OR role IS NULL"))
        seeded_names = {vendor["name"] for vendor in SEED_VENDORS}
        city_coordinates = coordinates_for_location("Puducherry")
        vendors = connection.execute(
            text("SELECT id, name, location, latitude, longitude, owner_id FROM vendors")
        ).mappings()
        for vendor in vendors:
            coordinates = coordinates_for_location(vendor["location"] or "")
            existing_coordinates = (vendor["latitude"], vendor["longitude"])
            missing_coordinates = None in existing_coordinates
            needs_seed_correction = (
                vendor["owner_id"] is None
                and vendor["name"] in seeded_names
                and existing_coordinates == city_coordinates
                and coordinates != city_coordinates
            )
            if coordinates and (missing_coordinates or needs_seed_correction):
                connection.execute(
                    text("UPDATE vendors SET latitude = :latitude, longitude = :longitude WHERE id = :id"),
                    {"latitude": coordinates[0], "longitude": coordinates[1], "id": vendor["id"]},
                )
        connection.exec_driver_sql(
            "CREATE UNIQUE INDEX IF NOT EXISTS ux_reviews_booking_id "
            "ON reviews (booking_id) WHERE booking_id IS NOT NULL"
        )
        connection.exec_driver_sql(
            "CREATE UNIQUE INDEX IF NOT EXISTS ux_vendors_owner_id "
            "ON vendors (owner_id) WHERE owner_id IS NOT NULL"
        )
