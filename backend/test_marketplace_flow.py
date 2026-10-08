import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from database import Base, get_db
from main import app
from seed import seed_vendors
from services.location import coordinates_for_location
from services.sentiment import load_sentiment_model


class MarketplaceFlowTests(unittest.TestCase):
    def setUp(self):
        self.temp_directory = tempfile.TemporaryDirectory()
        database_path = Path(self.temp_directory.name) / "marketplace.db"
        self.engine = create_engine(
            f"sqlite:///{database_path.as_posix()}",
            connect_args={"check_same_thread": False},
        )
        Base.metadata.create_all(bind=self.engine)
        with self.engine.begin() as connection:
            connection.exec_driver_sql(
                "CREATE TRIGGER enforce_legacy_vendor_distance "
                "BEFORE INSERT ON vendors WHEN NEW.distance IS NULL "
                "BEGIN SELECT RAISE(ABORT, 'legacy vendors.distance is NOT NULL'); END"
            )
        self.session_factory = sessionmaker(bind=self.engine)

        def override_get_db():
            session = self.session_factory()
            try:
                yield session
            finally:
                session.close()

        app.dependency_overrides[get_db] = override_get_db
        session = self.session_factory()
        try:
            seed_vendors(session)
        finally:
            session.close()
        self.client = TestClient(app)
        load_sentiment_model()
        self.customer = self.register_customer()
        self.customer_headers = self.headers(self.customer)

    def tearDown(self):
        app.dependency_overrides.clear()
        self.engine.dispose()
        self.temp_directory.cleanup()

    @staticmethod
    def headers(user):
        return {"Authorization": f"Bearer {user['access_token']}"}

    def register_customer(self):
        response = self.client.post(
            "/api/auth/register",
            json={
                "name": "Demo Customer",
                "email": "demo.customer@example.test",
                "password": "PondyDemo123!",
                "role": "customer",
                "phone": "+91 90000 00001",
                "location": "Puducherry",
            },
        )
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()

    def register_vendor(self, **overrides):
        values = {
            "name": "Green Home Cleaning",
            "email": "green.cleaning@example.test",
            "password": "PondyVendor123!",
            "role": "vendor",
            "phone": "+91 90000 00002",
            "category": "Cleaner",
            "description": "Home cleaning for local residents.",
            "service_details": "Deep clean and regular maintenance.",
            "location": "White Town, Puducherry",
            "price": "Rs. 600-1500",
            "available": True,
        }
        values.update(overrides)
        response = self.client.post("/api/auth/register", json=values)
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()

    def test_registration_login_and_new_vendor_search(self):
        vendor = self.register_vendor()
        self.assertEqual(vendor["role"], "vendor")
        self.assertIsNotNone(vendor["vendor_id"])
        session = self.session_factory()
        try:
            from models import Vendor

            saved_vendor = session.get(Vendor, vendor["vendor_id"])
            self.assertGreater(saved_vendor.distance, 0)
        finally:
            session.close()

        login = self.client.post(
            "/api/auth/login",
            json={"email": vendor["email"], "password": "PondyVendor123!"},
        )
        self.assertEqual(login.status_code, 200, login.text)
        self.assertEqual(login.json()["role"], "vendor")
        self.assertEqual(login.json()["vendor_id"], vendor["vendor_id"])
        customer_login = self.client.post(
            "/api/auth/login",
            json={"email": self.customer["email"], "password": "PondyDemo123!"},
        )
        self.assertEqual(customer_login.status_code, 200, customer_login.text)
        self.assertEqual(customer_login.json()["role"], "customer")
        duplicate = self.client.post(
            "/api/auth/register",
            json={
                "name": "Duplicate Customer",
                "email": self.customer["email"].upper(),
                "password": "AnotherDemo123!",
                "role": "customer",
                "phone": "+91 90000 00003",
                "location": "Puducherry",
            },
        )
        self.assertEqual(duplicate.status_code, 409)
        self.assertTrue(self.client.get("/api/vendors").json())

        results = self.client.post(
            "/api/search", json={"query": "I need someone to clean my house"}
        )
        self.assertEqual(results.status_code, 200, results.text)
        self.assertTrue(results.json())
        self.assertEqual({item["category"] for item in results.json()}, {"Cleaner"})
        self.assertIn(vendor["name"], {item["name"] for item in results.json()})

        electrician = self.client.post(
            "/api/search", json={"query": "Fix my ceiling fan"}
        )
        self.assertEqual({item["category"] for item in electrician.json()}, {"Electrician"})
        plumber = self.client.post(
            "/api/search", json={"query": "My kitchen tap is leaking"}
        )
        self.assertEqual({item["category"] for item in plumber.json()}, {"Plumber"})

    def test_local_area_coordinates_override_city_fallback(self):
        self.assertEqual(
            coordinates_for_location("Muthialpet, Puducherry"), (11.9482, 79.8357)
        )
        self.assertEqual(coordinates_for_location("Puducherry"), (11.9416, 79.8083))

    def test_preferences_distance_and_ambiguous_search(self):
        self.register_vendor()
        available = self.client.post(
            "/api/search",
            json={"query": "clean my house", "available_only": True},
        )
        self.assertEqual(available.status_code, 200, available.text)
        self.assertTrue(all(item["available"] for item in available.json()))
        self.assertTrue(all(item["category"] == "Cleaner" for item in available.json()))

        close_results = self.client.post(
            "/api/search",
            json={
                "query": "clean my house",
                "latitude": 11.9343,
                "longitude": 79.8344,
                "nearby": True,
            },
        )
        self.assertEqual(close_results.status_code, 200, close_results.text)
        self.assertTrue(all(item["distance_is_estimate"] is False for item in close_results.json()))
        self.assertTrue(all(item["distance"] <= 5 for item in close_results.json()))

        unknown_area = self.client.post(
            "/api/search",
            json={"query": "clean my house", "location": "Unmapped City"},
        )
        self.assertTrue(all(item["distance"] is None for item in unknown_area.json()))

        ambiguous = self.client.post("/api/search", json={"query": "I need help"})
        self.assertEqual(ambiguous.status_code, 422)
        self.assertIn("specify the service", ambiguous.json()["detail"])

    def test_booking_completion_review_and_sentiment(self):
        vendor = self.register_vendor()
        vendor_headers = self.headers(vendor)
        vendor_id = vendor["vendor_id"]
        coords = coordinates_for_location("White Town, Puducherry")
        self.assertIsNotNone(coords)

        profile = self.client.get(f"/api/vendors/{vendor_id}")
        self.assertEqual(profile.status_code, 200, profile.text)
        self.assertEqual(profile.json()["category"], "Cleaner")
        self.assertEqual(profile.json()["rating"], 0)
        self.assertEqual(profile.json()["review_count"], 0)

        booking = self.client.post(
            "/api/bookings",
            headers=self.customer_headers,
            json={
                "vendor_id": vendor_id,
                "service": "House cleaning",
                "booking_date": "2026-10-08T10:00:00",
                "message": "Please focus on the kitchen.",
            },
        )
        self.assertEqual(booking.status_code, 201, booking.text)
        booking_id = booking.json()["id"]
        unauthorized = self.client.patch(
            f"/api/bookings/{booking_id}", json={"status": "accepted"}
        )
        self.assertEqual(unauthorized.status_code, 401)
        skipped_status = self.client.patch(
            f"/api/bookings/{booking_id}",
            headers=vendor_headers,
            json={"status": "completed"},
        )
        self.assertEqual(skipped_status.status_code, 400)

        dashboard = self.client.get("/api/vendor/dashboard", headers=vendor_headers)
        self.assertEqual(dashboard.status_code, 200, dashboard.text)
        self.assertEqual(dashboard.json()["requests"][0]["id"], booking_id)

        accepted = self.client.patch(
            f"/api/bookings/{booking_id}",
            headers=vendor_headers,
            json={"status": "accepted"},
        )
        self.assertEqual(accepted.status_code, 200, accepted.text)
        completed = self.client.patch(
            f"/api/bookings/{booking_id}",
            headers=vendor_headers,
            json={"status": "completed"},
        )
        self.assertEqual(completed.status_code, 200, completed.text)

        review = self.client.post(
            "/api/reviews",
            headers=self.customer_headers,
            json={
                "vendor_id": vendor_id + 999,
                "booking_id": booking_id,
                "rating": 5,
                "review": "Excellent service. Very professional and friendly.",
            },
        )
        self.assertEqual(review.status_code, 201, review.text)
        self.assertTrue(review.json()["success"])
        self.assertEqual(review.json()["review"]["vendor_id"], vendor_id)
        self.assertEqual(review.json()["review"]["booking_id"], booking_id)
        self.assertEqual(review.json()["review"]["rating"], 5)
        self.assertEqual(review.json()["review"]["sentiment"], "positive")
        self.assertIsInstance(review.json()["review"]["confidence"], float)
        self.assertGreater(review.json()["review"]["confidence"], 0)
        self.assertEqual(review.json()["updated_vendor"]["rating"], 5.0)
        self.assertEqual(review.json()["updated_vendor"]["review_count"], 1)

        duplicate = self.client.post(
            "/api/reviews",
            headers=self.customer_headers,
            json={
                "vendor_id": vendor_id,
                "booking_id": booking_id,
                "rating": 5,
                "review": "Duplicate review attempt.",
            },
        )
        self.assertEqual(duplicate.status_code, 409)

        second_booking = self.client.post(
            "/api/bookings",
            headers=self.customer_headers,
            json={
                "vendor_id": vendor_id,
                "service": "House cleaning",
                "booking_date": "2026-10-09T10:00:00",
            },
        )
        self.assertEqual(second_booking.status_code, 201, second_booking.text)
        second_booking_id = second_booking.json()["id"]
        for status_value in ("accepted", "completed"):
            transition = self.client.patch(
                f"/api/bookings/{second_booking_id}",
                headers=vendor_headers,
                json={"status": status_value},
            )
            self.assertEqual(transition.status_code, 200, transition.text)
        second_review = self.client.post(
            "/api/reviews",
            headers=self.customer_headers,
            json={
                "vendor_id": vendor_id + 99,
                "booking_id": second_booking_id,
                "rating": 4,
                "review": "Good service and the work was completed properly.",
            },
        )
        self.assertEqual(second_review.status_code, 201, second_review.text)
        self.assertEqual(second_review.json()["sentiment"], "positive")
        self.assertIsInstance(second_review.json()["confidence"], float)
        self.assertEqual(second_review.json()["updated_vendor"]["rating"], 4.5)
        self.assertEqual(second_review.json()["updated_vendor"]["review_count"], 2)

        listed_reviews = self.client.get(f"/api/reviews/{vendor_id}")
        self.assertEqual(len(listed_reviews.json()), 2)
        self.assertTrue(all(item["created_at"] for item in listed_reviews.json()))
        updated_profile = self.client.get(f"/api/vendors/{vendor_id}")
        self.assertEqual(updated_profile.json()["review_count"], 2)
        self.assertEqual(updated_profile.json()["rating"], 4.5)

        vendor_update = self.client.patch(
            "/api/vendors/me",
            headers=vendor_headers,
            json={"available": False, "price": "Rs. 700-1600"},
        )
        self.assertEqual(vendor_update.status_code, 200, vendor_update.text)
        self.assertFalse(vendor_update.json()["available"])
        self.assertEqual(vendor_update.json()["price"], "Rs. 700-1600")
        available_cleaners = self.client.get("/api/vendors?category=Cleaner&available=true")
        self.assertNotIn(vendor["name"], {item["name"] for item in available_cleaners.json()})

        customer_bookings = self.client.get(
            f"/api/bookings/{self.customer['id']}", headers=self.customer_headers
        )
        self.assertTrue(all(booking["reviewed"] for booking in customer_bookings.json()))

        sentiment = self.client.post(
            "/api/sentiment",
            json={"review": "the plumber was excellent and fixed everything quickly"},
        )
        self.assertEqual(sentiment.status_code, 200, sentiment.text)
        self.assertEqual(sentiment.json()["sentiment"], "positive")
        self.assertIsInstance(sentiment.json()["confidence"], float)


if __name__ == "__main__":
    unittest.main()
