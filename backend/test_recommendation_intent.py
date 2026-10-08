import unittest
from types import SimpleNamespace
from unittest.mock import patch

import services.recommendation as recommendation
from services.recommendation import (
    detect_category,
    generate_recommendation_reasons,
    generate_recommendation_summary,
    rank_vendors,
)


INTENT_CASES = (
    ("my kitchen tap is leaking", "Plumber"),
    ("leaking faucet under the window", "Plumber"),
    ("broken tap in the utility room", "Plumber"),
    ("there is a water leak behind the sink", "Plumber"),
    ("pipe leak in the bathroom", "Plumber"),
    ("could someone clear this clogged drain", "Plumber"),
    ("blocked sink after washing dishes", "Plumber"),
    ("need a plumber for low water pressure", "Plumber"),
    ("faucet repair please", "Plumber"),
    ("water pipe is loose", "Plumber"),
    ("toilet cistern keeps running", "Plumber"),
    ("plumbing issue at my flat", "Plumber"),
    ("fan not working", "Electrician"),
    ("fix my ceiling fan", "Electrician"),
    ("ceiling fan problem in the bedroom", "Electrician"),
    ("wiring issue near the doorway", "Electrician"),
    ("electrical problem at home", "Electrician"),
    ("switch not working", "Electrician"),
    ("socket problem by the desk", "Electrician"),
    ("power issue in the kitchen", "Electrician"),
    ("light not working in the hall", "Electrician"),
    ("electrician needed this afternoon", "Electrician"),
    ("short circuit in the living room", "Electrician"),
    ("breaker keeps tripping", "Electrician"),
    ("need someone to clean my house", "Cleaner"),
    ("home cleaning before the weekend", "Cleaner"),
    ("room cleaning after painting", "Cleaner"),
    ("deep cleaning for the kitchen", "Cleaner"),
    ("bathroom cleaning service", "Cleaner"),
    ("cleaner for a two bedroom flat", "Cleaner"),
    ("please arrange house cleaning", "Cleaner"),
    ("cleaning service near me", "Cleaner"),
    ("cupboard repair after the hinge came loose", "Carpenter"),
    ("repair my cupboard door", "Carpenter"),
    ("cabinet repair in the kitchen", "Carpenter"),
    ("wooden door repair", "Carpenter"),
    ("furniture repair for an old chair", "Carpenter"),
    ("table repair", "Carpenter"),
    ("wood work for new shelves", "Carpenter"),
    ("carpenter to fix a drawer", "Carpenter"),
    ("need lunch for 10 people", "Food Vendor"),
    ("dinner for the family tonight", "Food Vendor"),
    ("breakfast delivery for six", "Food Vendor"),
    ("food delivery for our meeting", "Food Vendor"),
    ("local food for visiting relatives", "Food Vendor"),
    ("snacks for a neighborhood event", "Food Vendor"),
    ("restaurant food for a small party", "Food Vendor"),
    ("catering for the office", "Food Vendor"),
    ("washing machine is not working", "Appliance Repair"),
    ("washing machine repair", "Appliance Repair"),
    ("refrigerator repair after it stopped cooling", "Appliance Repair"),
    ("fridge repair please", "Appliance Repair"),
    ("AC repair for the bedroom", "Appliance Repair"),
    ("microwave repair", "Appliance Repair"),
    ("mixer repair needed", "Appliance Repair"),
    ("TV repair for a blank screen", "Appliance Repair"),
    ("appliance not working", "Appliance Repair"),
    ("aircon repair", "Appliance Repair"),
    ("fridge won't cool", "Appliance Repair"),
)

AMBIGUOUS_CASES = (
    "need repair",
    "help with a problem",
    "I need a service",
    "please fix this",
    "something is wrong",
    "the issue is getting worse",
    "need help at home",
    "repair service please",
    "can someone take a look",
    "I have a problem with something",
)


class RecommendationIntentTests(unittest.TestCase):
    def test_supported_natural_language_intents(self):
        for query, expected in INTENT_CASES:
            with self.subTest(query=query):
                self.assertEqual(detect_category(query), expected)

    def test_generic_or_ambiguous_terms_do_not_select_a_category(self):
        for query in AMBIGUOUS_CASES:
            with self.subTest(query=query):
                self.assertIsNone(detect_category(query))

    def test_ties_use_deterministic_category_priority(self):
        tied_keywords = {
            "Plumber": {"phrases": {"shared fixture request": 5}, "terms": {}},
            "Electrician": {"phrases": {"shared fixture request": 5}, "terms": {}},
        }
        with patch.dict(recommendation.CATEGORY_KEYWORDS, tied_keywords, clear=True):
            self.assertEqual(detect_category("shared fixture request"), "Electrician")
            self.assertEqual(detect_category("shared fixture request"), "Electrician")

    def test_weighted_ranking_formula_is_unchanged(self):
        vendor = SimpleNamespace(
            id=999,
            name="Test Plumbing",
            category="Plumber",
            description="Fixture for score regression test",
            location="Test Area",
            distance=5.0,
            rating=4.0,
            review_count=50,
            price="Test",
            available=True,
            phone="000",
        )
        ranked = rank_vendors([vendor], "leaking tap")
        expected_score = 55 + 15 + 12 + 5 + 2.5
        self.assertEqual(ranked[0]["recommendation_score"], expected_score)
        self.assertEqual(ranked[0]["matched_category"], "Plumber")

    def test_category_match_is_a_hard_filter(self):
        vendors = [make_vendor(category="Cleaner"), make_vendor(category="Plumber")]

        ranked = rank_vendors(vendors, "I need someone to clean my house")

        self.assertEqual([vendor["category"] for vendor in ranked], ["Cleaner"])
        self.assertEqual(rank_vendors(vendors, "I need help"), [])


def make_vendor(**overrides):
    values = {
        "category": "Cleaner",
        "available": True,
        "rating": 3.8,
        "distance": 8.0,
        "review_count": 5,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class RecommendationReasonTests(unittest.TestCase):
    def test_exact_service_match(self):
        reasons = generate_recommendation_reasons(make_vendor(category="Plumber"), "Plumber")
        self.assertEqual(reasons[0], "Exact service match")

    def test_available_vendor(self):
        reasons = generate_recommendation_reasons(make_vendor(available=True), None)
        self.assertIn("Currently available", reasons)

    def test_unavailable_vendor(self):
        reasons = generate_recommendation_reasons(make_vendor(available=False), None)
        self.assertIn("Currently unavailable", reasons)

    def test_highly_rated_vendor(self):
        reasons = generate_recommendation_reasons(make_vendor(rating=4.5), None)
        self.assertIn("Highly rated", reasons)
        self.assertNotIn("Good customer rating", reasons)

    def test_medium_rated_vendor(self):
        reasons = generate_recommendation_reasons(make_vendor(rating=4.2), None)
        self.assertIn("Good customer rating", reasons)
        self.assertNotIn("Highly rated", reasons)
        self.assertEqual(
            generate_recommendation_summary(reasons),
            "Vendor is currently available, well rated, and farther away.",
        )

    def test_multiple_reviews_summary_is_grammatical(self):
        reasons = generate_recommendation_reasons(
            make_vendor(available=False, rating=3.8, review_count=25), None
        )
        self.assertIn("Multiple customer reviews", reasons)
        self.assertEqual(
            generate_recommendation_summary(reasons),
            "Vendor is not currently available, farther away, and supported by multiple customer reviews.",
        )

    def test_very_close_vendor(self):
        reasons = generate_recommendation_reasons(make_vendor(distance=2.0), None)
        self.assertIn("Very close", reasons)
        self.assertNotIn("Nearby", reasons)

    def test_nearby_vendor(self):
        reasons = generate_recommendation_reasons(make_vendor(distance=4.0), None)
        self.assertIn("Nearby", reasons)
        self.assertNotIn("Very close", reasons)

    def test_well_reviewed_vendor(self):
        reasons = generate_recommendation_reasons(make_vendor(review_count=50), None)
        self.assertIn("Well reviewed", reasons)
        self.assertNotIn("Multiple customer reviews", reasons)

    def test_multiple_reasons_and_summary(self):
        vendor = make_vendor(
            category="Plumber",
            available=True,
            rating=4.8,
            distance=1.5,
            review_count=80,
        )
        reasons = generate_recommendation_reasons(vendor, "Plumber")
        self.assertEqual(
            reasons,
            [
                "Exact service match",
                "Currently available",
                "Highly rated",
                "Very close",
                "Well reviewed",
            ],
        )
        self.assertEqual(
            generate_recommendation_summary(reasons),
            "Vendor is an exact service match, currently available, highly rated, very close, and well reviewed.",
        )

    def test_vendor_with_no_strong_reason(self):
        reasons = generate_recommendation_reasons(
            make_vendor(available=False, rating=3.5, distance=8.0, review_count=4),
            None,
        )
        self.assertEqual(reasons, ["Currently unavailable", "Farther away"])
        self.assertEqual(
            generate_recommendation_summary(reasons),
            "Vendor is not currently available and farther away.",
        )


if __name__ == "__main__":
    unittest.main()