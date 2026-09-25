import re

import pandas as pd

CATEGORY_KEYWORDS = {
    "Plumber": ["leaking tap", "pipe", "water problem", "plumb", "leak"],
    "Electrician": ["wiring", "electrical", "electrician", "fan", "switch"],
    "Cleaner": ["house cleaning", "cleaning", "cleaner", "deep clean"],
    "Carpenter": ["furniture", "wood", "cupboard", "carpenter", "cabinet"],
    "Food Vendor": ["food", "meal", "lunch", "dinner", "catering"],
    "Appliance Repair": ["washing machine", "refrigerator", "appliance", "fridge", "repair"],
}


def detect_category(query: str) -> str | None:
    normalized = re.sub(r"[^a-z0-9 ]", " ", query.lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    matches = {
        category: sum(1 for keyword in keywords if keyword in normalized)
        for category, keywords in CATEGORY_KEYWORDS.items()
    }
    best_category, best_matches = max(matches.items(), key=lambda item: item[1])
    return best_category if best_matches else None


def rank_vendors(vendors, query: str) -> list[dict]:
    category = detect_category(query)
    rows = [
        {
            "vendor": vendor,
            "category_match": 1.0 if category and vendor.category == category else 0.0,
            "availability": 1.0 if vendor.available else 0.0,
            "rating_score": min(max(vendor.rating, 0.0), 5.0) / 5.0,
            "distance_score": 1.0 / (1.0 + max(vendor.distance, 0.0) / 5.0),
            "review_score": min(max(vendor.review_count, 0), 100) / 100.0,
        }
        for vendor in vendors
    ]
    if not rows:
        return []

    frame = pd.DataFrame(rows)
    frame["recommendation_score"] = (
        frame["category_match"] * 55
        + frame["availability"] * 15
        + frame["rating_score"] * 15
        + frame["distance_score"] * 10
        + frame["review_score"] * 5
    ).round(2)
    frame = frame.sort_values(
        by=["recommendation_score", "category_match", "rating_score"],
        ascending=False,
    )

    return [
        {
            **row["vendor"].__dict__,
            "recommendation_score": float(row["recommendation_score"]),
            "matched_category": category,
        }
        for _, row in frame.iterrows()
    ]
