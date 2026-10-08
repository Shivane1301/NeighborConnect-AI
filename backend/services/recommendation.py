import re

import pandas as pd

from services.location import distance_in_km

CATEGORY_KEYWORDS = {
    "Plumber": {
        "phrases": {
            "leaking tap": 5,
            "tap leaking": 5,
            "tap is leaking": 5,
            "leaky tap": 5,
            "broken tap": 5,
            "tap repair": 5,
            "leaking faucet": 5,
            "faucet leaking": 5,
            "faucet repair": 5,
            "water leak": 5,
            "pipe leak": 5,
            "leaking pipe": 5,
            "water pipe": 4,
            "clogged drain": 5,
            "blocked drain": 5,
            "blocked sink": 5,
            "bathroom pipe": 4,
            "plumbing issue": 5,
            "plumbing problem": 5,
            "sink is blocked": 5,
            "drain is clogged": 5,
            "toilet is leaking": 5,
            "water pressure": 4,
        },
        "terms": {
            "plumber": 5,
            "plumbing": 5,
            "faucet": 3,
            "tap": 3,
            "pipe": 3,
            "drain": 3,
            "sink": 3,
            "cistern": 3,
        },
    },
    "Electrician": {
        "phrases": {
            "fan not working": 5,
            "ceiling fan problem": 5,
            "ceiling fan repair": 5,
            "fan repair": 5,
            "fan installation": 5,
            "wiring issue": 5,
            "electrical problem": 5,
            "switch not working": 5,
            "socket problem": 5,
            "power issue": 5,
            "light not working": 5,
            "short circuit": 5,
            "circuit breaker": 5,
            "power outage": 5,
            "faulty wiring": 5,
            "electrical fault": 5,
            "light fixture": 4,
            "broken switch": 5,
            "dead outlet": 5,
            "no electricity": 5,
        },
        "terms": {
            "electrician": 5,
            "electrical": 4,
            "wiring": 4,
            "switch": 3,
            "socket": 3,
            "outlet": 3,
            "breaker": 3,
            "fuse": 3,
            "fan": 3,
        },
    },
    "Cleaner": {
        "phrases": {
            "house cleaning": 5,
            "home cleaning": 5,
            "room cleaning": 5,
            "deep cleaning": 5,
            "deep clean": 5,
            "bathroom cleaning": 5,
            "kitchen cleaning": 5,
            "cleaning service": 5,
            "cleaning my house": 5,
            "clean my house": 5,
            "clean my home": 5,
            "move out cleaning": 5,
            "move-in cleaning": 5,
            "sofa cleaning": 5,
            "window cleaning": 5,
            "carpet cleaning": 5,
        },
        "terms": {
            "cleaner": 5,
            "cleaning": 5,
            "housekeeping": 4,
        },
    },
    "Carpenter": {
        "phrases": {
            "furniture repair": 5,
            "cupboard repair": 5,
            "cabinet repair": 5,
            "wooden door repair": 5,
            "wood door repair": 5,
            "table repair": 5,
            "chair repair": 5,
            "wood work": 5,
            "woodwork repair": 5,
            "door frame repair": 5,
            "shelf installation": 5,
            "drawer repair": 5,
            "loose hinge": 4,
            "sticking door": 4,
            "broken chair": 4,
        },
        "terms": {
            "carpenter": 5,
            "carpentry": 5,
            "furniture": 3,
            "cupboard": 3,
            "cabinet": 3,
            "woodwork": 4,
            "woodworking": 4,
        },
    },
    "Food Vendor": {
        "phrases": {
            "food delivery": 5,
            "local food": 5,
            "restaurant food": 5,
            "food for a party": 5,
            "lunch for": 5,
            "dinner for": 5,
            "breakfast for": 5,
            "meal delivery": 5,
            "meal for": 5,
            "catering service": 5,
            "office catering": 5,
            "event catering": 5,
            "snacks for": 5,
        },
        "terms": {
            "food": 3,
            "meal": 3,
            "lunch": 3,
            "dinner": 3,
            "breakfast": 3,
            "catering": 4,
            "restaurant": 4,
            "snacks": 3,
        },
    },
    "Appliance Repair": {
        "phrases": {
            "washing machine repair": 5,
            "washing machine not working": 5,
            "washing machine is not working": 5,
            "refrigerator repair": 5,
            "fridge repair": 5,
            "ac repair": 5,
            "aircon repair": 5,
            "air conditioner repair": 5,
            "air conditioning repair": 5,
            "appliance repair": 5,
            "microwave repair": 5,
            "mixer repair": 5,
            "tv repair": 5,
            "television repair": 5,
            "appliance not working": 5,
            "dishwasher repair": 5,
            "oven repair": 5,
            "dryer repair": 5,
            "refrigerator not working": 5,
        },
        "terms": {
            "appliance": 4,
            "refrigerator": 4,
            "fridge": 4,
            "microwave": 4,
            "mixer": 4,
            "washing": 3,
            "dishwasher": 4,
            "oven": 3,
            "dryer": 3,
            "television": 4,
            "tv": 3,
            "ac": 3,
            "aircon": 3,
        },
    },
}

MIN_CATEGORY_SCORE = 3
CATEGORY_TIE_PRIORITY = tuple(sorted(CATEGORY_KEYWORDS))


def normalize_query(query: str) -> str:
    normalized = re.sub(r"[^a-z0-9 ]", " ", query.lower())
    return re.sub(r"\s+", " ", normalized).strip()


def category_scores(query: str) -> dict[str, int]:
    normalized = normalize_query(query)
    scores = {}
    for category, mapping in CATEGORY_KEYWORDS.items():
        score = 0
        for phrase, weight in mapping["phrases"].items():
            if re.search(rf"\b{re.escape(phrase)}\b", normalized):
                score += weight
        for keyword, weight in mapping["terms"].items():
            if re.search(rf"\b{re.escape(keyword)}\b", normalized):
                score += weight
        scores[category] = score
    return scores


def detect_category(query: str) -> str | None:
    scores = category_scores(query)
    best_score = max(scores.values(), default=0)
    if best_score < MIN_CATEGORY_SCORE:
        return None
    tied_categories = [category for category, score in scores.items() if score == best_score]
    return min(tied_categories, key=CATEGORY_TIE_PRIORITY.index)


def generate_recommendation_reasons(
    vendor, matched_category: str | None, distance: float | None = None
) -> list[str]:
    reasons = []
    if matched_category is not None and vendor.category == matched_category:
        reasons.append("Exact service match")

    reasons.append("Currently available" if vendor.available else "Currently unavailable")

    if vendor.rating >= 4.5:
        reasons.append("Highly rated")
    elif vendor.rating >= 4.0:
        reasons.append("Good customer rating")

    effective_distance = vendor.distance if distance is None else distance
    if effective_distance is not None:
        if effective_distance <= 2:
            reasons.append("Very close")
        elif effective_distance <= 5:
            reasons.append("Nearby")
        else:
            reasons.append("Farther away")

    if vendor.review_count >= 50:
        reasons.append("Well reviewed")
    elif vendor.review_count >= 20:
        reasons.append("Multiple customer reviews")
    return reasons


def generate_recommendation_summary(reasons: list[str]) -> str:
    summary_phrases = {
        "Exact service match": "an exact service match",
        "Currently available": "currently available",
        "Currently unavailable": "not currently available",
        "Highly rated": "highly rated",
        "Good customer rating": "well rated",
        "Very close": "very close",
        "Nearby": "nearby",
        "Farther away": "farther away",
        "Well reviewed": "well reviewed",
        "Multiple customer reviews": "supported by multiple customer reviews",
    }
    phrases = [summary_phrases[reason] for reason in reasons]
    if not phrases:
        return "No standout service, availability, rating, distance, or review signals."
    if len(phrases) == 1:
        joined_phrases = phrases[0]
    elif len(phrases) == 2:
        joined_phrases = f"{phrases[0]} and {phrases[1]}"
    else:
        joined_phrases = f"{', '.join(phrases[:-1])}, and {phrases[-1]}"
    return f"Vendor is {joined_phrases}."


def rank_vendors(
    vendors,
    query: str,
    *,
    customer_latitude: float | None = None,
    customer_longitude: float | None = None,
    customer_location_is_estimate: bool = False,
    customer_location_provided: bool = False,
    available_only: bool = False,
    nearby: bool = False,
    highly_rated: bool = False,
    max_price: float | None = None,
) -> list[dict]:
    category = detect_category(query)
    if category is None:
        return []

    filtered_vendors = []
    for vendor in vendors:
        if str(vendor.category).strip().casefold() != category.casefold():
            continue
        if available_only and not vendor.available:
            continue
        if highly_rated and vendor.rating < 4.5:
            continue

        vendor_distance = vendor.distance
        has_customer_location = customer_latitude is not None and customer_longitude is not None
        if has_customer_location and vendor.latitude is not None and vendor.longitude is not None:
            vendor_distance = distance_in_km(
                customer_latitude, customer_longitude, vendor.latitude, vendor.longitude
            )
        elif has_customer_location:
            vendor_distance = None
        elif customer_location_provided:
            vendor_distance = None
        if nearby and (vendor_distance is None or vendor_distance > 5):
            continue
        if max_price is not None:
            prices = re.findall(r"\d+(?:\.\d+)?", vendor.price or "")
            if not prices or max(map(float, prices)) > max_price:
                continue
        filtered_vendors.append(
            (vendor, vendor_distance, not has_customer_location or customer_location_is_estimate)
        )
    rows = [
        {
            "vendor": vendor,
            "distance": distance,
            "distance_is_estimate": distance_is_estimate,
            "category_match": 1.0,
            "availability": 1.0 if vendor.available else 0.0,
            "rating_score": min(max(vendor.rating, 0.0), 5.0) / 5.0,
            "distance_score": 1.0 / (1.0 + max(distance, 0.0) / 5.0)
            if distance is not None
            else 0.0,
            "review_score": min(max(vendor.review_count, 0), 100) / 100.0,
        }
        for vendor, distance, distance_is_estimate in filtered_vendors
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

    ranked_vendors = []
    for _, row in frame.iterrows():
        result = {
            **row["vendor"].__dict__,
            "distance": row["distance"],
            "distance_is_estimate": row["distance_is_estimate"],
            "recommendation_score": float(row["recommendation_score"]),
            "matched_category": category,
        }
        reasons = generate_recommendation_reasons(row["vendor"], category, row["distance"])
        result["recommendation_reasons"] = reasons
        result["recommendation_summary"] = generate_recommendation_summary(reasons)
        ranked_vendors.append(result)
    return ranked_vendors
