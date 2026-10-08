LOCATION_COORDINATES = {
    "puducherry": (11.9416, 79.8083),
    "pondicherry": (11.9416, 79.8083),
    "white town": (11.9343, 79.8344),
    "heritage town": (11.9307, 79.8301),
    "muthialpet": (11.9482, 79.8357),
    "lawspet": (11.9541, 79.8125),
    "oulgaret": (11.9400, 79.7900),
    "ariyankuppam": (11.8950, 79.8140),
    "auroville": (12.0060, 79.8100),
    "villupuram": (11.9401, 79.4861),
}


def distance_in_km(
    latitude_a: float, longitude_a: float, latitude_b: float, longitude_b: float
) -> float:
    from math import asin, cos, radians, sin, sqrt

    latitude_delta = radians(latitude_b - latitude_a)
    longitude_delta = radians(longitude_b - longitude_a)
    haversine = (
        sin(latitude_delta / 2) ** 2
        + cos(radians(latitude_a))
        * cos(radians(latitude_b))
        * sin(longitude_delta / 2) ** 2
    )
    return 6371.0 * 2 * asin(sqrt(haversine))


def coordinates_for_location(location: str) -> tuple[float, float] | None:
    normalized = location.casefold().strip()
    area_names = [name for name in LOCATION_COORDINATES if name not in {"puducherry", "pondicherry"}]
    for name in sorted(area_names, key=len, reverse=True):
        if name in normalized:
            return LOCATION_COORDINATES[name]
    for name in ("puducherry", "pondicherry"):
        if name in normalized:
            return LOCATION_COORDINATES[name]
    return None


def resolve_coordinates(
    location: str | None, latitude: float | None, longitude: float | None
) -> tuple[float | None, float | None]:
    if (latitude is None) != (longitude is None):
        raise ValueError("Latitude and longitude must be provided together.")
    if latitude is not None and longitude is not None:
        if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
            raise ValueError("The provided coordinates are outside valid ranges.")
        return latitude, longitude
    coordinates = coordinates_for_location(location or "")
    return coordinates if coordinates else (None, None)