from geopy.distance import geodesic

# Known coordinates for demo cities (Latitude, Longitude)
CITY_COORDINATES = {
    "New York": (40.7128, -74.0060),
    "London": (51.5074, -0.1278),
    "Paris": (48.8566, 2.3514),
    "Tokyo": (35.6762, 139.6503),
    "Sydney": (-33.8688, 151.2093),
    "Los Angeles": (34.0522, -118.2437)
}

def detect_impossible_travel(city1: str, city2: str, time_elapsed_hours: float, max_flight_speed_mph: float = 550.0):
    """
    Calculates distance between two cities and checks if required travel speed 
    exceeds commercial aircraft speeds (~550 mph).
    """
    if city1 not in CITY_COORDINATES or city2 not in CITY_COORDINATES:
        return {"error": "Invalid City Selection"}
        
    coords1 = CITY_COORDINATES[city1]
    coords2 = CITY_COORDINATES[city2]
    
    # Calculate geodesic distance in miles
    distance_miles = geodesic(coords1, coords2).miles
    
    # Prevent division by zero
    if time_elapsed_hours <= 0:
        required_speed = float('inf') if distance_miles > 0 else 0.0
    else:
        required_speed = distance_miles / time_elapsed_hours
        
    is_impossible = required_speed > max_flight_speed_mph
    
    return {
        "distance_miles": round(distance_miles, 2),
        "time_elapsed_hours": time_elapsed_hours,
        "required_speed_mph": round(required_speed, 2),
        "is_impossible": is_impossible
    }