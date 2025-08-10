import os
import json
from pathlib import Path
from typing import Dict, Optional
from fastmcp import FastMCP
from dotenv import load_dotenv
from aiohttp import ClientSession

# Load environment variables
load_dotenv()

# Initialize FastMCP
mcp = FastMCP("mcp-weather")

# Cache configuration
CACHE_DIR = Path.home() / ".cache" / "weather"
LOCATION_CACHE_FILE = CACHE_DIR / "location_cache.json"

def get_cached_location_key(location: str) -> Optional[str]:
    """Get location key from cache."""
    if not LOCATION_CACHE_FILE.exists():
        return None
    
    try:
        with open(LOCATION_CACHE_FILE, "r") as f:
            cache = json.load(f)
            return cache.get(location)
    except (json.JSONDecodeError, FileNotFoundError):
        return None

def cache_location_key(location: str, location_key: str):
    """Cache location key for future use."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        if LOCATION_CACHE_FILE.exists():
            with open(LOCATION_CACHE_FILE, "r") as f:
                cache = json.load(f)
        else:
            cache = {}
        
        cache[location] = location_key
        
        with open(LOCATION_CACHE_FILE, "w") as f:
            json.dump(cache, f, indent=2)
    except Exception as e:
        print(f"Warning: Failed to cache location key: {e}")

@mcp.tool()
async def get_hourly_weather(location: str) -> Dict:
    """Get hourly weather forecast for a location."""
    api_key = os.getenv("ACCUWEATHER_API_KEY")
    base_url = "http://dataservice.accuweather.com"
    
    # Try to get location key from cache first
    location_key = get_cached_location_key(location)
    
    async with ClientSession() as session:
        if not location_key:
            location_search_url = f"{base_url}/locations/v1/cities/search"
            params = {
                "apikey": api_key,
                "q": location,
            }
            async with session.get(location_search_url, params=params) as response:
                locations = await response.json()
                if response.status != 200:
                    raise Exception(f"Error fetching location data: {response.status}, {locations}")
                if not locations or len(locations) == 0:
                    raise Exception("Location not found")
            
            location_key = locations[0]["Key"]
            # Cache the location key for future use
            cache_location_key(location, location_key)
        
        # Get current conditions
        current_conditions_url = f"{base_url}/currentconditions/v1/{location_key}"
        params = {
            "apikey": api_key,
        }
        async with session.get(current_conditions_url, params=params) as response:
            current_conditions = await response.json()
            
        # Get hourly forecast
        forecast_url = f"{base_url}/forecasts/v1/hourly/12hour/{location_key}"
        params = {
            "apikey": api_key,
            "metric": "true",
        }
        async with session.get(forecast_url, params=params) as response:
            forecast = await response.json()
        
        # Format response
        hourly_data = []
        for i, hour in enumerate(forecast, 1):
            hourly_data.append({
                "relative_time": f"+{i} hour{'s' if i > 1 else ''}",
                "temperature": {
                    "value": hour["Temperature"]["Value"],
                    "unit": hour["Temperature"]["Unit"]
                },
                "weather_text": hour["IconPhrase"],
                "precipitation_probability": hour["PrecipitationProbability"],
                "precipitation_type": hour.get("PrecipitationType"),
                "precipitation_intensity": hour.get("PrecipitationIntensity"),
            })
        
        # Format current conditions
        if current_conditions and len(current_conditions) > 0:
            current = current_conditions[0]
            current_data = {
                "temperature": {
                    "value": current["Temperature"]["Metric"]["Value"],
                    "unit": current["Temperature"]["Metric"]["Unit"]
                },
                "weather_text": current["WeatherText"],
                "relative_humidity": current.get("RelativeHumidity"),
                "precipitation": current.get("HasPrecipitation", False),
                "observation_time": current["LocalObservationDateTime"]
            }
        else:
            current_data = "No current conditions available"
        
        return {
            "location": locations[0]["LocalizedName"],
            "location_key": location_key,
            "country": locations[0]["Country"]["LocalizedName"],
            "current_conditions": current_data,
            "hourly_forecast": hourly_data
        } 