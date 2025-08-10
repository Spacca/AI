from pathlib import Path
from typing import Dict
from fastmcp import FastMCP
from dotenv import load_dotenv
from aiohttp import ClientSession
from datetime import datetime

# Load environment variables
load_dotenv()

# Initialize FastMCP
mcp = FastMCP("mcp-weather")

# Cache configuration
CACHE_DIR = Path.home() / ".cache" / "weather"
LOCATION_CACHE_FILE = CACHE_DIR / "location_cache.json"


@mcp.tool()
async def current_date_time() -> Dict:
    """Get the current hour in format %H:%M:%S and date in format %Y-%m-%d."""
    now = datetime.now()
    return {"date": now.strftime("%Y-%m-%d"), "time": now.strftime("%H:%M:%S")}


@mcp.tool()
async def get_lat_long(city: str) -> Dict:
    """Get latitute and longitude of a city"""
    async with ClientSession() as session:
        geocode_url = (
            f"https://nominatim.openstreetmap.org/search?q={city}&format=json&limit=1"
        )
        async with session.get(geocode_url) as resp:
            data = await resp.json()
            if data:
                return {"latitude": data[0]["lat"], "longitude": data[0]["lon"]}
            else:
                return {"error": "Location not found"}


@mcp.tool()
async def temperature(latitude: float, longitude: float) -> Dict:
    """Get temperature forecast for given latitude and longitude."""
    async with ClientSession() as session:
        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&hourly=temperature_2m&forecast_days=1"
        async with session.get(weather_url) as resp:
            weather_data = await resp.json()
            return weather_data.get("hourly", {})


if __name__ == "__main__":
    mcp.run(
        transport="http", host="127.0.0.1", port=8000, path="/mcp", stateless_http=True
    )
