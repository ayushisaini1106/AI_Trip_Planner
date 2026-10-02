import os
from utils.weather_info import WeatherForecastTool
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv


class CityInput(BaseModel):
    """Input schema for weather tools. Accepts 'city' as the destination name."""
    city: str = Field(..., description="The name of the city to get weather information for (e.g., 'Paris', 'Goa')")


class WeatherInfoTool:
    def __init__(self):
        load_dotenv(override=True)
        self.api_key = os.environ.get("OPENWEATHERMAP_API_KEY")
        self.weather_service = WeatherForecastTool(self.api_key)
        self.weather_tool_list = self._setup_tools()

    def _setup_tools(self) -> List:
        """Setup all tools for the weather forecast tool with explicit schemas."""

        def _get_current_weather(city: str) -> str:
            """Get current weather for a city"""
            if not self.api_key:
                return (
                    f"Weather information for {city} is currently unavailable "
                    f"(OPENWEATHERMAP_API_KEY not configured). "
                    f"Please note weather data as 'N/A' in the itinerary."
                )
            try:
                weather_data = self.weather_service.get_current_weather(city)
                if weather_data and weather_data.get('main'):
                    temp = weather_data.get('main', {}).get('temp', 'N/A')
                    desc = weather_data.get('weather', [{}])[0].get('description', 'N/A')
                    return f"Current weather in {city}: {temp}°C, {desc}"
                return f"Could not fetch weather for {city}. Include weather as 'N/A' in the plan."
            except Exception as e:
                return (
                    f"Weather data for {city} is temporarily unavailable (error: {str(e)}). "
                    f"Continue with the itinerary and mark weather as 'unavailable'."
                )

        def _get_weather_forecast(city: str) -> str:
            """Get weather forecast for a city"""
            if not self.api_key:
                return (
                    f"Weather forecast for {city} is currently unavailable "
                    f"(OPENWEATHERMAP_API_KEY not configured). "
                    f"Please note weather as 'N/A' in the itinerary."
                )
            try:
                forecast_data = self.weather_service.get_forecast_weather(city)
                if forecast_data and 'list' in forecast_data:
                    forecast_summary = []
                    for item in forecast_data['list']:
                        date = item['dt_txt'].split(' ')[0]
                        temp = item['main']['temp']
                        desc = item['weather'][0]['description']
                        forecast_summary.append(f"{date}: {temp}°C, {desc}")
                    return f"Weather forecast for {city}:\n" + "\n".join(forecast_summary)
                return f"Could not fetch forecast for {city}. Mark weather as 'unavailable' in the plan."
            except Exception as e:
                return (
                    f"Weather forecast for {city} is temporarily unavailable (error: {str(e)}). "
                    f"Continue with the itinerary and mark weather as 'unavailable'."
                )

        get_current_weather = StructuredTool.from_function(
            func=_get_current_weather,
            name="get_current_weather",
            description="Get the current weather conditions for a city.",
            args_schema=CityInput,
        )

        get_weather_forecast = StructuredTool.from_function(
            func=_get_weather_forecast,
            name="get_weather_forecast",
            description="Get a multi-day weather forecast for a city.",
            args_schema=CityInput,
        )

        return [get_current_weather, get_weather_forecast]