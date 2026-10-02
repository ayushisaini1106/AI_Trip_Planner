import os
from utils.place_info_search import GooglePlaceSearchTool, TavilyPlaceSearchTool
from typing import List
from langchain.tools import tool
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from dotenv import load_dotenv


class PlaceInput(BaseModel):
    """Input schema for place search tools. Accepts 'place' as the destination name."""
    place: str = Field(..., description="The name of the city or destination to search (e.g., 'Paris', 'Goa', 'Manali')")


class PlaceSearchTool:
    def __init__(self):
        load_dotenv(override=True)
        self.google_api_key = os.environ.get("GPLACES_API_KEY")
        self.google_places_search = GooglePlaceSearchTool(self.google_api_key)
        self.tavily_search = TavilyPlaceSearchTool()
        self.place_search_tool_list = self._setup_tools()

    def _setup_tools(self) -> List:
        """Setup all tools for the place search tool with explicit Pydantic schemas."""

        def _search_attractions(place: str) -> str:
            """Search attractions of a place"""
            try:
                result = self.google_places_search.google_search_attractions(place)
                if result:
                    return f"Following are the attractions of {place} as suggested by Google: {result}"
            except Exception as google_err:
                pass
            try:
                result = self.tavily_search.tavily_search_attractions(place)
                if result:
                    return f"Following are the attractions of {place}: {result}"
            except Exception as tavily_err:
                pass
            return (
                f"Live search for attractions in {place} is currently unavailable "
                f"(TAVILY_API_KEY or GPLACES_API_KEY not configured). "
                f"Please use general knowledge to suggest popular attractions in {place}."
            )

        def _search_restaurants(place: str) -> str:
            """Search restaurants of a place"""
            try:
                result = self.google_places_search.google_search_restaurants(place)
                if result:
                    return f"Following are the restaurants of {place} as suggested by Google: {result}"
            except Exception:
                pass
            try:
                result = self.tavily_search.tavily_search_restaurants(place)
                if result:
                    return f"Following are the restaurants of {place}: {result}"
            except Exception:
                pass
            return (
                f"Live restaurant search for {place} is currently unavailable. "
                f"Please use general knowledge to suggest popular restaurants in {place}."
            )

        def _search_activities(place: str) -> str:
            """Search activities of a place"""
            try:
                result = self.google_places_search.google_search_activity(place)
                if result:
                    return f"Following are the activities in and around {place} as suggested by Google: {result}"
            except Exception:
                pass
            try:
                result = self.tavily_search.tavily_search_activity(place)
                if result:
                    return f"Following are the activities of {place}: {result}"
            except Exception:
                pass
            return (
                f"Live activity search for {place} is currently unavailable. "
                f"Please use general knowledge to suggest popular activities in {place}."
            )

        def _search_transportation(place: str) -> str:
            """Search transportation of a place"""
            try:
                result = self.google_places_search.google_search_transportation(place)
                if result:
                    return f"Following are the modes of transportation available in {place} as suggested by Google: {result}"
            except Exception:
                pass
            try:
                result = self.tavily_search.tavily_search_transportation(place)
                if result:
                    return f"Following are the modes of transportation available in {place}: {result}"
            except Exception:
                pass
            return (
                f"Live transportation search for {place} is currently unavailable. "
                f"Please use general knowledge to suggest transportation options in {place}."
            )

        # Use StructuredTool with explicit args_schema to lock parameter name to 'place'
        search_attractions = StructuredTool.from_function(
            func=_search_attractions,
            name="search_attractions",
            description="Search for top tourist attractions and sightseeing spots in a destination city or place.",
            args_schema=PlaceInput,
        )

        search_restaurants = StructuredTool.from_function(
            func=_search_restaurants,
            name="search_restaurants",
            description="Search for top restaurants and food spots in a destination city or place.",
            args_schema=PlaceInput,
        )

        search_activities = StructuredTool.from_function(
            func=_search_activities,
            name="search_activities",
            description="Search for popular activities and things to do in a destination city or place.",
            args_schema=PlaceInput,
        )

        search_transportation = StructuredTool.from_function(
            func=_search_transportation,
            name="search_transportation",
            description="Search for available modes of transportation in a destination city or place.",
            args_schema=PlaceInput,
        )

        return [search_attractions, search_restaurants, search_activities, search_transportation]