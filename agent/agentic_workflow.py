import os
import re
import time
import concurrent.futures
from langchain_core.messages import AIMessage, HumanMessage
from utils.model_loader import ModelLoader
from prompt_library.prompt import SYSTEM_PROMPT
from tools.weather_info_tool import WeatherInfoTool
from tools.place_search_tool import PlaceSearchTool
from tools.expense_calculator_tool import CalculatorTool
from tools.currency_conversion_tool import CurrencyConverterTool


# ─────────────────────────────────────────────────────────────────────────────
#  TripPlannerApp  –  Replaces the LangGraph ReAct loop.
#
#  OLD approach (slow):  LLM → call tool 1 → LLM → call tool 2 → LLM → ...
#                        ~8-10 sequential LLM calls = 60-120 seconds
#
#  NEW approach (fast):  Extract params (no LLM) → run ALL tools in PARALLEL
#                        → ONE final LLM call  =  10-20 seconds total
# ─────────────────────────────────────────────────────────────────────────────
class TripPlannerApp:
    """
    Lightweight executor:
      1. Parse trip params from user query (regex, zero LLM calls).
      2. Run all tools concurrently with ThreadPoolExecutor + per-tool timeout.
      3. Make ONE LLM call to generate the complete, structured itinerary.
    Compatible with the existing main.py interface (invoke / get_graph).
    """

    TOOL_TIMEOUT = 8.0   # seconds per individual tool call
    TOTAL_TIMEOUT = 20.0 # seconds for the whole parallel batch

    def __init__(self, llm, tools_map: dict):
        self.llm = llm
        self.tools_map = tools_map

    # ── Param extraction ──────────────────────────────────────────────────────

    def _extract_params(self, query: str) -> dict:
        """Extract destination, days, travelers, budget from query using regex.
        No LLM call required — keeps latency near zero for this step."""
        q = query.strip()
        ql = q.lower()

        # Days
        days = 3
        m = re.search(r'(\d+)\s*(?:day|din|days|dino|raat|night|nights)', ql)
        if m:
            days = max(1, min(int(m.group(1)), 30))

        # Travelers
        travelers = None
        m = re.search(r'(\d+)\s*(?:person|people|log|traveler|adult|logon|vyakti)', ql)
        if m:
            travelers = max(1, int(m.group(1)))

        # Budget type
        budget_type = "mid-range"
        if any(w in ql for w in ['budget', 'cheap', 'sasta', 'affordable', 'low cost', 'backpack', 'economy']):
            budget_type = "budget"
        elif any(w in ql for w in ['luxury', 'premium', 'expensive', 'five star', '5 star', 'lavish']):
            budget_type = "luxury"

        # Destination — try multiple patterns in order on lowercase query
        destination = ""
        patterns = [
            r'\b(?:trip|travel|visit|tour|yatra|go)\s+(?:to\s+)?([a-z\s]+?)(?:\s+for|\s+in|\s+with|\s+on|\,|\.|$)',
            r'\bto\s+([a-z\s]+?)(?:\s+for|\s+in|\s+with|\s+on|\,|\.|$)',
            r'\b([a-z\s]+?)\s+(?:trip|tour|visit|ke\s+liye|mein|travel)',
        ]
        
        exclude = {
            'a', 'the', 'my', 'our', 'your', 'plan', 'day', 'trip', 'travel',
            'people', 'person', 'budget', 'days', 'night', 'nights', 'for',
            'with', 'and', 'please', 'make', 'give', 'create', 'generate',
            'tour', 'visit', 'yatra', 'go', 'to', 'in', 'on', 'me', 'us',
            'budget', 'cheap', 'luxury', 'expensive', 'mid-range', 'midrange',
            'friends', 'family', 'couple', 'solo', 'group', 'hindi', 'english',
            'an', 'some', 'any', 'that', 'this', 'there', 'here', 'now', 'then',
            'of', 'is', 'are', 'was', 'were', 'am', 'i', 'we', 'he', 'she', 'it', 'they'
        }
        
        for pattern in patterns:
            m = re.search(pattern, ql)
            if m:
                # Get candidate, clean it, split into words
                candidate_raw = m.group(1).strip().rstrip('.,')
                words = candidate_raw.split()
                
                # Filter out exclude words from the candidate
                filtered_words = [w for w in words if w not in exclude and len(w) > 1]
                
                if filtered_words:
                    # Join back and capitalize nicely (e.g. 'new delhi' -> 'New Delhi')
                    destination = " ".join(filtered_words).title()
                    break

        # Fallback if patterns fail (just find the first non-excluded word >= 3 chars)
        if not destination:
            words = re.findall(r'\b([a-z]{3,})\b', ql)
            candidates = [w for w in words if w not in exclude]
            if candidates:
                destination = candidates[-1].title() # usually destination is near the end

        # Detect Hindi / Hinglish
        hindi_chars = bool(re.search(r'[\u0900-\u097F]', q))
        hindi_words = any(w in ql for w in [
            'ke liye', 'ka plan', 'banao', 'btao', 'din ka', ' log',
            'logon', 'yatra', 'ghoomna', 'trip plan karo', 'batao',
        ])
        language = "Hindi" if (hindi_chars or hindi_words) else "English"

        return {
            "destination": destination,
            "days": days,
            "travelers": travelers,
            "budget_type": budget_type,
            "language": language,
            "original_query": query,
        }

    # ── Parallel tool execution ───────────────────────────────────────────────

    def _call_tool(self, tool_name: str, args: dict) -> str:
        """Call a single tool safely; return a graceful string on any error."""
        tool = self.tools_map.get(tool_name)
        if not tool:
            return f"[{tool_name}] not available"
        try:
            result = tool.invoke(args)
            return str(result) if result else f"[{tool_name}] returned empty result"
        except Exception as e:
            return f"[{tool_name}] unavailable: {str(e)[:120]}"

    def _run_all_tools_parallel(self, destination: str) -> dict:
        """Submit all tool calls to a thread pool and collect results."""
        calls = [
            ("weather_current",  "get_current_weather",   {"city": destination}),
            ("weather_forecast", "get_weather_forecast",  {"city": destination}),
            ("attractions",      "search_attractions",    {"place": destination}),
            ("restaurants",      "search_restaurants",    {"place": destination}),
            ("activities",       "search_activities",     {"place": destination}),
            ("transportation",   "search_transportation", {"place": destination}),
        ]

        results = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
            future_map = {
                pool.submit(self._call_tool, tool_name, args): key
                for key, tool_name, args in calls
            }
            done, _ = concurrent.futures.wait(
                future_map.keys(),
                timeout=self.TOTAL_TIMEOUT
            )
            for future in done:
                key = future_map[future]
                try:
                    results[key] = future.result(timeout=1)
                except Exception as e:
                    results[key] = f"error: {e}"
            for future in future_map:
                if future not in done:
                    key = future_map[future]
                    results[key] = f"[{key}] timed out — using general knowledge"

        return results

    # ── Final prompt assembly ─────────────────────────────────────────────────

    @staticmethod
    def _is_useful(text: str) -> bool:
        """Return True if tool result contains real data (not just error msgs)."""
        lower = text.lower()
        bad_markers = ["unavailable", "not available", "timed out", "error:", "timeout", "missing"]
        return not any(m in lower for m in bad_markers)

    def _build_final_prompt(self, params: dict, tool_results: dict) -> str:
        dest      = params["destination"]
        days      = params["days"]
        travelers = params["travelers"]
        budget    = params["budget_type"]
        language  = params["language"]
        query     = params["original_query"]

        travelers_text = f"for {travelers} traveler(s)" if travelers else "for the total trip (per person if generic)"

        # Collect live data that's actually useful
        data_parts = []
        label_map = {
            "weather_current":  "CURRENT WEATHER",
            "weather_forecast": "WEATHER FORECAST",
            "attractions":      "ATTRACTIONS",
            "restaurants":      "RESTAURANTS & FOOD",
            "activities":       "ACTIVITIES",
            "transportation":   "LOCAL TRANSPORT",
        }
        for key, label in label_map.items():
            val = tool_results.get(key, "")
            if val and self._is_useful(val):
                # Truncate very long tool outputs to keep tokens under control
                data_parts.append(f"### {label}\n{val[:800]}")

        live_data = "\n\n".join(data_parts) if data_parts else \
            "No live tool data — use your own knowledge for this destination."

        return f"""USER REQUEST: {query}

TRIP PARAMETERS:
- Destination: {dest}
- Duration: {days} days
- Travelers: {travelers if travelers else 'Not specified (assume standard generic budget)'}
- Budget type: {budget}
- Response language: {language} — respond ENTIRELY in {language}

LIVE DATA COLLECTED BY TOOLS:
{live_data}

─────────────────────────────────────────
Generate a COMPLETE and PRACTICAL {days}-day travel plan for {dest}.
Include ALL sections below. Use Markdown with emoji headers.
─────────────────────────────────────────

1. 🌍 DESTINATION OVERVIEW
   Best time to visit | Why the destination is worth visiting | Approximate weather/season | Local currency | Language | Basic travel tips

2. 📋 TRIP SUMMARY
   Exact duration ({days} days / {days - 1} nights) | Recommended number of nights | Suggested trip pace | Estimated total budget range {travelers_text}

3. 🗓 DAY-BY-DAY ITINERARY  ← MOST IMPORTANT — generate exactly {days} days
   For EVERY requested day include a detailed text description (DO NOT USE TABLES for the itinerary):
   - **Day X: [Theme/Title]**
   - **Morning:** Detailed description of where to go, what to see, specific attractions, time needed, and approx costs. EXPLICITLY explain HOW to travel from your hotel to these places and between them (e.g., take a specific bus, metro line, book an Uber, or ride a rented bike) with approx commute time and cost.
   - **Afternoon:** Detailed description of places to visit, suggested order, food/lunch suggestions with approx costs, and explicit local travel instructions (how to get there from the morning location).
   - **Evening:** Detailed description of evening activities, dinner suggestions, night stay area, and how to travel back to the hotel.
   Write this as a highly detailed, descriptive narrative with bullet points.

4. 🏛 BEST PLACES TO VISIT
   For each important attraction:
   - Why it is worth visiting
   - Approximate time required
   - Whether it is free or paid (when reliable information is available)

5. 💰 BUDGET BREAKDOWN ({travelers_text}, {days} days)
   Give an approximate budget in the relevant local currency (and convert to INR if available).
   Clearly distinguish Budget, Mid-range, and Luxury trips.
   Show price ranges when exact live prices are unavailable (clearly mark as estimates). Never invent exact prices.
   Format exactly like this for each tier:
   Accommodation + Food + Local transportation + Attractions/entry tickets + Miscellaneous = TOTAL TRIP COST (state if per person or for total travelers).
   Also include approximate daily expenses.

6. 💡 EXPENSIVE VS BUDGET OPTIONS
   Explain practical alternatives:
   - Budget accommodation vs expensive accommodation
   - Public transport vs taxis/private transport
   - Free attractions vs paid attractions
   - Budget food vs expensive dining

7. 🏨 WHERE TO STAY
   Recommend suitable neighborhoods/areas based on:
   - Budget, Mid-range, and Luxury accommodation options
   - Location and Accessibility to major attractions

8. 🚌 TRANSPORTATION & HOW TO REACH
   **A. How to reach {dest} (Compare all options with approx costs):**
   - ✈️ **By Aeroplane:** Nearest airport, approx flight cost, and how to reach the city from the airport.
   - 🚆 **By Train:** Nearest railway station, approx ticket cost, and travel time.
   - 🚌 **By Bus:** Availability of sleeper/volvo buses and approx ticket cost.
   - 🏍️/🚗 **By Bike/Car (Road Trip):** Road conditions, estimated fuel cost, and whether a road trip is recommended.

   **B. Local Commute (Getting around the city):**
   - Explain practically how to get around locally (e.g., where and how to rent scooties/bikes, booking cabs like Uber/Ola/Grab, or using local public transport like Metro/Bus). Give specific examples and approximate costs for renting per day or commuting per trip.

9. 🍽 FOOD
   - Local foods to try
   - Budget-friendly food options
   - Important food/dining tips and daily food budget

10. 📝 IMPORTANT TRAVEL TIPS
    - Safety/general precautions
    - Booking advice
    - Local transport tips
    - Common tourist mistakes
    - What to carry
    - Visa/passport information (only when relevant and reliable)

11. 💸 HOW TO SAVE MONEY & UPGRADES
    - How to save money
    - Optional premium upgrades
    - What is included/excluded from the total estimated budget

12. ✅ QUICK FINAL SUMMARY
    Recommended stay | Approximate budget | Must-visit places | Best areas to stay | Main things to avoid | Overall practical recommendation based on preferences
"""

    # ── Public interface ──────────────────────────────────────────────────────

    def invoke(self, messages: dict) -> dict:
        """
        Main entry point called by main.py.
        Returns dict with {"messages": [AIMessage(content=...)]}
        """
        raw = messages.get("messages", [])
        if not raw:
            return {"messages": [AIMessage(content="Please provide a destination for your trip plan.")]}

        last = raw[-1]
        query = last if isinstance(last, str) else getattr(last, "content", str(last))

        # Step 1 — extract params (no LLM)
        params = self._extract_params(query)

        # Step 2 — run tools in parallel
        tool_results = {}
        if params["destination"]:
            tool_results = self._run_all_tools_parallel(params["destination"])
        else:
            # No destination found — ask user
            msg = (
                "I need a destination to plan your trip. "
                "Please mention where you want to go, e.g. 'Plan a 5-day trip to Jaipur for 2 people'."
            )
            return {"messages": [AIMessage(content=msg)]}

        # Step 3 — ONE final LLM call
        final_prompt = self._build_final_prompt(params, tool_results)
        response = self.llm.invoke([
            SYSTEM_PROMPT,
            HumanMessage(content=final_prompt),
        ])
        return {"messages": [response]}

    def get_graph(self):
        """Stub for compatibility with main.py graph-PNG code."""
        return _DummyGraph()


class _DummyGraph:
    def draw_mermaid_png(self):
        raise Exception("Graph PNG not available in optimized mode")


# ─────────────────────────────────────────────────────────────────────────────
#  GraphBuilder  –  Public factory, interface unchanged for main.py
# ─────────────────────────────────────────────────────────────────────────────
class GraphBuilder:
    def __init__(self, model_provider: str = "groq"):
        self.model_loader = ModelLoader(model_provider=model_provider)
        self.llm = self.model_loader.load_llm()

        # Initialise tool providers
        weather_tools    = WeatherInfoTool()
        place_tools      = PlaceSearchTool()
        calculator_tools = CalculatorTool()
        currency_tools   = CurrencyConverterTool()

        all_tools = (
            weather_tools.weather_tool_list +
            place_tools.place_search_tool_list +
            calculator_tools.calculator_tool_list +
            currency_tools.currency_converter_tool_list
        )
        self.tools_map = {t.name: t for t in all_tools}

    def __call__(self) -> TripPlannerApp:
        return TripPlannerApp(llm=self.llm, tools_map=self.tools_map)