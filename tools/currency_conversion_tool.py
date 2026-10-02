import os
from utils.currency_converter import CurrencyConverter
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
from typing import List, Any
from dotenv import load_dotenv


class CurrencyInput(BaseModel):
    """Input schema for currency conversion tool."""
    amount: Any = Field(..., description="The amount to convert (e.g., 100)")
    from_currency: str = Field(..., description="Source currency code (e.g., 'USD', 'EUR')")
    to_currency: str = Field(..., description="Target currency code (e.g., 'INR', 'GBP')")


class CurrencyConverterTool:
    def __init__(self):
        load_dotenv(override=True)
        self.api_key = os.environ.get("EXCHANGE_RATE_API_KEY")
        self.currency_service = CurrencyConverter(self.api_key) if self.api_key else None
        self.currency_converter_tool_list = self._setup_tools()

    def _setup_tools(self) -> List:
        """Setup all tools for the currency converter tool with explicit schema."""

        def _convert_currency(amount: Any, from_currency: str, to_currency: str) -> str:
            """Convert amount from one currency to another"""
            if not self.api_key or not self.currency_service:
                return (
                    f"Currency conversion is currently unavailable "
                    f"(EXCHANGE_RATE_API_KEY not configured). "
                    f"Please provide approximate costs without live conversion."
                )
            try:
                result = self.currency_service.convert(amount, from_currency, to_currency)
                return f"{amount} {from_currency} = {result:.2f} {to_currency}"
            except Exception as e:
                return (
                    f"Currency conversion from {from_currency} to {to_currency} failed: {str(e)}. "
                    f"Please provide approximate costs without live conversion."
                )

        convert_currency = StructuredTool.from_function(
            func=_convert_currency,
            name="convert_currency",
            description="Convert an amount from one currency to another using live exchange rates.",
            args_schema=CurrencyInput,
        )

        return [convert_currency]