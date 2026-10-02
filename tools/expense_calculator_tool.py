from utils.expense_calculator import Calculator
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
from typing import List, Any


class HotelCostInput(BaseModel):
    """Input schema for hotel cost estimation."""
    price_per_night: Any = Field(..., description="Price per night for the hotel (number or string like '5000' or '$150')")
    total_days: Any = Field(..., description="Total number of days of the stay")


class TotalExpenseInput(BaseModel):
    """Input schema for total expense calculation."""
    costs: List[Any] = Field(..., description="List of all individual costs to be summed up")


class DailyBudgetInput(BaseModel):
    """Input schema for daily budget calculation."""
    total_cost: Any = Field(..., description="Total cost of the trip")
    days: Any = Field(..., description="Total number of days")


class CalculatorTool:

    def __init__(self):
        self.calculator = Calculator()
        self.calculator_tool_list = self._setup_tools()

    def _setup_tools(self) -> List:
        """Setup all tools for the calculator tool with explicit Pydantic schemas."""

        def _estimate_total_hotel_cost(price_per_night: Any, total_days: Any) -> float:
            """Calculate total hotel cost given price per night and total days"""
            try:
                return self.calculator.multiply(price_per_night, total_days)
            except Exception as e:
                return 0.0

        def _calculate_total_expense(costs: List[Any]) -> float:
            """Calculate total expense of the trip given a list of individual costs"""
            try:
                return self.calculator.calculate_total(*costs)
            except Exception as e:
                return 0.0

        def _calculate_daily_expense_budget(total_cost: Any, days: Any) -> float:
            """Calculate daily expense budget given total cost and number of days"""
            try:
                return self.calculator.calculate_daily_budget(total_cost, days)
            except Exception as e:
                return 0.0

        estimate_total_hotel_cost = StructuredTool.from_function(
            func=_estimate_total_hotel_cost,
            name="estimate_total_hotel_cost",
            description="Calculate total hotel cost by multiplying price per night by number of days.",
            args_schema=HotelCostInput,
        )

        calculate_total_expense = StructuredTool.from_function(
            func=_calculate_total_expense,
            name="calculate_total_expense",
            description="Calculate the total expense of a trip by summing up a list of individual costs.",
            args_schema=TotalExpenseInput,
        )

        calculate_daily_expense_budget = StructuredTool.from_function(
            func=_calculate_daily_expense_budget,
            name="calculate_daily_expense_budget",
            description="Calculate the daily expense budget by dividing total cost by number of days.",
            args_schema=DailyBudgetInput,
        )

        return [estimate_total_hotel_cost, calculate_total_expense, calculate_daily_expense_budget]