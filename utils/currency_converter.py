import requests
from typing import Any

class CurrencyConverter:
    def __init__(self, api_key: str):
        self.base_url = f"https://v6.exchangerate-api.com/v6/{api_key}/latest/"
    
    def convert(self, amount: Any, from_currency: str, to_currency: str):
        """Convert the amount from one currency to another"""
        try:
            amount = float(amount)
        except Exception:
            amount = 0.0
        url = f"{self.base_url}/{from_currency}"
        response = requests.get(url)
        if response.status_code != 200:
            raise Exception("API call failed:", response.json())
        rates = response.json()["conversion_rates"]
        if to_currency not in rates:
            raise ValueError(f"{to_currency} not found in exchange rates.")
        return amount * rates[to_currency]