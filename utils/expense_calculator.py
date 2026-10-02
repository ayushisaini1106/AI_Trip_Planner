from typing import Any

class Calculator:
    @staticmethod
    def multiply(a: Any, b: Any) -> float:
        """
        Multiply two numbers safely.
        """
        try:
            if isinstance(a, str):
                a = a.replace('$', '').replace('₹', '').replace(',', '').strip()
            if isinstance(b, str):
                b = b.replace('$', '').replace('₹', '').replace(',', '').strip()
            return float(a) * float(b)
        except Exception:
            return 0.0
    
    @staticmethod
    def calculate_total(*x: Any) -> float:
        """
        Calculate sum of the given list of numbers
        """
        total = 0.0
        for item in x:
            try:
                if isinstance(item, str):
                    item = item.replace('$', '').replace('₹', '').replace(',', '').strip()
                total += float(item)
            except Exception:
                pass
        return total
    
    @staticmethod
    def calculate_daily_budget(total: Any, days: Any) -> float:
        """
        Calculate daily budget
        """
        try:
            t = float(total)
            d = float(days)
            return t / d if d > 0 else 0.0
        except Exception:
            return 0.0
    
    