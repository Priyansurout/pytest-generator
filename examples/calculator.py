"""
Simple Calculator Module - Demo for TestGen
"""

from typing import Union, Optional
import asyncio


def add(a: Union[int, float], b: Union[int, float]) -> Union[int, float]:
    """Add two numbers.
    
    Args:
        a: First number
        b: Second number
        
    Returns:
        Sum of a and b
        
    Raises:
        TypeError: If inputs are not numbers
    """
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        raise TypeError("Inputs must be numbers")
    return a + b


def subtract(a: float, b: float) -> float:
    """Subtract b from a.
    
    Args:
        a: Minuend
        b: Subtrahend
        
    Returns:
        Difference (a - b)
    """
    return a - b


def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


def divide(dividend: float, divisor: float) -> float:
    """Divide dividend by divisor.
    
    Args:
        dividend: Number to be divided
        divisor: Number to divide by
        
    Returns:
        Quotient
        
    Raises:
        ZeroDivisionError: If divisor is zero
        TypeError: If inputs are not numbers
    """
    if divisor == 0:
        raise ZeroDivisionError("Cannot divide by zero")
    return dividend / divisor


def calculate_discount(price: float, percentage: float) -> float:
    """Calculate discounted price.
    
    Args:
        price: Original price (must be positive)
        percentage: Discount percentage 0-100
        
    Returns:
        Discounted price
        
    Raises:
        ValueError: If price is negative or percentage invalid
    """
    if price < 0:
        raise ValueError("Price cannot be negative")
    if not (0 <= percentage <= 100):
        raise ValueError("Percentage must be between 0 and 100")
    return price * (1 - percentage / 100)


async def async_calculate_bulk(
    numbers: list,
    operation: str,
    db_client: Optional[object] = None
) -> dict:
    """Perform bulk calculation asynchronously with database logging.
    
    Args:
        numbers: List of numbers to process
        operation: Operation to perform ('sum', 'avg', 'max')
        db_client: Database client for logging (optional)
        
    Returns:
        Dictionary with results and metadata
        
    Raises:
        ValueError: If operation is invalid
        ConnectionError: If database logging fails
        
    # Dependencies: db_client.connect(), db_client.log_result()
    """
    if operation not in ['sum', 'avg', 'max']:
        raise ValueError(f"Invalid operation: {operation}")
    
    if not numbers:
        return {'result': 0, 'count': 0, 'operation': operation}
    
    if operation == 'sum':
        result = sum(numbers)
    elif operation == 'avg':
        result = sum(numbers) / len(numbers)
    else:  # max
        result = max(numbers)
    
    # Log to database if client provided
    if db_client:
        try:
            await db_client.connect()
            await db_client.log_result(operation, result)
        except Exception as e:
            raise ConnectionError(f"Failed to log to database: {e}")
    
    return {
        'result': result,
        'count': len(numbers),
        'operation': operation
    }


def validate_expression(expr: str) -> bool:
    """Validate if string is a valid mathematical expression.
    
    Args:
        expr: Expression string to validate
        
    Returns:
        True if valid, False otherwise
        
    Raises:
        TypeError: If input is not string
    """
    if not isinstance(expr, str):
        raise TypeError("Expression must be a string")
    
    allowed_chars = set('0123456789+-*/.() ')
    return all(c in allowed_chars for c in expr.strip())


# Save this as calculator.py
if __name__ == "__main__":
    print("Calculator module loaded")
    print(f"2 + 3 = {add(2, 3)}")
    print(f"10 / 2 = {divide(10, 2)}")