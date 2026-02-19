"""
Utility functions for the application.
This file demonstrates simple functions without dependencies.
"""

from typing import List, Dict


def calculate_discount(price: float, discount_percent: float) -> float:
    """
    Calculate discounted price.
    
    Args:
        price: Original price
        discount_percent: Discount percentage (0-100)
        
    Returns:
        Discounted price
        
    Raises:
        ValueError: If discount is invalid
    """
    if discount_percent < 0 or discount_percent > 100:
        raise ValueError("Discount must be between 0 and 100")
    
    discount_amount = price * (discount_percent / 100)
    return price - discount_amount


def format_currency(amount: float, symbol: str = "$") -> str:
    """
    Format amount as currency string.
    
    Args:
        amount: Amount to format
        symbol: Currency symbol
        
    Returns:
        Formatted currency string
    """
    return f"{symbol}{amount:.2f}"


def validate_email(email: str) -> bool:
    """
    Validate email address format.
    
    Args:
        email: Email to validate
        
    Returns:
        True if valid, False otherwise
    """
    if not email or "@" not in email:
        return False
    
    parts = email.split("@")
    if len(parts) != 2:
        return False
    
    return len(parts[0]) > 0 and len(parts[1]) > 0


async def fetch_user_data(user_id: int) -> Dict[str, str]:
    """
    Asynchronously fetch user data.
    
    Args:
        user_id: User ID to fetch
        
    Returns:
        User data dictionary
        
    Raises:
        ValueError: If user_id is invalid
    """
    if user_id <= 0:
        raise ValueError("Invalid user ID")
    
    # Simulate async operation
    return {"id": str(user_id), "name": "Test User"}
