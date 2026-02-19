"""
User service for business logic.
This file demonstrates local class dependencies.
"""

from typing import Optional, List
from app.models.database import User, Order


def create_user(name: str, email: str) -> User:
    """
    Create a new user.
    
    Args:
        name: User's full name
        email: User's email address
        
    Returns:
        Created User object
        
    Raises:
        ValueError: If email is invalid
    """
    if "@" not in email:
        raise ValueError("Invalid email address")
    
    user = User(id=1, name=name, email=email)
    return user


def update_user_email(user: User, new_email: str) -> bool:
    """
    Update a user's email address.
    
    Args:
        user: User object to update
        new_email: New email address
        
    Returns:
        True if update successful
        
    Raises:
        ValueError: If email is invalid
    """
    if "@" not in new_email:
        raise ValueError("Invalid email address")
    
    return user.update_email(new_email)


def get_user_orders(user: User, orders: List[Order]) -> List[Order]:
    """
    Get all orders for a specific user.
    
    Args:
        user: User object
        orders: List of all orders
        
    Returns:
        List of orders belonging to the user
    """
    user_orders = [order for order in orders if order.user_id == user.id]
    return user_orders


def calculate_user_total(user: User, orders: List[Order]) -> float:
    """
    Calculate total amount for all user orders.
    
    Args:
        user: User object
        orders: List of all orders
        
    Returns:
        Total amount
    """
    user_orders = get_user_orders(user, orders)
    total = sum(order.total for order in user_orders)
    return total
