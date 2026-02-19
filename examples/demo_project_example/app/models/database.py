"""
Database models for the demo application.
This file demonstrates local class dependencies.
"""

from typing import Optional, List


class User:
    """User model."""
    
    def __init__(self, id: int, name: str, email: str):
        self.id = id
        self.name = name
        self.email = email
    
    def update_email(self, new_email: str) -> bool:
        """Update user email."""
        self.email = new_email
        return True
    
    def get_display_name(self) -> str:
        """Get formatted display name."""
        return f"{self.name} <{self.email}>"


class Order:
    """Order model."""
    
    def __init__(self, id: int, user_id: int, total: float):
        self.id = id
        self.user_id = user_id
        self.total = total
        self.status = "pending"
    
    def update_status(self, status: str) -> None:
        """Update order status."""
        self.status = status
    
    def calculate_tax(self, rate: float) -> float:
        """Calculate tax amount."""
        return self.total * rate
