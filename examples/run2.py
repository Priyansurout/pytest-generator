"""
Comprehensive Test Suite for pytest-generator
Contains 10 functions: Easy (3), Medium (4), Hard (3)
"""
import asyncio
from typing import List, Dict, Optional, Union, Tuple
from datetime import datetime, timedelta
import json


# ============================================================================
# EASY LEVEL (3 functions)
# ============================================================================

def celsius_to_fahrenheit(celsius: float) -> float:
    """Convert Celsius to Fahrenheit.
    
    Args:
        celsius: Temperature in Celsius
        
    Returns:
        Temperature in Fahrenheit
        
    Raises:
        ValueError: If celsius is below absolute zero (-273.15)
    """
    if celsius < -273.15:
        raise ValueError("Temperature cannot be below absolute zero")
    return (celsius * 9/5) + 32


def is_palindrome(text: str) -> bool:
    """Check if a string is a palindrome (case-insensitive).
    
    Args:
        text: String to check
        
    Returns:
        True if palindrome, False otherwise
        
    Raises:
        TypeError: If input is not a string
        ValueError: If string is empty
    """
    if not isinstance(text, str):
        raise TypeError("Input must be a string")
    if not text:
        raise ValueError("String cannot be empty")
    
    cleaned = text.lower().replace(" ", "")
    return cleaned == cleaned[::-1]


def find_max(numbers: List[Union[int, float]]) -> Union[int, float]:
    """Find the maximum value in a list of numbers.
    
    Args:
        numbers: List of numbers
        
    Returns:
        Maximum value
        
    Raises:
        ValueError: If list is empty
        TypeError: If list contains non-numeric values
    """
    if not numbers:
        raise ValueError("List cannot be empty")
    
    for num in numbers:
        if not isinstance(num, (int, float)):
            raise TypeError("All elements must be numbers")
    
    return max(numbers)


# ============================================================================
# MEDIUM LEVEL (4 functions)
# ============================================================================

def parse_json_config(config_string: str, strict: bool = False) -> dict:
    """Parse JSON configuration string with error handling.
    
    Args:
        config_string: JSON string to parse
        strict: If True, raise error on invalid JSON. If False, return empty dict.
        
    Returns:
        Parsed configuration dictionary
        
    Raises:
        ValueError: If JSON is invalid and strict=True
        TypeError: If input is not a string
        
    # Dependencies: json.loads()
    """
    if not isinstance(config_string, str):
        raise TypeError("Config must be a string")
    
    try:
        config = json.loads(config_string)
        return config
    except json.JSONDecodeError as e:
        if strict:
            raise ValueError(f"Invalid JSON: {e}")
        return {}


async def fetch_user_profile(user_id: int, include_posts: bool = False) -> dict:
    """Fetch user profile asynchronously with optional posts.
    
    Args:
        user_id: User identifier
        include_posts: Whether to include user's posts
        
    Returns:
        User profile dictionary with optional posts
        
    Raises:
        ValueError: If user_id is negative
        UserNotFoundError: If user doesn't exist
        DatabaseError: If database query fails
        
    # Dependencies: database.get_user(), database.get_posts()
    """
    if user_id < 0:
        raise ValueError("User ID must be positive")
    
    # Simulate async database call
    await asyncio.sleep(0.1)
    
    # Mock user data (in real implementation, this would be from database)
    user = {
        "id": user_id,
        "name": f"User{user_id}",
        "email": f"user{user_id}@example.com"
    }
    
    if include_posts:
        user["posts"] = [
            {"id": 1, "title": "Post 1"},
            {"id": 2, "title": "Post 2"}
        ]
    
    return user


def calculate_shopping_cart(
    items: List[Dict[str, Union[str, float, int]]],
    discount_code: Optional[str] = None,
    tax_rate: float = 0.1
) -> Dict[str, float]:
    """Calculate shopping cart totals with discounts and tax.
    
    Args:
        items: List of items with 'price' and 'quantity' keys
        discount_code: Optional discount code ('SAVE10', 'SAVE20', 'SAVE50')
        tax_rate: Tax rate (default 0.1 for 10%)
        
    Returns:
        Dictionary with subtotal, discount, tax, and total
        
    Raises:
        ValueError: If items list is empty or invalid discount code
        KeyError: If items missing required keys
        
    # Dependencies: discount_validator.validate()
    """
    if not items:
        raise ValueError("Cart cannot be empty")
    
    # Calculate subtotal
    subtotal = 0
    for item in items:
        if 'price' not in item or 'quantity' not in item:
            raise KeyError("Items must have 'price' and 'quantity'")
        subtotal += item['price'] * item['quantity']
    
    # Apply discount
    discount = 0
    if discount_code:
        if discount_code == 'SAVE10':
            discount = subtotal * 0.10
        elif discount_code == 'SAVE20':
            discount = subtotal * 0.20
        elif discount_code == 'SAVE50':
            discount = subtotal * 0.50
        else:
            raise ValueError(f"Invalid discount code: {discount_code}")
    
    # Calculate tax
    taxable_amount = subtotal - discount
    tax = taxable_amount * tax_rate
    
    # Calculate total
    total = taxable_amount + tax
    
    return {
        "subtotal": round(subtotal, 2),
        "discount": round(discount, 2),
        "tax": round(tax, 2),
        "total": round(total, 2)
    }


def validate_email_batch(
    emails: List[str],
    allow_duplicates: bool = False
) -> Tuple[List[str], List[str]]:
    """Validate a batch of email addresses.
    
    Args:
        emails: List of email addresses to validate
        allow_duplicates: If False, mark duplicates as invalid
        
    Returns:
        Tuple of (valid_emails, invalid_emails)
        
    Raises:
        TypeError: If emails is not a list
        ValueError: If emails list is empty
        
    # Dependencies: email_validator.is_valid()
    """
    if not isinstance(emails, list):
        raise TypeError("Emails must be a list")
    if not emails:
        raise ValueError("Email list cannot be empty")
    
    valid = []
    invalid = []
    seen = set()
    
    for email in emails:
        # Basic email validation
        if not isinstance(email, str):
            invalid.append(email)
            continue
        
        if '@' not in email or '.' not in email:
            invalid.append(email)
            continue
        
        # Check for duplicates
        if not allow_duplicates and email in seen:
            invalid.append(email)
            continue
        
        seen.add(email)
        valid.append(email)
    
    return (valid, invalid)


# ============================================================================
# HARD LEVEL (3 functions)
# ============================================================================

async def process_payment_transaction(
    order_id: str,
    amount: float,
    payment_method: str,
    customer_id: int,
    retry_on_failure: bool = True,
    max_retries: int = 3
) -> Dict[str, Union[str, float, bool]]:
    """Process payment transaction with retry logic and fraud detection.
    
    Args:
        order_id: Unique order identifier
        amount: Payment amount (must be positive)
        payment_method: Payment method ('card', 'paypal', 'crypto')
        customer_id: Customer identifier
        retry_on_failure: Whether to retry failed transactions
        max_retries: Maximum number of retry attempts
        
    Returns:
        Transaction result with status, transaction_id, and timestamp
        
    Raises:
        ValueError: If amount is negative or payment_method invalid
        FraudDetectedError: If transaction flagged as fraudulent
        PaymentGatewayError: If payment gateway unavailable
        InsufficientFundsError: If customer has insufficient funds
        RetryExhaustedError: If all retry attempts fail
        
    # Dependencies: fraud_detector.check(), payment_gateway.charge(), 
    #               order_service.get_order(), notification_service.send(),
    #               audit_logger.log()
    """
    if amount <= 0:
        raise ValueError("Amount must be positive")
    
    if payment_method not in ['card', 'paypal', 'crypto']:
        raise ValueError(f"Invalid payment method: {payment_method}")
    
    # Simulate async processing
    await asyncio.sleep(0.1)
    
    # Mock successful transaction
    return {
        "status": "success",
        "transaction_id": f"TXN_{order_id}",
        "amount": amount,
        "timestamp": datetime.now().isoformat(),
        "fraud_checked": True
    }


def generate_report_with_filters(
    data_source: str,
    start_date: datetime,
    end_date: datetime,
    filters: Dict[str, List[str]],
    aggregation: str = 'sum',
    group_by: Optional[List[str]] = None,
    export_format: str = 'json'
) -> Union[Dict, str]:
    """Generate analytical report with complex filtering and aggregation.
    
    Args:
        data_source: Data source name ('sales', 'users', 'inventory', 'analytics')
        start_date: Report start date
        end_date: Report end date
        filters: Dictionary of filter criteria (e.g., {'region': ['US', 'EU'], 'category': ['electronics']})
        aggregation: Aggregation method ('sum', 'avg', 'count', 'min', 'max')
        group_by: Optional list of fields to group by
        export_format: Output format ('json', 'csv', 'excel', 'pdf')
        
    Returns:
        Report data in requested format (dict for json, str for others)
        
    Raises:
        ValueError: If date range invalid or data_source not supported
        InvalidFilterError: If filters contain unsupported fields
        DataFetchError: If unable to fetch data from source
        AggregationError: If aggregation method not supported
        ExportError: If unable to export in requested format
        
    # Dependencies: data_warehouse.query(), aggregator.aggregate(),
    #               filter_validator.validate(), exporter.export(),
    #               cache_manager.get(), cache_manager.set()
    """
    if end_date <= start_date:
        raise ValueError("End date must be after start date")
    
    if data_source not in ['sales', 'users', 'inventory', 'analytics']:
        raise ValueError(f"Unsupported data source: {data_source}")
    
    if aggregation not in ['sum', 'avg', 'count', 'min', 'max']:
        raise ValueError(f"Unsupported aggregation: {aggregation}")
    
    if export_format not in ['json', 'csv', 'excel', 'pdf']:
        raise ValueError(f"Unsupported export format: {export_format}")
    
    # Mock report data
    report = {
        "data_source": data_source,
        "date_range": {
            "start": start_date.isoformat(),
            "end": end_date.isoformat()
        },
        "filters": filters,
        "aggregation": aggregation,
        "group_by": group_by,
        "results": [
            {"category": "electronics", "total": 15000},
            {"category": "clothing", "total": 8500}
        ],
        "generated_at": datetime.now().isoformat()
    }
    
    if export_format == 'json':
        return report
    else:
        return f"Report exported as {export_format}"


async def sync_data_across_systems(
    source_system: str,
    target_systems: List[str],
    entity_type: str,
    entity_ids: List[int],
    conflict_resolution: str = 'source_wins',
    batch_size: int = 100,
    enable_rollback: bool = True
) -> Dict[str, Union[int, List[str], bool]]:
    """Synchronize data across multiple systems with conflict resolution.
    
    Args:
        source_system: Source system name ('crm', 'erp', 'warehouse')
        target_systems: List of target system names
        entity_type: Type of entity to sync ('user', 'product', 'order')
        entity_ids: List of entity IDs to synchronize
        conflict_resolution: Strategy ('source_wins', 'target_wins', 'merge', 'manual')
        batch_size: Number of entities per batch (max 1000)
        enable_rollback: Whether to rollback on failure
        
    Returns:
        Sync result with success count, failed IDs, and rollback status
        
    Raises:
        ValueError: If batch_size exceeds limit or invalid system names
        ConnectionError: If unable to connect to any system
        SyncConflictError: If conflicts detected and resolution is 'manual'
        DataIntegrityError: If data validation fails
        RollbackError: If rollback fails after sync failure
        
    # Dependencies: source_api.get_entities(), target_api.update_entities(),
    #               conflict_resolver.resolve(), validator.validate(),
    #               transaction_manager.begin(), transaction_manager.rollback(),
    #               logger.log(), metrics.record()
    """
    if batch_size > 1000:
        raise ValueError("Batch size cannot exceed 1000")
    
    valid_systems = ['crm', 'erp', 'warehouse', 'analytics']
    if source_system not in valid_systems:
        raise ValueError(f"Invalid source system: {source_system}")
    
    for target in target_systems:
        if target not in valid_systems:
            raise ValueError(f"Invalid target system: {target}")
    
    if entity_type not in ['user', 'product', 'order']:
        raise ValueError(f"Invalid entity type: {entity_type}")
    
    if conflict_resolution not in ['source_wins', 'target_wins', 'merge', 'manual']:
        raise ValueError(f"Invalid conflict resolution strategy: {conflict_resolution}")
    
    # Simulate async sync
    await asyncio.sleep(0.2)
    
    # Mock successful sync
    return {
        "synced_count": len(entity_ids),
        "failed_ids": [],
        "conflicts_resolved": 0,
        "rollback_performed": False,
        "timestamp": datetime.now().isoformat()
    }


# Custom exceptions for hard functions
class FraudDetectedError(Exception):
    """Raised when fraudulent transaction detected"""
    pass

class PaymentGatewayError(Exception):
    """Raised when payment gateway is unavailable"""
    pass

class InsufficientFundsError(Exception):
    """Raised when customer has insufficient funds"""
    pass

class RetryExhaustedError(Exception):
    """Raised when all retry attempts exhausted"""
    pass

class UserNotFoundError(Exception):
    """Raised when user not found"""
    pass

class DatabaseError(Exception):
    """Raised when database operation fails"""
    pass

class InvalidFilterError(Exception):
    """Raised when invalid filter provided"""
    pass

class DataFetchError(Exception):
    """Raised when unable to fetch data"""
    pass

class AggregationError(Exception):
    """Raised when aggregation fails"""
    pass

class ExportError(Exception):
    """Raised when export fails"""
    pass

class SyncConflictError(Exception):
    """Raised when sync conflict detected"""
    pass

class DataIntegrityError(Exception):
    """Raised when data validation fails"""
    pass

class RollbackError(Exception):
    """Raised when rollback operation fails"""
    pass


if __name__ == "__main__":
    print("Test suite loaded successfully!")
    print("\nEasy functions: celsius_to_fahrenheit, is_palindrome, find_max")
    print("Medium functions: parse_json_config, fetch_user_profile, calculate_shopping_cart, validate_email_batch")
    print("Hard functions: process_payment_transaction, generate_report_with_filters, sync_data_across_systems")