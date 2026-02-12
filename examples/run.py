def calculate_discount(price: float, discount_percent: float) -> float:
    """Calculate discounted price.
    
    Args:
        price: Original price
        discount_percent: Discount percentage (0-100)
    
    Returns:
        Final price after discount
    
    Raises:
        ValueError: If discount_percent is negative or > 100
    
    # Dependencies: logger.log()  # <-- OUTSIDE the docstring!
    """
# 2. Basic string operation
def format_username(first_name: str, last_name: str) -> str:
    """Format user's full name.
    
    Args:
        first_name: User's first name
        last_name: User's last name
    
    Returns:
        Formatted username (lowercase, underscore-separated)
    
    Raises:
        ValueError: If either name is empty

    # Dependencies: validator.check_name()
    """
  # 3. Simple list operation
def get_active_users(user_ids: list[int]) -> list[dict]:
    """Get active users by IDs.
    
    Args:
        user_ids: List of user IDs
    
    Returns:
        List of active user dictionaries
    
    Raises:
        ValueError: If user_ids is empty

    # Dependencies: database.query()
    """
    

# 4. Multiple dependencies with conditions
def process_payment(order_id: str, amount: float, method: str) -> dict:
    """Process payment for an order.
    
    Args:
        order_id: Order identifier
        amount: Payment amount
        method: Payment method ('card' or 'paypal')
    
    Returns:
        Payment result with transaction_id and status
    
    Raises:
        InvalidOrderError: If order not found
        PaymentFailedError: If payment processing fails
        ValueError: If amount is negative or method is invalid

    # Dependencies: payment_gateway.charge(), order_service.get_order(), notification.send()
    """
# 5. Async function with caching
async def fetch_product_details(product_id: int, use_cache: bool = True) -> dict:
    """Fetch product details with optional caching.
    
    Args:
        product_id: Product identifier
        use_cache: Whether to check cache first
    
    Returns:
        Product details dictionary
    
    Raises:
        ProductNotFoundError: If product doesn't exist
        CacheError: If cache operation fails

    # Dependencies: cache.get(), database.fetch_product()
    """

# 6. File processing with validation
def parse_csv_report(file_path: str, skip_errors: bool = False) -> list[dict]:
    """Parse CSV report file.
    
    Args:
        file_path: Path to CSV file
        skip_errors: Whether to skip invalid rows
    
    Returns:
        List of parsed row dictionaries
    
    Raises:
        FileNotFoundError: If file doesn't exist
        ValidationError: If CSV format is invalid and skip_errors is False

    # Dependencies: file_reader.read(), validator.validate_row()
    """

# 7. Multiple return scenarios
def create_user_account(email: str, password: str, role: str = "user") -> dict:
    """Create new user account.
    
    Args:
        email: User email address
        password: User password (min 8 characters)
        role: User role ('user' or 'admin')
    
    Returns:
        Created user data with user_id and created_at
    
    Raises:
        UserExistsError: If email already registered
        WeakPasswordError: If password doesn't meet requirements
        InvalidRoleError: If role is not valid

      # Dependencies: database.insert(), email_service.send_welcome(), password_validator.check()
    """


# 8. Complex async with multiple error paths
async def sync_user_data(user_id: int, source: str, target: str, retry: int = 3) -> dict:
    """Sync user data between systems with retry logic.
    
    Args:
        user_id: User identifier
        source: Source system name ('crm', 'warehouse', 'analytics')
        target: Target system name
        retry: Number of retry attempts
    
    Returns:
        Sync result with synced_fields count and timestamp
    
    Raises:
        UserNotFoundError: If user doesn't exist in source
        SyncFailedError: If sync fails after all retries
        ConnectionError: If unable to connect to systems
        DataConflictError: If data conflicts between systems

    # Dependencies: source_api.get_data(), target_api.update_data(), conflict_resolver.resolve(), logger.log()
    """
# 9. Batch processing with transactions
async def process_batch_orders(order_ids: list[str], priority: bool = False) -> dict:
    """Process multiple orders in a batch with transaction support.
    
    Args:
        order_ids: List of order identifiers (max 100)
        priority: Whether to process as high priority
    
    Returns:
        Result dict with successful, failed, and skipped order counts
    
    Raises:
        BatchSizeError: If order_ids exceeds 100
        TransactionError: If database transaction fails
        InventoryError: If inventory insufficient for any order
        PaymentError: If payment processing fails for any order

    # Dependencies: transaction.begin(), inventory.reserve_batch(), payment.charge_batch(), notification.send_batch()
    """
# 10. Complex conditional logic with external APIs
async def generate_report(report_type: str, filters: dict, export_format: str = "pdf") -> str:
    """Generate analytical report with data aggregation.
    
    Args:
        report_type: Type of report ('sales', 'inventory', 'users')
        filters: Filter criteria (date_range, categories, regions)
        export_format: Output format ('pdf', 'excel', 'csv')
    
    Returns:
        File path of generated report
    
    Raises:
        InvalidReportTypeError: If report_type is not supported
        InvalidFilterError: If required filters are missing
        DataFetchError: If unable to fetch data from analytics
        ExportError: If report generation fails
        StorageError: If unable to save report file
    # Dependencies: analytics_api.fetch_data(), data_processor.aggregate(), report_generator.create(), storage.save()
    """
