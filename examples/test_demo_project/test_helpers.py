"""
Generated tests for helpers.py
Created by pytest-generator
"""
import pytest
import importlib

module = importlib.import_module("helpers")
calculate_discount = getattr(module, "calculate_discount")
format_currency = getattr(module, "format_currency")
validate_email = getattr(module, "validate_email")
fetch_user_data = getattr(module, "fetch_user_data")


# Test 1: calculate_discount
@pytest.mark.parametrize("price,discount_percent,expected", [
    (100.0, 10.0, 90.0),
    (50.0, 20.0, 40.0),
    (200.0, 0.0, 200.0),
])
def test_calculate_discount(price, discount_percent, expected):
    result = calculate_discount(price, discount_percent)
    assert result == expected

def test_calculate_discount_negative_discount():
    with pytest.raises(ValueError):
        calculate_discount(100.0, -5.0)

def test_calculate_discount_over_100_discount():
    with pytest.raises(ValueError):
        calculate_discount(100.0, 105.0)


# Test 2: format_currency
@pytest.mark.parametrize("amount,symbol,expected", [
    (100.0, "$", "$100.00"),
    (1234.56, "€", "€1234.56"),
    (0.0, "£", "£0.00"),
])
def test_format_currency(amount, symbol, expected):
    result = format_currency(amount, symbol)
    assert result == expected

def test_format_currency_default_symbol():
    result = format_currency(42.5)
    assert result == "$42.50"


# Test 3: validate_email
@pytest.mark.parametrize("email,expected", [
    ("test@example.com", True),
    ("user.name@domain.co.uk", True),
    ("invalid-email", False),
    ("@domain.com", False),
    ("", False),
])
def test_validate_email(email, expected):
    result = validate_email(email)
    assert result == expected


# Test 4: fetch_user_data
@pytest.mark.asyncio
@pytest.mark.parametrize("user_id", [1, 2, 100])
async def test_fetch_user_data(user_id):
    result = await fetch_user_data(user_id)
    assert result == {"id": str(user_id), "name": "Test User"}

@pytest.mark.asyncio
async def test_fetch_user_data_invalid_id():
    with pytest.raises(ValueError):
        await fetch_user_data(-1)

@pytest.mark.asyncio
async def test_fetch_user_data_zero_id():
    with pytest.raises(ValueError):
        await fetch_user_data(0)
