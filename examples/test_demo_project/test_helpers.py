"""
Generated tests for helpers.py
Created by pytest-generator

NOTE: This is unedited raw model output. It contains known 8B model
limitations (incorrect expected values) that require developer review
and refinement before use.
"""
# Test 1: calculate_discount
import pytest
import importlib

module = importlib.import_module("helpers")
calculate_discount = getattr(module, "calculate_discount")


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
import pytest
import importlib

module = importlib.import_module("helpers")
format_currency = getattr(module, "format_currency")


@pytest.mark.parametrize("amount,symbol,expected", [
    (100.0, "$", "$100.00"),
    (1234.56, "€", "€1,234.56"),
    (0.0, "£", "£0.00"),
])
def test_format_currency(amount, symbol, expected):
    result = format_currency(amount, symbol)
    assert result == expected

@pytest.mark.parametrize("amount,symbol", [
    (100.0, "$"),
    (50.0, "€"),
    (75.0, "£"),
])
def test_format_currency_default_symbol(amount, symbol):
    result = format_currency(amount, symbol)
    assert result == f"{symbol}{amount:.2f}"

# Test 3: validate_email
import pytest
import importlib

module = importlib.import_module("helpers")
validate_email = getattr(module, "validate_email")


@pytest.mark.parametrize("email,expected", [
    ("test@example.com", True),
    ("user.name@domain.co.uk", True),
    ("invalid-email", False),
    ("missing@tld.", False),
    ("@domain.com", False),
])
def test_validate_email(email, expected):
    result = validate_email(email)
    assert result == expected

# Test 4: fetch_user_data
import pytest
import importlib
from typing import Dict

module = importlib.import_module("helpers")
fetch_user_data = getattr(module, "fetch_user_data")
ValueError = getattr(module, "ValueError", Exception)


@pytest.mark.asyncio
@pytest.mark.parametrize("user_id,expected", [
    (1, {"id": 1, "name": "Alice"}),
    (2, {"id": 2, "name": "Bob"}),
    (100, {"id": 100, "name": "Charlie"}),
])
async def test_fetch_user_data(user_id, expected):
    result = await fetch_user_data(user_id)
    assert result == expected

@pytest.mark.asyncio
async def test_fetch_user_data_invalid_id():
    with pytest.raises(ValueError):
        await fetch_user_data(-1)
