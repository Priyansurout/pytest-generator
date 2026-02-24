"""
Generated tests for user_service.py
Created by pytest-generator
"""
import pytest
from unittest.mock import MagicMock
import importlib

module = importlib.import_module("user_service")
create_user = getattr(module, "create_user")
update_user_email = getattr(module, "update_user_email")
get_user_orders = getattr(module, "get_user_orders")
calculate_user_total = getattr(module, "calculate_user_total")


# Test 1: create_user
@pytest.mark.parametrize("name,email", [
    ("Alice Smith", "alice.smith@example.com"),
    ("Bob Johnson", "bob.johnson@test.org"),
])
def test_create_user(name, email):
    result = create_user(name, email)
    assert result.name == name
    assert result.email == email

def test_create_user_invalid_email():
    with pytest.raises(ValueError):
        create_user("John Doe", "invalid-email")


# Test 2: update_user_email
# Dependencies: user.update_email()
@pytest.fixture
def user():
    return MagicMock()

def test_update_user_email_success(user):
    user.update_email.return_value = True
    result = update_user_email(user, "new@example.com")
    assert result is True
    user.update_email.assert_called_once_with("new@example.com")

def test_update_user_email_invalid_email():
    user = MagicMock()
    with pytest.raises(ValueError):
        update_user_email(user, "invalid-email")


# Test 3: get_user_orders
def test_get_user_orders():
    user = MagicMock()
    user.id = 1

    order1 = MagicMock()
    order1.user_id = 1
    order2 = MagicMock()
    order2.user_id = 2
    order3 = MagicMock()
    order3.user_id = 1

    result = get_user_orders(user, [order1, order2, order3])
    assert len(result) == 2
    assert order1 in result
    assert order3 in result

def test_get_user_orders_no_orders():
    user = MagicMock()
    user.id = 1

    result = get_user_orders(user, [])
    assert result == []

def test_get_user_orders_no_matching_orders():
    user = MagicMock()
    user.id = 99

    order1 = MagicMock()
    order1.user_id = 1

    result = get_user_orders(user, [order1])
    assert result == []


# Test 4: calculate_user_total
def test_calculate_user_total():
    user = MagicMock()
    user.id = 1

    order1 = MagicMock()
    order1.user_id = 1
    order1.total = 100.0
    order2 = MagicMock()
    order2.user_id = 1
    order2.total = 200.0
    order3 = MagicMock()
    order3.user_id = 2
    order3.total = 50.0

    result = calculate_user_total(user, [order1, order2, order3])
    assert result == 300.0

def test_calculate_user_total_no_orders():
    user = MagicMock()
    user.id = 1

    result = calculate_user_total(user, [])
    assert result == 0.0
