"""
Generated tests for user_service.py
Created by pytest-generator

NOTE: This is unedited raw model output. It contains known 8B model
limitations (wrong constructor args, fabricated assertions) that
require developer review and refinement before use.
"""
# Test 1: create_user
import pytest
import importlib

module = importlib.import_module("user_service")
create_user = getattr(module, "create_user")
ValueError = getattr(module, "ValueError", Exception)


@pytest.mark.parametrize("name,email,expected", [
    ("Alice Smith", "alice.smith@example.com", {"name": "Alice Smith", "email": "alice.smith@example.com"}),
    ("Bob Johnson", "bob.johnson@test.org", {"name": "Bob Johnson", "email": "bob.johnson@test.org"}),
    ("", "user@domain.com", {"name": "", "email": "user@domain.com"}),
])
def test_create_user(name, email, expected):
    result = create_user(name, email)
    assert result == expected

def test_create_user_invalid_email():
    with pytest.raises(ValueError):
        create_user("John Doe", "invalid-email")

# Test 2: update_user_email
import pytest
import importlib
from unittest.mock import MagicMock

module = importlib.import_module("user_service")
update_user_email = getattr(module, "update_user_email")
User = getattr(module, "User", object)
ValueError = getattr(module, "ValueError", Exception)


@pytest.fixture
def user():
    return MagicMock(spec=User)

@pytest.mark.parametrize("user_instance,new_email,expected", [
    (MagicMock(), "valid@email.com", True),
    (MagicMock(), "another@domain.org", True),
])
def test_update_user_email_success(user, new_email, expected):
    user.update_email.return_value = expected
    result = update_user_email(user, new_email)
    assert result == expected
    user.update_email.assert_called_once_with(new_email)

def test_update_user_email_invalid_email():
    user = MagicMock()
    with pytest.raises(ValueError):
        update_user_email(user, "invalid-email")

# Test 3: get_user_orders
import pytest
import importlib
from unittest.mock import MagicMock

module = importlib.import_module("user_service")
get_user_orders = getattr(module, "get_user_orders")
User = getattr(module, "User")
Order = getattr(module, "Order")


@pytest.fixture
def user():
    mock = MagicMock()
    mock.__init__ = MagicMock()
    mock.update_email = MagicMock()
    mock.get_display_name = MagicMock(return_value="Alice")
    return mock

@pytest.fixture
def orders():
    return [
        Order(user_id=1, product="Book"),
        Order(user_id=2, product="Laptop"),
        Order(user_id=1, product="Phone"),
    ]

def test_get_user_orders():
    user = User()
    user.__init__ = MagicMock()
    user.get_display_name = MagicMock(return_value="Alice")
    
    orders = [
        Order(user_id=1, product="Book"),
        Order(user_id=2, product="Laptop"),
        Order(user_id=1, product="Phone"),
    ]
    
    result = get_user_orders(user, orders)
    assert len(result) == 2
    assert result[0].product == "Book"
    assert result[1].product == "Phone"

def test_get_user_orders_no_orders():
    user = User()
    user.__init__ = MagicMock()
    user.get_display_name = MagicMock(return_value="Bob")
    
    orders = []
    
    result = get_user_orders(user, orders)
    assert result == []

def test_get_user_orders_with_update_email():
    user = User()
    user.__init__ = MagicMock()
    user.update_email = MagicMock()
    user.get_display_name = MagicMock(return_value="Charlie")
    
    orders = [
        Order(user_id=3, product="Tablet"),
    ]
    
    result = get_user_orders(user, orders)
    assert len(result) == 1
    user.update_email.assert_called_once()

# Test 4: calculate_user_total
import pytest
import importlib

module = importlib.import_module("user_service")
calculate_user_total = getattr(module, "calculate_user_total")
User = getattr(module, "User")
Order = getattr(module, "Order")
NotFoundError = getattr(module, "NotFoundError", Exception)


@pytest.fixture
def user():
    mock = MagicMock()
    mock.__init__ = MagicMock()
    mock.update_email = MagicMock()
    mock.get_display_name = MagicMock(return_value="Alice")
    return mock

@pytest.fixture
def orders():
    return [MagicMock(), MagicMock()]

@pytest.mark.parametrize("orders_total,expected", [
    ([{"amount": 100}, {"amount": 200}], 300.0),
    ([{"amount": 50}], 50.0),
    ([], 0.0),
])
def test_calculate_user_total(user, orders, orders_total, expected):
    for order in orders:
        order.amount = orders_total[0]["amount"] if len(orders_total) == 1 else orders_total[1]["amount"]
    result = calculate_user_total(user, orders)
    assert result == expected

def test_calculate_user_total_not_found():
    user.get_display_name.side_effect = NotFoundError("User not found")
    with pytest.raises(NotFoundError):
        calculate_user_total(user, [])
