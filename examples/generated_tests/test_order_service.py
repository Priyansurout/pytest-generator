"""
Generated tests for order_service.py
Created by pytest-generator
"""
import pytest
from unittest.mock import MagicMock
import importlib

module = importlib.import_module("order_service")
get_order_details = getattr(module, "get_order_details")
place_order = getattr(module, "place_order")
cancel_order = getattr(module, "cancel_order")


# Test 1: get_order_details
# Dependencies: repo.get_order()
@pytest.fixture
def repo():
    return MagicMock()

@pytest.fixture
def payment():
    return MagicMock()


def test_get_order_details_success(repo):
    repo.get_order.return_value = {"id": 1, "status": "pending", "total": 99.99}

    result = get_order_details(repo, 1)

    assert result["id"] == 1
    assert result["total"] == 99.99
    repo.get_order.assert_called_once_with(1)

def test_get_order_details_invalid_id(repo):
    with pytest.raises(ValueError):
        get_order_details(repo, 0)
    with pytest.raises(ValueError):
        get_order_details(repo, -5)

def test_get_order_details_not_found(repo):
    repo.get_order.return_value = {}

    with pytest.raises(KeyError):
        get_order_details(repo, 999)


# Test 2: place_order
# Dependencies: repo.create_order(), payment.validate_card(), payment.charge()
def test_place_order_success(repo, payment):
    payment.validate_card.return_value = True
    payment.charge.return_value = {"id": "txn_123", "status": "success"}
    repo.create_order.return_value = {"id": 1, "user_id": 42, "items": [{"product_id": 1, "qty": 2, "price": 25.0}]}

    items = [{"product_id": 1, "qty": 2, "price": 25.0}]
    result = place_order(repo, payment, 42, items, "tok_valid")

    assert result["id"] == 1
    assert result["transaction"]["status"] == "success"
    payment.validate_card.assert_called_once_with("tok_valid")
    payment.charge.assert_called_once_with(50.0, "USD", "tok_valid")
    repo.create_order.assert_called_once_with(42, items)

def test_place_order_empty_items(repo, payment):
    with pytest.raises(ValueError):
        place_order(repo, payment, 42, [], "tok_valid")

def test_place_order_invalid_card(repo, payment):
    payment.validate_card.return_value = False

    with pytest.raises(ValueError):
        place_order(repo, payment, 42, [{"product_id": 1, "qty": 1, "price": 10.0}], "tok_invalid")

def test_place_order_charge_fails(repo, payment):
    payment.validate_card.return_value = True
    payment.charge.return_value = {"id": "txn_fail", "status": "declined"}

    with pytest.raises(RuntimeError):
        place_order(repo, payment, 42, [{"product_id": 1, "qty": 1, "price": 10.0}], "tok_valid")


# Test 3: cancel_order
# Dependencies: repo.get_order(), repo.update_status(), payment.refund()
def test_cancel_order_paid(repo, payment):
    repo.get_order.return_value = {"id": 1, "status": "paid", "transaction_id": "txn_123", "total": 50.0}
    payment.refund.return_value = {"status": "success"}
    repo.update_status.return_value = True

    result = cancel_order(repo, payment, 1)

    assert result is True
    payment.refund.assert_called_once_with("txn_123", 50.0)
    repo.update_status.assert_called_once_with(1, "cancelled")

def test_cancel_order_unpaid(repo, payment):
    repo.get_order.return_value = {"id": 2, "status": "pending"}
    repo.update_status.return_value = True

    result = cancel_order(repo, payment, 2)

    assert result is True
    payment.refund.assert_not_called()
    repo.update_status.assert_called_once_with(2, "cancelled")

def test_cancel_order_not_found(repo, payment):
    repo.get_order.return_value = {}

    with pytest.raises(KeyError):
        cancel_order(repo, payment, 999)

def test_cancel_order_refund_fails(repo, payment):
    repo.get_order.return_value = {"id": 1, "status": "paid", "transaction_id": "txn_123", "total": 50.0}
    payment.refund.return_value = {"status": "failed"}

    with pytest.raises(RuntimeError):
        cancel_order(repo, payment, 1)
