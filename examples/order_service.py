"""
Order Service — tests Stage 1 (AST codebase index / local classes).

Run with:
    python pytest_generator.py examples/order_service.py -o examples/generated_tests/

The tool will scan the project, find OrderRepository and PaymentProcessor in this
file, and inject real method names into the # Dependencies: comment before inference.
"""

from typing import Optional


# ── local classes that the indexer will pick up ──────────────────────────────

class OrderRepository:
    """Thin data-access layer for orders."""

    def get_order(self, order_id: int) -> dict:
        """Fetch a single order by ID. Returns {} if not found."""
        raise NotImplementedError

    def create_order(self, user_id: int, items: list) -> dict:
        """Persist a new order and return it with its generated ID."""
        raise NotImplementedError

    def update_status(self, order_id: int, status: str) -> bool:
        """Update order status. Returns True on success, False if not found."""
        raise NotImplementedError

    def delete_order(self, order_id: int) -> bool:
        """Hard-delete an order. Returns True if deleted, False if not found."""
        raise NotImplementedError


class PaymentProcessor:
    """Handles payment operations against an external gateway."""

    def charge(self, amount: float, currency: str, card_token: str) -> dict:
        """Charge a card. Returns transaction dict with 'id' and 'status'."""
        raise NotImplementedError

    def refund(self, transaction_id: str, amount: float) -> dict:
        """Issue a full or partial refund. Returns refund confirmation dict."""
        raise NotImplementedError

    def validate_card(self, card_token: str) -> bool:
        """Check whether a card token is valid and chargeable."""
        raise NotImplementedError


# ── business logic functions (what the generator will target) ─────────────────

def get_order_details(repo: OrderRepository, order_id: int) -> dict:
    """Return order details, raising ValueError if not found.

    Args:
        repo: OrderRepository instance
        order_id: ID of the order to look up

    Returns:
        Order dict

    Raises:
        ValueError: If order_id is not positive
        KeyError: If order does not exist
    """
    if order_id <= 0:
        raise ValueError("order_id must be positive")
    order = repo.get_order(order_id)
    if not order:
        raise KeyError(f"Order {order_id} not found")
    return order


def place_order(
    repo: OrderRepository,
    payment: PaymentProcessor,
    user_id: int,
    items: list,
    card_token: str,
) -> dict:
    """Validate card, charge it, then persist the order.

    Args:
        repo: OrderRepository instance
        payment: PaymentProcessor instance
        user_id: ID of the user placing the order
        items: List of item dicts with 'product_id' and 'qty'
        card_token: Payment card token

    Returns:
        Created order dict with payment transaction embedded

    Raises:
        ValueError: If items list is empty or card is invalid
        RuntimeError: If payment charge fails
    """
    if not items:
        raise ValueError("Order must contain at least one item")
    if not payment.validate_card(card_token):
        raise ValueError("Invalid card token")

    total = sum(item.get("price", 0) * item.get("qty", 1) for item in items)
    txn = payment.charge(total, "USD", card_token)
    if txn.get("status") != "success":
        raise RuntimeError(f"Payment failed: {txn}")

    order = repo.create_order(user_id, items)
    order["transaction"] = txn
    return order


def cancel_order(
    repo: OrderRepository,
    payment: PaymentProcessor,
    order_id: int,
) -> bool:
    """Cancel an order and refund the charge if it was already paid.

    Args:
        repo: OrderRepository instance
        payment: PaymentProcessor instance
        order_id: ID of the order to cancel

    Returns:
        True if successfully cancelled

    Raises:
        KeyError: If order does not exist
        RuntimeError: If refund fails
    """
    order = repo.get_order(order_id)
    if not order:
        raise KeyError(f"Order {order_id} not found")

    if order.get("status") == "paid" and order.get("transaction_id"):
        refund = payment.refund(order["transaction_id"], order["total"])
        if refund.get("status") != "success":
            raise RuntimeError("Refund failed")

    return repo.update_status(order_id, "cancelled")
