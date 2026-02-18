"""
HTTP Fetcher — tests Stage 2 (RuntimeInspector / pip-installed packages).

Run with:
    python pytest_generator.py examples/http_fetcher.py -o examples/generated_tests/

httpx.AsyncClient and requests.Session are pip packages — the tool imports them at
runtime with importlib and uses inspect.signature() to get real method names and
argument names, then injects them as # Dependencies: comments.
"""

import httpx
import requests
from typing import Optional


# ── async HTTP functions (httpx.AsyncClient) ─────────────────────────────────

async def fetch_user(client: httpx.AsyncClient, user_id: int) -> dict:
    """Fetch a user profile from the API.

    Args:
        client: Async HTTP client
        user_id: User ID to look up

    Returns:
        User dict with at least 'id' and 'name'

    Raises:
        ValueError: If user_id is not positive
        httpx.HTTPStatusError: If the API returns 4xx/5xx
    """
    if user_id <= 0:
        raise ValueError("user_id must be positive")
    resp = await client.get(f"/users/{user_id}")
    resp.raise_for_status()
    return resp.json()


async def create_user(client: httpx.AsyncClient, name: str, email: str) -> dict:
    """Create a new user via POST.

    Args:
        client: Async HTTP client
        name: Display name for the new user
        email: Email address (must be non-empty)

    Returns:
        Created user dict including the new 'id'

    Raises:
        ValueError: If name or email is empty
        httpx.HTTPStatusError: If creation fails
    """
    if not name or not email:
        raise ValueError("name and email are required")
    resp = await client.post("/users", json={"name": name, "email": email})
    resp.raise_for_status()
    return resp.json()


async def delete_user(client: httpx.AsyncClient, user_id: int) -> bool:
    """Delete a user by ID.

    Args:
        client: Async HTTP client
        user_id: ID of the user to delete

    Returns:
        True if deleted successfully

    Raises:
        ValueError: If user_id is not positive
        httpx.HTTPStatusError: On 4xx/5xx responses
    """
    if user_id <= 0:
        raise ValueError("user_id must be positive")
    resp = await client.delete(f"/users/{user_id}")
    resp.raise_for_status()
    return resp.status_code == 204


# ── sync HTTP functions (requests.Session) ───────────────────────────────────

def get_product(session: requests.Session, product_id: int) -> dict:
    """Fetch a product from a REST API using a persistent session.

    Args:
        session: Requests session (handles auth headers, cookies, retries)
        product_id: Product ID to look up

    Returns:
        Product dict

    Raises:
        ValueError: If product_id is not positive
        requests.HTTPError: If the server returns an error status
    """
    if product_id <= 0:
        raise ValueError("product_id must be positive")
    resp = session.get(f"/products/{product_id}")
    resp.raise_for_status()
    return resp.json()


def search_products(
    session: requests.Session,
    query: str,
    max_results: int = 20,
) -> list:
    """Search products by keyword.

    Args:
        session: Requests session
        query: Search keyword (non-empty)
        max_results: Maximum number of results to return (1-100)

    Returns:
        List of matching product dicts

    Raises:
        ValueError: If query is empty or max_results is out of range
        requests.HTTPError: On API error
    """
    if not query:
        raise ValueError("query must be non-empty")
    if not (1 <= max_results <= 100):
        raise ValueError("max_results must be between 1 and 100")
    resp = session.get("/products/search", params={"q": query, "limit": max_results})
    resp.raise_for_status()
    return resp.json().get("results", [])
