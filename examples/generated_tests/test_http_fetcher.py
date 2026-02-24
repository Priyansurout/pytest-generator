"""
Generated tests for http_fetcher.py
Created by pytest-generator
"""
import pytest
from unittest.mock import MagicMock, AsyncMock
import importlib

module = importlib.import_module("http_fetcher")
fetch_user = getattr(module, "fetch_user")
create_user = getattr(module, "create_user")
delete_user = getattr(module, "delete_user")
get_product = getattr(module, "get_product")
search_products = getattr(module, "search_products")

httpx = importlib.import_module("httpx")
HTTPStatusError = getattr(httpx, "HTTPStatusError")

requests = importlib.import_module("requests")
HTTPError = getattr(requests, "HTTPError")


# Test 1: fetch_user (async)
# Dependencies: client.get()
@pytest.fixture
def client():
    mock = MagicMock()
    mock.get = AsyncMock()
    mock.post = AsyncMock()
    mock.delete = AsyncMock()
    return mock

@pytest.fixture
def session():
    return MagicMock()


@pytest.mark.asyncio
async def test_fetch_user_success(client):
    resp = MagicMock()
    resp.json.return_value = {"id": 1, "name": "Alice"}
    client.get.return_value = resp

    result = await fetch_user(client, 1)

    assert result == {"id": 1, "name": "Alice"}
    client.get.assert_called_once_with("/users/1")
    resp.raise_for_status.assert_called_once()

@pytest.mark.asyncio
async def test_fetch_user_invalid_id(client):
    with pytest.raises(ValueError):
        await fetch_user(client, 0)
    with pytest.raises(ValueError):
        await fetch_user(client, -1)

@pytest.mark.asyncio
async def test_fetch_user_http_error(client):
    resp = MagicMock()
    resp.raise_for_status.side_effect = HTTPStatusError("Not found", request=MagicMock(), response=MagicMock())
    client.get.return_value = resp

    with pytest.raises(HTTPStatusError):
        await fetch_user(client, 999)


# Test 2: create_user (async)
# Dependencies: client.post()
@pytest.mark.asyncio
async def test_create_user_success(client):
    resp = MagicMock()
    resp.json.return_value = {"id": 10, "name": "Bob", "email": "bob@test.com"}
    client.post.return_value = resp

    result = await create_user(client, "Bob", "bob@test.com")

    assert result["id"] == 10
    assert result["name"] == "Bob"
    client.post.assert_called_once_with("/users", json={"name": "Bob", "email": "bob@test.com"})

@pytest.mark.asyncio
async def test_create_user_empty_name(client):
    with pytest.raises(ValueError):
        await create_user(client, "", "bob@test.com")

@pytest.mark.asyncio
async def test_create_user_empty_email(client):
    with pytest.raises(ValueError):
        await create_user(client, "Bob", "")


# Test 3: delete_user (async)
# Dependencies: client.delete()
@pytest.mark.asyncio
async def test_delete_user_success(client):
    resp = MagicMock()
    resp.status_code = 204
    client.delete.return_value = resp

    result = await delete_user(client, 1)

    assert result is True
    client.delete.assert_called_once_with("/users/1")

@pytest.mark.asyncio
async def test_delete_user_invalid_id(client):
    with pytest.raises(ValueError):
        await delete_user(client, 0)


# Test 4: get_product (sync)
# Dependencies: session.get()
def test_get_product_success(session):
    resp = MagicMock()
    resp.json.return_value = {"id": 5, "name": "Widget"}
    session.get.return_value = resp

    result = get_product(session, 5)

    assert result == {"id": 5, "name": "Widget"}
    session.get.assert_called_once_with("/products/5")
    resp.raise_for_status.assert_called_once()

def test_get_product_invalid_id(session):
    with pytest.raises(ValueError):
        get_product(session, 0)

def test_get_product_http_error(session):
    resp = MagicMock()
    resp.raise_for_status.side_effect = HTTPError("Server error")
    session.get.return_value = resp

    with pytest.raises(HTTPError):
        get_product(session, 1)


# Test 5: search_products (sync)
# Dependencies: session.get()
def test_search_products_success(session):
    resp = MagicMock()
    resp.json.return_value = {"results": [{"id": 1, "name": "Widget"}, {"id": 2, "name": "Gadget"}]}
    session.get.return_value = resp

    result = search_products(session, "wid", max_results=10)

    assert len(result) == 2
    session.get.assert_called_once_with("/products/search", params={"q": "wid", "limit": 10})

def test_search_products_empty_query(session):
    with pytest.raises(ValueError):
        search_products(session, "")

@pytest.mark.parametrize("max_results", [0, -1, 101, 200])
def test_search_products_invalid_max_results(session, max_results):
    with pytest.raises(ValueError):
        search_products(session, "test", max_results=max_results)
