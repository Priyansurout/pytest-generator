"""
Generated tests for mixed_service.py
Created by pytest-generator
"""
import pytest
from unittest.mock import MagicMock, AsyncMock
import importlib

module = importlib.import_module("mixed_service")
register_user_and_notify = getattr(module, "register_user_and_notify")
upload_report = getattr(module, "upload_report")

httpx = importlib.import_module("httpx")
HTTPStatusError = getattr(httpx, "HTTPStatusError")


# Test 1: register_user_and_notify (async)
# Dependencies: client.post(), notifier.send_email()
@pytest.fixture
def client():
    mock = MagicMock()
    mock.post = AsyncMock()
    return mock

@pytest.fixture
def notifier():
    return MagicMock()

@pytest.fixture
def s3_client():
    return MagicMock()


@pytest.mark.asyncio
async def test_register_user_and_notify_success(client, notifier):
    resp = MagicMock()
    resp.json.return_value = {"id": 1, "name": "Alice", "email": "alice@test.com"}
    client.post.return_value = resp
    notifier.send_email.return_value = True

    result = await register_user_and_notify(client, notifier, "Alice", "alice@test.com")

    assert result["id"] == 1
    client.post.assert_called_once_with("/users", json={"name": "Alice", "email": "alice@test.com"})
    notifier.send_email.assert_called_once_with(
        "alice@test.com",
        "Welcome!",
        "Hi Alice, your account is ready.",
    )

@pytest.mark.asyncio
async def test_register_user_and_notify_empty_name(client, notifier):
    with pytest.raises(ValueError):
        await register_user_and_notify(client, notifier, "", "alice@test.com")

@pytest.mark.asyncio
async def test_register_user_and_notify_empty_email(client, notifier):
    with pytest.raises(ValueError):
        await register_user_and_notify(client, notifier, "Alice", "")

@pytest.mark.asyncio
async def test_register_user_and_notify_http_error(client, notifier):
    resp = MagicMock()
    resp.raise_for_status.side_effect = HTTPStatusError("Failed", request=MagicMock(), response=MagicMock())
    client.post.return_value = resp

    with pytest.raises(HTTPStatusError):
        await register_user_and_notify(client, notifier, "Alice", "alice@test.com")

@pytest.mark.asyncio
async def test_register_user_and_notify_email_fails(client, notifier):
    resp = MagicMock()
    resp.json.return_value = {"id": 1, "name": "Alice"}
    client.post.return_value = resp
    notifier.send_email.return_value = False

    with pytest.raises(RuntimeError):
        await register_user_and_notify(client, notifier, "Alice", "alice@test.com")


# Test 2: upload_report (sync)
# Dependencies: s3_client.put_object(), notifier.send_email()
def test_upload_report_success(s3_client, notifier):
    notifier.send_email.return_value = True

    result = upload_report(s3_client, notifier, "my-bucket", "reports/q1.pdf", b"pdf data", "bob@test.com")

    assert result == "https://my-bucket.s3.amazonaws.com/reports/q1.pdf"
    s3_client.put_object.assert_called_once_with(Bucket="my-bucket", Key="reports/q1.pdf", Body=b"pdf data")
    notifier.send_email.assert_called_once_with(
        "bob@test.com",
        "Report ready",
        "Your report is available at: https://my-bucket.s3.amazonaws.com/reports/q1.pdf",
    )

@pytest.mark.parametrize("bucket,key,data", [
    ("", "key", b"data"),
    ("bucket", "", b"data"),
    ("bucket", "key", b""),
])
def test_upload_report_missing_args(s3_client, notifier, bucket, key, data):
    with pytest.raises(ValueError):
        upload_report(s3_client, notifier, bucket, key, data, "bob@test.com")

def test_upload_report_notification_fails(s3_client, notifier):
    notifier.send_email.return_value = False

    with pytest.raises(RuntimeError):
        upload_report(s3_client, notifier, "my-bucket", "report.pdf", b"data", "bob@test.com")
