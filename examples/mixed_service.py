"""
Mixed Service — tests BOTH resolution paths in one file.

Run with:
    python pytest_generator.py examples/mixed_service.py -o examples/generated_tests/

Expected behaviour:
  - NotificationSender (local class)  → resolved via AST codebase index (Stage 1)
  - httpx.AsyncClient (pip package)   → resolved via RuntimeInspector (Stage 2)
  - boto3 S3 client (pip package)     → resolved via RuntimeInspector (Stage 2)
"""

import httpx
import boto3
from typing import Optional


# ── local class the indexer will pick up ─────────────────────────────────────

class NotificationSender:
    """Sends email / push notifications to users."""

    def send_email(self, to: str, subject: str, body: str) -> bool:
        """Send an email. Returns True on success."""
        raise NotImplementedError

    def send_push(self, user_id: int, message: str) -> bool:
        """Send a push notification. Returns True on success."""
        raise NotImplementedError

    def send_sms(self, phone: str, message: str) -> bool:
        """Send an SMS. Returns True on success."""
        raise NotImplementedError


# ── functions that mix local + pip dependencies ───────────────────────────────

async def register_user_and_notify(
    client: httpx.AsyncClient,
    notifier: NotificationSender,
    name: str,
    email: str,
) -> dict:
    """Create a user via the API and send a welcome email.

    Args:
        client: Async HTTP client for the user API
        notifier: NotificationSender for welcome messages
        name: New user's display name
        email: New user's email address

    Returns:
        Created user dict

    Raises:
        ValueError: If name or email is empty
        httpx.HTTPStatusError: If user creation fails
        RuntimeError: If welcome email cannot be sent
    """
    if not name or not email:
        raise ValueError("name and email are required")

    resp = await client.post("/users", json={"name": name, "email": email})
    resp.raise_for_status()
    user = resp.json()

    sent = notifier.send_email(
        email,
        "Welcome!",
        f"Hi {name}, your account is ready.",
    )
    if not sent:
        raise RuntimeError("Failed to send welcome email")

    return user


def upload_report(
    s3_client: boto3.client,
    notifier: NotificationSender,
    bucket: str,
    key: str,
    data: bytes,
    recipient_email: str,
) -> str:
    """Upload a report to S3 and notify the recipient.

    Args:
        s3_client: Boto3 S3 client
        notifier: NotificationSender for delivery notification
        bucket: S3 bucket name
        key: S3 object key (path)
        data: Raw bytes to upload
        recipient_email: Email address to notify on success

    Returns:
        Public URL of the uploaded object

    Raises:
        ValueError: If bucket, key, or data is empty
        RuntimeError: If upload or notification fails
    """
    if not bucket or not key or not data:
        raise ValueError("bucket, key, and data are required")

    s3_client.put_object(Bucket=bucket, Key=key, Body=data)
    url = f"https://{bucket}.s3.amazonaws.com/{key}"

    sent = notifier.send_email(
        recipient_email,
        "Report ready",
        f"Your report is available at: {url}",
    )
    if not sent:
        raise RuntimeError("Report uploaded but notification failed")

    return url
