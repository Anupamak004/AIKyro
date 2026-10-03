"""MCP stdio server exposing a single SMTP email dispatch tool."""
from __future__ import annotations

import os
import re
import smtplib
from email.message import EmailMessage

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("movie-email")


@mcp.tool()
def send_email(to: str, subject: str, body: str) -> str:
    """Send a movie-scene summary email using the configured SMTP account."""
    if not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", to):
        raise ValueError("A valid recipient email address is required.")
    host = os.getenv("SMTP_HOST")
    username = os.getenv("SMTP_USERNAME")
    password = os.getenv("SMTP_PASSWORD")
    sender = os.getenv("SMTP_FROM") or username
    if not all((host, username, password, sender)):
        raise RuntimeError("SMTP is not configured. Set SMTP_HOST, SMTP_USERNAME, SMTP_PASSWORD, and SMTP_FROM.")
    message = EmailMessage()
    message["From"], message["To"], message["Subject"] = sender, to, subject
    message.set_content(body)
    port = int(os.getenv("SMTP_PORT", "587"))
    use_tls = os.getenv("SMTP_USE_TLS", "true").lower() in {"1", "true", "yes"}
    with smtplib.SMTP(host, port, timeout=30) as server:
        if use_tls:
            server.starttls()
        server.login(username, password)
        server.send_message(message)
    return f"Email sent to {to}."


if __name__ == "__main__":
    mcp.run(transport="stdio")
