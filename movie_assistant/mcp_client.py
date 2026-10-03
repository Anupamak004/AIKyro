from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def _send_email(to: str, subject: str, body: str) -> str:
    load_dotenv()
    server = StdioServerParameters(
        command=sys.executable,
        args=["-m", "movie_assistant.email_mcp_server"],
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parent.parent)},
    )
    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool("send_email", {"to": to, "subject": subject, "body": body})
            if result.isError:
                message = " ".join(getattr(item, "text", "") for item in result.content).strip()
                raise RuntimeError(message or "MCP email tool returned an error.")
            return " ".join(getattr(item, "text", "") for item in result.content).strip() or "Email sent."


def send_email(to: str, subject: str, body: str) -> str:
    """Start a fresh stdio MCP session and invoke its send_email tool."""
    return asyncio.run(_send_email(to, subject, body))
