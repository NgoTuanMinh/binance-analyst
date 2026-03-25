"""Minimal Telegram Bot API sender (HTTPS); no extra SDK."""

from __future__ import annotations

from typing import Any

import requests


def send_message(
    token: str,
    chat_id: str,
    text: str,
    *,
    parse_mode: str | None = None,
    disable_web_page_preview: bool = True,
    session: requests.Session | None = None,
    timeout: int = 30,
) -> dict[str, Any]:
    sess = session or requests.Session()
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload: dict[str, Any] = {
        "chat_id": chat_id,
        "text": text,
        "disable_web_page_preview": disable_web_page_preview,
    }
    if parse_mode:
        payload["parse_mode"] = parse_mode
    r = sess.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    if not data.get("ok"):
        raise RuntimeError(f"Telegram API error: {data!r}")
    return data
