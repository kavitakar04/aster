"""Kalshi I/O: REST + WebSocket utilities."""

from __future__ import annotations
from urllib.parse import urlparse

import asyncio
import base64
import json
import os
import time
from datetime import datetime, timedelta
from typing import Iterator, Iterable, List, Dict, Any, Optional, AsyncIterator

import jwt
import requests
import websockets
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.backends import default_backend
from dotenv import load_dotenv

from orderbook import RawEvent, EventType


load_dotenv()

KALSHI_API_BASE = os.getenv(
    "KALSHI_API_BASE",
    "https://api.elections.kalshi.com/trade-api/v2",
)
KALSHI_WS_URL = os.getenv(
    "KALSHI_WS_URL",
    "wss://api.elections.kalshi.com/trade-api/ws/v2",
)

# cached auth state
_jwt_token_cache: Optional[str] = None
_jwt_token_expiry: Optional[datetime] = None
_private_key_cache = None


def _get_private_key():
    """Load and cache private key for signing."""
    global _private_key_cache
    if _private_key_cache is None:
        private_key_path = os.getenv("KALSHI_PRIVATE_KEY_PATH")
        if not private_key_path:
            raise ValueError("KALSHI_PRIVATE_KEY_PATH must be set in environment")
        with open(private_key_path, "rb") as f:
            _private_key_cache = serialization.load_pem_private_key(
                f.read(), password=None, backend=default_backend()
            )
    return _private_key_cache


def _get_jwt_token() -> str:
    """Return cached JWT or mint a new one."""
    global _jwt_token_cache, _jwt_token_expiry

    if _jwt_token_cache and _jwt_token_expiry and datetime.utcnow() < _jwt_token_expiry:
        return _jwt_token_cache

    api_key_id = os.getenv("KALSHI_API_KEY_ID")
    if not api_key_id:
        raise ValueError("KALSHI_API_KEY_ID must be set in environment")

    private_key = _get_private_key()
    now = datetime.utcnow()
    expiry = now + timedelta(hours=1)

    payload = {
        "iss": api_key_id,
        "iat": int(now.timestamp()),
        "exp": int(expiry.timestamp()),
    }

    token = jwt.encode(payload, private_key, algorithm="RS256")

    _jwt_token_cache = token
    _jwt_token_expiry = expiry - timedelta(minutes=5)  # refresh window
    return token


def _auth_headers() -> Dict[str, str]:
    """Auth headers for REST."""
    token = os.getenv("KALSHI_AUTH_TOKEN") or _get_jwt_token()
    return {"Authorization": f"Bearer {token}"} if token else {}


def _ws_auth_headers(path: str = "/trade-api/ws/v2") -> Dict[str, str]:
    """Auth headers for WebSocket."""
    api_key_id = os.getenv("KALSHI_API_KEY_ID")
    if not api_key_id:
        raise RuntimeError("KALSHI_API_KEY_ID not found in environment.")

    ts = str(int(datetime.now().timestamp() * 1000))
    msg = f"{ts}GET{path.split('?')[0]}".encode("utf-8")

    signature = _get_private_key().sign(
        msg,
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
        hashes.SHA256(),
    )

    return {
        "KALSHI-ACCESS-KEY": api_key_id,
        "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode("utf-8"),
        "KALSHI-ACCESS-TIMESTAMP": ts,
    }


# ---------------------------------------------------------------------------
# REST: market listing
# ---------------------------------------------------------------------------

def fetch_series_markets(
    series_ticker: str = "KXNCAAFGAME",
    status: str = "open",
    limit: int = 1000,
) -> List[Dict[str, Any]]:
    """
    Fetch all markets in a specific series (e.g., KXNCAAFGAME).
    Useful for batch training all active games.
    """
    url = f"{KALSHI_API_BASE}/markets"
    params = {
        "series_ticker": series_ticker,
        "status": status,
        "limit": limit,
    }
    resp = requests.get(url, params=params, headers=_auth_headers(), timeout=10)
    resp.raise_for_status()
    return resp.json().get("markets", [])


# Alias for backwards compatibility
def fetch_ncaaf_markets(status: str = "open", limit: int = 200) -> List[Dict[str, Any]]:
    """Alias for fetch_series_markets with KXNCAAFGAME series."""
    return fetch_series_markets(series_ticker="KXNCAAFGAME", status=status, limit=limit)


# ---------------------------------------------------------------------------
# REST: historical orderbook / trade history
# ---------------------------------------------------------------------------

def fetch_orderbook_history(
    market_ticker: str,
    start_ts: Optional[float] = None,
    end_ts: Optional[float] = None,
    limit: int = 1000,
) -> List[Dict[str, Any]]:
    """
    Fetch historical trades for a market.
    API Endpoint: GET /markets/trades
    """
    url = f"{KALSHI_API_BASE}/markets/trades"
    
    # Kalshi expects 'ticker', 'min_ts', 'max_ts' (integers)
    params: Dict[str, Any] = {
        "ticker": market_ticker,
        "limit": limit
    }
    def _to_epoch(val: Any) -> int:
        try:
            return int(val.timestamp())  # pandas.Timestamp or datetime
        except Exception:
            return int(val)
    if start_ts is not None:
        params["min_ts"] = _to_epoch(start_ts)
    if end_ts is not None:
        params["max_ts"] = _to_epoch(end_ts)

    resp = requests.get(url, params=params, headers=_auth_headers(), timeout=30)
    resp.raise_for_status()
    return resp.json().get("trades", [])
def _normalize_rest_message(msg: Dict[str, Any], default_market_id: str) -> Optional[RawEvent]:
    """Normalize REST trade event."""
    if "trade_id" in msg or "taker_side" in msg:
        # Created time is ISO string in REST, convert to timestamp
        c_time = msg.get("created_time")
        if c_time:
            ts = datetime.fromisoformat(c_time.replace("Z", "+00:00")).timestamp()
        else:
            ts = time.time()

        normalized_side = "BUY" if msg.get("taker_side") == "yes" else "SELL"
        
        return RawEvent(
            ts_exchange=float(ts),
            ts_ingest=float(time.time()),
            market_id=str(msg.get("ticker", default_market_id)),
            type=EventType.TRADE,
            payload={
                "price": float(msg.get("yes_price", 0)) if "yes_price" in msg else float(msg.get("price", 0)),
                "size": float(msg.get("count", 0)),
                "side": normalized_side,
            },
            seq=None,
        )
    return None
def historical_event_stream(
    market_tickers: Iterable[str],
    start_ts: Optional[float] = None,
    end_ts: Optional[float] = None,
    limit_per_market: int = 10_000,
) -> Iterator[RawEvent]:
    """
    Synchronous iterator of RawEvent from REST history.

    Intended for offline training: pipe into OrderbookState/TickStore.
    """
    for ticker in market_tickers:
        raw_events = fetch_orderbook_history(
            market_ticker=ticker,
            start_ts=start_ts,
            end_ts=end_ts,
            limit=limit_per_market,
        )
        for msg in raw_events:
            evt = _normalize_rest_message(msg, default_market_id=ticker)
            if evt is not None:
                yield evt


# ---------------------------------------------------------------------------
# WebSocket: realtime event stream
# ---------------------------------------------------------------------------

async def _subscribe_stream(
    meta_registry: Dict[str, Any],
) -> AsyncIterator[RawEvent]:
    markets = list(meta_registry.keys())
    if not markets:
        return

    print(f"[WS] Connecting to Kalshi WebSocket...")
    headers = _ws_auth_headers()
    async with websockets.connect(KALSHI_WS_URL, additional_headers=headers) as ws:
        sub_msg = {
            "id": 1,
            "cmd": "subscribe",
            "params": {
                "channels": ["orderbook_delta", "trade"],  # "trade" not "trades"
                "market_tickers": markets,
            },
        }
        print(f"[WS] Subscribing to {len(markets)} markets on channels: orderbook_delta, trade")
        await ws.send(json.dumps(sub_msg))

        msg_count = 0
        event_count = 0
        async for raw in ws:
            msg_count += 1
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                print(f"[WS] Failed to parse message {msg_count}")
                continue

            if msg_count <= 3:
                print(f"[WS] Message {msg_count}: {msg.get('type', 'unknown')} - {str(msg)[:100]}")

            evt = _normalize_ws_message(msg, meta_registry=meta_registry)
            if evt is not None:
                event_count += 1
                if event_count <= 5:
                    print(f"[WS] Event {event_count}: {evt.type.value} for {evt.market_id}")
                yield evt
            elif msg_count % 100 == 0:
                print(f"[WS] Received {msg_count} messages, {event_count} events processed")


def _normalize_ws_message(
    msg: Dict[str, Any],
    meta_registry: Optional[Dict[str, Any]] = None,  # meta reserved for future enrichment
) -> Optional[RawEvent]:
    """
    Transform raw WebSocket messages into RawEvent.

    Quote payload:
      {"kalshi_msg_type": str, "yes": [...], "no": [...]}
    Trade payload:
      {"price": float, "size": float, "side": "BUY"/"SELL"}
    """
    m_type = msg.get("type")
    inner = msg.get("msg", {})
    ticker = inner.get("market_ticker") or msg.get("market_id") or msg.get("ticker")
    seq = msg.get("seq")

    if not ticker:
        return None

    # hook for meta if/when needed
    if meta_registry:
        _ = meta_registry.get(ticker)

    ts = time.time()

    if m_type in ("orderbook_snapshot", "orderbook_delta"):
        yes_levels = inner.get("yes") or []
        no_levels = inner.get("no") or []

        return RawEvent(
            ts_exchange=ts,
            ts_ingest=ts,
            market_id=ticker,
            type=EventType.QUOTE,
            payload={
                "kalshi_msg_type": m_type,
                "yes": yes_levels,  # [(price_cents, qty), ...]
                "no": no_levels,
            },
            seq=seq,
        )

    if m_type == "trade":
        price = inner.get("price")
        size = inner.get("size")
        side = inner.get("side")  # "yes"/"no" from Kalshi

        if price is None or size is None or side is None:
            return None

        normalized_side = "BUY" if side.lower() == "yes" else "SELL"

        return RawEvent(
            ts_exchange=ts,
            ts_ingest=ts,
            market_id=ticker,
            type=EventType.TRADE,
            payload={
                "price": float(price),
                "size": float(size),
                "side": normalized_side,
            },
            seq=seq,
        )

    return None


def kalshi_event_stream(meta_registry: Dict[str, Any]) -> Iterator[RawEvent]:
    """Sync wrapper around the async WS subscriber."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def runner():
        async for evt in _subscribe_stream(meta_registry):
            yield evt

    gen = runner()
    try:
        while True:
            yield loop.run_until_complete(gen.__anext__())
    except StopAsyncIteration:
        pass
    finally:
        loop.close()
