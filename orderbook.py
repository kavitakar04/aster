from __future__ import annotations

import json
import os
from collections import deque, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Dict, Iterable, Literal, List, Any, Optional

import pandas as pd


class EventType(str, Enum):
    QUOTE = "QUOTE"
    TRADE = "TRADE"


@dataclass
class RawEvent:
    """Canonical event container between transport and orderbook."""
    ts_exchange: float
    ts_ingest: float
    market_id: str
    type: EventType
    payload: dict
    seq: int | None = None


@dataclass
class Trade:
    """Normalized trade tick for microstructure features."""
    ts: float
    price: float
    size: float
    side: Literal["BUY", "SELL"]


def _build_levels(levels: Iterable[tuple[int, float]] | None) -> Dict[int, float]:
    """Convert [(price_cents, qty), ...] to {price_cents: qty}."""
    return {
        int(price_cents): float(qty)
        for price_cents, qty in (levels or [])
        if qty > 0
    }


def _apply_deltas(book: Dict[int, float], deltas: Iterable[tuple[int, float]] | None) -> None:
    """In-place update of price levels; qty <= 0 removes the level."""
    for price_cents, qty in (deltas or []):
        price_cents = int(price_cents)
        if qty <= 0:
            book.pop(price_cents, None)
        else:
            book[price_cents] = float(qty)


@dataclass
class OrderbookState:
    """
    Kalshi orderbook in YES probability space.

    yes_levels / no_levels: depth in cents 0–100
        price_cents -> resting size

    best_*: derived best quotes as YES probabilities in [0,1].
    """

    market_id: str

    yes_levels: Dict[int, float] = field(default_factory=dict)
    no_levels: Dict[int, float] = field(default_factory=dict)

    best_bid_prob: float | None = None
    best_bid_size: float | None = None
    best_ask_prob: float | None = None
    best_ask_size: float | None = None

    recent_trades: Deque[Trade] = field(default_factory=lambda: deque(maxlen=256))
    recent_quote_ts: Deque[float] = field(default_factory=lambda: deque(maxlen=512))

    last_quote_ts: float = 0.0
    last_trade_ts: float = 0.0
    last_update_seq: int | None = None

    def apply_event(self, evt: RawEvent) -> None:
        """Apply a RawEvent to update state; ignore stale seq numbers."""
        if evt.seq is not None and self.last_update_seq is not None:
            if evt.seq <= self.last_update_seq:
                return

        if evt.type is EventType.QUOTE:
            kind = evt.payload.get("kalshi_msg_type")
            handlers = {
                "orderbook_snapshot": self._apply_snapshot,
                "orderbook_delta": self._apply_delta,
            }
            handler = handlers.get(kind)
            if not handler:
                return

            handler(evt.payload)
            self.last_quote_ts = evt.ts_exchange
            self.recent_quote_ts.append(evt.ts_exchange)

        elif evt.type is EventType.TRADE:
            p = evt.payload
            self.recent_trades.append(
                Trade(
                    ts=evt.ts_exchange,
                    price=float(p["price"]),
                    size=float(p["size"]),
                    side=p["side"],
                )
            )
            self.last_trade_ts = evt.ts_exchange

        self.last_update_seq = evt.seq

    def _apply_snapshot(self, payload: dict) -> None:
        """Replace ladders from snapshot payload."""
        self.yes_levels = _build_levels(payload.get("yes"))
        self.no_levels = _build_levels(payload.get("no"))
        self._recompute_best_levels()

    def _apply_delta(self, payload: dict) -> None:
        """Apply incremental level updates."""
        _apply_deltas(self.yes_levels, payload.get("yes"))
        _apply_deltas(self.no_levels, payload.get("no"))
        self._recompute_best_levels()

    def _recompute_best_levels(self) -> None:
        """
        YES bid: max YES price
        YES ask: 1 - best NO bid
        """
        if self.yes_levels:
            best_yes_cents = max(self.yes_levels)
            self.best_bid_prob = best_yes_cents / 100.0
            self.best_bid_size = self.yes_levels[best_yes_cents]
        else:
            self.best_bid_prob = self.best_bid_size = None

        if self.no_levels:
            best_no_cents = max(self.no_levels)
            self.best_ask_prob = (100 - best_no_cents) / 100.0
            self.best_ask_size = self.no_levels[best_no_cents]
        else:
            self.best_ask_prob = self.best_ask_size = None

    def midpoint_prob(self) -> float:
        """Midpoint YES probability; 0.0 if one side is missing."""
        return (
            0.0
            if self.best_bid_prob is None or self.best_ask_prob is None
            else 0.5 * (self.best_bid_prob + self.best_ask_prob)
        )

    def spread_prob(self) -> float:
        """YES spread width; 0.0 if one side is missing."""
        return (
            0.0
            if self.best_bid_prob is None or self.best_ask_prob is None
            else self.best_ask_prob - self.best_bid_prob
        )


# Config for real-time window tracking
QUOTE_VEL_WINDOW = 10.0  # seconds


class TickStore:
    """
    High-frequency event store.

    Optimized with a dedicated 'fast path' for O(1) quote velocity lookups
    while maintaining the full event history for Parquet persistence.
    """

    def __init__(
        self,
        storage_path: str = "data/ticks/",
        max_events_per_market: int = 50_000,
        flush_threshold: int = 2_000,
    ) -> None:
        self.storage_path = storage_path
        os.makedirs(self.storage_path, exist_ok=True)

        self.flush_threshold = flush_threshold
        self._buffer: List[Dict[str, Any]] = []

        # 1. Heavy Store: Full event history for window_events() and debugging
        self._events: Dict[str, Deque[RawEvent]] = defaultdict(
            lambda: deque(maxlen=max_events_per_market)
        )

        # 2. Fast Path: Lightweight timestamp-only queues for O(1) velocity
        # Maps market_id -> Deque[float] (timestamps only)
        self._quote_times: Dict[str, Deque[float]] = defaultdict(deque)

    def record_event(self, evt: RawEvent, team_info: Optional[Dict[str, str]] = None) -> None:
        """Log event and update real-time counters. Extracts structured data instead of serializing full payload."""
        teams = team_info or {}

        record: Dict[str, Any] = {
            "ts_exchange": evt.ts_exchange,
            "ts_ingest": evt.ts_ingest,
            "market_id": evt.market_id,
            "type": evt.type.value,
            "seq": evt.seq,
            "team_a": teams.get("a"),
            "team_b": teams.get("b"),
        }

        if evt.type == EventType.QUOTE:
            msg_type = evt.payload.get("kalshi_msg_type")
            record["msg_type"] = msg_type

            yes_levels = evt.payload.get("yes", [])
            no_levels = evt.payload.get("no", [])

            record["best_yes_price"] = max((p for p, q in yes_levels if q > 0), default=None) if yes_levels else None
            record["best_yes_size"] = next((q for p, q in yes_levels if p == record["best_yes_price"]), None) if record["best_yes_price"] else None

            record["best_no_price"] = max((p for p, q in no_levels if q > 0), default=None) if no_levels else None
            record["best_no_size"] = next((q for p, q in no_levels if p == record["best_no_price"]), None) if record["best_no_price"] else None

        elif evt.type == EventType.TRADE:
            record["trade_price"] = float(evt.payload.get("price", 0))
            record["trade_size"] = float(evt.payload.get("size", 0))
            record["trade_side"] = evt.payload.get("side")

        self._buffer.append(record)

        if len(self._buffer) >= self.flush_threshold:
            self.flush_to_disk()

        self._events[evt.market_id].append(evt)

        if evt.type == EventType.QUOTE:
            q_times = self._quote_times[evt.market_id]
            q_times.append(evt.ts_exchange)

            cutoff = evt.ts_exchange - QUOTE_VEL_WINDOW
            while q_times and q_times[0] < cutoff:
                q_times.popleft()

    def get_velocity(self, market_id: str) -> float:
        """O(1) lookup for quote velocity (ticks/sec) over the last 10s."""
        q_times = self._quote_times.get(market_id)
        if not q_times:
            return 0.0

        now = q_times[-1]
        cutoff = now - QUOTE_VEL_WINDOW
        while q_times and q_times[0] < cutoff:
            q_times.popleft()

        return len(q_times) / QUOTE_VEL_WINDOW

    def flush_to_disk(self) -> None:
        """Commit buffered events to compressed Parquet files."""
        if not self._buffer:
            return

        df = pd.DataFrame(self._buffer)
        ts = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        path = os.path.join(self.storage_path, f"ticks_{ts}.parquet")

        df.to_parquet(path, index=False, compression="snappy")
        self._buffer = []

    def window_events(self, market_id: str, start_ts: float, end_ts: float) -> List[RawEvent]:
        """Full retrieval for complex features (e.g. trade imbalance)."""
        dq = self._events.get(market_id)
        if not dq:
            return []
        return [e for e in dq if start_ts <= e.ts_exchange <= end_ts]
