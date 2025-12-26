from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Dict, Iterable, Literal


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
