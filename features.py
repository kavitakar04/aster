from __future__ import annotations

import math
import torch
from typing import Optional, Iterable, Dict, Any

from orderbook import OrderbookState, Trade, EventType
from pipeline_logger import TickStore

TRADE_IMB_WINDOW_SEC: float = 30.0
QUOTE_VEL_WINDOW_SEC: float = 10.0
MAX_STALENESS: float = 60.0

_EPS: float = 1e-9

def _compute_trade_imbalance(trades: Iterable[Trade], as_of_ts: float) -> float:
    start_ts = as_of_ts - TRADE_IMB_WINDOW_SEC
    buy_vol, sell_vol = 0.0, 0.0

    for tr in trades:
        if start_ts <= tr.ts <= as_of_ts:
            side = tr.side.upper()
            if side == "BUY":
                buy_vol += tr.size
            elif side == "SELL":
                sell_vol += tr.size

    denom = buy_vol + sell_vol + _EPS
    return (buy_vol - sell_vol) / denom


def _compute_quote_velocity(
    ticks: TickStore, 
    market_id: str, 
    as_of_ts: float, 
    override: Optional[float] = None
) -> float:
    """Event arrival rate (ticks/sec)."""
    if override is not None:
        return override
    return ticks.get_velocity(market_id)

def compute_micro_features(
    book: OrderbookState,
    ticks: TickStore,
    as_of_ts: float,
    meta: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    meta = meta or {}

    q_stale = (
        min(as_of_ts - book.last_quote_ts, MAX_STALENESS)
        if book.last_quote_ts > 0 else MAX_STALENESS
    )
    t_stale = (
        min(as_of_ts - book.last_trade_ts, MAX_STALENESS)
        if book.last_trade_ts > 0 else MAX_STALENESS
    )
    start_ts = meta.get("event_start_ts")
    time_to_start = (start_ts - as_of_ts) if start_ts else 0.0
    rating_diff = meta.get("rating_diff", 0.0)
    bid_sz = book.best_bid_size or 0.0
    ask_sz = book.best_ask_size or 0.0
    depth_imb = (bid_sz - ask_sz) / (bid_sz + ask_sz + _EPS)

    values = [
        book.midpoint_prob(),
        book.spread_prob(),
        depth_imb,
        _compute_trade_imbalance(book.recent_trades, as_of_ts),
        _compute_quote_velocity(ticks, book.market_id, as_of_ts),
        max(0.0, q_stale),
        max(0.0, t_stale),
        time_to_start,
        rating_diff,
    ]
    def _sanitize(x: float) -> float:
        return x if math.isfinite(x) else 0.0

    return torch.tensor([_sanitize(v) for v in values], dtype=torch.float32)
