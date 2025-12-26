from __future__ import annotations

import os
import json
import pandas as pd
from collections import defaultdict, deque
from typing import Deque, Dict, List, Optional, Any
from orderbook import RawEvent, EventType

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
        """
        Log event and update real-time counters.
        """
        # --- Slow Path: Disk Persistence ---
        teams = team_info or {}
        self._buffer.append({
            "ts_exchange": evt.ts_exchange,
            "market_id": evt.market_id,
            "type": evt.type.value,
            "seq": evt.seq,
            "team_a": teams.get("a"),
            "team_b": teams.get("b"),
            "payload": json.dumps(evt.payload)
        })
        
        if len(self._buffer) >= self.flush_threshold:
            self.flush_to_disk()

        # --- Fast Path: Memory State ---
        self._events[evt.market_id].append(evt)

        # Optimize Quote Velocity:
        # If this is a QUOTE, add to the fast buffer and prune old ones
        if evt.type == EventType.QUOTE:
            q_times = self._quote_times[evt.market_id]
            q_times.append(evt.ts_exchange)
            
            # Amortized O(1) pruning: Remove timestamps older than window
            # We only check the head (oldest) since time is monotonic
            cutoff = evt.ts_exchange - QUOTE_VEL_WINDOW
            while q_times and q_times[0] < cutoff:
                q_times.popleft()

    def get_velocity(self, market_id: str) -> float:
        """
        O(1) lookup for quote velocity (ticks/sec) over the last 10s.
        """
        q_times = self._quote_times.get(market_id)
        if not q_times:
            return 0.0
            
        # Optional: Lazy prune on read to ensure accuracy if no new ticks arrived
        # (Useful if the market goes quiet)
        now = q_times[-1] # Approximation: use last tick time as 'now'
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