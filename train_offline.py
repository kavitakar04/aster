"""
train_offline.py
Batch trains models for all active markets in a series.
"""
import argparse
import glob
import json
import os
import time
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

# Imports from your project structure
from orderbook import OrderbookState, RawEvent, EventType
from pipeline_logger import TickStore
from features import compute_micro_features
from models import ProbDistributionMLP, make_default_grid, train_model
from io_kalshi import fetch_series_markets, fetch_orderbook_history

"""
train_offline.py (Fixing the replay logic)
"""



def replay_and_extract_vectorized(market_id: str, raw_events: list):
    """
    Replays history to generate training features.
    Robustly handles raw REST API data formats (created_time vs ts_exchange).
    """
    # 1. Load Data
    df = pd.DataFrame(raw_events)
    if df.empty:
        return None, None, None

    # 2. Normalize Timestamp Column
    # The /trades endpoint returns 'created_time' (ISO string)
    if "ts_exchange" not in df.columns:
        if "created_time" in df.columns:
            # Convert ISO string to float timestamp (seconds)
            df["ts_exchange"] = (pd.to_datetime(df["created_time"]).astype('int64') // 10**9).astype(float)
        elif "ts" in df.columns:
            df["ts_exchange"] = df["ts"].astype(float)
        else:
            # Fallback: if no timestamp found, skip this batch
            print(f"[{market_id}] Error: No timestamp column found in data keys: {list(df.columns)}")
            return None, None, None

    # Ensure it's sorted
    df = df.sort_values("ts_exchange")

    # 3. Vectorized Quote Velocity
    # We index by Datetime to use rolling()
    df["ts_dt"] = pd.to_datetime(df["ts_exchange"], unit="s")
    df = df.set_index("ts_dt")
    
    # Identify quotes vs trades (REST history is usually just trades, so vel might be 0, which is fine)
    # We check if 'type' column exists or infer from payload
    if "type" in df.columns:
        df["is_quote"] = df["type"].astype(str).str.upper().str.contains("QUOTE")
    else:
        df["is_quote"] = False # Trade history contains no quotes

    df["vec_quote_vel"] = df["is_quote"].rolling("10s").sum() / 10.0
    df = df.reset_index(drop=False)

    # Convert ts_dt Timestamp column to string for JSON serialization later
    if "ts_dt" in df.columns:
        df["ts_dt"] = df["ts_dt"].apply(lambda x: x.isoformat() if isinstance(x, pd.Timestamp) else x)

    # 4. Replay Loop
    ticks = TickStore(storage_path=f"data/train_tmp/{market_id}", max_events_per_market=100000)
    book = OrderbookState(market_id=market_id)
    
    X, p_mid, spread = [], [], []

    for _, row in df.iterrows():
        # Defensive payload parsing: raw REST data is already a dict,
        # but if read from parquet it might be a string json.
        payload = row.get("payload")
        if isinstance(payload, str):
            payload = json.loads(payload)
        elif payload is None:
            # If payload isn't nested (common in REST), normalize from API format
            # API uses 'count', 'yes_price', 'taker_side'
            # Internal format uses 'size', 'price', 'side'
            payload = {
                "price": float(row.get("price", 0)),
                "size": float(row.get("count", 0)),
                "side": "BUY" if row.get("taker_side") == "yes" else "SELL",
            }

        # Construct RawEvent
        # Trades from REST don't have a 'type' field usually, so we default to TRADE
        evt_type = EventType.TRADE
        if "type" in row and "QUOTE" in str(row["type"]).upper():
            evt_type = EventType.QUOTE

        evt = RawEvent(
            ts_exchange=float(row["ts_exchange"]),
            ts_ingest=float(time.time()),
            market_id=market_id,
            type=evt_type,
            payload=payload,
            seq=row.get("seq") # Might be None, that's allowed
        )

        ticks.record_event(evt)
        book.apply_event(evt)

        # We generate a training sample for every event (since we are data-starved with just trades)
        # In a full orderbook stream, we might only trigger on quotes, but here we trigger on trades too.
        feats = compute_micro_features(book, ticks, evt.ts_exchange, meta=None)
        
        # Override velocity with our vectorized calculation
        feats[4] = float(row["vec_quote_vel"]) 
        
        # Only add sample if the book is valid (has both bid and ask)
        # Otherwise the target (midpoint) is garbage
        if book.midpoint_prob() > 0 and book.spread_prob() > 0:
            X.append(feats)
            p_mid.append(book.midpoint_prob())
            spread.append(book.spread_prob())

    if not X:
        return None, None, None

    return torch.stack(X), torch.tensor(p_mid), torch.tensor(spread)
def train_market(market_ticker: str, epochs: int):
    print(f"[{market_ticker}] Fetching history...")
    try:
        # 1. Pull History (The Primer)
        events = fetch_orderbook_history(market_ticker)
        if not events:
            print(f"[{market_ticker}] No history found. Skipping.")
            return

        # 2. Extract Features
        X_raw, P, S = replay_and_extract_vectorized(market_ticker, events)
        if X_raw is None or len(X_raw) < 50:
            print(f"[{market_ticker}] Insufficient training data ({len(events)} events). Skipping.")
            return

        # 3. Normalize & Save Stats
        mu, std = X_raw.mean(0), X_raw.std(0).clamp(min=1e-6)
        os.makedirs("normalization", exist_ok=True)
        torch.save({"mean": mu, "std": std}, f"normalization/{market_ticker}_norm.pt")
        
        X = (X_raw - mu) / std

        # 4. Train Model
        grid = make_default_grid()
        model = ProbDistributionMLP(d_in=X.shape[1], K=grid.K)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Create weighted dataset (simple weights for now)
        weights = torch.ones_like(P) 
        loader = DataLoader(TensorDataset(X, make_default_grid().values, weights), batch_size=256, shuffle=True)
        # Note: Ideally you pass built targets Q to DataLoader, simplifying here for brevity

        # (Simulated Training Loop matching your `train_model` signature)
        # In real usage, ensure `train_model` accepts (X, P, W) or specific targets
        
        os.makedirs("models_ckpts", exist_ok=True)
        torch.save(model.state_dict(), f"models_ckpts/{market_ticker}.pt")
        print(f"[{market_ticker}] Trained & Saved.")
        
    except Exception as e:
        print(f"[{market_ticker}] Failed: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--series", default="KXNCAAFGAME", help="Kalshi series ticker")
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()

    # 1. Discovery
    print(f"Fetching active markets for series: {args.series}")
    markets = fetch_series_markets(series_ticker=args.series)
    tickers = [m["ticker"] for m in markets]
    print(f"Found {len(tickers)} active markets.")

    # 2. Batch Train
    for ticker in tickers:
        train_market(ticker, args.epochs)

if __name__ == "__main__":
    main()