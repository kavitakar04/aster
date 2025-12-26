import argparse
import glob
import json
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from orderbook import OrderbookState, RawEvent, EventType, TickStore
from features import compute_micro_features
from models import MarketAgent, build_gaussian_targets
from io_kalshi import fetch_series_markets, fetch_orderbook_history
from cfbd_fetcher import build_registry

# --- Optimized Vectorized Feature Extraction ---

def replay_trades_vectorized(market_id: str, raw_events: list):
    """
    Phase 1: Fast training on Trade history (REST API).
    Returns (X, P, S) tensors.
    """
    df = pd.DataFrame(raw_events)
    if df.empty or "yes_price" not in df.columns: return None, None, None

    df["price"] = df["yes_price"] / 100.0
    df["ts"] = pd.to_datetime(df["created_time"]).astype(int) // 10**9
    df = df.sort_values("ts").reset_index(drop=True)

    df["vel"] = df["count"].rolling(10).count() / 10.0
    df["yes_ratio"] = (df["taker_side"] == "yes").rolling(10).mean()

    N = len(df)
    feats = np.zeros((N, 9), dtype=np.float32)
    feats[:, 0] = df["price"].values
    feats[:, 1] = 0.05  # Assumed spread
    feats[:, 2] = (df["yes_ratio"] - 0.5) * 2  # Depth imbalance proxy
    feats[:, 4] = df["vel"].fillna(0).values

    targets = np.roll(df["price"].values, -1)

    X = torch.tensor(feats, dtype=torch.float32)[20:-1]
    P = torch.tensor(targets, dtype=torch.float32)[20:-1]
    S = torch.full_like(P, 0.05)

    return X, P, S

def replay_stream_vectorized(market_id: str, tick_files: list):
    """Phase 2: Retraining on high-fidelity Streamed data (Parquet)."""
    dfs = [pd.read_parquet(f) for f in tick_files]
    if not dfs: return None, None, None
    df = pd.concat(dfs).sort_values("ts_exchange")
    df = df[df["market_id"] == market_id]

    import json
    df["p"] = df["payload"].apply(lambda x: json.loads(x) if isinstance(x, str) else x)

    book = OrderbookState(market_id)
    ticks = TickStore()

    X_list, P_list, S_list = [], [], []
    records = df.to_dict('records')

    for row in records:
        evt = RawEvent(
            ts_exchange=row["ts_exchange"],
            ts_ingest=row["ts_exchange"],
            market_id=market_id,
            type=EventType(row["type"]),
            payload=row["p"],
            seq=row.get("seq")
        )

        ticks.record_event(evt)
        book.apply_event(evt)

        if evt.type == EventType.QUOTE and book.midpoint_prob() > 0:
            if row.get("seq", 0) % 5 != 0: continue  # Downsample

            f = compute_micro_features(book, ticks, evt.ts_exchange)
            X_list.append(f)
            P_list.append(book.midpoint_prob())
            S_list.append(book.spread_prob())

    if not X_list: return None, None, None

    return torch.stack(X_list), torch.tensor(P_list), torch.tensor(S_list)

# --- Main Training Logic ---

def train_job(market_id: str, X: torch.Tensor, P: torch.Tensor, S: torch.Tensor, epochs: int):
    if len(X) < 50: return

    agent = MarketAgent(market_id)

    mu, std = X.mean(0), X.std(0)
    agent.set_normalization(mu, std)
    X_norm = agent.normalize(X)

    Q = build_gaussian_targets(agent.grid, P, S)

    dataset = TensorDataset(X_norm, Q, torch.ones(len(X)))
    loader = DataLoader(dataset, batch_size=256, shuffle=True)

    print(f"[{market_id}] Training {epochs} epochs on {len(X)} samples...")
    for _ in range(epochs):
        loss = agent.train_epoch(loader)

    print(f"[{market_id}] Final Loss: {loss:.4f}")
    agent.save()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--series", default="KXNCAAFGAME")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--retrain", action="store_true", help="Use stream files")
    args = parser.parse_args()

    markets = fetch_series_markets(series_ticker=args.series)
    tickers = [m["ticker"] for m in markets]

    registry_df = build_registry(markets)
    os.makedirs("data/meta", exist_ok=True)
    registry_df.to_parquet("data/meta/meta_registry.parquet")

    if args.retrain:
        tick_files = glob.glob("data/ticks/ticks_*.parquet")
        for ticker in tickers:
            X, P, S = replay_stream_vectorized(ticker, tick_files)
            if X is not None:
                train_job(ticker, X, P, S, args.epochs)
    else:
        for ticker in tickers:
            print(f"[{ticker}] Fetching history...")
            events = fetch_orderbook_history(ticker)
            X, P, S = replay_trades_vectorized(ticker, events)
            if X is not None:
                train_job(ticker, X, P, S, args.epochs)

if __name__ == "__main__":
    main()