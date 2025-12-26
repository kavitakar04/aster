import argparse
import json
import os
from typing import Dict, Any, Callable, Iterable

import pandas as pd
import torch

from orderbook import OrderbookState, RawEvent, EventType
from features import compute_micro_features
from models import ProbGrid, ProbDistributionMLP, make_default_grid, predict_with_uncertainty
from pipeline_logger import TickStore
from io_kalshi import kalshi_event_stream

# We now store normalization stats per market in a dictionary
MARKET_NORMS: Dict[str, Dict[str, torch.Tensor]] = {}
MARKET_MODELS: Dict[str, ProbDistributionMLP] = {}

def load_deployment(config_path: str, grid_K: int):
    """Load models and metadata from the deployment JSON."""
    with open(config_path, "r") as f:
        config = json.load(f)

    meta_registry = {}
    
    print(f"Loading deployment from {config['generated_at']}...")

    for ticker, data in config["markets"].items():
        # 1. Rehydrate Metadata
        meta_registry[ticker] = data["meta"]

        # 2. Load Normalization Stats
        norm = torch.load(data["norm_path"], map_location="cpu")
        MARKET_NORMS[ticker] = {
            "mean": norm["mean"],
            "std": norm["std"]
        }

        # 3. Load Model
        # Assuming input dim 9 based on feature set
        model = ProbDistributionMLP(d_in=9, K=grid_K)
        model.load_state_dict(torch.load(data["model_path"], map_location="cpu"))
        model.eval() # Set to eval mode!
        MARKET_MODELS[ticker] = model

    return meta_registry

def load_meta_registry(path: str) -> Dict[str, Dict[str, Any]]:
    df = pd.read_parquet(path)
    if "ticker" in df.columns:
        df = df.set_index("ticker")
    return df.to_dict(orient="index")

def run_realtime(
    event_stream: Callable[[], Iterable[RawEvent]],
    grid: ProbGrid,
    meta_registry: Dict[str, Any],
    on_result: Callable[[Dict[str, Any]], None],
    n_samples: int = 10,
) -> None:
    
    ticks = TickStore(storage_path="data/ticks/")
    books: Dict[str, OrderbookState] = {}

    print(f"Listening for {len(meta_registry)} markets...")

    try:
        for evt in event_stream():
            if evt.market_id not in meta_registry:
                continue

            # [cite_start]Standard state updates [cite: 13, 14]
            m_meta = meta_registry[evt.market_id]
            ticks.record_event(evt, team_info=m_meta.get("teams", {}))
            
            book = books.setdefault(evt.market_id, OrderbookState(market_id=evt.market_id))
            book.apply_event(evt)

            if evt.type is not EventType.QUOTE:
                continue

            # [cite_start]Feature Extraction [cite: 23-29]
            feat_vec = compute_micro_features(
                book=book,
                ticks=ticks,
                as_of_ts=evt.ts_exchange,
                meta=m_meta,
            )

            # --- UPDATED: Per-Market Normalization ---
            # [cite_start]We fetch the specific norm stats for *this* market [cite: 59]
            norms = MARKET_NORMS.get(evt.market_id)
            if norms:
                feat_vec = (feat_vec - norms["mean"]) / (norms["std"] + 1e-6)

            # --- UPDATED: Per-Market Inference ---
            # [cite_start]We fetch the specific model for *this* market [cite: 33]
            model = MARKET_MODELS.get(evt.market_id)
            if model:
                mean_probs, mean_p, var_p = predict_with_uncertainty(
                    model=model,
                    grid=grid,
                    features=feat_vec,
                    n_samples=n_samples,
                )

                on_result({
                    "ts_exchange": evt.ts_exchange,
                    "market_id": evt.market_id,
                    "mean_p": mean_p,
                    "var_p": var_p,
                    "probs": mean_probs,
                })

    finally:
        ticks.flush_to_disk()

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to deployment config.json")
    parser.add_argument("--markets", nargs="*", default=[])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--samples", type=int, default=10)
    args = parser.parse_args()

    # [cite_start]Grid setup [cite: 6]
    grid = make_default_grid(K=21)

    if args.config:
        # Load everything from the bundle
        meta_registry = load_deployment(args.config, grid.K)
        if not meta_registry:
            print("No markets found in config. Exiting.")
            return
    else:
        meta_path = os.path.join("data", "meta", "meta_registry.parquet")
        if not os.path.exists(meta_path):
            raise SystemExit(f"Meta registry not found: {meta_path}")
        meta_registry = load_meta_registry(meta_path)
        if args.markets:
            meta_registry = {mkt: meta_registry.get(mkt, {}) for mkt in args.markets}
        elif args.all:
            pass
        else:
            raise SystemExit("Provide --config, --markets, or --all.")

    # [cite_start]Pass the registry to the stream [cite: 16]
    run_realtime(
        event_stream=lambda: kalshi_event_stream(meta_registry),
        grid=grid,
        meta_registry=meta_registry,
        on_result=lambda res: print(f"{res['ts_exchange']:.3f} | {res['market_id']} | μ={res['mean_p']:.4f} | σ²={res['var_p']:.6f}"),
        n_samples=args.samples,
    )

if __name__ == "__main__":
    main()
