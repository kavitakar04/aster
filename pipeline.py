import argparse
import os
import torch
from typing import Dict, Any, Callable, Iterable, List
from collections import defaultdict
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

from orderbook import OrderbookState, RawEvent, EventType, TickStore
from features import compute_micro_features
from models import MarketAgent, make_default_grid, build_gaussian_targets
from io_kalshi import kalshi_event_stream, load_dotenv

# --- Global State ---
AGENTS: Dict[str, MarketAgent] = {}
BUFFERS: Dict[str, List] = defaultdict(list)  # Stores tuples (features, target, spread)

CONFIG = {
    "finetune": False,
    "buffer_size": 500,
    "device": "cpu"
}

def get_or_create_agent(market_id: str) -> MarketAgent:
    """Lazy loader for agents."""
    if market_id not in AGENTS:
        agent = MarketAgent(market_id, device=CONFIG["device"])
        success = agent.load()
        if not success:
            print(f"[pipeline] Warning: No pre-trained model for {market_id}. Starting fresh.")
        AGENTS[market_id] = agent
    return AGENTS[market_id]

def process_online_learning(agent: MarketAgent):
    """Flushes buffer to update the agent."""
    buffer = BUFFERS[agent.market_id]
    if len(buffer) < CONFIG["buffer_size"]:
        return

    # Unpack buffer
    X_raw = torch.stack([b[0] for b in buffer])
    P = torch.tensor([b[1] for b in buffer], dtype=torch.float32)
    S = torch.tensor([b[2] for b in buffer], dtype=torch.float32).clamp(min=0.01)
    
    # Normalize inputs (using agent's stored stats)
    X = agent.normalize(X_raw)
    Q = build_gaussian_targets(agent.grid, p_mid=P, tau=S)
    dataset = TensorDataset(X, Q, torch.ones(len(X)))
    loader = DataLoader(dataset, batch_size=256, shuffle=True)
    loss = agent.train_epoch(loader)
    print(f"[finetune] {agent.market_id} updated. Loss: {loss:.4f}")
    
    buffer.clear()
 

def run_realtime(meta_registry: Dict[str, Any], n_samples: int = 10):
    ticks = TickStore(storage_path="data/ticks/")
    books: Dict[str, OrderbookState] = {}
    
    print(f"Starting pipeline for {len(meta_registry)} markets...")

    stream = kalshi_event_stream(meta_registry)
    
    try:
        for evt in stream:
            if evt.market_id not in meta_registry:
                continue

            # 1. Update State
            ticks.record_event(evt)
            book = books.setdefault(evt.market_id, OrderbookState(market_id=evt.market_id))
            book.apply_event(evt)

            # 2. Inference (Only on Quote updates)
            if evt.type == EventType.QUOTE:
                # Features
                feat_vec = compute_micro_features(book, ticks, evt.ts_exchange, meta=meta_registry[evt.market_id])
                
                # Inference
                agent = get_or_create_agent(evt.market_id)
                probs, price, var = agent.predict(feat_vec, n_samples=n_samples)

                # Format probability distribution over grid
                # Show top 3 most likely price bins
                grid = agent.grid
                top_k = 3
                top_indices = probs.argsort(descending=True)[:top_k]
                prob_dist_str = ", ".join([
                    f"P({grid.values[i]:.2f})={probs[i]:.3f}"
                    for i in top_indices
                ])

                print(f"{evt.ts_exchange:.3f} | {evt.market_id} | E[p]={price:.3f} | [{prob_dist_str}] | σ²={var:.4f}")

                # 3. Online Learning Collection
                if CONFIG["finetune"] and book.midpoint_prob() > 0:
                    target = book.midpoint_prob()
                    spread = book.spread_prob()
                    BUFFERS[evt.market_id].append((feat_vec.cpu(), target, spread))
                    
                    if len(BUFFERS[evt.market_id]) >= CONFIG["buffer_size"]:
                        process_online_learning(agent)

    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        ticks.flush_to_disk()
        if CONFIG["finetune"]:
            for m_id, agent in AGENTS.items():
                if len(BUFFERS[m_id]) > 50: # Only flush if significant data left
                    process_online_learning(agent)
                    agent.save()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--markets", nargs="*", default=[])
    parser.add_argument("--finetune", action="store_true")
    args = parser.parse_args()

    load_dotenv()
    CONFIG["finetune"] = args.finetune
    
    meta_path = "data/meta/meta_registry.parquet"
    if not os.path.exists(meta_path):
        print("No meta registry found. Run train_offline.py first.")
        return
        
    df = pd.read_parquet(meta_path)
    if "ticker" in df.columns: df = df.set_index("ticker")
    registry = df.to_dict(orient="index")
    
    if args.markets:
        registry = {k: v for k, v in registry.items() if k in args.markets}

    run_realtime(registry)

if __name__ == "__main__":
    main()