"""
deploy.py
Bundles trained models and metadata for deployment.
"""
import argparse
import json
import os
import shutil
import pandas as pd
from datetime import datetime
from io_kalshi import fetch_series_markets
from cfbd_fetcher import build_registry   

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--series", default="KXNCAAFGAME", help="Series to package")
    parser.add_argument("--out-dir", default="deployment_bundle", help="Output directory")
    args = parser.parse_args()

    # 1. Setup
    timestamp = datetime.utcnow().isoformat()
    models_out = os.path.join(args.out_dir, "models")
    if os.path.exists(args.out_dir):
        shutil.rmtree(args.out_dir)
    os.makedirs(models_out)

    # 2. Get Metadata for the Series
    print(f"Fetching live metadata for {args.series}...")
    markets_raw = fetch_series_markets(series_ticker=args.series)
    # Build rich registry (Ticker -> Teams/Ratings)
    registry_df = build_registry(markets_raw) 
    registry = registry_df.to_dict(orient="index")

    config = {
        "generated_at": timestamp,
        "series": args.series,
        "markets": {}
    }

    # 3. Bundle Artifacts
    print(f"Packaging artifacts...")
    for ticker in registry.keys():
        src_model = f"models_ckpts/{ticker}.pt"
        src_norm = f"normalization/{ticker}_norm.pt"

        # Only bundle if we successfully trained it
        if os.path.exists(src_model) and os.path.exists(src_norm):
            shutil.copy2(src_model, os.path.join(models_out, f"{ticker}.pt"))
            shutil.copy2(src_norm, os.path.join(models_out, f"{ticker}_norm.pt"))

            config["markets"][ticker] = {
                "model_path": f"models/{ticker}.pt", # Relative path for portability
                "norm_path": f"models/{ticker}_norm.pt",
                "meta": registry[ticker] # Embeds rating_diff, teams, etc.
            }

    # 4. Write Config
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"Success! Bundled {len(config['markets'])} markets into '{args.out_dir}'.")

if __name__ == "__main__":
    main()