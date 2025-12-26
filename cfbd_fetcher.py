"""
cfbd_fetcher.py: CFBD Ratings & Market Registry Builder.
"""
from __future__ import annotations
import os
import cfbd
import pandas as pd
from fuzzywuzzy import process
from dotenv import load_dotenv

load_dotenv()
META_DIR = os.getenv("META_DIR", "data/meta")
os.makedirs(META_DIR, exist_ok=True)

# --- 1. CFBD API Interaction ---

def fetch_sp_ratings(year: int = 2025) -> pd.DataFrame:
    """Fetch SP+ ratings from CFBD and return as DataFrame."""
    config = cfbd.Configuration()
    config.access_token = os.getenv("CFBD_API_KEY")
    client = cfbd.ApiClient(config)
    
    # Fetch Data
    ratings = cfbd.RatingsApi(client).get_sp(year=year)
    teams = cfbd.TeamsApi(client).get_teams(year=year)
    
    # Map School Name -> Canonical Abbreviation
    school_to_abbr = {t.school: t.abbreviation for t in teams if t.abbreviation}
    
    data = []
    for r in ratings:
        abbr = school_to_abbr.get(r.team)
        if abbr:
            data.append({
                "team_code": abbr.upper(),
                "team_name": r.team,
                "rating": r.rating,
                "offense": r.offense.rating if r.offense else None,
                "defense": r.defense.rating if r.defense else None,
            })
            
    return pd.DataFrame(data)

def ensure_ratings_file(year: int = 2025) -> str:
    """Ensure ratings CSV exists on disk, fetching if necessary."""
    path = os.path.join(META_DIR, f"cfb_ratings_{year}.csv")
    if not os.path.exists(path):
        print(f"[cfbd] Fetching ratings for {year}...")
        df = fetch_sp_ratings(year)
        df.to_csv(path, index=False)
        print(f"[cfbd] Saved {len(df)} teams to {path}")
    return path


# --- 2. Market Registry Construction ---

KNOWN_ALIASES = {
    "BAMA": "ALABAMA", 
    "OLEMISS": "MISSISSIPPI", 
    "WASH": "WASHINGTON"
}

def _get_canonical_team(raw_code: str, valid_teams: list[str]) -> str | None:
    """Fuzzy match raw ticker codes to CFBD abbreviations."""
    query = KNOWN_ALIASES.get(raw_code.upper(), raw_code.upper())
    
    # 1. Exact Match
    if query in valid_teams: 
        return query
    
    # 2. Fuzzy Match
    match, score = process.extractOne(query, valid_teams)
    return match if score >= 80 else None

import pandas as pd

def build_registry(markets: list[dict], year: int = 2025) -> pd.DataFrame:
    """
    Build a meta registry for Kalshi markets.

    - Every market in `markets` gets a row.
    - CFBD ratings are attached when both teams can be resolved.
    - Missing ratings / times are allowed and stored as None/NaN.
    """
    # Load ratings and index by team_code
    ratings_path = ensure_ratings_file(year)
    ratings_df = pd.read_csv(ratings_path)

    if "team_code" not in ratings_df.columns:
        raise ValueError(f"'team_code' column not found in ratings file: {ratings_path}")

    ratings_df = ratings_df.set_index("team_code")
    valid_teams = set(ratings_df.index)

    rows = []

    for m in markets:
        ticker = m.get("ticker", "")
        parts = ticker.split("-")

        # Basic parse: last two segments are usually teams, but we don't drop if it's weird
        team_a_raw = parts[-2] if len(parts) >= 4 else None
        team_b_raw = parts[-1] if len(parts) >= 4 else None

        team_a = _get_canonical_team(team_a_raw, valid_teams) if team_a_raw else None
        team_b = _get_canonical_team(team_b_raw, valid_teams) if team_b_raw else None

        has_ratings = (team_a in valid_teams) and (team_b in valid_teams)

        ra = rb = rating_diff = None
        if has_ratings:
            ra = float(ratings_df.at[team_a, "rating"])
            rb = float(ratings_df.at[team_b, "rating"])
            rating_diff = ra - rb

        # Event start: use open_time, then expiration_time, else None
        ts = m.get("open_time") or m.get("expiration_time")
        event_start_ts = pd.Timestamp(ts).timestamp() if ts is not None else None

        rows.append(
            {
                "ticker": ticker,
                # Teams nested dict for compatibility with pipeline
                "teams": {"a": team_a, "b": team_b},
                # raw team codes from ticker
                "team_a_raw": team_a_raw,
                "team_b_raw": team_b_raw,
                # canonical team codes (may be None)
                "team_a": team_a,
                "team_b": team_b,
                # ratings info
                "has_ratings": has_ratings,
                "team_a_rating": ra,
                "team_b_rating": rb,
                "rating_diff": rating_diff,
                # timing + strike
                "event_start_ts": event_start_ts,
                "strike_price": m.get("strike_price"),
            }
        )

    df = pd.DataFrame(rows)
    # Use ticker as index if present
    if not df.empty and "ticker" in df.columns:
        df = df.set_index("ticker")

    return df

if __name__ == "__main__":
    from io_kalshi import fetch_ncaaf_markets
    
    print("[cfbd] Fetching active markets...")
    markets = fetch_ncaaf_markets(status="open", limit=200)
    
    print("[cfbd] Building registry...")
    df = build_registry(markets, year=2025)
    
    out_path = os.path.join(META_DIR, "meta_registry.parquet")
    df.to_parquet(out_path)
    print(f"[cfbd] Registry saved with {len(df)} markets to {out_path}")