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

def build_registry(markets: list[dict], year: int = 2025) -> pd.DataFrame:
    """Match Kalshi markets to CFBD ratings and return registry DataFrame."""
    ratings_path = ensure_ratings_file(year)
    ratings_df = pd.read_csv(ratings_path).set_index("team_code")
    valid_teams = set(ratings_df.index)

    registry = {}
    for m in markets:
        # Parse ticker: KXNCAAF-YYYYMMDD-TEAM1-TEAM2
        parts = m["ticker"].split("-")
        if len(parts) < 4: 
            continue
        
        team_a = _get_canonical_team(parts[-2], valid_teams)
        team_b = _get_canonical_team(parts[-1], valid_teams)
        
        if team_a and team_b:
            ra = ratings_df.loc[team_a].rating
            rb = ratings_df.loc[team_b].rating
            
            # Determine start time (Open or Expiry)
            ts = m.get("open_time") or m.get("expiration_time")
            
            registry[m["ticker"]] = {
                "teams": {"a": team_a, "b": team_b},
                "rating_diff": float(ra - rb),
                "event_start_ts": pd.Timestamp(ts).timestamp(),
                "strike_price": m.get("strike_price")
            }

    return pd.DataFrame.from_dict(registry, orient="index")


if __name__ == "__main__":
    # Standalone usage: Build registry from live Kalshi markets
    # Note: Requires io_kalshi to be present in the environment
    from io_kalshi import fetch_ncaaf_markets
    
    print("[cfbd] Fetching active markets...")
    markets = fetch_ncaaf_markets(status="open", limit=200)
    
    print("[cfbd] Building registry...")
    df = build_registry(markets, year=2025)
    
    out_path = os.path.join(META_DIR, "meta_registry.parquet")
    df.to_parquet(out_path)
    print(f"[cfbd] Registry saved with {len(df)} markets to {out_path}")