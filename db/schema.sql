-- LoL T2 Prediction Model — SQLite Schema

CREATE TABLE IF NOT EXISTS players (
    id INTEGER PRIMARY KEY,
    player_name TEXT NOT NULL UNIQUE,
    role TEXT,
    team TEXT,
    region TEXT,
    updated_at TEXT
);

CREATE TABLE IF NOT EXISTS accounts (
    id INTEGER PRIMARY KEY,
    player_id INTEGER REFERENCES players(id),
    summoner_name TEXT,
    server TEXT,
    rank_tier TEXT,
    lp INTEGER,
    soloq_rating REAL,
    snapshot_date TEXT
);

CREATE TABLE IF NOT EXISTS teams (
    id INTEGER PRIMARY KEY,
    team_name TEXT NOT NULL UNIQUE,
    region TEXT,
    league TEXT,
    pro_elo REAL DEFAULT 1500.0,
    games_played INTEGER DEFAULT 0,
    updated_at TEXT
);

CREATE TABLE IF NOT EXISTS matches (
    id INTEGER PRIMARY KEY,
    gameid TEXT UNIQUE,
    date TEXT,
    league TEXT,
    patch TEXT,
    playoffs INTEGER,
    blue_team TEXT,
    red_team TEXT,
    winner TEXT,
    gamelength INTEGER
);

CREATE TABLE IF NOT EXISTS rosters (
    id INTEGER PRIMARY KEY,
    team TEXT,
    player_name TEXT,
    role TEXT,
    snapshot_date TEXT,
    tournament TEXT
);

CREATE TABLE IF NOT EXISTS paper_trades (
    id INTEGER PRIMARY KEY,
    market_id TEXT NOT NULL,
    team_a TEXT,
    team_b TEXT,
    bet_team TEXT,
    side TEXT,
    amount REAL,
    entry_price REAL,
    model_prob REAL,
    edge REAL,
    kelly_fraction REAL,
    entry_time TEXT,
    market_url TEXT,
    exit_price REAL,
    match_winner TEXT,
    profit_loss REAL,
    status TEXT DEFAULT 'open'
);

CREATE TABLE IF NOT EXISTS paper_portfolio (
    id INTEGER PRIMARY KEY,
    date TEXT UNIQUE,
    bankroll REAL,
    daily_pnl REAL,
    open_positions INTEGER,
    total_bets INTEGER,
    wins INTEGER,
    losses INTEGER
);

CREATE INDEX IF NOT EXISTS idx_matches_date ON matches(date);
CREATE INDEX IF NOT EXISTS idx_matches_league ON matches(league);
CREATE INDEX IF NOT EXISTS idx_accounts_player ON accounts(player_id);
CREATE INDEX IF NOT EXISTS idx_accounts_snapshot ON accounts(snapshot_date);
CREATE INDEX IF NOT EXISTS idx_rosters_team ON rosters(team, snapshot_date);
CREATE INDEX IF NOT EXISTS idx_paper_trades_status ON paper_trades(status);
CREATE INDEX IF NOT EXISTS idx_paper_trades_market ON paper_trades(market_id);

-- ===== Polymarket Forward Price Collection =====

CREATE TABLE IF NOT EXISTS polymarket_markets (
    id INTEGER PRIMARY KEY,
    market_id TEXT NOT NULL UNIQUE,
    condition_id TEXT,
    slug TEXT,
    question TEXT,
    team_a TEXT,
    team_b TEXT,
    db_team_a TEXT,
    db_team_b TEXT,
    token_id_a TEXT,
    token_id_b TEXT,
    url TEXT,
    first_seen TEXT,
    last_seen TEXT,
    status TEXT DEFAULT 'active',
    resolution_winner TEXT,
    resolution_time TEXT,
    closing_price_a REAL,
    closing_price_b REAL
);

CREATE TABLE IF NOT EXISTS polymarket_prices (
    id INTEGER PRIMARY KEY,
    market_id TEXT NOT NULL REFERENCES polymarket_markets(market_id),
    timestamp TEXT NOT NULL,
    price_a REAL NOT NULL,
    price_b REAL NOT NULL,
    spread REAL,
    volume REAL,
    source TEXT DEFAULT 'gamma'
);

CREATE TABLE IF NOT EXISTS polymarket_price_history (
    id INTEGER PRIMARY KEY,
    market_id TEXT NOT NULL REFERENCES polymarket_markets(market_id),
    token_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    price REAL NOT NULL,
    interval TEXT NOT NULL,
    UNIQUE(market_id, token_id, timestamp, interval)
);

CREATE INDEX IF NOT EXISTS idx_pm_markets_status ON polymarket_markets(status);
CREATE INDEX IF NOT EXISTS idx_pm_prices_market ON polymarket_prices(market_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_pm_price_hist ON polymarket_price_history(market_id, token_id, timestamp);

-- ===== Historical Bookmaker Odds =====

CREATE TABLE IF NOT EXISTS bookmaker_odds (
    id INTEGER PRIMARY KEY,
    source TEXT NOT NULL,
    match_date TEXT NOT NULL,
    league TEXT,
    team_a_raw TEXT NOT NULL,
    team_b_raw TEXT NOT NULL,
    team_a_db TEXT,
    team_b_db TEXT,
    odds_a REAL,
    odds_b REAL,
    implied_prob_a REAL,
    implied_prob_b REAL,
    no_vig_prob_a REAL,
    no_vig_prob_b REAL,
    winner_raw TEXT,
    winner_db TEXT,
    match_id INTEGER REFERENCES matches(id),
    scraped_at TEXT,
    UNIQUE(source, match_date, team_a_raw, team_b_raw)
);

CREATE TABLE IF NOT EXISTS team_name_aliases (
    id INTEGER PRIMARY KEY,
    external_name TEXT NOT NULL,
    source TEXT NOT NULL,
    db_team_name TEXT NOT NULL,
    UNIQUE(external_name, source)
);

CREATE INDEX IF NOT EXISTS idx_bm_odds_match ON bookmaker_odds(match_date, team_a_db, team_b_db);
CREATE INDEX IF NOT EXISTS idx_bm_odds_source ON bookmaker_odds(source);
CREATE INDEX IF NOT EXISTS idx_bm_odds_match_id ON bookmaker_odds(match_id);
CREATE INDEX IF NOT EXISTS idx_aliases_source ON team_name_aliases(source, external_name);
