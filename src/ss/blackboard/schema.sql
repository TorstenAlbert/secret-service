-- SS Blackboard Schema
-- All tables use IF NOT EXISTS for idempotent migration

-- Sessions
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    client_id TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    problem_text TEXT NOT NULL,
    problem_context TEXT,  -- JSON
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    completed_at TEXT,
    duration_ms INTEGER,
    total_llm_calls INTEGER NOT NULL DEFAULT 0,
    total_events INTEGER NOT NULL DEFAULT 0
);

-- Issues
CREATE TABLE IF NOT EXISTS issues (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    summary TEXT NOT NULL,
    classification TEXT NOT NULL,
    severity TEXT NOT NULL DEFAULT 'medium',
    who TEXT NOT NULL,
    where_location TEXT NOT NULL,
    why_reason TEXT NOT NULL,
    precondition TEXT NOT NULL,
    postcondition TEXT NOT NULL,
    key_points TEXT NOT NULL,  -- JSON array
    tags TEXT NOT NULL DEFAULT '[]',  -- JSON array
    created_at TEXT NOT NULL
);

-- Strategies
CREATE TABLE IF NOT EXISTS strategies (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    description TEXT NOT NULL,
    objective TEXT NOT NULL,
    approach_type TEXT,
    rank INTEGER NOT NULL,
    confidence REAL,
    jury_score REAL,
    jury_metrics TEXT,  -- JSON
    status TEXT NOT NULL DEFAULT 'planned',
    rating_label TEXT,
    failure_reason TEXT,
    created_at TEXT NOT NULL,
    completed_at TEXT
);

-- Taktiks
CREATE TABLE IF NOT EXISTS taktiks (
    id TEXT PRIMARY KEY,
    strategy_id TEXT NOT NULL REFERENCES strategies(id),
    session_id TEXT NOT NULL REFERENCES sessions(id),
    steps TEXT NOT NULL,  -- JSON array of TaktikStep
    required_skills TEXT NOT NULL DEFAULT '[]',  -- JSON array
    estimated_complexity TEXT,
    judge_verification TEXT,  -- JSON
    verified INTEGER NOT NULL DEFAULT 0,
    attempt_number INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL
);

-- Missions
CREATE TABLE IF NOT EXISTS missions (
    id TEXT PRIMARY KEY,
    taktik_id TEXT NOT NULL REFERENCES taktiks(id),
    strategy_id TEXT NOT NULL REFERENCES strategies(id),
    session_id TEXT NOT NULL REFERENCES sessions(id),
    status TEXT NOT NULL DEFAULT 'running',
    attempt_number INTEGER NOT NULL DEFAULT 1,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    duration_ms INTEGER
);

-- Mission Results
CREATE TABLE IF NOT EXISTS mission_results (
    id TEXT PRIMARY KEY,
    mission_id TEXT NOT NULL REFERENCES missions(id),
    step_index INTEGER NOT NULL,
    action TEXT NOT NULL,
    expected_outcome TEXT,
    actual_outcome TEXT NOT NULL,
    success INTEGER NOT NULL,
    error_detail TEXT,
    artifacts TEXT,  -- JSON
    duration_ms INTEGER,
    created_at TEXT NOT NULL
);

-- Memories
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    scope TEXT NOT NULL,
    source_session_id TEXT,
    source_agent TEXT NOT NULL,
    content TEXT NOT NULL,
    structured_content TEXT,  -- JSON
    tags TEXT NOT NULL DEFAULT '[]',  -- JSON array
    relevance_count INTEGER NOT NULL DEFAULT 0,
    last_recalled_at TEXT,
    confidence REAL NOT NULL DEFAULT 1.0,
    superseded_by TEXT,
    created_at TEXT NOT NULL,
    expires_at TEXT,
    is_active INTEGER NOT NULL DEFAULT 1
);

-- Agent Notes
CREATE TABLE IF NOT EXISTS agent_notes (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    agent_name TEXT NOT NULL,
    note_type TEXT NOT NULL DEFAULT 'observation',
    content TEXT NOT NULL,
    note_references TEXT NOT NULL DEFAULT '[]',  -- JSON array
    created_at TEXT NOT NULL
);

-- Client Profiles
CREATE TABLE IF NOT EXISTS client_profiles (
    client_id TEXT PRIMARY KEY,
    display_name TEXT,
    expertise_level TEXT,
    known_domains TEXT NOT NULL DEFAULT '[]',  -- JSON array
    communication_style TEXT,
    preferences TEXT NOT NULL DEFAULT '{}',  -- JSON
    total_sessions INTEGER NOT NULL DEFAULT 0,
    first_seen_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL
);

-- Client Issue History
CREATE TABLE IF NOT EXISTS client_issue_history (
    id TEXT PRIMARY KEY,
    client_id TEXT NOT NULL REFERENCES client_profiles(client_id),
    session_id TEXT NOT NULL REFERENCES sessions(id),
    issue_summary TEXT NOT NULL,
    classification TEXT NOT NULL,
    outcome TEXT NOT NULL,
    winning_strategy_summary TEXT,
    jury_score REAL,
    created_at TEXT NOT NULL
);

-- Strategy Scores
CREATE TABLE IF NOT EXISTS strategy_scores (
    id TEXT PRIMARY KEY,
    strategy_id TEXT NOT NULL REFERENCES strategies(id),
    session_id TEXT NOT NULL REFERENCES sessions(id),
    correctness REAL NOT NULL,
    completeness REAL NOT NULL,
    elegance REAL NOT NULL,
    robustness REAL NOT NULL,
    efficiency REAL NOT NULL,
    weighted_total REAL NOT NULL,
    reasoning TEXT NOT NULL,
    created_at TEXT NOT NULL
);

-- Session Events
CREATE TABLE IF NOT EXISTS session_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    agent_name TEXT NOT NULL,
    event_type TEXT NOT NULL,
    phase TEXT,
    payload TEXT NOT NULL,  -- JSON
    timestamp TEXT NOT NULL
);

-- Embedding Registry
CREATE TABLE IF NOT EXISTS embedding_registry (
    rowid INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_type TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    embedded_at TEXT NOT NULL,
    UNIQUE(entity_type, entity_id)
);

-- -----------------------------------------------------------------------
-- Indexes
-- -----------------------------------------------------------------------

CREATE INDEX IF NOT EXISTS idx_sessions_client_id ON sessions(client_id);
CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions(status);
CREATE INDEX IF NOT EXISTS idx_strategies_session_id ON strategies(session_id);
CREATE INDEX IF NOT EXISTS idx_strategies_rating ON strategies(rating_label, jury_score);
CREATE INDEX IF NOT EXISTS idx_taktiks_strategy_id ON taktiks(strategy_id);
CREATE INDEX IF NOT EXISTS idx_missions_strategy_id ON missions(strategy_id);
CREATE INDEX IF NOT EXISTS idx_memories_type_scope ON memories(type, scope, is_active);
CREATE INDEX IF NOT EXISTS idx_memories_source_session ON memories(source_session_id);
CREATE INDEX IF NOT EXISTS idx_memories_active ON memories(is_active, type, scope);
CREATE INDEX IF NOT EXISTS idx_agent_notes_session_agent ON agent_notes(session_id, agent_name);
CREATE INDEX IF NOT EXISTS idx_client_issue_history_client ON client_issue_history(client_id);
CREATE INDEX IF NOT EXISTS idx_session_events_session_id ON session_events(session_id, id);
CREATE INDEX IF NOT EXISTS idx_session_events_type ON session_events(event_type);
CREATE INDEX IF NOT EXISTS idx_embedding_registry ON embedding_registry(entity_type, entity_id);

-- -----------------------------------------------------------------------
-- VSS Virtual Tables (vector similarity search, embedding dim=384)
-- -----------------------------------------------------------------------

CREATE VIRTUAL TABLE IF NOT EXISTS vss_issues USING vss0(embedding(384));
CREATE VIRTUAL TABLE IF NOT EXISTS vss_strategies USING vss0(embedding(384));
CREATE VIRTUAL TABLE IF NOT EXISTS vss_taktiks USING vss0(embedding(384));
CREATE VIRTUAL TABLE IF NOT EXISTS vss_missions USING vss0(embedding(384));
CREATE VIRTUAL TABLE IF NOT EXISTS vss_memories USING vss0(embedding(384));
CREATE VIRTUAL TABLE IF NOT EXISTS vss_agent_notes USING vss0(embedding(384));
