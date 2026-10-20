"""Results database. Every experiment writes one row per metric per run.
No number enters the manuscript unless it exists here with a run_id."""
import sqlite3, time, glob, os

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL, wp TEXT, task TEXT, dataset TEXT, tier TEXT,
    method TEXT, model TEXT, seed INTEGER, split TEXT,
    metric TEXT NOT NULL, value REAL, unit TEXT,
    gpu_hours REAL, created_at TEXT, config_path TEXT, git_hash TEXT,
    UNIQUE(run_id, metric, split)
);
CREATE INDEX IF NOT EXISTS idx_runs_run_id ON runs(run_id);
"""

def connect(db_path):
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    con = sqlite3.connect(db_path)
    con.executescript(SCHEMA)
    return con

def log_metric(db_path, run_id, metric, value, *, wp=None, task=None,
               dataset=None, tier=None, method=None, model=None, seed=None,
               split=None, unit=None, gpu_hours=None, config_path=None,
               git_hash=None):
    con = connect(db_path)
    with con:
        con.execute(
            "INSERT OR REPLACE INTO runs (run_id, wp, task, dataset, tier, method, model,"
            " seed, split, metric, value, unit, gpu_hours, created_at, config_path, git_hash)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (run_id, wp, task, dataset, tier, method, model, seed, split, metric,
             value, unit, gpu_hours, time.strftime("%Y-%m-%dT%H:%M:%S"), config_path, git_hash))
    con.close()

def to_pandas(db_path, query="SELECT * FROM runs"):
    import pandas as pd
    con = connect(db_path)
    df = pd.read_sql_query(query, con)
    con.close()
    return df

def merge_parts(master_path, parts_glob="results/parts/*.sqlite"):
    """Fold per-job part databases (SLURM arrays) into the master."""
    master = connect(master_path)
    n = 0
    for part in sorted(glob.glob(parts_glob)):
        pcon = sqlite3.connect(part)
        rows = pcon.execute(
            "SELECT run_id, wp, task, dataset, tier, method, model, seed, split,"
            " metric, value, unit, gpu_hours, created_at, config_path, git_hash FROM runs").fetchall()
        pcon.close()
        with master:
            master.executemany(
                "INSERT OR REPLACE INTO runs (run_id, wp, task, dataset, tier, method, model,"
                " seed, split, metric, value, unit, gpu_hours, created_at, config_path, git_hash)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
        n += len(rows)
    master.close()
    return n
