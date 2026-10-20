import os, sys, sqlite3, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.common.db import connect, log_metric, merge_parts

def test_round_trip():
    with tempfile.TemporaryDirectory() as td:
        db = os.path.join(td, "r.sqlite")
        log_metric(db, "wp0_smoke_s0", "mAP50", 0.9957, wp="WP0", model="wsn-yolo",
                   seed=0, split="test", unit="fraction")
        con = connect(db)
        rows = con.execute("SELECT run_id, metric, value FROM runs").fetchall()
        con.close()
        assert rows == [("wp0_smoke_s0", "mAP50", 0.9957)]

def test_upsert_no_duplicates():
    with tempfile.TemporaryDirectory() as td:
        db = os.path.join(td, "r.sqlite")
        log_metric(db, "run1", "mAP50", 0.5, split="val")
        log_metric(db, "run1", "mAP50", 0.6, split="val")
        con = connect(db)
        rows = con.execute("SELECT value FROM runs").fetchall()
        con.close()
        assert rows == [(0.6,)]

def test_merge_parts():
    with tempfile.TemporaryDirectory() as td:
        parts = os.path.join(td, "parts")
        os.makedirs(parts)
        for i in range(3):
            log_metric(os.path.join(parts, f"p{i}.sqlite"), f"run_{i}", "fitness", float(i), seed=i)
        master = os.path.join(td, "master.sqlite")
        n = merge_parts(master, os.path.join(parts, "*.sqlite"))
        assert n == 3
        con = sqlite3.connect(master)
        assert con.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 3
        con.close()
