import os, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import make_splits

def test_regeneration_byte_identical():
    with tempfile.TemporaryDirectory() as t1, tempfile.TemporaryDirectory() as t2:
        cs1 = make_splits.build(out_dir=t1)
        cs2 = make_splits.build(out_dir=t2)
        assert cs1 == cs2
        for split in ("train", "val", "test"):
            a = open(os.path.join(t1, f"{split}.txt"), "rb").read()
            b = open(os.path.join(t2, f"{split}.txt"), "rb").read()
            assert a == b

def test_counts():
    with tempfile.TemporaryDirectory() as td:
        make_splits.build(out_dir=td)
        for split, n in make_splits.EXPECTED.items():
            lines = open(os.path.join(td, f"{split}.txt")).read().strip().split("\n")
            assert len(lines) == n
