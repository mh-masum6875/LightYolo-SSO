"""Freeze the published dataset split (24,853 / 3,105 / 3,109) as index files.

The split already exists on disk in GitHub/Dataset/Dataset/images/{train,val,test}.
We freeze it by recording sorted file lists plus SHA256 checksums.
Regeneration must be byte-identical; tests/test_splits.py proves it.
Never re-split. The test set is untouched until final evaluation.
"""
import hashlib, os, sys

DATASET_ROOT = os.path.join(os.path.dirname(__file__), "..", "GitHub", "Dataset", "Dataset")
OUT_DIR = os.path.join(os.path.dirname(__file__), "data", "splits")
EXPECTED = {"train": 24853, "val": 3105, "test": 3109}

def build(dataset_root=DATASET_ROOT, out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    checksums = {}
    for split, expected_n in EXPECTED.items():
        img_dir = os.path.join(dataset_root, "images", split)
        names = sorted(os.listdir(img_dir))
        if len(names) != expected_n:
            sys.exit(f"FATAL: {split} has {len(names)} images, expected {expected_n}. Aborting.")
        content = "\n".join(names) + "\n"
        path = os.path.join(out_dir, f"{split}.txt")
        with open(path, "w", newline="\n") as f:
            f.write(content)
        checksums[split] = hashlib.sha256(content.encode()).hexdigest()
    with open(os.path.join(out_dir, "checksums.sha256"), "w", newline="\n") as f:
        for split in ("train", "val", "test"):
            f.write(f"{checksums[split]}  {split}.txt\n")
    return checksums

if __name__ == "__main__":
    cs = build()
    for k, v in cs.items():
        print(f"{k}: {v}")
    print("Split files frozen in", OUT_DIR)
