"""Print fields + one example per benchmark. Run before anything else."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets import load_dataset
from vtcb.data import BENCHES, iter_items
only = sys.argv[1:]
for name, (p, c, s) in BENCHES.items():
    if only and name not in only:
        continue
    print("=" * 70, name, p, c, s)
    try:
        ds = load_dataset(p, c, split=s, streaming=True)
        r = next(iter(ds))
        for k, v in r.items():
            print(f"  {k:20s} {type(v).__name__:12s} {str(v)[:80]!r}")
        it = next(iter_items(name, 3))
        print("  -> adapted:", {k: (v if k != "image" else (v.size if v else None)) for k, v in it.items()})
    except Exception as e:
        print("  !! FAILED:", type(e).__name__, e)
