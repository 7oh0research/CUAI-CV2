"""Estimate the full grid duration from measured smoke-test wall time and batch latency."""
import glob
import json
import os
import statistics
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
config = yaml.safe_load(open("configs/methods.yaml", encoding="utf-8"))
methods_cfg = config["methods"]
active = [m for m in methods_cfg if m != "anchorprune" or os.path.isdir("third_party/anchorprune")]
active = [m for m in active if os.path.isdir("third_party/" + m) or m == "baseline"]
# The benchmark has one baseline plus six pruning methods in the default checkout.
active = [m for m in active if m in ("baseline", "fastv", "pdrop", "sparsevlm", "visionzip", "divprune", "cdpruner", "anchorprune")]

runtime = {}
path = "logs/runtime_runs.tsv"
if os.path.exists(path):
    for line in open(path, encoding="utf-8"):
        cols = line.rstrip("\n").split("\t")
        if len(cols) != 8:
            continue
        method, _r, attn, _bs, _rep, elapsed, rc, out = cols
        if not out.startswith("results/smoke/"):
            continue
        try:
            runtime[(method, attn)] = float(elapsed)
        except ValueError:
            pass

# Read actual smoke batch latency (4 examples per batch).
latency = {}
for file in glob.glob("results/smoke/*.jsonl"):
    try:
        rows = [json.loads(x) for x in open(file, encoding="utf-8") if x.strip()]
        if not rows or rows[0].get("status") not in ("ok", "fallback"):
            continue
        s = rows[0]
        batches = [x for x in rows if x.get("type") == "batch"]
        if not batches:
            continue
        # Smoke uses batch size 1 and n=4; e2e_ms is batch elapsed time.
        per_sample = statistics.median(
            x["e2e_ms"] / max(1, x.get("n_items", 1)) for x in batches
        )
        latency[(s["method"], s["attn_requested"])] = per_sample / 1000.0
    except Exception:
        pass

# Determine supported attention modes independently for each method environment.
method_attns = {}
for method in active:
    env_name = methods_cfg[method].get("env", "h1_llava")
    env_file = f"env/{env_name}.json"
    has_flash = False
    if os.path.exists(env_file):
        try:
            has_flash = bool(json.load(open(env_file, encoding="utf-8")).get("flash_attn", False))
        except Exception:
            pass
    method_attns[method] = ["eager", "sdpa"] + (["flash_attention_2"] if has_flash else [])

# Per method/attention pair. These include all warmup and measured samples in run_one.py.
# bs=16 drops the incomplete final batch (192 measured examples out of 200).
base_samples = 3 * 210 + 3 * (240 + 280 + 352)  # step 3 + step 4, 3 repetitions
prune_samples = base_samples + 3 * (240 + 280 + 352) * 3  # step 5: 3 R values, 3 reps
base_runs = 12
prune_runs = 48

missing = []
seconds = 0.0
runs = 0
planned_examples = 0
for method in active:
    for attn in method_attns[method]:
        sample_s = latency.get((method, attn))
        # If this attention was skipped in smoke, do not count it as an executable configuration.
        if sample_s is None:
            continue
        run_count = base_runs if method == "baseline" else prune_runs
        sample_count = base_samples if method == "baseline" else prune_samples
        # Smoke wall-time includes model loading + a 4-image batch. Subtract that batch's measured latency
        # to approximate per-process startup, then apply it to every benchmark process.
        wall = runtime.get((method, attn), 0.0)
        # If smoke output predates runtime logging, assume a conservative 30 s model-load/startup cost.
        startup = max(0.0, wall - 4 * sample_s) if wall > 0 else 30.0
        seconds += run_count * startup + sample_count * sample_s
        runs += run_count
        planned_examples += sample_count

if not latency:
    print("[WARN] No usable smoke timings. Run bash step2_smoke_test.sh first.")
    raise SystemExit(0)

# MME prediction generation is a separate full pass at 16 tokens, not part of the raw-grid sample count.
# Count it when the configured dataset can be loaded; estimate conservatively at 30% of a 128-token sample.
mme_count = None
mme_root = os.environ.get("MME_ROOT", "")
if os.path.isfile(".env"):
    for line in open(".env", encoding="utf-8"):
        if line.strip().startswith("MME_ROOT="):
            mme_root = line.strip().split("=", 1)[1].strip().strip("\"' ")
try:
    if mme_root and os.path.isdir(mme_root):
        from h1bench.data import load_mme
        mme_count = len(load_mme(mme_root))
        seconds += len([m for m in active if m != "anchorprune"]) * mme_count * statistics.median(latency.values()) * 0.30
except Exception:
    mme_count = None

# Add 25% uncertainty margin for variation in model loading, I/O, clocks and batch efficiency.
low = seconds / 3600
high = seconds * 1.25 / 3600
print("\n========== FULL BENCHMARK RUNTIME ESTIMATE ==========")
print(f"Active methods: {', '.join(active)}")
print("Attention modes by method: " + "; ".join(f"{m}={','.join(method_attns[m])}" for m in active))
print(f"Benchmark subprocesses (planned): {runs}")
print(f"Approx. input examples incl. warmup: {planned_examples:,}")
if mme_count is not None:
    print(f"MME questions detected: {mme_count:,} (prediction generation included approximately)")
else:
    print("MME prediction generation: not included (MME_ROOT not readable in this estimator)")
print(f"Estimated wall time: {low:.1f}–{high:.1f} hours (based on this GPU's smoke timings)")
if missing:
    print("Missing timing samples: " + ", ".join(missing))
print("NOTE: This estimate does not guarantee every batch size fits in VRAM. OOM can reduce the run count;")
print("      unsupported attention modes are excluded. First model download is not included.")
print("      If no smoke wall-time log exists, a 30 s per-process startup allowance is used.")
print("=====================================================\n")
