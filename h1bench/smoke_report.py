import glob, json, os

cfg = json.load(open("configs/methods.json")) if os.path.exists("configs/methods.json") else None
lines = []
files = sorted(glob.glob("results/smoke/*.jsonl"))
if not files:
    raise SystemExit("[ERROR] No smoke-test result files were found.")
for path in files:
    try:
        rows = [json.loads(line) for line in open(path) if line.strip()]
        if not rows:
            lines.append(f"{path}: FAIL empty result")
            continue
        s = rows[0]
        batches = [r for r in rows if r.get("type") == "batch"]
        status = s.get("status", "unknown")
        method, R, attn = s.get("method", "?"), s.get("R", "?"), s.get("attn_requested", "?")
        if status == "unsupported_env":
            lines.append(f"{method:12s} R={R:3} {attn:18s} SKIP {s.get('error', 'environment unavailable')}")
            continue
        if status in ("oom", "failed", "unsupported") or not batches:
            lines.append(f"{method:12s} R={R:3} {attn:18s} FAIL {status} {s.get('error', 'no batch records')}")
            continue
        vis = batches[0].get("layer_visual_tokens") or []
        final = vis[-1] if vis else None
        expected = 576 if method == "baseline" else R
        tag = "OK" if final == expected else "MISMATCH"
        lines.append(f"{method:12s} R={R:3} {attn:18s} {status:9s} final_vis={final} expected={expected} {tag} layers={vis}")
    except Exception as exc:
        lines.append(f"{path}: FAIL parse error: {exc}")
os.makedirs("results/smoke", exist_ok=True)
report = "\n".join(lines) + "\n"
open("results/smoke/smoke_report.txt", "w").write(report)
print(report, end="")
if any("FAIL" in line or "MISMATCH" in line for line in lines):
    raise SystemExit(2)
