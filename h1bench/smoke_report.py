import json, glob, yaml
cfg = yaml.safe_load(open("configs/methods.yaml"))
lines = []
for p in sorted(glob.glob("results/smoke/*.jsonl")):
    rows = [json.loads(l) for l in open(p)]
    s = rows[0]; b = [r for r in rows if r["type"] == "batch"]
    if s["status"] in ("oom", "failed", "unsupported"):
        lines.append(f"{s['method']:12s} R={s['R']:3d} {s['attn_requested']:18s} {s['status'].upper()} {s.get('error','')[:150]}"); continue
    vis = b[0]["layer_visual_tokens"]
    final = vis[-1]
    exp = 576 if s["method"] == "baseline" else s["R"]
    tag = "OK" if final == exp else "MISMATCH"
    lines.append(f"{s['method']:12s} R={s['R']:3d} {s['attn_requested']:18s} {s['status']:9s} final_vis={final} expected={exp} {tag} layers={vis}")
open("results/smoke/smoke_report.txt", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
