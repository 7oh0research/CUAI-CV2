"""Load MME and POPE samples for H3. Supports common local POPE JSONL/CSV layouts."""
from pathlib import Path
import csv, json


def load_mme(root):
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"MME_ROOT is not a directory: {root}")
    items = []
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        qa_dir = sub / "questions_answers_YN"
        img_dir = sub / "images" if (sub / "images").is_dir() else sub
        txts = sorted(qa_dir.glob("*.txt")) if qa_dir.is_dir() else sorted(sub.glob("*.txt"))
        for txt in txts:
            stem = txt.stem
            img = next((img_dir / f"{stem}{ext}" for ext in (".jpg", ".png", ".jpeg", ".JPG", ".PNG") if (img_dir / f"{stem}{ext}").is_file()), None)
            if img is None:
                continue
            for i, line in enumerate(txt.read_text(encoding="utf-8").splitlines()):
                if not line.strip():
                    continue
                parts = line.split("\t")
                if len(parts) < 2:
                    continue
                items.append({"id": f"MME:{sub.name}/{stem}/{i}", "dataset": "MME", "image": str(img),
                              "question": parts[0].strip(), "answer": parts[1].strip(), "subtask": sub.name})
    if not items:
        raise RuntimeError(f"No MME QA/image pairs found under {root}")
    return items


def _resolve_image(value, root):
    p = Path(str(value))
    if p.is_file():
        return str(p.resolve())
    candidates = [Path(root) / p, Path(root) / "images" / p, Path(root) / "coco" / "val2014" / p]
    for c in candidates:
        if c.is_file():
            return str(c.resolve())
    # POPE records often contain COCO image names; recursively search once as fallback.
    matches = list(Path(root).rglob(p.name)) if p.name else []
    return str(matches[0].resolve()) if matches else None


def load_pope(root, explicit_file=None):
    root = Path(root)
    candidates = [Path(explicit_file)] if explicit_file else []
    if root.is_file():
        candidates.append(root)
    elif root.is_dir():
        candidates += [root / n for n in ("coco_pope.json", "pope.json", "pope_test.json", "questions.jsonl", "questions.csv")]
        candidates += sorted(root.glob("*.jsonl")) + sorted(root.glob("*.csv")) + sorted(root.glob("*.json"))
    path = next((p for p in candidates if p and p.is_file()), None)
    if path is None:
        raise FileNotFoundError("POPE data file not found. Set POPE_FILE to a local JSONL/CSV/JSON file.")
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as f: rows = list(csv.DictReader(f))
    elif path.suffix.lower() == ".json":
        obj = json.loads(path.read_text(encoding="utf-8")); rows = obj if isinstance(obj, list) else obj.get("data", obj.get("questions", []))
    else:
        rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    items = []
    for i, r in enumerate(rows):
        image = r.get("image", r.get("image_path", r.get("image_id", r.get("filename"))))
        question = r.get("question", r.get("text"))
        answer = r.get("answer", r.get("label", r.get("ground_truth")))
        if image is None or question is None or answer is None:
            continue
        image_path = _resolve_image(image, root.parent if root.is_file() else root)
        if not image_path:
            continue
        items.append({"id": f"POPE:{r.get('question_id', i)}", "dataset": "POPE", "image": image_path,
                      "question": str(question).strip(), "answer": str(answer).strip(), "subtask": "POPE"})
    if not items:
        raise RuntimeError(f"No usable POPE rows found in {path}. Expected image/question/answer fields.")
    return items


def load_h3_data(mme_root, pope_root=None, pope_file=None, max_mme=0, max_pope=0, seed=42):
    import random
    mme = load_mme(mme_root)
    pope = load_pope(pope_root, pope_file) if pope_root or pope_file else []
    rng = random.Random(seed)
    if max_mme and len(mme) > max_mme: mme = sorted(rng.sample(mme, max_mme), key=lambda x: x["id"])
    if max_pope and len(pope) > max_pope: pope = sorted(rng.sample(pope, max_pope), key=lambda x: x["id"])
    return mme + pope
