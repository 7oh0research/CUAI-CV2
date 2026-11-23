import os, random, glob

def load_mme(root):
    """MME_Benchmark_release_version 구조를 읽어 (id, image_path, question, answer, subtask) 목록 반환.
    두 가지 폴더 구조(이미지와 txt가 같은 폴더 / images, questions_answers_YN 하위 폴더)를 모두 처리."""
    items = []
    for sub in sorted(os.listdir(root)):
        d = os.path.join(root, sub)
        if not os.path.isdir(d):
            continue
        qa_dir = os.path.join(d, "questions_answers_YN")
        img_dir = os.path.join(d, "images")
        if os.path.isdir(qa_dir):
            txts = glob.glob(os.path.join(qa_dir, "*.txt"))
        else:
            txts = glob.glob(os.path.join(d, "*.txt")); img_dir = d
        for t in sorted(txts):
            stem = os.path.splitext(os.path.basename(t))[0]
            img = None
            for ext in (".jpg", ".png", ".jpeg", ".JPG", ".PNG"):
                p = os.path.join(img_dir, stem + ext)
                if os.path.exists(p):
                    img = p; break
            if img is None:
                continue
            with open(t, encoding="utf-8") as f:
                for i, line in enumerate(l for l in f if l.strip()):
                    q, a = line.rstrip("\n").split("\t")[:2]
                    items.append({"id": f"{sub}/{stem}/{i}", "image": img, "question": q,
                                  "answer": a.strip(), "subtask": sub})
    return items

def eff_subset(root, n=200, seed=0, ids_path="data/eff_ids.txt"):
    items = load_mme(root)
    if os.path.exists(ids_path):
        ids = [l.strip() for l in open(ids_path) if l.strip()]
        m = {x["id"]: x for x in items}
        return [m[i] for i in ids]
    sel = random.Random(seed).sample(items, n)
    os.makedirs(os.path.dirname(ids_path), exist_ok=True)
    with open(ids_path, "w") as f:
        f.write("\n".join(x["id"] for x in sel))
    return sel
