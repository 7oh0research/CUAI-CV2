"""각 데이터셋의 필드 구조를 출력한다. 로더가 기대하는 필드와 다르면 h3/data.py를 수정해야 한다."""
import yaml
from datasets import load_dataset
cfg = yaml.safe_load(open("configs/default.yaml"))
for name, b in cfg["benchmarks"].items():
    try:
        ds = load_dataset(b["hf_id"], b.get("hf_config"), split=b["split"], streaming=True)
        ex = next(iter(ds))
        print(f"== {name}: {b['hf_id']} [{b['split']}]")
        for k, v in ex.items():
            s = type(v).__name__ if k == "image" else repr(v)[:80]
            print(f"   {k}: {s}")
    except Exception as e:
        print(f"== {name}: 실패 -> {e}")
