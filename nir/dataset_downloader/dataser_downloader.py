import json
from pathlib import Path
from datasets import load_dataset

BASE = Path("G:/ITMO/NIR/DATASETS")
SAVE = BASE / "processed"
CACHE = BASE / "hf_cache"
SAVE.mkdir(exist_ok=True, parents=True)
CACHE.mkdir(exist_ok=True, parents=True)

def save_jsonl(data, file):
    with open(file, "w", encoding="utf-8") as fw:
        for x in data:
            json.dump(x, fw, ensure_ascii=False)
            fw.write("\n")

# ===== HumanEval =====
def fetch_humaneval(n=100):
    print("🔹 Loading HumanEval…")
    ds = load_dataset("openai_humaneval", split="test", cache_dir=str(CACHE))
    out = []
    for i, x in enumerate(ds):
        if i >= n: break
        out.append({"input": x["prompt"], "target": x["canonical_solution"]})
    save_jsonl(out, SAVE / "humaneval.jsonl")
    print(f"✅ Saved {len(out)} examples to humaneval.jsonl")

# ===== CoNaLa (curated) =====
def fetch_conala(n=100):
    print("🔹 Loading CoNaLa curated…")
    ds = load_dataset("neulab/conala", split="train", cache_dir=str(CACHE), trust_remote_code=True)
    out = []
    for i, x in enumerate(ds):
        if i >= n: break
        if x.get("snippet") and x.get("rewritten_intent"):
            out.append({"input": x["rewritten_intent"], "target": x["snippet"]})
    save_jsonl(out, SAVE / "conala.jsonl")
    print(f"✅ Saved {len(out)} examples to conala.jsonl")

# ===== CoNaLa mined (600k, берём 5000) =====
def fetch_conala_mined(n=5000):
    print("🔹 Loading CoNaLa mined-curated…")
    ds = load_dataset("codeparrot/conala-mined-curated", split="train", cache_dir=str(CACHE))
    out = []
    for i, x in enumerate(ds):
        if i >= n: break
        if x.get("intent") and x.get("snippet"):
            out.append({"input": x["intent"], "target": x["snippet"]})
    save_jsonl(out, SAVE / "conala_mined.jsonl")
    print(f"✅ Saved {len(out)} examples to conala_mined.jsonl")

# ===== MBPP =====
def fetch_mbpp():
    print("🔹 Loading MBPP…")
    ds = load_dataset("mbpp", split="train", cache_dir=str(CACHE))
    out = []
    for x in ds:
        if x.get("text") and x.get("code"):
            out.append({"input": x["text"], "target": x["code"]})
    save_jsonl(out, SAVE / "mbpp.jsonl")
    print(f"✅ Saved {len(out)} examples to mbpp.jsonl")


if __name__ == "__main__":
    fetch_humaneval(100)
    fetch_conala(100)
    fetch_conala_mined(5000)
    fetch_mbpp()
