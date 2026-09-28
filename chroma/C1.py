"""
C1 - External embedding generation and decoupled storage in Chroma (for L2 and Cosine collections).
"""
import json
import os
import platform
import time
from pathlib import Path

import chromadb
import numpy as np
from chromadb.config import Settings
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

RESULTS_DIR = ROOT / "results"
CHROMA_PATH = ROOT / "chroma_db"
BATCH_SIZE = int(os.getenv("BATCH_SIZE"))

METRICS = ["l2", "cosine"]
MODEL_NAME = "all-MiniLM-L6-v2"


def stats(times):
    a = np.asarray(times)
    return {
        "min": float(a.min()),
        "max": float(a.max()),
        "mean": float(a.mean()),
        "std": float(a.std())
    }

def dir_size_mb(path):
    return sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file()) / 1e6

def main():
    
    with open(ROOT / "bookcorpus_sentences.json", encoding="utf-8") as f:
        sentences = json.load(f)

    model = SentenceTransformer(MODEL_NAME, device="cpu")
    model.encode(sentences[:8], show_progress_bar=False)  # calentamiento (no se mide)

    client = chromadb.PersistentClient(
        path=CHROMA_PATH, settings=Settings(anonymized_telemetry=False))
    cols = {}
    for m in METRICS:
        try:
            client.delete_collection(f"corpus_{m}")
        except Exception:
            pass
        cols[m] = client.create_collection(name=f"corpus_{m}", metadata={"hnsw:space": m}, embedding_function=None)
    
    gen_times = []
    store_times = {m: [] for m in METRICS}
    n_calls = 0

    for start in range(0, len(sentences), BATCH_SIZE):
        texts = sentences[start:start + BATCH_SIZE]

        # external embedding generation
        t0 = time.perf_counter()
        embs = model.encode(texts, batch_size=64, show_progress_bar=False)
        gen_times.append(time.perf_counter() - t0)

        # store embeddings and texts in each collection
        for m in METRICS:
            t0 = time.perf_counter()
            cols[m].add(ids=[str(start + j) for j in range(len(texts))],
                        embeddings=list(embs), documents=texts)
            store_times[m].append(time.perf_counter() - t0)
            n_calls += 1

    n_rows = {m: cols[m].count() for m in METRICS}
    size = dir_size_mb(CHROMA_PATH)
    gen_s = stats(gen_times)
    store_s = {m: stats(store_times[m]) for m in METRICS}

    print(f"\nRecords: {n_rows} | add() calls: {n_calls} | Disk size (all collections): {size:.1f} MB")

    print(f"\n--- [C1] EMBEDDING GENERATION (per batch of {BATCH_SIZE}, device=cpu) ---")
    print(f"Min:  {gen_s['min']:.6f} s")
    print(f"Max:  {gen_s['max']:.6f} s")
    print(f"Mean: {gen_s['mean']:.6f} s")
    print(f"Standard deviation: {gen_s['std']:.6f} s")
    print(f"Total time: {sum(gen_times):.4f} s")

    for m in METRICS:
        s = store_s[m]
        print(f"\n--- [C1] STORAGE vector+text (collection {m.upper()}, per batch) ---")
        print(f"Min:  {s['min']:.6f} s")
        print(f"Max:  {s['max']:.6f} s")
        print(f"Mean: {s['mean']:.6f} s")
        print(f"Standard deviation: {s['std']:.6f} s")
        print(f"Total time: {sum(store_times[m]):.4f} s")

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "chroma_C1.json", "w", encoding="utf-8") as f:
        json.dump({
            "chromadb": chromadb.__version__,
            "python": platform.python_version(),
            "machine": platform.platform(),
            "model": MODEL_NAME,
            "device": "cpu",
            "metrics": METRICS,
            "batch_size": BATCH_SIZE,
            "rows": n_rows,
            "add_calls": n_calls,
            "size_mb": size,
            "generation_stats_seconds": gen_s,
            "storage_stats_seconds": store_s,
            "generation_times_seconds": gen_times,
            "storage_times_seconds": store_times
        }, f, indent=2)


if __name__ == "__main__":
    main()