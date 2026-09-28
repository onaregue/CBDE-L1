"""
C0 - Load text into Chroma with implicit embedding generation
"""
import json
import os
import platform
import time
from pathlib import Path

import chromadb
import numpy as np
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

RESULTS_DIR = ROOT / "results"
CHROMA_PATH = ROOT / "chroma_db_c0"
COLLECTION = "corpus_c0"
BATCH_SIZE = int(os.getenv("BATCH_SIZE"))

METRIC = "l2"
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

    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=MODEL_NAME, device="cpu")
    ef(sentences[:8])  # calentamiento (no se mide)

    client = chromadb.PersistentClient(
        path=CHROMA_PATH, settings=Settings(anonymized_telemetry=False))
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    col = client.create_collection(name=COLLECTION, metadata={"hnsw:space": METRIC}, embedding_function=ef)

    times = []
    n_calls = 0
    for start in range(0, len(sentences), BATCH_SIZE):
        chunk = sentences[start:start + BATCH_SIZE]
        t0 = time.perf_counter()
        col.add(ids=[str(start + j) for j in range(len(chunk))], documents=chunk)
        times.append(time.perf_counter() - t0)
        n_calls += 1

    n_rows = col.count()
    size = dir_size_mb(CHROMA_PATH)
    s = stats(times)

    print(f"\nRecords in Chroma: {n_rows} | add() calls: {n_calls} | metric: {METRIC} | disk size: {size:.1f} MB")
    print(f"\n--- [C0] TEXT + IMPLICIT EMBEDDING (per batch of {BATCH_SIZE}, device=cpu) ---")
    print("(coupled operation: add() computes embeddings, writes documents and updates HNSW index)")
    print(f"Min:  {s['min']:.6f} s")
    print(f"Max:  {s['max']:.6f} s")
    print(f"Mean: {s['mean']:.6f} s")
    print(f"Standard deviation: {s['std']:.6f} s")
    print(f"Total time: {sum(times):.4f} s")

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "chroma_C0.json", "w", encoding="utf-8") as f:
        json.dump({
            "chromadb": chromadb.__version__,
            "python": platform.python_version(),
            "machine": platform.platform(),
            "model": MODEL_NAME,
            "device": "cpu",
            "metric": METRIC,
            "batch_size": BATCH_SIZE,
            "rows": n_rows,
            "add_calls": n_calls,
            "size_mb": size,
            "stats_seconds": s,
            "batch_times_seconds": times
        }, f, indent=2)



if __name__ == "__main__":
    main()