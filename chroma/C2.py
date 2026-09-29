"""
C2 - Compute top-2 most similar sentences for 10 queries in Chroma using HNSW (L2 and Cosine).
"""
import json
import math
import os
import platform
import time
from pathlib import Path

import chromadb
import numpy as np
from chromadb.config import Settings
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

RESULTS_DIR = ROOT / "results"
QUERIES_PATH = ROOT / "query_sentences.json"
PG_RESULTS_PATH = RESULTS_DIR / "postgres_P2.json"
CHROMA_PATH = ROOT / "chroma_db"

METRICS = ["l2", "cosine"]
TOP_K = 2


def stats(times):
    a = np.asarray(times)
    return {
        "min": float(a.min()),
        "max": float(a.max()),
        "mean": float(a.mean()),
        "std": float(a.std())
    }

def main():
    with open(QUERIES_PATH, encoding="utf-8") as f:
        queries = json.load(f)

    client = chromadb.PersistentClient(
        path=CHROMA_PATH, settings=Settings(anonymized_telemetry=False))

    results, all_stats = {}, {}
    for metric in METRICS:
        col = client.get_collection(f"corpus_{metric}", embedding_function=None)

        # Warm-up run
        col.query(query_embeddings=[col.get(ids=[str(queries[0]["id"])], include=["embeddings"])["embeddings"][0]], n_results=TOP_K + 1)

        times, per_query = [], []
        for q in queries:
            t0 = time.perf_counter()
            got = col.get(ids=[str(q["id"])], include=["embeddings"])
            t1 = time.perf_counter()
            res = col.query(query_embeddings=[got["embeddings"][0]], n_results=TOP_K + 1, include=["documents", "distances"])
            t2 = time.perf_counter()

            times.append(t2 - t0)

            neighbors = []
            for i, doc, d in zip(res["ids"][0], res["documents"][0], res["distances"][0]):
                if i != str(q["id"]):
                    dist = math.sqrt(max(d, 0.0)) if metric == "l2" else float(d)
                    neighbors.append({"id": int(i), "text": doc, "distance_raw": float(d), "distance": dist})
                if len(neighbors) == TOP_K:
                    break

            per_query.append({
                "query_id": q["id"],
                "query_text": q["text"],
                "neighbors": neighbors,
                "time_seconds": times[-1],
                "time_get_seconds": t1 - t0,
                "time_query_seconds": t2 - t1
            })

        results[metric] = per_query
        all_stats[metric] = stats(times)

    for metric in METRICS:
        print(f"\n=== Metric: {metric.upper()} ===")
        for r in results[metric]:
            print(f"[{r['query_id']}] {r['query_text'][:70]}")
            for n in r["neighbors"]:
                print(f"     -> id {n['id']} (d={n['distance']:.4f}): {n['text'][:70]}")
        s = all_stats[metric]
        print(f"\n--- [C2] TOP-{TOP_K} QUERY TIME ({metric.upper()}, {len(queries)} queries, get+query) ---")
        print(f"Min:  {s['min'] * 1000:.3f} ms")
        print(f"Max:  {s['max'] * 1000:.3f} ms")
        print(f"Mean: {s['mean'] * 1000:.3f} ms")
        print(f"Standard deviation: {s['std'] * 1000:.3f} ms")
        print(f"Total time: {sum(r['time_seconds'] for r in results[metric]) * 1000:.3f} ms")

    same = sum(
        [n["id"] for n in ra["neighbors"]] == [n["id"] for n in rb["neighbors"]]
        for ra, rb in zip(results["l2"], results["cosine"])
    )
    print(f"\nQueries with identical top-{TOP_K} ranking between L2 and Cosine: {same}/{len(queries)}")

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "chroma_C2.json", "w", encoding="utf-8") as f:
        json.dump({
            "chromadb": chromadb.__version__,
            "python": platform.python_version(),
            "machine": platform.platform(),
            "metrics": METRICS,
            "stats_seconds": all_stats,
            "results": results
        }, f, indent=2)

if __name__ == "__main__":
    main()